"""
NetrAi Classifier — Core Model
================================
Architecture:
  Image (512×512)  +  Anomaly Map (512×512, [0,1])
        │                      │
   MIT-B3 Encoder              │
   Custom Decode Head          │
   → F_concat (B, 1024, 128, 128)
        │                      │
        └──── Late Spatial Gate ┘
              F_gated = F + α·(F ⊙ A)
              α: learned scalar
                     │
              Global Avg Pool → (B, 1024)
                     │
        ┌────────────┴────────────┐
     Path A                   Path B (VIB)
  Linear(1024→384)         Linear(1024→768)
      384-D                  μ(384) + log_σ(384)
      (raw ctx)              z = μ + σε  [train]
                             z = μ       [infer]
        └────────────┬────────────┘
                     │ + global scalar (1-D)
                     │   mean(A_scaled)
                   cat → 769-D vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


# ---------------------------------------------------------------------------
# Decode Head
# ---------------------------------------------------------------------------

class SegFormerDecodeHead(nn.Module):
    """
    Custom decode head for feature extraction (not segmentation).

    Takes the 4 spatial hidden states from the MIT-B3 encoder:
      Stage 0: (B, 64,  H/4,  W/4 )
      Stage 1: (B, 128, H/8,  W/8 )
      Stage 2: (B, 320, H/16, W/16)
      Stage 3: (B, 512, H/32, W/32)

    Projects each to `decoder_embed_dim` channels, upsamples all to H/4×W/4,
    concatenates and fuses → F_concat of shape (B, 4·decoder_embed_dim, H/4, W/4).
    """

    def __init__(
        self,
        in_channels: tuple = (64, 128, 320, 512),
        decoder_embed_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.n_stages  = len(in_channels)

        # Per-stage linear projections (applied on flattened spatial tokens)
        self.linear_c = nn.ModuleList([
            nn.Linear(c, decoder_embed_dim, bias=False) for c in in_channels
        ])

        # Lightweight channel fusion after concatenation
        out_ch = decoder_embed_dim * self.n_stages
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, hidden_states: tuple) -> torch.Tensor:
        """
        hidden_states: tuple of 4 tensors, each (B, C_i, H_i, W_i)
        Returns: F_concat  (B, 4·decoder_embed_dim, H/4, W/4)
        """
        target_h, target_w = hidden_states[0].shape[-2:]  # H/4, W/4

        projected = []
        for feat, linear in zip(hidden_states, self.linear_c):
            B, C, H, W = feat.shape
            # (B, C, H, W) → (B, H·W, C) → project → (B, H·W, E) → (B, E, H, W)
            x = feat.flatten(2).transpose(1, 2)   # (B, H·W, C)
            x = linear(x)                          # (B, H·W, E)
            x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

            if (H, W) != (target_h, target_w):
                x = F.interpolate(
                    x, size=(target_h, target_w),
                    mode='bilinear', align_corners=False,
                )
            projected.append(x)

        F_concat = torch.cat(projected, dim=1)     # (B, 4E, H/4, W/4)
        return self.fuse(F_concat)


# ---------------------------------------------------------------------------
# Late Spatial Gate
# ---------------------------------------------------------------------------

class LateSpatialGate(nn.Module):
    """
    Merges diffusion model output with SegFormer features.

        F_gated = F_concat + α · (F_concat ⊙ A_scaled)

    α is a single learned scalar initialised to 1.0.
    Residual connection (+ F_concat) ensures healthy tissue is preserved.

    Note on sign semantics:
        F_concat values can be negative (post-GELU / LayerNorm).
        A_scaled ∈ [0, 1] ⟹ multiplication amplifies magnitude in whatever
        direction each feature already points — sign is never flipped.
    """

    def __init__(self, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self, F_concat: torch.Tensor, anomaly_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        F_concat:    (B, C, H, W)
        anomaly_map: (B, 1, H_orig, W_orig)  ∈ [0, 1]

        Returns:
            F_gated  (B, C, H, W)
            A_scaled (B, 1, H, W)   — resampled anomaly map at feature resolution
        """
        A_scaled = F.interpolate(
            anomaly_map,
            size=F_concat.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )  # (B, 1, H/4, W/4)

        F_gated = F_concat + self.alpha * (F_concat * A_scaled)
        return F_gated, A_scaled


# ---------------------------------------------------------------------------
# VIB Path
# ---------------------------------------------------------------------------

class VIBPath(nn.Module):
    """
    Variational Information Bottleneck.

    Training:  z = μ + σ ⊙ ε,  ε ~ N(0, I)
    Inference: z = μ            (ε = 0, fully deterministic)

    The KL divergence penalty KL(q(z|x) ∥ N(0,I)) is computed externally
    in losses.py using (μ, log_σ²).
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 384):
        super().__init__()
        self.mu_head      = nn.Linear(in_dim, out_dim)
        self.log_var_head = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, in_dim)
        Returns: z (B, out_dim), mu (B, out_dim), log_var (B, out_dim)
        """
        mu      = self.mu_head(x)
        log_var = self.log_var_head(x)

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z   = mu + std * eps
        else:
            z = mu  # deterministic at inference

        return z, mu, log_var


# ---------------------------------------------------------------------------
# Dual-Path Bottleneck
# ---------------------------------------------------------------------------

class DualPathBottleneck(nn.Module):
    """
    Compresses pooled F_gated (1024-D) into the final 769-D vector.

    Path A → Linear(1024, 384) → 384-D  raw contextual features
    Path B → VIB               → 384-D  μ  (strongest disease signals)
    Scalar → mean(A_scaled)    →   1-D  global anomaly severity

    Concatenated: [384 | 384 | 1] = 769-D
    """

    def __init__(self, in_dim: int = 1024, path_dim: int = 384):
        super().__init__()
        self.path_a = nn.Linear(in_dim, path_dim)
        self.path_b = VIBPath(in_dim=in_dim, out_dim=path_dim)

    def forward(
        self, F_gated: torch.Tensor, A_scaled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        F_gated:  (B, C, H, W)
        A_scaled: (B, 1, H, W)

        Returns:
            vector_769 (B, 769)
            mu         (B, 384)   ← target of Ortho penalty
            log_var    (B, 384)   ← target of KL loss
        """
        # Global average pool: (B, C, H, W) → (B, C)
        pooled = F.adaptive_avg_pool2d(F_gated, 1).flatten(1)

        # Path A — raw context (NO orthogonal penalty applied here)
        a_out = self.path_a(pooled)                        # (B, 384)

        # Path B — VIB
        z, mu, log_var = self.path_b(pooled)               # (B, 384)

        # Global anomaly severity scalar
        scalar = A_scaled.mean(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
        scalar = scalar.view(-1, 1)                            # (B, 1)

        # Concatenate → 769-D
        vector_769 = torch.cat([a_out, z, scalar], dim=1)

        return vector_769, mu, log_var


# ---------------------------------------------------------------------------
# NetrAi Encoder (full model)
# ---------------------------------------------------------------------------

class NetrAiEncoder(nn.Module):
    """
    Full NetrAi feature extractor.

    Inputs:
        image       (B, 3, 512, 512)  — ImageNet-normalised retina image
        anomaly_map (B, 1, 512, 512)  — Clean residual heatmap ∈ [0, 1]

    Outputs:
        vector_769  (B, 769)  — final feature vector for downstream XGBoost
        mu          (B, 384)  — VIB mean   (used by Ortho loss)
        log_var     (B, 384)  — VIB log σ² (used by KL  loss)
    """

    MIT_B3_CHANNELS = (64, 128, 320, 512)

    def __init__(
        self,
        backbone_name:      str   = "nvidia/mit-b3",
        decoder_embed_dim:  int   = 256,
        path_a_dim:         int   = 384,
        path_b_dim:         int   = 384,
        alpha_init:         float = 1.0,
    ):
        super().__init__()

        # MIT-B3 encoder — ImageNet pretrained
        self.encoder = SegformerModel.from_pretrained(
            backbone_name,
            output_hidden_states=True,
        )

        # F_concat channels: 4 × decoder_embed_dim = 1024
        self.decoder = SegFormerDecodeHead(
            in_channels=self.MIT_B3_CHANNELS,
            decoder_embed_dim=decoder_embed_dim,
        )

        # Late spatial gate
        self.gate = LateSpatialGate(alpha_init=alpha_init)

        # Dual-path bottleneck: 1024 → 769
        bottleneck_in = decoder_embed_dim * len(self.MIT_B3_CHANNELS)  # 1024
        self.bottleneck = DualPathBottleneck(
            in_dim=bottleneck_in,
            path_dim=path_b_dim,  # path_a_dim == path_b_dim == 384
        )

        # Override path_a projection if dims differ
        if path_a_dim != path_b_dim:
            self.bottleneck.path_a = nn.Linear(bottleneck_in, path_a_dim)

    def forward(
        self,
        image:       torch.Tensor,
        anomaly_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Encode — get 4 spatial feature maps
        enc_out = self.encoder(
            pixel_values=image,
            output_hidden_states=True,
        )
        # hidden_states: tuple of 4 tensors, each (B, C_i, H_i, W_i)
        hidden_states = enc_out.hidden_states

        # 2. Decode → F_concat (B, 1024, H/4, W/4)
        F_concat = self.decoder(hidden_states)

        # 3. Late spatial gate
        F_gated, A_scaled = self.gate(F_concat, anomaly_map)

        # 4. Dual-path bottleneck → 769-D
        vector_769, mu, log_var = self.bottleneck(F_gated, A_scaled)

        return vector_769, mu, log_var


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NetrAiEncoder().to(device)

    dummy_img   = torch.randn(2, 3, 512, 512, device=device)
    dummy_amap  = torch.rand (2, 1, 512, 512, device=device)

    model.train()
    v, mu, lv = model(dummy_img, dummy_amap)
    print(f"[TRAIN] vector: {v.shape}  mu: {mu.shape}  log_var: {lv.shape}")
    assert v.shape == (2, 769), f"Expected (2, 769), got {v.shape}"

    model.eval()
    with torch.no_grad():
        v, mu, lv = model(dummy_img, dummy_amap)
    print(f"[INFER] vector: {v.shape}  mu: {mu.shape}  log_var: {lv.shape}")
    print("model.py — all checks passed ✓")
