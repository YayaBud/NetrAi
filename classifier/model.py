"""
NetrAi Classifier — Core Model (v2)
=====================================
Architecture:

  Stream A: 6-Channel Stack (3ch RGB + 3ch Clean Residual)
                      │
            [ MIT-B3 SegFormer ]  (6-ch input, trains)
                      │
      ┌───────────────┼───────────────┐
   Scale[0]       Scale[2]        Scale[3]
  (128×128×64)  (32×32×320)    (16×16×512)
      │               │               │
  [ DR Head ]   [ Glauc Head ]   [ PM Head ]
  1x1→3x3→CBAM  1x1→3x3→CBAM   SE→GAP→MLP
     Pool            Pool
      │               │               │
    256-D           256-D           256-D
      └───────────────┴───────────────┘
                      │
                   768-D
                      │
                  [ VIB 1 ]
                 (768→256→128)
                      │
             128-D (z1, μ1, logσ²1)
                 ┌────┴────┐
           [Aux Linear]     │              Stream B: pre-cached RETFound
            (Training)      │              1024-D embedding from disk
                 │          │                       │
              (L_aux)       │                   [ VIB 2 ]
                            │              (1024→256→128)
                            │                       │
                            │            128-D (z2, μ2, logσ²2)
                            │                       │
                    z1 (128) ⊕ z2 (128) = 256-D fused
                                    │
                            [ Main Linear ]
                              (Training)
                                    │
                                (L_main)

After training: drop Aux + Main classifiers.
Freeze all. Extract 256-D → XGBoost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


# ---------------------------------------------------------------------------
# CBAM — Channel + Spatial Attention
# ---------------------------------------------------------------------------

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).
    Applies channel attention then spatial attention sequentially.
    ~2 × C × (C / reduction) parameters — very lightweight.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)

        # Shared MLP for channel attention (applied to avg-pool and max-pool)
        self.ca_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # 7×7 conv for spatial attention
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Channel attention ─────────────────────────────────────────────
        avg_c = F.adaptive_avg_pool2d(x, 1).view(B, C)    # (B, C)
        max_c = F.adaptive_max_pool2d(x, 1).view(B, C)    # (B, C)
        ca    = torch.sigmoid(
            self.ca_mlp(avg_c) + self.ca_mlp(max_c)
        ).view(B, C, 1, 1)
        x = x * ca

        # ── Spatial attention ─────────────────────────────────────────────
        avg_s = x.mean(dim=1, keepdim=True)               # (B, 1, H, W)
        max_s = x.amax(dim=1, keepdim=True)               # (B, 1, H, W)
        sa    = torch.sigmoid(
            self.sa_conv(torch.cat([avg_s, max_s], dim=1))
        )
        return x * sa


# ---------------------------------------------------------------------------
# SE Block — Channel-only attention (no spatial conv)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    Pure channel re-weighting via global average pooling.
    Correct inductive bias for PM: "which channels fire globally?"
    rather than "where do they fire?" — no spatial convolutions.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Squeeze
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),              # Excitation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Expert Heads
# ---------------------------------------------------------------------------

class DRHead(nn.Module):
    """
    Diabetic Retinopathy Expert Head.

    Routed from Scale 0 (1/4 resolution, 128×128×64).
    High resolution is necessary: DR features are micro-aneurysms,
    dot haemorrhages, and hard exudates — small and focal.

    1×1 conv (channel expand) → 3×3 conv → BatchNorm+GELU
    → CBAM (spatial + channel attention) → GAP → FC → 256-D
    """

    def __init__(self, in_channels: int = 64, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        mid = 128
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.attn = CBAM(mid, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(mid, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class GlaucHead(nn.Module):
    """
    Glaucoma Expert Head.

    Routed from Scale 2 (1/16 resolution, 32×32×320).
    Mid resolution isolates the optic disc region well.
    Glaucoma signature: cup-to-disc ratio enlargement, rim thinning.

    1×1 conv (channel reduce) → 3×3 conv → BatchNorm+GELU
    → CBAM → GAP → FC → 256-D
    """

    def __init__(self, in_channels: int = 320, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        mid = 256
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.attn = CBAM(mid, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(mid, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attn(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class PMHead(nn.Module):
    """
    Pathological Myopia Expert Head.

    Routed from Scale 3 (1/32 resolution, 16×16×512).
    PM is a GLOBAL structural deformation — axial elongation, myopic
    crescent, posterior staphyloma. SegFormer's Scale 3 features already
    encode global self-attention context. Applying spatial convolutions
    here is redundant and adds wrong inductive bias.

    Instead: SE-Block (pure channel re-weighting) → GAP → MLP → 256-D
    Asks: "which channels indicate global structural deformation?"
    NOT: "where are the deformations?"
    """

    def __init__(self, in_channels: int = 512, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.se   = SEBlock(in_channels, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# VIB Module
# ---------------------------------------------------------------------------

class VIB(nn.Module):
    """
    Variational Information Bottleneck.

    Training:  z = μ + σ ⊙ ε,  ε ~ N(0, I)
        Noise forces the network to encode robust, generalizable features.
        Cannot free-ride on any single dimension.

    Inference (eval mode):  z = μ
        Deterministic — XGBoost requires static, stable tabular vectors.
        Stochastic embeddings would destroy tree split consistency.

    Architecture:
        in_dim → Linear(in_dim, hidden_dim) → GELU
                        ↓
              ┌─── mu_head(hidden_dim, out_dim)      → μ
              └── log_var_head(hidden_dim, out_dim)  → log σ²

    KL divergence KL(q(z|x) ∥ N(0,I)) is returned to losses.py.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, out_dim)
        self.log_var_head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, in_dim)
        Returns:
            z       (B, out_dim)  — sampled (train) or deterministic μ (eval)
            mu      (B, out_dim)  — VIB mean for KL loss
            log_var (B, out_dim)  — VIB log σ² for KL loss
        """
        h       = self.pre(x)
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h)

        if self.training:
            std = torch.exp(0.5 * log_var).clamp(max=10.0)   # clamp prevents exploding σ
            eps = torch.randn_like(std)
            z   = mu + std * eps
        else:
            z = mu   # deterministic μ — noise off for XGBoost extraction

        return z, mu, log_var


# ---------------------------------------------------------------------------
# NetrAi Encoder v2
# ---------------------------------------------------------------------------

class NetrAiEncoder(nn.Module):
    """
    Full NetrAi dual-stream feature extractor.

    Input isolation (hard rule):
        six_ch       → MIT-B3 SegFormer ONLY  (6-channel, trains)
        retfound_emb → VIB 2 ONLY             (pre-cached 1024-D, never touches SegFormer)

    Inputs:
        six_ch       (B, 6, 512, 512)  — stacked tensor: RGB + clean residual (each 3ch)
        retfound_emb (B, 1024)         — pre-cached frozen RETFound [CLS] embedding

    Outputs:
        z_fused  (B, 256)   — 128-D z1 ⊕ 128-D z2 (main XGBoost input after Phase 2)
        z1       (B, 128)   — VIB1 output (Aux Classifier input during Phase 1)
        mu1      (B, 128)   — VIB1 mean for KL1 loss
        log_var1 (B, 128)   — VIB1 log σ² for KL1 loss
        mu2      (B, 128)   — VIB2 mean for KL2 loss
        log_var2 (B, 128)   — VIB2 log σ² for KL2 loss

    Temporary classifiers (Phase 1 training scaffolding — dropped after):
        aux_classifier   Linear(128, 3)  — on z1 only  → L_aux
        main_classifier  Linear(256, 3)  — on z_fused  → L_main
    """

    # MIT-B3 channel counts per stage
    MIT_B3_STAGE_CHANNELS = (64, 128, 320, 512)

    def __init__(
        self,
        backbone_name: str   = "nvidia/mit-b3",
        head_out_dim:  int   = 256,
        vib_hidden:    int   = 256,
        vib_out_dim:   int   = 128,
        dropout:       float = 0.3,
    ):
        super().__init__()

        # ── MIT-B3 backbone ───────────────────────────────────────────────
        self.encoder = SegformerModel.from_pretrained(
            backbone_name,
            output_hidden_states=True,
        )
        self._adapt_input_channels(in_channels=6)

        # ── Scale routing → Expert heads ──────────────────────────────────
        C0, _C1, C2, C3 = self.MIT_B3_STAGE_CHANNELS
        self.dr_head    = DRHead(in_channels=C0,  out_dim=head_out_dim, dropout=dropout)
        self.glauc_head = GlaucHead(in_channels=C2, out_dim=head_out_dim, dropout=dropout)
        self.pm_head    = PMHead(in_channels=C3,  out_dim=head_out_dim, dropout=dropout)

        # ── Dual VIB ─────────────────────────────────────────────────────
        fused_in = head_out_dim * 3    # 768
        self.vib1 = VIB(in_dim=fused_in, hidden_dim=vib_hidden, out_dim=vib_out_dim)
        self.vib2 = VIB(in_dim=1024,     hidden_dim=vib_hidden, out_dim=vib_out_dim)

        # ── Temporary Phase-1 classifiers (dropped after training) ────────
        fused_out = vib_out_dim * 2    # 256
        self.aux_classifier  = nn.Linear(vib_out_dim, 3)   # on z1 alone → L_aux
        self.main_classifier = nn.Linear(fused_out,   3)   # on z_fused  → L_main

    def _adapt_input_channels(self, in_channels: int = 6) -> None:
        """
        Replace the first patch embedding projection to accept `in_channels`
        input channels instead of the pretrained 3.

        All other backbone weights are untouched.

        Initialisation strategy:
            Channels 0-2 (RGB):      copy pretrained weights exactly.
            Channels 3-5 (residual): zero-init so the model starts with
                                     exactly the pretrained RGB behaviour.
                                     Residual channels learn from scratch.
        """
        proj     = self.encoder.encoder.patch_embeddings[0].proj
        old_w    = proj.weight.data   # (out_ch, 3, kH, kW)
        out_ch   = proj.out_channels

        new_proj = nn.Conv2d(
            in_channels,
            out_ch,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=(proj.bias is not None),
        )

        with torch.no_grad():
            new_proj.weight[:, :3].copy_(old_w)      # preserve RGB pretrained weights
            new_proj.weight[:, 3:].zero_()            # residual channels start at zero
            if proj.bias is not None:
                new_proj.bias.copy_(proj.bias)

        self.encoder.encoder.patch_embeddings[0].proj = new_proj
        # Update config so SegformerModel knows about the channel count
        self.encoder.config.num_channels = in_channels

    def forward(
        self,
        six_ch:       torch.Tensor,   # (B, 6, 512, 512)
        retfound_emb: torch.Tensor,   # (B, 1024)
    ) -> tuple[torch.Tensor, ...]:
        """
        Returns:
            z_fused  (B, 256)
            z1       (B, 128)
            mu1      (B, 128)
            log_var1 (B, 128)
            mu2      (B, 128)
            log_var2 (B, 128)
        """
        # ── Stream A: SegFormer ───────────────────────────────────────────
        enc_out      = self.encoder(pixel_values=six_ch, output_hidden_states=True)
        hidden       = enc_out.hidden_states   # tuple: (s0, s1, s2, s3)

        # Scale routing (by disease inductive bias):
        #   s0 → (B,  64, H/4,  W/4 ) = (B,  64, 128, 128) → DR  (fine-grained)
        #   s2 → (B, 320, H/16, W/16) = (B, 320,  32,  32) → Glauc (optic disc)
        #   s3 → (B, 512, H/32, W/32) = (B, 512,  16,  16) → PM  (global shape)
        dr_feat    = self.dr_head(hidden[0])      # (B, 256)
        glauc_feat = self.glauc_head(hidden[2])   # (B, 256)
        pm_feat    = self.pm_head(hidden[3])      # (B, 256)

        fused_768 = torch.cat([dr_feat, glauc_feat, pm_feat], dim=1)  # (B, 768)

        # ── Dual VIB ─────────────────────────────────────────────────────
        z1, mu1, log_var1 = self.vib1(fused_768)       # (B, 128) each
        z2, mu2, log_var2 = self.vib2(retfound_emb)    # (B, 128) each

        # Fuse
        z_fused = torch.cat([z1, z2], dim=1)            # (B, 256)

        return z_fused, z1, mu1, log_var1, mu2, log_var2


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NetrAiEncoder().to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train / 1e6:.1f}M")

    dummy_six  = torch.randn(2, 6, 512, 512, device=device)
    dummy_retf = torch.randn(2, 1024, device=device)

    # ── Train mode ────────────────────────────────────────────────────────
    model.train()
    z_fused, z1, mu1, lv1, mu2, lv2 = model(dummy_six, dummy_retf)
    aux_logits  = model.aux_classifier(z1)
    main_logits = model.main_classifier(z_fused)

    assert z_fused.shape   == (2, 256), f"z_fused: {z_fused.shape}"
    assert z1.shape        == (2, 128), f"z1: {z1.shape}"
    assert aux_logits.shape == (2, 3),  f"aux: {aux_logits.shape}"
    assert main_logits.shape == (2, 3), f"main: {main_logits.shape}"
    print(f"[TRAIN] z_fused={z_fused.shape}  aux={aux_logits.shape}  main={main_logits.shape}")

    # ── Eval mode ─────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        z_fused, z1, mu1, lv1, mu2, lv2 = model(dummy_six, dummy_retf)
    assert z_fused.shape == (2, 256)
    print(f"[EVAL]  z_fused={z_fused.shape}")
    print("model.py — all checks passed ✓")
