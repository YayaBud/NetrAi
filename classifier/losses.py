"""
NetrAi Classifier — Loss Functions
=====================================
Three losses, applied during SegFormer training:

  L_total = L_SupCon(1.0) + λ_kl · L_KL(β) + λ_ortho · L_Ortho

L_SupCon  — Supervised Contrastive Loss (class-aware temperature)
              Applied to: full 769-D vector
L_KL      — VIB KL divergence with β-annealing
              Applied to: Path B  μ, log_var
L_Ortho   — Orthogonal Feature Projection (cosine similarity penalty)
              Applied to: Path B  μ ONLY (Path A stays untouched)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Supervised Contrastive Loss  (class-aware temperature)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020) with per-class
    temperature to handle class imbalance without aggressive oversampling.

    Standard formulation:
        L_i = -1/|P(i)| · Σ_{p∈P(i)}  log [
            exp(z_i·z_p / τ_i)
            ──────────────────────────────────
            Σ_{a≠i} exp(z_i·z_a / τ_i)
        ]

    τ_i = class_temperatures[label_i].  Lower τ → sharper penalty →
    minority class errors are punished more.

    Args:
        class_temperatures  dict  {class_idx: float}
                            e.g. {0: 0.07, 1: 0.07, 2: 0.04}
        base_temperature    float  normalisation constant (keeps loss scale
                                   comparable across temperature values)
    """

    def __init__(
        self,
        class_temperatures: dict,
        base_temperature:   float = 0.07,
    ):
        super().__init__()
        self.class_temperatures = class_temperatures
        self.base_temperature   = base_temperature

    def forward(
        self,
        features: torch.Tensor,   # (B, D) — the 769-D vectors (L2-normalised inside)
        labels:   torch.Tensor,   # (B,)   — integer class indices
    ) -> torch.Tensor:
        """
        Returns scalar SupCon loss.
        """
        device = features.device
        B      = features.shape[0]

        if B < 2:
            # Return zero loss connected to the computation graph
            # so backward() doesn't produce None gradients
            return (features * 0).sum()

        # L2 normalise features onto the unit hypersphere
        z = F.normalize(features, dim=1)              # (B, D)

        # Build per-sample temperature vector
        tau = torch.tensor(
            [self.class_temperatures.get(int(lbl.item()), 0.07) for lbl in labels],
            dtype=torch.float32, device=device,
        )                                              # (B,)

        # Pairwise cosine similarity matrix, scaled by per-sample τ
        # sim[i,j] = z_i · z_j / τ_i
        sim = torch.mm(z, z.T)                        # (B, B)
        sim = sim / tau.unsqueeze(1)                  # broadcast τ_i over columns

        # Mask out the diagonal (self-similarity)
        eye  = torch.eye(B, dtype=torch.bool, device=device)
        sim  = sim.masked_fill(eye, float('-inf'))

        # Log-sum-exp denominator (all pairs except self)
        log_denom = torch.logsumexp(sim, dim=1)       # (B,)

        # Positive mask: same class, different index
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        pos_mask   = (labels_row == labels_col) & ~eye  # (B, B)

        # Count positives per anchor
        n_pos = pos_mask.sum(dim=1).float()            # (B,)
        # Anchors with no positive pair contribute zero
        valid = n_pos > 0

        if not valid.any():
            return (features * 0).sum()

        # Sum log-probabilities over positives
        log_prob = sim - log_denom.unsqueeze(1)        # (B, B)
        loss_per_anchor = -(pos_mask.float() * log_prob).sum(dim=1)  # (B,)

        # Normalise by |P(i)| and base_temperature factor
        loss_per_anchor = loss_per_anchor / n_pos.clamp(min=1)
        loss_per_anchor = loss_per_anchor * (self.base_temperature / tau)

        loss = loss_per_anchor[valid].mean()
        return loss


# ---------------------------------------------------------------------------
# 2. VIB KL Divergence Loss
# ---------------------------------------------------------------------------

def vib_kl_loss(
    mu:      torch.Tensor,   # (B, D)
    log_var: torch.Tensor,   # (B, D)
    beta:    float = 1.0,
) -> torch.Tensor:
    """
    KL divergence between N(μ, σ²) and N(0, I), scaled by β.

        L_KL = β · mean( -½ · Σ_d (1 + log σ²_d - μ²_d - σ²_d) )

    β is annealed from 0 → β_target during training (see BetaScheduler).
    When β=0 the VIB path is unconstrained; as β rises the bottleneck
    forces Path B to discard noisy features and keep only the strongest
    disease signal in μ.
    """
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    return beta * kl.mean()


# ---------------------------------------------------------------------------
# 3. Orthogonal Feature Projection (applied to μ only)
# ---------------------------------------------------------------------------

def orthogonal_loss(
    mu:          torch.Tensor,   # (B, 384) — VIB mean vectors only
    labels:      torch.Tensor,   # (B,)
    num_classes: int = 3,
    eps:         float = 1e-8,
) -> torch.Tensor:
    """
    Cosine similarity penalty between class-mean μ vectors.

    Forces Path B to learn maximally distinct (non-overlapping) features
    for each disease class.  NOT applied to Path A (raw context preserved).

        L_Ortho = Σ_{i<j}  |cos(μ_centroid_i, μ_centroid_j)|

    When L_Ortho → 0, all class centroids are orthogonal in the embedding
    space → XGBoost sees maximally separable clusters.
    """
    device = mu.device

    # Compute class centroids
    centroids = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            # No samples for this class in the batch — use zero centroid
            centroids.append(torch.zeros(mu.shape[1], device=device))
        else:
            centroids.append(mu[mask].mean(dim=0))

    loss = (mu * 0).sum()   # graph-connected zero
    n_pairs = 0

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            ci = centroids[i]
            cj = centroids[j]

            # Only penalise if both centroids are non-zero (both classes in batch)
            if ci.norm() < eps or cj.norm() < eps:
                continue

            cos_sim = F.cosine_similarity(
                ci.unsqueeze(0), cj.unsqueeze(0)
            ).abs()
            loss    = loss + cos_sim
            n_pairs += 1

    return loss / max(n_pairs, 1)


# ---------------------------------------------------------------------------
# 4. β-Annealing Scheduler
# ---------------------------------------------------------------------------

class BetaScheduler:
    """
    Controls VIB β during training:

        Epochs [0,  warmup)                → β = 0
        Epochs [warmup, warmup + anneal)   → β linearly ramps 0 → β_target
        Epochs [warmup + anneal, ∞)        → β = β_target

    This ensures SupCon establishes initial clusters BEFORE the bottleneck
    starts compressing, preventing early collapse to N(0, I).
    """

    def __init__(
        self,
        warmup_epochs:  int   = 10,
        anneal_epochs:  int   = 20,
        beta_target:    float = 0.001,
    ):
        self.warmup  = warmup_epochs
        self.anneal  = anneal_epochs
        self.target  = beta_target

    def get_beta(self, epoch: int) -> float:
        """epoch is 0-indexed."""
        if epoch < self.warmup:
            return 0.0
        prog = min(epoch - self.warmup, self.anneal) / self.anneal
        return float(prog * self.target)


# ---------------------------------------------------------------------------
# 5. Combined Loss
# ---------------------------------------------------------------------------

class NetrAiLoss(nn.Module):
    """
    Wraps all three losses into a single callable.

    Usage:
        criterion = NetrAiLoss(cfg)
        loss, breakdown = criterion(vector_769, mu, log_var, labels, epoch)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        tc = cfg['training']
        cc = cfg['classes']

        class_temps = {
            cc['DR']:       tc['supcon_temperatures'][0],
            cc['Glaucoma']: tc['supcon_temperatures'][1],
            cc['PM']:       tc['supcon_temperatures'][2],
        }

        self.supcon    = SupConLoss(
            class_temperatures=class_temps,
            base_temperature=tc['supcon_base_temperature'],
        )
        self.beta_sched = BetaScheduler(
            warmup_epochs=tc['beta_warmup_epochs'],
            anneal_epochs=tc['beta_anneal_epochs'],
            beta_target=tc['beta_target'],
        )
        self.lambda_kl    = tc['lambda_kl']
        self.lambda_ortho = tc['lambda_ortho']
        self.supcon_w     = tc['supcon_weight']
        self.num_classes  = len(cc['names'])

    def forward(
        self,
        vector_769: torch.Tensor,   # (B, 769)
        mu:         torch.Tensor,   # (B, 384)
        log_var:    torch.Tensor,   # (B, 384)
        labels:     torch.Tensor,   # (B,)
        epoch:      int,
    ) -> tuple[torch.Tensor, dict]:

        beta = self.beta_sched.get_beta(epoch)

        l_supcon = self.supcon(vector_769, labels)
        l_kl     = vib_kl_loss(mu, log_var, beta=beta)
        l_ortho  = orthogonal_loss(mu, labels, num_classes=self.num_classes)

        total = (
            self.supcon_w     * l_supcon
            + self.lambda_kl  * l_kl
            + self.lambda_ortho * l_ortho
        )

        breakdown = {
            'loss':       total.item(),
            'l_supcon':   l_supcon.item(),
            'l_kl':       l_kl.item(),
            'l_ortho':    l_ortho.item(),
            'beta':       beta,
        }
        return total, breakdown
