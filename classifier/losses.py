"""
NetrAi Classifier — Loss Functions (v2)
=========================================
Three losses, applied during Phase 1 differentiable training:

    L_total = L_main + λ_aux · L_aux + β · (KL₁ + KL₂)

L_main   — BCEWithLogitsLoss on main_classifier(z_fused) vs label_vec
              Multi-label: 3 independent binary signals, one per disease.
              Applied to the full fused 256-D path.

L_aux    — BCEWithLogitsLoss on aux_classifier(z1) vs label_vec
              Forces VIB1 to independently encode disease signal from the
              custom SegFormer heads. WITHOUT this, VIB1 collapses to N(0,I)
              and the optimizer free-rides entirely on the frozen RETFound
              stream (VIB2).

KL₁ + KL₂ — Dual VIB KL divergence with β-annealing.
              Compresses each stream independently.
              β is annealed 0 → β_target over training to prevent the
              bottleneck from collapsing before the classifiers converge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. VIB KL Divergence Loss
# ---------------------------------------------------------------------------

def vib_kl_loss(
    mu:      torch.Tensor,   # (B, D)
    log_var: torch.Tensor,   # (B, D)
    beta:    float = 1.0,
) -> torch.Tensor:
    """
    KL divergence between N(μ, σ²) and the prior N(0, I), scaled by β.

        KL = -½ · Σ_d (1 + log σ²_d - μ²_d - σ²_d)

    β is annealed from 0 → β_target over training (see BetaScheduler).
    Starting at β=0 lets the classifiers establish meaningful clusters
    before the bottleneck starts compressing, preventing premature collapse.

    Applied separately to VIB1 and VIB2 — each stream must justify
    its own compression budget.
    """
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    return beta * kl.mean()


# ---------------------------------------------------------------------------
# 2. β-Annealing Scheduler
# ---------------------------------------------------------------------------

class BetaScheduler:
    """
    Controls the VIB KL penalty weight β during training.

        Epochs [0, warmup)                  → β = 0
        Epochs [warmup, warmup + anneal)    → β linearly ramps 0 → β_target
        Epochs [warmup + anneal, ∞)         → β = β_target (constant)

    The warmup phase ensures the BCEWithLogitsLoss classifiers establish
    useful disease-separating features BEFORE the bottleneck compresses
    anything. Starting with high β causes posterior collapse in epoch 1
    because collapsing to N(0,I) immediately drops the KL term to zero.
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
        prog = min(epoch - self.warmup, self.anneal) / max(self.anneal, 1)
        return float(prog * self.target)


# ---------------------------------------------------------------------------
# 3. Combined Loss
# ---------------------------------------------------------------------------

class NetrAiLoss(nn.Module):
    """
    Wraps all losses for Phase 1 training.

    Usage:
        criterion = NetrAiLoss(cfg)
        loss, breakdown = criterion(
            main_logits, aux_logits,
            mu1, log_var1, mu2, log_var2,
            label_vec, epoch
        )

    Args:
        main_logits  (B, 3)  — from main_classifier(z_fused)
        aux_logits   (B, 3)  — from aux_classifier(z1)
        mu1          (B, 128) — VIB1 mean
        log_var1     (B, 128) — VIB1 log σ²
        mu2          (B, 128) — VIB2 mean
        log_var2     (B, 128) — VIB2 log σ²
        label_vec    (B, 3)  — multi-hot float [dr, glauc, pm] ∈ {0.0, 1.0}
        epoch        int     — current epoch (0-indexed) for β scheduler
    """

    def __init__(self, cfg: dict):
        super().__init__()
        tc = cfg['training']

        self.bce = nn.BCEWithLogitsLoss()   # multi-label: reduction='mean' over all B×3 elements

        self.beta_sched  = BetaScheduler(
            warmup_epochs = tc['beta_warmup_epochs'],
            anneal_epochs = tc['beta_anneal_epochs'],
            beta_target   = tc['beta_target'],
        )

        self.lambda_aux = tc['lambda_aux']
        self.lambda_kl  = tc['lambda_kl']

    def forward(
        self,
        main_logits: torch.Tensor,   # (B, 3)
        aux_logits:  torch.Tensor,   # (B, 3)
        mu1:         torch.Tensor,   # (B, 128)
        log_var1:    torch.Tensor,   # (B, 128)
        mu2:         torch.Tensor,   # (B, 128)
        log_var2:    torch.Tensor,   # (B, 128)
        label_vec:   torch.Tensor,   # (B, 3) float multi-hot
        epoch:       int,
    ) -> tuple[torch.Tensor, dict]:

        beta = self.beta_sched.get_beta(epoch)

        # Multi-label BCE losses
        l_main = self.bce(main_logits, label_vec)
        l_aux  = self.bce(aux_logits,  label_vec)

        # Dual KL penalties
        l_kl1  = vib_kl_loss(mu1, log_var1, beta=1.0)   # β applied via scalar below
        l_kl2  = vib_kl_loss(mu2, log_var2, beta=1.0)
        l_kl   = beta * self.lambda_kl * (l_kl1 + l_kl2)

        total = l_main + self.lambda_aux * l_aux + l_kl

        breakdown = {
            'loss':    total.item(),
            'l_main':  l_main.item(),
            'l_aux':   l_aux.item(),
            'l_kl1':   l_kl1.item(),
            'l_kl2':   l_kl2.item(),
            'l_kl':    l_kl.item(),
            'beta':    beta,
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = {
        'training': {
            'lambda_aux':          0.4,
            'lambda_kl':           1.0,
            'beta_warmup_epochs':  10,
            'beta_anneal_epochs':  20,
            'beta_target':         0.001,
        }
    }
    criterion = NetrAiLoss(cfg)
    B = 4

    main_logits = torch.randn(B, 3)
    aux_logits  = torch.randn(B, 3)
    mu1         = torch.randn(B, 128)
    lv1         = torch.randn(B, 128)
    mu2         = torch.randn(B, 128)
    lv2         = torch.randn(B, 128)
    labels      = torch.zeros(B, 3)
    labels[torch.arange(B), torch.randint(0, 3, (B,))] = 1.0   # one-hot

    # β=0 during warmup
    loss, bd = criterion(main_logits, aux_logits, mu1, lv1, mu2, lv2, labels, epoch=5)
    assert bd['beta'] == 0.0
    print(f"epoch=5  loss={bd['loss']:.4f}  β={bd['beta']:.5f}  ✓ (KL should be 0)")

    # β>0 after warmup
    loss, bd = criterion(main_logits, aux_logits, mu1, lv1, mu2, lv2, labels, epoch=20)
    assert bd['beta'] > 0.0
    print(f"epoch=20 loss={bd['loss']:.4f}  β={bd['beta']:.5f}  ✓")

    print("losses.py — all checks passed ✓")
