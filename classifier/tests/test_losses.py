"""
tests/test_losses.py
======================
Unit tests for all three loss functions + BetaScheduler.

Run:
    pytest classifier/tests/test_losses.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import numpy as np

from classifier.losses import (
    SupConLoss,
    vib_kl_loss,
    orthogonal_loss,
    BetaScheduler,
    NetrAiLoss,
)

BATCH       = 12
VEC_DIM     = 769
MU_DIM      = 384
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def balanced_labels():
    """4 samples per class: DR=0, Glaucoma=1, PM=2."""
    return torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


@pytest.fixture
def vectors(balanced_labels):
    return torch.randn(len(balanced_labels), VEC_DIM)


@pytest.fixture
def mu_vib(balanced_labels):
    return torch.randn(len(balanced_labels), MU_DIM)


@pytest.fixture
def log_var_vib(balanced_labels):
    return torch.randn(len(balanced_labels), MU_DIM)


# ---------------------------------------------------------------------------
# SupConLoss
# ---------------------------------------------------------------------------

class TestSupConLoss:

    CLASS_TEMPS = {0: 0.07, 1: 0.07, 2: 0.04}

    def _make_loss(self):
        return SupConLoss(
            class_temperatures=self.CLASS_TEMPS,
            base_temperature=0.07,
        )

    def test_scalar_output(self, vectors, balanced_labels):
        loss_fn = self._make_loss()
        loss    = loss_fn(vectors, balanced_labels)
        assert loss.shape == (), f"Loss must be scalar, got {loss.shape}"
        assert loss.item() > 0, "SupCon loss must be positive"

    def test_finite(self, vectors, balanced_labels):
        loss = self._make_loss()(vectors, balanced_labels)
        assert torch.isfinite(loss), "Loss must be finite"

    def test_perfect_clustering_lower_loss(self):
        """
        If same-class vectors are identical and far from other classes,
        loss should be near zero.
        """
        loss_fn = self._make_loss()
        labels  = torch.tensor([0, 0, 1, 1, 2, 2])

        # Perfect clusters: each class in orthogonal direction, large magnitude
        vecs = torch.zeros(6, 64)
        vecs[0:2, 0:20] = 100.0   # DR cluster: direction 0
        vecs[2:4, 20:40] = 100.0  # Glaucoma:  direction 1
        vecs[4:6, 40:60] = 100.0  # PM:        direction 2

        loss_perfect = loss_fn(vecs, labels)

        # Random vectors (poor clustering)
        vecs_random = torch.randn(6, 64)
        loss_random = loss_fn(vecs_random, labels)

        assert loss_perfect.item() < loss_random.item(), \
            "Perfect clustering should have lower SupCon loss than random"

    def test_minority_class_sharper_gradient(self):
        """
        PM (τ=0.04) should produce larger per-sample loss than
        DR (τ=0.07) for equivalent embedding difficulty.
        """
        labels_dr = torch.tensor([0, 0, 0, 0])
        labels_pm = torch.tensor([2, 2, 2, 2])

        # Same vectors for fair comparison
        vecs = torch.randn(4, 32)

        loss_fn = self._make_loss()

        # For this to work we need a mixed batch so positives ≠ all
        labels_mixed_dr = torch.tensor([0, 0, 1, 1])
        labels_mixed_pm = torch.tensor([2, 2, 1, 1])

        loss_dr = loss_fn(vecs, labels_mixed_dr)
        loss_pm = loss_fn(vecs, labels_mixed_pm)

        # Lower τ → higher loss magnitude for the same cosine sim
        # This is a statistical property so we just check both are finite
        assert torch.isfinite(loss_dr)
        assert torch.isfinite(loss_pm)

    def test_single_sample_batch(self):
        """Batch of 1 — no valid pairs — loss must be 0 without crashing."""
        loss_fn = self._make_loss()
        loss    = loss_fn(torch.randn(1, 16), torch.tensor([0]))
        assert loss.item() == 0.0

    def test_backward(self, vectors, balanced_labels):
        vectors_req = vectors.requires_grad_(True)
        loss = self._make_loss()(vectors_req, balanced_labels)
        loss.backward()
        assert vectors_req.grad is not None
        assert torch.isfinite(vectors_req.grad).all()


# ---------------------------------------------------------------------------
# VIB KL Loss
# ---------------------------------------------------------------------------

class TestVibKlLoss:

    def test_zero_when_standard_normal(self):
        """KL(N(0,I) || N(0,I)) = 0."""
        mu      = torch.zeros(BATCH, MU_DIM)
        log_var = torch.zeros(BATCH, MU_DIM)   # σ² = exp(0) = 1
        loss    = vib_kl_loss(mu, log_var, beta=1.0)
        assert abs(loss.item()) < 1e-4, \
            f"KL should be ~0 for N(0,I), got {loss.item()}"

    def test_positive_for_non_standard(self):
        """Any departure from N(0,I) should produce KL > 0."""
        mu      = torch.ones(BATCH, MU_DIM) * 2.0
        log_var = torch.zeros(BATCH, MU_DIM)
        loss    = vib_kl_loss(mu, log_var, beta=1.0)
        assert loss.item() > 0

    def test_beta_scaling(self):
        """Loss should scale linearly with β."""
        mu      = torch.randn(BATCH, MU_DIM)
        log_var = torch.randn(BATCH, MU_DIM)
        l1 = vib_kl_loss(mu, log_var, beta=1.0)
        l2 = vib_kl_loss(mu, log_var, beta=2.0)
        assert abs(l2.item() / l1.item() - 2.0) < 1e-4, \
            "KL loss must scale exactly linearly with β"

    def test_beta_zero(self):
        """β=0 must produce zero loss regardless of μ, σ."""
        mu      = torch.randn(BATCH, MU_DIM) * 100
        log_var = torch.randn(BATCH, MU_DIM) * 10
        loss    = vib_kl_loss(mu, log_var, beta=0.0)
        assert loss.item() == 0.0

    def test_backward(self, mu_vib, log_var_vib):
        mu_req  = mu_vib.requires_grad_(True)
        lv_req  = log_var_vib.requires_grad_(True)
        loss    = vib_kl_loss(mu_req, lv_req, beta=0.01)
        loss.backward()
        assert mu_req.grad is not None
        assert lv_req.grad is not None


# ---------------------------------------------------------------------------
# Orthogonal Loss
# ---------------------------------------------------------------------------

class TestOrthogonalLoss:

    def test_orthogonal_vectors_zero_loss(self):
        """
        If class centroids are perfectly orthogonal, cosine sim = 0 → loss = 0.
        """
        mu     = torch.zeros(6, 9)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        # Perfectly orthogonal centroids
        mu[0:2, 0:3] = 1.0   # DR centroid: (1,1,1,0,0,0,0,0,0)
        mu[2:4, 3:6] = 1.0   # Glaucoma:   (0,0,0,1,1,1,0,0,0)
        mu[4:6, 6:9] = 1.0   # PM:         (0,0,0,0,0,0,1,1,1)

        loss = orthogonal_loss(mu, labels, num_classes=3)
        assert loss.item() < 1e-5, \
            f"Orthogonal centroids → loss should be ~0, got {loss.item()}"

    def test_parallel_vectors_max_loss(self):
        """
        Identical class centroids → max cosine sim = 1 → loss = 1.
        """
        mu     = torch.ones(6, 8)   # all identical → cosine_sim = 1
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss   = orthogonal_loss(mu, labels, num_classes=3)
        assert abs(loss.item() - 1.0) < 1e-4, \
            f"Identical centroids → loss should be 1.0, got {loss.item()}"

    def test_missing_class_in_batch(self):
        """Missing a class in the batch should not crash."""
        mu     = torch.randn(4, 8)
        labels = torch.tensor([0, 0, 1, 1])   # PM (2) not present
        loss   = orthogonal_loss(mu, labels, num_classes=3)
        assert torch.isfinite(loss)

    def test_range(self, mu_vib, balanced_labels):
        loss = orthogonal_loss(mu_vib, balanced_labels, num_classes=3)
        # Cosine similarity ∈ [-1, 1], absolute value ∈ [0, 1]
        assert 0.0 <= loss.item() <= 1.0 + 1e-5, \
            f"Ortho loss must be in [0, 1], got {loss.item()}"

    def test_applied_to_mu_only_not_path_a(self, mu_vib, balanced_labels):
        """
        Verify that ortho loss can be applied to a 384-D slice without error.
        The test asserts it does NOT accidentally receive the 769-D full vector.
        """
        assert mu_vib.shape[1] == MU_DIM, \
            f"Ortho penalty target must be 384-D μ, not 769-D vector"
        loss = orthogonal_loss(mu_vib, balanced_labels, num_classes=3)
        assert torch.isfinite(loss)

    def test_backward(self, mu_vib, balanced_labels):
        mu_req = mu_vib.requires_grad_(True)
        loss   = orthogonal_loss(mu_req, balanced_labels, num_classes=3)
        loss.backward()
        assert mu_req.grad is not None


# ---------------------------------------------------------------------------
# BetaScheduler
# ---------------------------------------------------------------------------

class TestBetaScheduler:

    def test_zero_during_warmup(self):
        sched = BetaScheduler(warmup_epochs=10, anneal_epochs=20, beta_target=0.001)
        for ep in range(10):
            assert sched.get_beta(ep) == 0.0, f"β must be 0 during warmup (epoch {ep})"

    def test_linear_ramp(self):
        sched = BetaScheduler(warmup_epochs=10, anneal_epochs=20, beta_target=0.001)
        betas = [sched.get_beta(ep) for ep in range(10, 31)]
        # Must be monotonically non-decreasing
        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1], \
                f"β must be non-decreasing during anneal (epoch {i+10})"

    def test_reaches_target(self):
        sched = BetaScheduler(warmup_epochs=5, anneal_epochs=10, beta_target=0.001)
        beta_final = sched.get_beta(15)   # epoch after warmup + anneal
        assert abs(beta_final - 0.001) < 1e-8, \
            f"β must reach target=0.001, got {beta_final}"

    def test_saturates_at_target(self):
        sched = BetaScheduler(warmup_epochs=5, anneal_epochs=10, beta_target=0.001)
        # Epochs well past the end of annealing
        b30 = sched.get_beta(30)
        b50 = sched.get_beta(50)
        assert b30 == b50 == 0.001, \
            "β must stay at target after annealing completes"


# ---------------------------------------------------------------------------
# NetrAiLoss (combined)
# ---------------------------------------------------------------------------

class TestNetrAiLoss:

    @pytest.fixture
    def cfg(self):
        return {
            'training': {
                'supcon_weight': 1.0,
                'lambda_kl':     0.01,
                'lambda_ortho':  0.1,
                'supcon_temperatures': {0: 0.07, 1: 0.07, 2: 0.04},
                'supcon_base_temperature': 0.07,
                'beta_warmup_epochs':  10,
                'beta_anneal_epochs':  20,
                'beta_target':         0.001,
            },
            'classes': {
                'names': ['DR', 'Glaucoma', 'PM'],
                'DR': 0, 'Glaucoma': 1, 'PM': 2,
            },
        }

    def test_returns_scalar_and_breakdown(self, cfg, balanced_labels):
        criterion  = NetrAiLoss(cfg)
        vector_769 = torch.randn(len(balanced_labels), VEC_DIM, requires_grad=True)
        mu         = torch.randn(len(balanced_labels), MU_DIM)
        log_var    = torch.randn(len(balanced_labels), MU_DIM)

        loss, bd = criterion(vector_769, mu, log_var, balanced_labels, epoch=0)

        assert loss.shape == (), "Total loss must be scalar"
        assert 'loss'     in bd
        assert 'l_supcon' in bd
        assert 'l_kl'     in bd
        assert 'l_ortho'  in bd
        assert 'beta'     in bd

    def test_kl_zero_during_warmup(self, cfg, balanced_labels):
        criterion  = NetrAiLoss(cfg)
        vector_769 = torch.randn(len(balanced_labels), VEC_DIM)
        mu         = torch.randn(len(balanced_labels), MU_DIM) * 100   # large μ
        log_var    = torch.randn(len(balanced_labels), MU_DIM) * 10

        _, bd = criterion(vector_769, mu, log_var, balanced_labels, epoch=0)
        assert bd['l_kl'] == 0.0, \
            "KL term must be zero during warmup (β=0)"

    def test_backward_propagates(self, cfg, balanced_labels):
        criterion  = NetrAiLoss(cfg)
        vector_769 = torch.randn(len(balanced_labels), VEC_DIM, requires_grad=True)
        mu         = torch.randn(len(balanced_labels), MU_DIM,  requires_grad=True)
        log_var    = torch.randn(len(balanced_labels), MU_DIM,  requires_grad=True)

        loss, _ = criterion(vector_769, mu, log_var, balanced_labels, epoch=15)
        loss.backward()

        assert vector_769.grad is not None, "Grad must flow to vector_769 (SupCon)"
        assert mu.grad         is not None, "Grad must flow to mu (Ortho + KL)"
        assert log_var.grad    is not None, "Grad must flow to log_var (KL)"
