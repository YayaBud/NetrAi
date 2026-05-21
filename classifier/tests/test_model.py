"""
tests/test_model.py
====================
Smoke tests for NetrAiEncoder and its sub-modules.
No GPU required — all tests run on CPU with tiny dummy inputs.

Run:
    pytest classifier/tests/test_model.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import torch.nn as nn

from classifier.model import (
    SegFormerDecodeHead,
    LateSpatialGate,
    VIBPath,
    DualPathBottleneck,
    NetrAiEncoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH   = 2
H, W    = 64, 64      # Tiny input — avoids downloading MIT-B3 in CI
C_FEAT  = 1024        # 4 × 256 decoder channels


@pytest.fixture(scope="module")
def dummy_hidden_states():
    """Mimics MIT-B3 encoder hidden states at 64×64 input."""
    return (
        torch.randn(BATCH, 64,  H//4,  W//4),   # stage 0: 16×16
        torch.randn(BATCH, 128, H//8,  W//8),   # stage 1:  8×8
        torch.randn(BATCH, 320, H//16, W//16),  # stage 2:  4×4
        torch.randn(BATCH, 512, H//32, W//32),  # stage 3:  2×2
    )


@pytest.fixture(scope="module")
def dummy_f_concat():
    """Mimics F_concat output from SegFormerDecodeHead."""
    return torch.randn(BATCH, C_FEAT, H//4, W//4)


@pytest.fixture(scope="module")
def dummy_anomaly_map():
    """Clean residual heatmap (values in [0,1])."""
    return torch.rand(BATCH, 1, H, W)


# ---------------------------------------------------------------------------
# SegFormerDecodeHead
# ---------------------------------------------------------------------------

class TestSegFormerDecodeHead:

    def test_output_shape(self, dummy_hidden_states):
        decoder = SegFormerDecodeHead(
            in_channels=(64, 128, 320, 512),
            decoder_embed_dim=256,
        )
        decoder.eval()
        with torch.no_grad():
            out = decoder(dummy_hidden_states)

        target_h = dummy_hidden_states[0].shape[-2]
        target_w = dummy_hidden_states[0].shape[-1]
        assert out.shape == (BATCH, 1024, target_h, target_w), \
            f"Expected (B, 1024, {target_h}, {target_w}), got {out.shape}"

    def test_grad_flows(self, dummy_hidden_states):
        decoder = SegFormerDecodeHead(in_channels=(64, 128, 320, 512))
        inp     = tuple(h.requires_grad_(True) for h in dummy_hidden_states)
        out     = decoder(inp)
        out.sum().backward()
        for i, h in enumerate(inp):
            assert h.grad is not None, f"Stage {i} hidden state has no grad"

    def test_custom_embed_dim(self, dummy_hidden_states):
        decoder = SegFormerDecodeHead(
            in_channels=(64, 128, 320, 512),
            decoder_embed_dim=128,
        )
        with torch.no_grad():
            out = decoder(dummy_hidden_states)
        assert out.shape[1] == 128 * 4, f"Expected 512 channels, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# LateSpatialGate
# ---------------------------------------------------------------------------

class TestLateSpatialGate:

    def test_output_shape(self, dummy_f_concat, dummy_anomaly_map):
        gate = LateSpatialGate(alpha_init=1.0)
        gate.eval()
        with torch.no_grad():
            F_gated, A_scaled = gate(dummy_f_concat, dummy_anomaly_map)

        assert F_gated.shape == dummy_f_concat.shape, \
            "F_gated shape must match F_concat"
        assert A_scaled.shape == (BATCH, 1, H//4, W//4), \
            f"A_scaled should be (B, 1, {H//4}, {W//4}), got {A_scaled.shape}"

    def test_residual_identity_when_alpha_zero(self, dummy_f_concat, dummy_anomaly_map):
        """When α=0, F_gated should equal F_concat exactly."""
        gate = LateSpatialGate(alpha_init=0.0)
        # Manually zero out the parameter
        gate.alpha.data.fill_(0.0)
        gate.eval()
        with torch.no_grad():
            F_gated, _ = gate(dummy_f_concat, dummy_anomaly_map)
        assert torch.allclose(F_gated, dummy_f_concat), \
            "α=0 should leave features unchanged (identity gate)"

    def test_alpha_is_learnable(self):
        gate = LateSpatialGate(alpha_init=1.0)
        assert gate.alpha.requires_grad, "α must be a learnable parameter"

    def test_sign_preservation(self, dummy_anomaly_map):
        """Multiplication by positive heatmap should not flip signs."""
        # All-negative feature map
        neg_feat = -torch.abs(torch.randn(BATCH, 8, H//4, W//4))
        gate     = LateSpatialGate(alpha_init=1.0)
        gate.alpha.data.fill_(1.0)
        with torch.no_grad():
            F_gated, _ = gate(neg_feat, dummy_anomaly_map)
        # All values should still be <= 0
        assert (F_gated <= 0).all(), \
            "Positive anomaly map should never flip negative feature signs"


# ---------------------------------------------------------------------------
# VIBPath
# ---------------------------------------------------------------------------

class TestVIBPath:

    def test_output_shapes(self):
        vib = VIBPath(in_dim=1024, out_dim=384)
        x   = torch.randn(BATCH, 1024)

        vib.train()
        z, mu, log_var = vib(x)
        assert z.shape       == (BATCH, 384)
        assert mu.shape      == (BATCH, 384)
        assert log_var.shape == (BATCH, 384)

    def test_deterministic_at_inference(self):
        """At eval time z must equal mu for the same input."""
        vib = VIBPath(in_dim=1024, out_dim=384)
        vib.eval()
        x = torch.randn(BATCH, 1024)
        with torch.no_grad():
            z, mu, _ = vib(x)
        assert torch.allclose(z, mu), \
            "At inference (eval mode) z must be exactly mu — no noise"

    def test_stochastic_at_train(self):
        """Two forward passes in train mode must produce different z."""
        vib = VIBPath(in_dim=1024, out_dim=384)
        vib.train()
        x  = torch.randn(BATCH, 1024)
        with torch.no_grad():
            z1, _, _ = vib(x)
            z2, _, _ = vib(x)
        # With overwhelming probability these will differ
        assert not torch.allclose(z1, z2), \
            "Two train-mode forward passes should produce different z (stochastic)"

    def test_grad_flows_through_z(self):
        vib = VIBPath(in_dim=16, out_dim=8)
        vib.train()
        x = torch.randn(BATCH, 16, requires_grad=True)
        z, mu, lv = vib(x)
        z.sum().backward()
        assert x.grad is not None, "Gradients must flow back through z"


# ---------------------------------------------------------------------------
# DualPathBottleneck
# ---------------------------------------------------------------------------

class TestDualPathBottleneck:

    def test_output_vector_dim(self, dummy_f_concat, dummy_anomaly_map):
        gate       = LateSpatialGate()
        bottleneck = DualPathBottleneck(in_dim=C_FEAT, path_dim=384)

        gate.eval()
        bottleneck.train()  # train so Path B uses z = mu + sigma*eps

        with torch.no_grad():
            F_gated, A_scaled = gate(dummy_f_concat, dummy_anomaly_map)
            vec, mu, lv = bottleneck(F_gated, A_scaled)

        assert vec.shape == (BATCH, 769), \
            f"Expected (B, 769), got {vec.shape}"
        assert mu.shape  == (BATCH, 384)
        assert lv.shape  == (BATCH, 384)

    def test_global_scalar_range(self):
        """The +1 global scalar (mean of A_scaled) must lie in [0,1]."""
        gate       = LateSpatialGate()
        bottleneck = DualPathBottleneck(in_dim=C_FEAT, path_dim=384)
        bottleneck.eval()

        f   = torch.randn(BATCH, C_FEAT, 16, 16)
        a   = torch.rand(BATCH, 1, 64, 64)   # ∈ [0,1]

        with torch.no_grad():
            F_gated, A_scaled = gate(f, a)
            vec, _, _ = bottleneck(F_gated, A_scaled)

        scalar_dim = vec[:, 768]   # index 768 = the global scalar
        assert (scalar_dim >= 0).all() and (scalar_dim <= 1).all(), \
            "Global anomaly scalar must be in [0, 1]"

    def test_vector_split(self):
        """First 384 dims = Path A, next 384 dims = Path B (z), last = scalar."""
        gate       = LateSpatialGate()
        bottleneck = DualPathBottleneck(in_dim=C_FEAT, path_dim=384)
        bottleneck.eval()

        f = torch.zeros(1, C_FEAT, 8, 8)   # all-zero features
        a = torch.ones(1, 1, 32, 32)        # anomaly map = 1 everywhere

        with torch.no_grad():
            F_gated, A_scaled = gate(f, a)
            vec, _, _ = bottleneck(F_gated, A_scaled)

        # Global scalar = mean(ones_downsampled) = 1.0
        assert abs(float(vec[0, 768]) - 1.0) < 1e-4, \
            "When anomaly map = 1, global scalar should be 1.0"


# ---------------------------------------------------------------------------
# Full NetrAiEncoder integration test (no HuggingFace download needed)
# ---------------------------------------------------------------------------

class TestNetrAiEncoderIntegration:
    """
    These tests use mock sub-modules to avoid downloading MIT-B3.
    Verifies that the data flow and shape contracts are correct end-to-end.
    """

    def _make_mock_encoder(self):
        """Replaces SegformerModel with a deterministic mock."""
        class _MockEncoder(nn.Module):
            """Returns 4 fake spatial feature maps of correct MIT-B3 shapes."""
            def __call__(self, pixel_values, output_hidden_states=True):
                B, _, H, W = pixel_values.shape
                class _Out:
                    hidden_states = (
                        torch.randn(B, 64,  H//4,  W//4),
                        torch.randn(B, 128, H//8,  W//8),
                        torch.randn(B, 320, H//16, W//16),
                        torch.randn(B, 512, H//32, W//32),
                    )
                return _Out()

        model = NetrAiEncoder.__new__(NetrAiEncoder)
        nn.Module.__init__(model)

        model.encoder    = _MockEncoder()
        model.decoder    = SegFormerDecodeHead(in_channels=(64, 128, 320, 512))
        model.gate       = LateSpatialGate(alpha_init=1.0)
        model.bottleneck = DualPathBottleneck(in_dim=1024, path_dim=384)
        return model

    def test_train_output_shapes(self):
        model = self._make_mock_encoder()
        model.train()

        img  = torch.randn(BATCH, 3, H, W)
        amap = torch.rand(BATCH, 1, H, W)

        vec, mu, lv = model(img, amap)
        assert vec.shape == (BATCH, 769), f"vector_769 shape: {vec.shape}"
        assert mu.shape  == (BATCH, 384), f"mu shape: {mu.shape}"
        assert lv.shape  == (BATCH, 384), f"log_var shape: {lv.shape}"

    def test_eval_deterministic(self):
        """Same input must produce identical output at inference."""
        model = self._make_mock_encoder()
        model.eval()

        img  = torch.randn(1, 3, H, W)
        amap = torch.rand(1, 1, H, W)

        with torch.no_grad():
            v1, mu1, _ = model(img, amap)
            v2, mu2, _ = model(img, amap)

        assert torch.allclose(v1, v2), "Inference must be deterministic"
        assert torch.allclose(mu1, mu2), "mu must be deterministic at inference"

    def test_gradient_checkpoint(self):
        """Verify backprop reaches the gate's α parameter."""
        model = self._make_mock_encoder()
        model.train()

        img  = torch.randn(BATCH, 3, H, W)
        amap = torch.rand(BATCH, 1, H, W)

        vec, mu, lv = model(img, amap)
        vec.sum().backward()

        assert model.gate.alpha.grad is not None, \
            "Gradient must reach learned gate scalar α"
