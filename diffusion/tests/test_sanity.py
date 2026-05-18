"""
Sanity tests for the diffusion training pipeline.
Run: python -m pytest diffusion/tests/test_sanity.py -v
"""
import sys
import os
import torch
import numpy as np
import pytest

# ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------
class TestConfig:
    def test_yaml_loads_without_string_none(self):
        """P0 regression: config.yaml must use YAML null, not string 'None'."""
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if not os.path.exists(cfg_path):
            pytest.skip("config.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for key in ("ddr_max_images", "ddr_max_seconds"):
            val = cfg["eval"][key]
            assert val is None or isinstance(val, (int, float)), \
                f"eval.{key} is {val!r} (string) -- use ~ or null in YAML"

    def test_required_paths_present(self):
        """Config must declare all expected path keys."""
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if not os.path.exists(cfg_path):
            pytest.skip("config.yaml not found")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        required = ["data_train", "data_val", "checkpoint_dir",
                     "pretrained_256", "ddr_images_dir", "ddr_masks_dir",
                     "retfound_weights"]
        for key in required:
            assert key in cfg["paths"], f"paths.{key} missing from config"


# ---------------------------------------------------------------------------
# 2. Losses -- shape and sync safety
# ---------------------------------------------------------------------------
class TestLosses:
    def test_snr_weight_shape(self):
        """Regression: weight * mse must not broadcast to [B,B,...]."""
        from diffusion.losses import snr_weighted_loss
        B = 4
        pred = torch.randn(B, 3, 64, 64)
        noise = torch.randn(B, 3, 64, 64)
        ac = torch.linspace(0.999, 0.001, 1000)
        ts = torch.randint(0, 1000, (B,))
        loss = snr_weighted_loss(pred, noise, ac, ts, gamma=2.0)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"

    def test_diffusion_loss_returns_detached(self):
        """comp dict values must be detached tensors, not Python floats."""
        from diffusion.losses import diffusion_loss
        B = 2
        pred = torch.randn(B, 3, 64, 64)
        noise = torch.randn(B, 3, 64, 64)
        pred_x0 = torch.randn(B, 3, 64, 64)
        x0 = torch.randn(B, 3, 64, 64)
        mask = torch.ones(B, 1, 64, 64)
        ac = torch.linspace(0.999, 0.001, 1000)
        ts = torch.randint(0, 1000, (B,))
        loss, comp = diffusion_loss(pred, noise, pred_x0, x0, mask, ac, ts)
        # must be tensors (not floats) to avoid per-call GPU sync
        assert isinstance(comp['snr'], torch.Tensor), "comp['snr'] should be a tensor"
        assert isinstance(comp['ms'], torch.Tensor), "comp['ms'] should be a tensor"
        assert not comp['snr'].requires_grad, "comp['snr'] should be detached"
        assert not comp['ms'].requires_grad, "comp['ms'] should be detached"


# ---------------------------------------------------------------------------
# 3. Data -- CSV column selection
# ---------------------------------------------------------------------------
class TestData:
    def test_csv_reads_correct_column(self, tmp_path):
        """Regression: must read 'path' column, not df.columns[0]."""
        import pandas as pd
        # CSV where 'path' is NOT the first column
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({
            "source": ["eyepacs", "eyepacs"],
            "path": [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
        }).to_csv(csv_file, index=False)
        # create dummy images
        from PIL import Image
        for name in ("a.jpg", "b.jpg"):
            Image.new("RGB", (64, 64), color="red").save(tmp_path / name)

        from diffusion.data import RetinaDataset
        ds = RetinaDataset(str(csv_file), crop_size=64, is_train=False)
        # verify it loaded the 'path' column, not 'source'
        for img_path in ds.images:
            assert img_path.endswith(".jpg"), \
                f"Loaded wrong column: {img_path}"

    def test_csv_image_column_also_works(self, tmp_path):
        """CSV with 'image' column should also work."""
        import pandas as pd
        csv_file = tmp_path / "test.csv"
        img_path = str(tmp_path / "x.jpg")
        pd.DataFrame({"image": [img_path]}).to_csv(csv_file, index=False)
        from PIL import Image
        Image.new("RGB", (64, 64)).save(tmp_path / "x.jpg")

        from diffusion.data import RetinaDataset
        ds = RetinaDataset(str(csv_file), crop_size=64, is_train=False)
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# 4. Diffusion -- tiling and DDIM
# ---------------------------------------------------------------------------
class TestDiffusion:
    def test_tile_weights_nonzero_everywhere(self):
        """Regression: edge pixels must have nonzero blend weight."""
        from diffusion.diffusion import make_linear_weight
        w = make_linear_weight(256, device='cpu')
        assert w.min().item() > 0, \
            f"Tile weight has zeros (min={w.min().item():.6f})"

    def test_ddim_final_step_gives_clean_output(self):
        """Final DDIM step with t_prev=-1 should target alpha_prev=1.0."""
        from diffusion.diffusion import simplex_ddim_step
        ac = torch.linspace(0.999, 0.001, 1000)
        x_t = torch.randn(1, 3, 32, 32)
        pred_noise = torch.randn(1, 3, 32, 32)
        t = torch.tensor([10], dtype=torch.long)
        t_prev = torch.tensor([-1], dtype=torch.long)  # sentinel
        result = simplex_ddim_step(x_t, pred_noise, t, t_prev, ac, 'cpu')
        # with ac_prev=1.0: coeff_x0=1.0, coeff_dir=0.0 -> result == pred_x0
        ac_t = ac[10].view(1, 1, 1, 1)
        expected_x0 = ((x_t - (1 - ac_t).sqrt() * pred_noise) /
                        (ac_t.sqrt() + 1e-8)).clamp(-1, 1)
        assert torch.allclose(result, expected_x0, atol=1e-5), \
            "Final step should return pure pred_x0 when ac_prev=1.0"

    def test_retinal_mask_no_margin_shrink(self):
        """Default margin=0.0 should not shrink the detected FOV."""
        from diffusion.diffusion import make_retinal_mask
        # white circle on black background
        img = torch.zeros(1, 3, 64, 64)
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        circle = ((yy - 32)**2 + (xx - 32)**2) < 28**2
        img[0, :, circle] = 1.0
        img = img * 2 - 1  # to [-1, 1]
        mask = make_retinal_mask(img, margin=0.0)
        # the mask should cover roughly the same area as the circle
        circle_area = circle.float().sum().item()
        mask_area = mask.sum().item()
        # allow 5% tolerance but it should NOT shrink by 10%+
        assert mask_area >= circle_area * 0.90, \
            f"Mask area {mask_area} is too small vs circle {circle_area}"


# ---------------------------------------------------------------------------
# 5. Models -- conditioner output shape
# ---------------------------------------------------------------------------
class TestModels:
    def test_conditioner_output_shape(self):
        """RETFoundConditioner must output (B, 1, 768) for cross-attention."""
        from diffusion.models import RETFoundConditioner
        cond = RETFoundConditioner(cross_attention_dim=768,
                                    retfound_weights=None)
        x = torch.randn(2, 3, 256, 256)
        out = cond(x)
        assert out.shape == (2, 1, 768), \
            f"Expected (2, 1, 768), got {out.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
