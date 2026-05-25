"""
NetrAi Classifier — Full Inference Pipeline (v2)
==================================================
Given a single retina image + its clean residual, produces:
  - Per-disease probabilities: {DR: 0.87, Glaucoma: 0.12, PM: 0.05}
  - Primary diagnosis (highest probability disease)
  - 256-D feature vector (for debugging / embedding visualisation)

Two RETFound embedding modes:
  A) Pre-cached .pt file from retfound_cache_dir (production — fast)
  B) On-the-fly RETFoundExtractor forward pass (demo — slow at startup)

6-channel stack built internally from (image_path, anomaly_path).

Run:
    python -m classifier infer --image path/to/retina.jpg \\
                               --anomaly path/to/anomaly.png \\
                               --config classifier/config.yaml
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .model       import NetrAiEncoder
from .retfound    import RETFoundExtractor, RETFOUND_TRANSFORM, load_cached_embedding, make_cache_key
from .xgboost_clf import NetrAiXGBoost
from .utils       import load_config, load_checkpoint, get_device, get_amp_dtype

CLASS_NAMES = ["DR", "Glaucoma", "PM"]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

RETINA_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def preprocess_six_channel(
    image_path:  str,
    anomaly_path: Optional[str],
    image_size:  int = 512,
) -> torch.Tensor:
    """
    Builds the 6-channel input tensor for the SegFormer.
    Channels 0-2: ImageNet-normalised RGB
    Channels 3-5: Clean residual replicated ×3

    Returns: (1, 6, H, W)
    """
    # RGB image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img_t = RETINA_TRANSFORM(img)   # (3, H, W)

    # Clean residual
    if anomaly_path and os.path.isfile(anomaly_path):
        amap = Image.open(anomaly_path).convert("L")
        amap = amap.resize((image_size, image_size), Image.BILINEAR)
        arr  = np.array(amap, dtype=np.float32) / 255.0
        amap_t = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)   # (3, H, W)
    else:
        amap_t = torch.zeros(3, image_size, image_size)

    six_ch = torch.cat([img_t, amap_t], dim=0).unsqueeze(0)   # (1, 6, H, W)
    return six_ch


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class NetrAiInference:
    """
    Loads all components once and provides predict() for inference.

    Components:
      1. NetrAiEncoder (frozen, eval mode) — 256-D feature extraction
      2. RETFoundExtractor (optional, for on-the-fly embedding)
      3. NetrAiXGBoost (3 binary classifiers) — final probabilities
    """

    def __init__(
        self,
        cfg:            dict,
        segformer_ckpt: Optional[str]         = None,
        xgboost_dir:    Optional[str]         = None,
        device:         Optional[torch.device] = None,
        load_retfound:  bool                   = False,
    ):
        self.cfg        = cfg
        self.device     = device or get_device()
        self.amp_dtype  = get_amp_dtype(self.device)
        self.image_size = cfg['data']['image_size']
        p = cfg['paths']

        # ── SegFormer encoder ─────────────────────────────────────────────
        mc = cfg['model']
        self.encoder = NetrAiEncoder(
            backbone_name = mc['backbone'],
            head_out_dim  = mc['head_out_dim'],
            vib_hidden    = mc['vib_hidden'],
            vib_out_dim   = mc['vib_out_dim'],
            dropout       = mc.get('dropout', 0.3),
        ).to(self.device)

        ckpt_path = segformer_ckpt or os.path.join(p['checkpoint_dir'], "best.pt")
        load_checkpoint(ckpt_path, self.encoder, device=self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        # ── RETFound (optional on-the-fly) ────────────────────────────────
        self.retfound_extractor  = None
        self.retfound_cache_dir  = p.get('retfound_cache_dir')

        if load_retfound:
            self.retfound_extractor = RETFoundExtractor(
                weights_path = p.get('retfound_weights'),
                device       = self.device,
            )

        # ── XGBoost classifiers ───────────────────────────────────────────
        self.xgb     = NetrAiXGBoost(cfg)
        xgb_dir      = xgboost_dir or os.path.join(p['checkpoint_dir'], "xgboost")
        self.xgb.load(xgb_dir)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image_path:   str,
        anomaly_path: Optional[str] = None,
    ) -> dict:
        """
        Single-image inference.

        Returns:
            {
                "diagnosis":     "DR",          # highest-probability disease
                "probabilities": {"DR": 0.87, "Glaucoma": 0.12, "PM": 0.05},
                "vector_256":    np.ndarray (256,),
            }
        """
        stem = Path(image_path).stem

        # ── Build 6-channel input ─────────────────────────────────────────
        six_ch = preprocess_six_channel(
            image_path, anomaly_path, self.image_size
        ).to(self.device)   # (1, 6, H, W)

        # ── Get RETFound embedding ────────────────────────────────────────
        retfound_np = self._get_retfound(image_path, stem)   # (1024,)
        retfound_t  = torch.from_numpy(retfound_np).unsqueeze(0).to(self.device)  # (1, 1024)

        # ── Forward pass → 256-D (deterministic μ in eval mode) ──────────
        with torch.autocast(
            device_type = self.device.type,
            dtype       = self.amp_dtype,
            enabled     = (self.device.type == "cuda"),
        ):
            z_fused, z1, mu1, lv1, mu2, lv2 = self.encoder(six_ch, retfound_t)

        vec_256 = z_fused.cpu().float().numpy().squeeze()   # (256,)

        # ── XGBoost → 3 independent probabilities ────────────────────────
        proba     = self.xgb.predict_proba(vec_256[np.newaxis, :])[0]  # (3,)
        pred_idx  = int(proba.argmax())
        diagnosis = CLASS_NAMES[pred_idx]

        return {
            "diagnosis":     diagnosis,
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, proba)},
            "vector_256":    vec_256,
        }

    def predict_batch(
        self,
        image_paths:   list[str],
        anomaly_paths: Optional[list[Optional[str]]] = None,
    ) -> list[dict]:
        if anomaly_paths is None:
            anomaly_paths = [None] * len(image_paths)
        return [self.predict(ip, ap) for ip, ap in zip(image_paths, anomaly_paths)]

    # ------------------------------------------------------------------
    # RETFound embedding resolution
    # ------------------------------------------------------------------

    def _get_retfound(self, image_path: str, stem: str) -> np.ndarray:
        """
        Priority:
          1. Load from cache (fast — no GPU needed)
          2. Compute on-the-fly with RETFoundExtractor
          3. Return zeros with a warning
        """
        if self.retfound_cache_dir:
            for key in (make_cache_key(image_path), stem):
                emb = load_cached_embedding(key, self.retfound_cache_dir)
                if emb is not None:
                    return emb.numpy()

        if self.retfound_extractor is not None:
            img   = Image.open(image_path).convert("RGB")
            img_t = RETFOUND_TRANSFORM(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.retfound_extractor(img_t)
            return emb.cpu().float().numpy().squeeze()

        print(f"  [WARN] No RETFound embedding for {stem} — using zeros.")
        return np.zeros(1024, dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Run NetrAi inference on a single image")
    parser.add_argument("--config",        default="classifier/config.yaml")
    parser.add_argument("--image",         required=True,  help="Path to retina image")
    parser.add_argument("--anomaly",       default=None,   help="Path to clean residual heatmap")
    parser.add_argument("--segformer-ckpt",default=None)
    parser.add_argument("--xgboost-dir",   default=None,   help="Dir containing xgb_*.pkl files")
    parser.add_argument("--load-retfound", action="store_true",
                        help="Load RETFound for on-the-fly embedding (slow at startup)")
    args = parser.parse_args(args)

    cfg    = load_config(args.config)
    engine = NetrAiInference(
        cfg            = cfg,
        segformer_ckpt = args.segformer_ckpt,
        xgboost_dir    = args.xgboost_dir,
        load_retfound  = args.load_retfound,
    )

    result = engine.predict(args.image, args.anomaly)

    print("\n" + "═" * 52)
    print(f"  DIAGNOSIS:  {result['diagnosis']}")
    print("  PROBABILITIES (independent per disease):")
    for cls, prob in result['probabilities'].items():
        bar = "█" * int(prob * 40)
        print(f"    {cls:10s}  {prob*100:5.1f}%  {bar}")
    print(f"  Vector dim: {result['vector_256'].shape}")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()
