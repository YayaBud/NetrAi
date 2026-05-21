"""
NetrAi Classifier — Full Inference Pipeline
=============================================
Given a single retina image + its anomaly map, produces:
  - Diagnosis label  (DR / Glaucoma / PM)
  - Confidence scores  {DR: 0.92, Glaucoma: 0.06, PM: 0.02}
  - 769-D SegFormer vector   (for debugging / embedding visualisation)
  - 1024-D RETFound vector   (if cache available, else computed on-the-fly)

Two usage modes:
  A) With RETFound cache (production):
       Load .pt file from cache_dir by image stem.
  B) Without cache (on-the-fly):
       Run RETFound forward pass — useful for single-image demo.

Run:
    python -m classifier infer --image path/to/retina.jpg \
                               --anomaly path/to/anomaly.png \
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

from .model     import NetrAiEncoder
from .retfound  import RETFoundExtractor, RETFOUND_TRANSFORM, load_cached_embedding, make_cache_key
from .xgboost_clf import NetrAiXGBoost
from .utils     import load_config, load_checkpoint, get_device, get_amp_dtype

CLASS_NAMES = ["DR", "Glaucoma", "PM"]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

RETINA_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def preprocess_image(
    image_path: str,
    image_size: int = 512,
) -> torch.Tensor:
    """Loads and normalises a retina image → (1, 3, H, W)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return RETINA_TRANSFORM(img).unsqueeze(0)  # (1, 3, H, W)


def preprocess_anomaly_map(
    map_path: Optional[str],
    image_size: int = 512,
) -> torch.Tensor:
    """
    Loads a clean residual heatmap → (1, 1, H, W) in [0, 1].
    Falls back to zeros if path is None or not found.
    """
    if map_path and os.path.isfile(map_path):
        img = Image.open(map_path).convert("L")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return torch.zeros(1, 1, image_size, image_size)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class NetrAiInference:
    """
    Loads all components once and provides a predict() method for
    single-image or batch inference.

    Components (all frozen, eval mode):
      1. NetrAiEncoder   — 769-D feature extraction
      2. RETFoundExtractor (optional, for on-the-fly embed)
      3. NetrAiXGBoost   — final classification
    """

    def __init__(
        self,
        cfg:              dict,
        segformer_ckpt:   Optional[str] = None,
        xgboost_ckpt:     Optional[str] = None,
        device:           Optional[torch.device] = None,
        load_retfound:    bool = False,   # True = on-the-fly; False = cache only
    ):
        self.cfg       = cfg
        self.device    = device or get_device()
        self.amp_dtype = get_amp_dtype(self.device)
        self.image_size = cfg['data']['image_size']
        p = cfg['paths']

        # ---- SegFormer encoder ----
        mc = cfg['model']
        self.encoder = NetrAiEncoder(
            backbone_name     = mc['backbone'],
            decoder_embed_dim = mc['decoder_embed_dim'],
            path_a_dim        = mc['path_a_dim'],
            path_b_dim        = mc['path_b_dim'],
            alpha_init        = mc['alpha_init'],
        ).to(self.device)

        ckpt_path = segformer_ckpt or os.path.join(p['checkpoint_dir'], "best.pt")
        load_checkpoint(ckpt_path, self.encoder, device=self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        # ---- RETFound (optional on-the-fly) ----
        self.retfound_extractor = None
        if load_retfound:
            self.retfound_extractor = RETFoundExtractor(
                weights_path=p.get('retfound_weights'),
                device=self.device,
            )

        self.retfound_cache_dir = p['retfound_cache_dir']

        # ---- XGBoost classifier ----
        self.xgb = NetrAiXGBoost(cfg)
        xgb_path = xgboost_ckpt or os.path.join(p['checkpoint_dir'], "xgboost_model.pkl")
        self.xgb.load(xgb_path)

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
                "diagnosis":    "DR",
                "confidence":   {"DR": 0.92, "Glaucoma": 0.06, "PM": 0.02},
                "probabilities": [0.92, 0.06, 0.02],
                "vector_769":   np.ndarray (769,),
                "vector_1793":  np.ndarray (1793,),
            }
        """
        stem = Path(image_path).stem

        # Preprocess
        image = preprocess_image(image_path, self.image_size).to(self.device)
        amap  = preprocess_anomaly_map(anomaly_path, self.image_size).to(self.device)

        # SegFormer → 769-D
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=(self.device.type == "cuda")
        ):
            vector_769, mu, _ = self.encoder(image, amap)

        seg_np = vector_769.cpu().float().numpy().squeeze()  # (769,)

        # RETFound → 1024-D
        retfound_np = self._get_retfound(image_path, stem).astype(np.float32)

        # Concatenate → 1793-D
        vec_1793 = np.concatenate([seg_np, retfound_np])  # (1793,)

        # XGBoost → probabilities
        proba     = self.xgb.predict_proba(vec_1793[np.newaxis, :])[0]  # (3,)
        pred_idx  = int(proba.argmax())
        diagnosis = CLASS_NAMES[pred_idx]

        return {
            "diagnosis":     diagnosis,
            "confidence":    {name: float(p) for name, p in zip(CLASS_NAMES, proba)},
            "probabilities": proba.tolist(),
            "vector_769":    seg_np,
            "vector_1793":   vec_1793,
        }

    def predict_batch(
        self,
        image_paths:   list[str],
        anomaly_paths: Optional[list[Optional[str]]] = None,
    ) -> list[dict]:
        """Runs predict() over a list of images."""
        if anomaly_paths is None:
            anomaly_paths = [None] * len(image_paths)
        return [
            self.predict(ip, ap)
            for ip, ap in zip(image_paths, anomaly_paths)
        ]

    # ------------------------------------------------------------------
    # RETFound embedding resolution
    # ------------------------------------------------------------------

    def _get_retfound(self, image_path: str, stem: str) -> np.ndarray:
        """
        Priority:
          1. Load from cache (fast — no GPU needed)
          2. Compute on-the-fly with loaded RETFoundExtractor
          3. Return zeros with a warning
        """
        # Try cache first — try collision-safe key, then bare stem as fallback
        for key in (make_cache_key(image_path), stem):
            emb = load_cached_embedding(key, self.retfound_cache_dir)
            if emb is not None:
                return emb.numpy()

        # Try on-the-fly
        if self.retfound_extractor is not None:
            img = Image.open(image_path).convert("RGB")
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
    parser.add_argument("--xgboost-ckpt",  default=None)
    parser.add_argument("--load-retfound", action="store_true",
                        help="Load RETFound for on-the-fly embedding (slow at startup)")
    args = parser.parse_args(args)

    cfg     = load_config(args.config)
    engine  = NetrAiInference(
        cfg             = cfg,
        segformer_ckpt  = args.segformer_ckpt,
        xgboost_ckpt    = args.xgboost_ckpt,
        load_retfound   = args.load_retfound,
    )

    result = engine.predict(args.image, args.anomaly)

    print("\n" + "═" * 50)
    print(f"  DIAGNOSIS:  {result['diagnosis']}")
    print("  CONFIDENCE:")
    for cls, prob in result['confidence'].items():
        bar = "█" * int(prob * 40)
        print(f"    {cls:10s}  {prob*100:5.1f}%  {bar}")
    print(f"  Vector dim: {result['vector_1793'].shape}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
