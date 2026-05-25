"""
NetrAi Classifier — Dataset & DataLoader (v2)
===============================================
Expected on-disk layout (unchanged):

    data/
    ├── train/
    │   ├── DR/
    │   ├── Glaucoma/
    │   └── PM/
    └── val/
        ├── DR/
        ├── Glaucoma/
        └── PM/

    data/anomaly_maps/
    │   ├── <image_stem>_anomaly.png   ← preferred naming
    │   └── <image_stem>.png           ← fallback naming
    └── ...

Key changes from v1:
  - Returns a 6-channel stacked tensor (3ch RGB + 3ch clean residual)
    instead of separate (image, anomaly_map) tensors.
  - Returns label_vec (3,) float multi-hot [dr, glauc, pm]
    instead of single integer label.
  - Loads pre-cached RETFound 1024-D embedding from disk.
    Falls back to zeros if cache miss (zero RETFound contribution
    via VIB2 → network leans on custom heads instead).
  - Augmentations now applied to both image and residual separately
    before stacking.

6-channel stack format:
    Channels 0-2: RGB image (ImageNet normalised)
    Channels 3-5: Clean residual replicated ×3 from greyscale
                  (preserves the spatial intensity map in all 3 channels)
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------
CLASS_TO_IDX = {"DR": 0, "Glaucoma": 1, "PM": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES   = len(CLASS_TO_IDX)


# ---------------------------------------------------------------------------
# Anomaly map loader
# ---------------------------------------------------------------------------

def _load_anomaly_map(stem: str, anomaly_dir: str, size: int) -> torch.Tensor:
    """
    Loads greyscale clean residual map → (3, size, size) float in [0, 1].
    Replicated across 3 channels to form the residual half of the 6-ch stack.
    Falls back to zeros (gate-equivalent: residual contribution = 0).
    """
    for suffix in (f"{stem}_anomaly.png", f"{stem}.png"):
        fpath = os.path.join(anomaly_dir, suffix)
        if os.path.isfile(fpath):
            try:
                img = Image.open(fpath).convert("L")
                img = img.resize((size, size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
                t   = torch.from_numpy(arr).unsqueeze(0)         # (1, H, W)
                return t.repeat(3, 1, 1)                         # (3, H, W)
            except Exception as e:
                warnings.warn(f"Failed to load anomaly map {fpath}: {e}")
    return torch.zeros(3, size, size)


# ---------------------------------------------------------------------------
# RETFound embedding loader
# ---------------------------------------------------------------------------

def _load_retfound_emb(
    image_path: Path,
    class_name: str,
    split:      str,
    cache_dir:  Optional[str],
) -> torch.Tensor:
    """
    Loads a pre-cached 1024-D RETFound embedding from disk.
    Cache filename format: <split>_<class>_<stem>.pt
    Falls back to zeros with a warning on cache miss.
    """
    if cache_dir is None:
        return torch.zeros(1024)

    stem      = image_path.stem
    cache_key = f"{split}_{class_name}_{stem}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.isfile(cache_path):
        try:
            return torch.load(cache_path, map_location="cpu").float()
        except Exception as e:
            warnings.warn(f"Failed to load RETFound cache {cache_path}: {e}")

    # Fallback: bare stem (no split/class prefix)
    alt_path = os.path.join(cache_dir, f"{stem}.pt")
    if os.path.isfile(alt_path):
        try:
            return torch.load(alt_path, map_location="cpu").float()
        except Exception as e:
            warnings.warn(f"Failed to load RETFound cache {alt_path}: {e}")

    return torch.zeros(1024)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RetinalDataset(Dataset):
    """
    Returns (six_ch, retfound_emb, label_vec, path) per image.

    Args:
        root              Path to class-organised folder (train/ or val/)
        anomaly_dir       Path to flat directory of anomaly map PNGs
        retfound_cache_dir Path to pre-cached RETFound .pt embedding files
                          (None → zero tensors returned for retfound_emb)
        split             "train" or "val" — used to build RETFound cache keys
        image_size        Target spatial resolution (default 512)
        augment           Apply random augmentation (train only)
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root:               str,
        anomaly_dir:        str,
        retfound_cache_dir: Optional[str] = None,
        split:              str  = "train",
        image_size:         int  = 512,
        augment:            bool = False,
        mean:               tuple = (0.485, 0.456, 0.406),
        std:                tuple = (0.229, 0.224, 0.225),
    ):
        self.root               = Path(root)
        self.anomaly_dir        = anomaly_dir
        self.retfound_cache_dir = retfound_cache_dir
        self.split              = split
        self.image_size         = image_size
        self.augment            = augment

        self.normalise = T.Normalize(mean=mean, std=std)

        # Collect (path, class_name, label_idx) triples
        self.samples: list[tuple[Path, str, int]] = []
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                warnings.warn(f"Class folder not found: {class_dir}")
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((p, class_name, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

        # Per-sample weights for WeightedRandomSampler
        self._class_counts = self._count_classes()
        self.weights       = self._compute_weights()

    def _count_classes(self) -> dict:
        counts = {c: 0 for c in CLASS_TO_IDX.values()}
        for _, _, lbl in self.samples:
            counts[lbl] += 1
        return counts

    def _compute_weights(self) -> list:
        """Weight = 1/class_count → balanced batches via WeightedRandomSampler."""
        total = len(self.samples)
        w = []
        for _, _, lbl in self.samples:
            cnt = self._class_counts[lbl]
            w.append(total / (NUM_CLASSES * cnt))
        return w

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        """Returns (3, H, W) float in [0, 1] at image_size resolution."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(img)   # (3, H, W) in [0, 1]

    def _augment(
        self,
        image:    torch.Tensor,   # (3, H, W)
        residual: torch.Tensor,   # (3, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial augmentations applied consistently to both image and residual.
        Colour jitter applied to image only (residual is a heatmap, not a photo).
        """
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image    = TF.hflip(image)
            residual = TF.hflip(residual)

        # Random vertical flip
        if torch.rand(1) < 0.3:
            image    = TF.vflip(image)
            residual = TF.vflip(residual)

        # Random rotation ±15°
        angle    = float(torch.FloatTensor(1).uniform_(-15, 15))
        image    = TF.rotate(image,    angle, interpolation=TF.InterpolationMode.BILINEAR)
        residual = TF.rotate(residual, angle, interpolation=TF.InterpolationMode.BILINEAR)

        # Colour jitter on image only (retina-safe ranges)
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
        image  = jitter(image)

        # Random erasing on image only (simulates imaging artefacts)
        eraser = T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)
        image  = eraser(image)

        return image, residual

    def __getitem__(self, idx: int) -> dict:
        path, class_name, label_idx = self.samples[idx]

        # ── Load image (3ch RGB) ──────────────────────────────────────────
        image    = self._load_image(path)                                              # (3, H, W) [0,1]

        # ── Load clean residual (3ch) ─────────────────────────────────────
        residual = _load_anomaly_map(path.stem, self.anomaly_dir, self.image_size)     # (3, H, W) [0,1]

        # ── Augment (jointly, same spatial transform) ─────────────────────
        if self.augment:
            image, residual = self._augment(image, residual)

        # ── Normalise image (ImageNet stats), keep residual in [0,1] ──────
        image = self.normalise(image)

        # ── Stack → 6-channel tensor ──────────────────────────────────────
        six_ch = torch.cat([image, residual], dim=0)   # (6, H, W)

        # ── Load RETFound embedding ───────────────────────────────────────
        retfound_emb = _load_retfound_emb(
            path, class_name, self.split, self.retfound_cache_dir
        )   # (1024,)

        # ── Multi-label one-hot vector ────────────────────────────────────
        # [1,0,0]=DR  [0,1,0]=Glauc  [0,0,1]=PM
        # In the future comorbid images can have multiple 1s.
        label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label_vec[label_idx] = 1.0

        return {
            "six_ch":       six_ch,          # (6, 512, 512)
            "retfound_emb": retfound_emb,    # (1024,)
            "label_vec":    label_vec,        # (3,) float multi-hot
            "label":        label_idx,        # int — kept for backwards compat / metrics
            "path":         str(path),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    root:               str,
    anomaly_dir:        str,
    retfound_cache_dir: Optional[str] = None,
    split:              str  = "train",
    image_size:         int  = 512,
    batch_size:         int  = 16,
    augment:            bool = False,
    balanced:           bool = True,
    num_workers:        int  = 4,
    pin_memory:         bool = True,
    drop_last:          bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for the retinal dataset.

    Training  (balanced=True):
        WeightedRandomSampler enforces 1:1:1 (DR:Glaucoma:PM) per batch.
        drop_last=True prevents incomplete final batches.

    Validation (balanced=False):
        Sequential iteration — every sample once, unbiased distribution.
        drop_last=False so no samples are skipped.
    """
    dataset = RetinalDataset(
        root               = root,
        anomaly_dir        = anomaly_dir,
        retfound_cache_dir = retfound_cache_dir,
        split              = split,
        image_size         = image_size,
        augment            = augment,
    )

    if balanced:
        sampler = WeightedRandomSampler(
            weights    = dataset.weights,
            num_samples = len(dataset),
            replacement = True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size   = batch_size,
        sampler      = sampler,
        shuffle      = shuffle,
        num_workers  = num_workers,
        pin_memory   = pin_memory,
        drop_last    = drop_last,
        persistent_workers = (num_workers > 0),
    )

    counts = dataset._class_counts
    print(
        f"  DataLoader [{('balanced' if balanced else 'sequential')}] [{split}]: "
        f"{len(dataset)} images | "
        f"DR={counts[0]}  Glaucoma={counts[1]}  PM={counts[2]} | "
        f"batch={batch_size} | "
        f"retfound_cache={'yes' if retfound_cache_dir else 'no (zeros)'}"
    )
    return loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import random

    with tempfile.TemporaryDirectory() as tmpdir:
        amap_dir = os.path.join(tmpdir, "amaps")
        os.makedirs(amap_dir)

        for cls in CLASS_TO_IDX:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir)
            for i in range(6):
                p  = os.path.join(cls_dir, f"img_{cls}_{i:03d}.png")
                ap = os.path.join(amap_dir, f"img_{cls}_{i:03d}_anomaly.png")
                Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                ).save(p)
                Image.fromarray(
                    np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                ).save(ap)

        loader = build_dataloader(
            root       = tmpdir,
            anomaly_dir = amap_dir,
            split      = "train",
            image_size = 64,
            batch_size = 4,
            augment    = True,
            num_workers = 0,
        )
        batch = next(iter(loader))
        print(f"six_ch:       {batch['six_ch'].shape}")         # (4, 6, 64, 64)
        print(f"retfound_emb: {batch['retfound_emb'].shape}")   # (4, 1024)
        print(f"label_vec:    {batch['label_vec'].shape}")       # (4, 3)
        print(f"label:        {batch['label']}")
        assert batch['six_ch'].shape      == (4, 6, 64, 64)
        assert batch['retfound_emb'].shape == (4, 1024)
        assert batch['label_vec'].shape   == (4, 3)
        print("data.py — all checks passed ✓")
