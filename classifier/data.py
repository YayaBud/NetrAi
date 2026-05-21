"""
NetrAi Classifier — Dataset & DataLoader
==========================================
Expected on-disk layout:

    data/
    ├── train/
    │   ├── DR/          ← class folder
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

Anomaly maps (clean residuals from the diffusion model):
  - Greyscale or hot-cmap PNG saved by visualization.py
  - Values in [0, 255] on disk → loaded and normalised to [0, 1]
  - If a map is not found, a zero tensor is used as fallback
    (the gate will pass F_concat unchanged: F + α·(F⊙0) = F)

Class balance:
  - WeightedRandomSampler enforces strict 1:1:1 (DR:Glaucoma:PM) per batch
"""

import os
import math
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


# ---------------------------------------------------------------------------
# Anomaly map loader helper
# ---------------------------------------------------------------------------

def _load_anomaly_map(stem: str, anomaly_dir: str, size: int) -> torch.Tensor:
    """
    Tries <stem>_anomaly.png then <stem>.png.
    Returns a (1, size, size) float tensor in [0, 1].
    Falls back to zeros if not found.
    """
    for suffix in (f"{stem}_anomaly.png", f"{stem}.png"):
        fpath = os.path.join(anomaly_dir, suffix)
        if os.path.isfile(fpath):
            try:
                img = Image.open(fpath).convert("L")           # greyscale
                img = img.resize((size, size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
                return torch.from_numpy(arr).unsqueeze(0)       # (1, H, W)
            except Exception as e:
                warnings.warn(f"Failed to load anomaly map {fpath}: {e}")
    return torch.zeros(1, size, size)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RetinalDataset(Dataset):
    """
    Loads (image, anomaly_map, label) triples.

    Args:
        root         Path to class-organised folder (train/ or val/)
        anomaly_dir  Path to flat directory of anomaly map PNGs
        image_size   Target spatial resolution (default 512)
        augment      Apply random augmentation (train only)
        mean / std   ImageNet statistics
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root:        str,
        anomaly_dir: str,
        image_size:  int   = 512,
        augment:     bool  = False,
        mean:        tuple = (0.485, 0.456, 0.406),
        std:         tuple = (0.229, 0.224, 0.225),
    ):
        self.root        = Path(root)
        self.anomaly_dir = anomaly_dir
        self.image_size  = image_size
        self.augment     = augment

        self.normalise = T.Normalize(mean=mean, std=std)

        # Collect all (path, label) pairs
        self.samples: list[tuple[Path, int]] = []
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                warnings.warn(f"Class folder not found: {class_dir}")
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((p, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

        # Pre-compute per-sample weights for WeightedRandomSampler
        self._class_counts = self._count_classes()
        self.weights = self._compute_weights()

    def _count_classes(self) -> dict:
        counts = {c: 0 for c in CLASS_TO_IDX.values()}
        for _, lbl in self.samples:
            counts[lbl] += 1
        return counts

    def _compute_weights(self) -> list:
        """Assigns weight = 1/class_count to each sample → balanced batches."""
        total = len(self.samples)
        w = []
        for _, lbl in self.samples:
            cnt = self._class_counts[lbl]
            w.append(total / (len(CLASS_TO_IDX) * cnt))
        return w

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return TF.to_tensor(img)  # (3, H, W) in [0, 1]

    def _augment(
        self, image: torch.Tensor, amap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial augmentations applied consistently to both image and anomaly map.
        Colour jitter applied to image only (anomaly map is greyscale intensity).
        """
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            amap  = TF.hflip(amap)

        # Random vertical flip
        if torch.rand(1) < 0.3:
            image = TF.vflip(image)
            amap  = TF.vflip(amap)

        # Random rotation ±15°
        angle = float(torch.FloatTensor(1).uniform_(-15, 15))
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        amap  = TF.rotate(amap,  angle, interpolation=TF.InterpolationMode.BILINEAR)

        # Colour jitter on image only (retina-safe ranges)
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
        image  = jitter(image)

        # Random erasing (simulates imaging artefacts)
        eraser = T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)
        image  = eraser(image)

        return image, amap

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]

        image = self._load_image(path)                                   # (3, H, W)
        amap  = _load_anomaly_map(path.stem, self.anomaly_dir, self.image_size)  # (1, H, W)

        if self.augment:
            image, amap = self._augment(image, amap)

        image = self.normalise(image)

        return {
            "image":       image,           # (3, 512, 512)
            "anomaly_map": amap,            # (1, 512, 512)  ∈ [0, 1]
            "label":       torch.tensor(label, dtype=torch.long),
            "path":        str(path),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    root:        str,
    anomaly_dir: str,
    image_size:  int  = 512,
    batch_size:  int  = 16,
    augment:     bool = False,
    balanced:    bool = True,    # True → WeightedRandomSampler (train)
                                  # False → sequential, every sample once (val)
    num_workers: int  = 4,
    pin_memory:  bool = True,
    drop_last:   bool = True,
) -> DataLoader:
    """
    Returns a DataLoader.

    Training  (balanced=True):
        WeightedRandomSampler enforces strict 1:1:1 (DR:Glaucoma:PM) sampling
        with replacement.  drop_last=True prevents incomplete final batches
        from destabilising SupCon loss.

    Validation (balanced=False):
        Plain sequential iteration — every sample is visited exactly once,
        no repetition.  Gives an unbiased evaluation on the real class
        distribution.  drop_last=False so no samples are silently skipped.
    """
    dataset = RetinalDataset(
        root=root,
        anomaly_dir=anomaly_dir,
        image_size=image_size,
        augment=augment,
    )

    if balanced:
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False   # sampler and shuffle are mutually exclusive
    else:
        sampler = None
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )

    counts = dataset._class_counts
    print(
        f"  DataLoader [{('balanced' if balanced else 'sequential')}]: "
        f"{len(dataset)} images | "
        f"DR={counts[0]}  Glaucoma={counts[1]}  PM={counts[2]} | "
        f"batch={batch_size}"
    )
    return loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, random
    from PIL import Image as PILImage

    # Create a tiny dummy dataset in a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        amap_dir = os.path.join(tmpdir, "amaps")
        os.makedirs(amap_dir)
        for cls in CLASS_TO_IDX:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir)
            for i in range(6):
                p = os.path.join(cls_dir, f"img_{cls}_{i:03d}.png")
                PILImage.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                ).save(p)
                # Matching anomaly map
                ap = os.path.join(amap_dir, f"img_{cls}_{i:03d}_anomaly.png")
                PILImage.fromarray(
                    np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                ).save(ap)

        loader = build_dataloader(
            root=tmpdir, anomaly_dir=amap_dir,
            image_size=64, batch_size=4, augment=True, num_workers=0,
        )
        batch = next(iter(loader))
        print(f"image:       {batch['image'].shape}")
        print(f"anomaly_map: {batch['anomaly_map'].shape}")
        print(f"label:       {batch['label']}")
        print("data.py — all checks passed ✓")
