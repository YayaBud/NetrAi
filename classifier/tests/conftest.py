"""
tests/conftest.py
==================
Shared pytest fixtures — temporary filesystem helpers that create a
minimal dummy dataset without touching any real data paths.

All tests that need a dataset should use the `dummy_dataset_root` fixture,
which creates:

    <tmpdir>/
    ├── train/
    │   ├── DR/         6 images
    │   ├── Glaucoma/   6 images
    │   └── PM/         6 images
    └── val/
        ├── DR/         3 images
        ├── Glaucoma/   3 images
        └── PM/         3 images

    <tmpdir>/anomaly_maps/
        <stem>_anomaly.png   — matching greyscale map for every image
"""

import os
import json
import tempfile
import shutil

import numpy as np
import pytest
from PIL import Image as PILImage


CLASS_NAMES = ["DR", "Glaucoma", "PM"]


def _make_dummy_image(path: str, size: int = 64) -> None:
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path)


def _make_dummy_anomaly(path: str, size: int = 64) -> None:
    arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)
    PILImage.fromarray(arr, mode="L").save(path)


@pytest.fixture(scope="session")
def dummy_dataset_root():
    """
    Session-scoped fixture: creates the full directory tree once and
    tears it down after the entire test session completes.
    """
    tmpdir = tempfile.mkdtemp(prefix="netr_ai_test_")
    amap_dir = os.path.join(tmpdir, "anomaly_maps")
    os.makedirs(amap_dir)

    counts = {"train": 6, "val": 3}

    for split, n in counts.items():
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(tmpdir, split, cls)
            os.makedirs(cls_dir)
            for i in range(n):
                stem   = f"img_{split}_{cls}_{i:03d}"
                img_p  = os.path.join(cls_dir,  f"{stem}.png")
                amap_p = os.path.join(amap_dir, f"{stem}_anomaly.png")
                _make_dummy_image(img_p)
                _make_dummy_anomaly(amap_p)

    yield {
        "root":       tmpdir,
        "train_dir":  os.path.join(tmpdir, "train"),
        "val_dir":    os.path.join(tmpdir, "val"),
        "anomaly_dir": amap_dir,
        "n_train":    len(CLASS_NAMES) * counts["train"],   # 18
        "n_val":      len(CLASS_NAMES) * counts["val"],     # 9
        "n_per_class_train": counts["train"],
        "n_per_class_val":   counts["val"],
    }

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def dummy_feature_dir(dummy_dataset_root):
    """
    Creates fake .npy feature files (1793-D) and matching label arrays,
    so tests that need XGBoost input data don't have to run the full pipeline.
    """
    feat_dir = os.path.join(dummy_dataset_root["root"], "features")
    os.makedirs(feat_dir)

    rng = np.random.default_rng(42)

    for prefix, n in [("train", 18), ("val", 9)]:
        feats  = rng.standard_normal((n, 1793)).astype(np.float32)
        labels = np.array([i % 3 for i in range(n)], dtype=np.int32)
        stems  = [f"img_{prefix}_{i:03d}" for i in range(n)]

        np.save(os.path.join(feat_dir, f"{prefix}_features.npy"), feats)
        np.save(os.path.join(feat_dir, f"{prefix}_labels.npy"),   labels)
        with open(os.path.join(feat_dir, f"{prefix}_stems.json"), "w") as f:
            json.dump(stems, f)

    yield feat_dir
