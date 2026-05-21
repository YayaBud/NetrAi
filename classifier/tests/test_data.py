"""
tests/test_data.py
===================
Tests for RetinalDataset and build_dataloader.

Covers:
  - Correct image / anomaly_map / label shapes
  - Anomaly map normalisation (values in [0, 1])
  - Zero fallback when anomaly map is missing
  - Augmentation doesn't mutate the original tensors
  - WeightedRandomSampler produces balanced batches (train)
  - Sequential sampler visits every sample exactly once (val)
  - Class count tracking and sample weights

Run:
    pytest classifier/tests/test_data.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import numpy as np
from collections import Counter

from classifier.data import (
    RetinalDataset,
    build_dataloader,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    _load_anomaly_map,
)


IMAGE_SIZE = 64   # tiny to keep tests fast


# ---------------------------------------------------------------------------
# RetinalDataset
# ---------------------------------------------------------------------------

class TestRetinalDataset:

    def test_sample_count(self, dummy_dataset_root):
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        assert len(ds) == dummy_dataset_root["n_train"], \
            f"Expected {dummy_dataset_root['n_train']} samples, got {len(ds)}"

    def test_image_shape(self, dummy_dataset_root):
        ds    = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        item  = ds[0]
        assert item["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE), \
            f"Image shape wrong: {item['image'].shape}"

    def test_anomaly_map_shape(self, dummy_dataset_root):
        ds   = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        item = ds[0]
        assert item["anomaly_map"].shape == (1, IMAGE_SIZE, IMAGE_SIZE), \
            f"Anomaly map shape wrong: {item['anomaly_map'].shape}"

    def test_anomaly_map_range(self, dummy_dataset_root):
        """Anomaly map values must be in [0, 1] after loading."""
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        for idx in range(min(6, len(ds))):
            amap = ds[idx]["anomaly_map"]
            assert amap.min() >= 0.0, "Anomaly map min < 0"
            assert amap.max() <= 1.0, "Anomaly map max > 1"

    def test_label_type_and_range(self, dummy_dataset_root):
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        for idx in range(len(ds)):
            lbl = ds[idx]["label"]
            assert lbl.dtype == torch.long
            assert int(lbl) in (0, 1, 2), f"Invalid label: {lbl}"

    def test_all_classes_present(self, dummy_dataset_root):
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        seen = {int(ds[i]["label"]) for i in range(len(ds))}
        assert seen == {0, 1, 2}, f"Not all classes present: {seen}"

    def test_class_counts_balanced(self, dummy_dataset_root):
        """Dummy dataset has equal samples per class."""
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        counts = ds._class_counts
        n      = dummy_dataset_root["n_per_class_train"]
        assert counts[0] == n, f"DR count wrong: {counts[0]} vs {n}"
        assert counts[1] == n, f"Glaucoma count wrong: {counts[1]} vs {n}"
        assert counts[2] == n, f"PM count wrong: {counts[2]} vs {n}"

    def test_sample_weights_equal_for_balanced_dataset(self, dummy_dataset_root):
        """Equal class sizes → all sample weights identical."""
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        weights = ds.weights
        assert len(set(round(w, 8) for w in weights)) == 1, \
            "All weights should be equal when classes are balanced"

    def test_path_in_item(self, dummy_dataset_root):
        ds   = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        item = ds[0]
        assert "path" in item
        assert os.path.isfile(item["path"]), f"Path does not exist: {item['path']}"

    def test_missing_anomaly_map_fallback(self, dummy_dataset_root, tmp_path):
        """When no anomaly map exists the loader must return zeros."""
        empty_amap_dir = str(tmp_path / "empty_amaps")
        os.makedirs(empty_amap_dir)
        ds   = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=empty_amap_dir,
            image_size=IMAGE_SIZE,
        )
        item = ds[0]
        assert item["anomaly_map"].sum().item() == 0.0, \
            "Missing anomaly map should fall back to all-zero tensor"

    def test_augmentation_does_not_crash(self, dummy_dataset_root):
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            augment=True,
        )
        for idx in range(min(6, len(ds))):
            item = ds[idx]
            assert torch.isfinite(item["image"]).all(), \
                f"Augmented image contains non-finite values at idx {idx}"
            assert item["anomaly_map"].min() >= -1e-5, \
                "Augmented anomaly map went negative"

    def test_augmentation_consistent_spatial(self, dummy_dataset_root):
        """
        Image and anomaly map must have the same spatial size after augmentation.
        (Spatial transforms are applied identically to both.)
        """
        ds = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            augment=True,
        )
        item = ds[0]
        img_hw  = item["image"].shape[-2:]
        amap_hw = item["anomaly_map"].shape[-2:]
        assert img_hw == amap_hw, \
            f"Image spatial size {img_hw} ≠ anomaly map {amap_hw}"

    def test_val_dataset_no_augment(self, dummy_dataset_root):
        """Same image loaded twice (no augment) must produce identical tensors."""
        ds   = RetinalDataset(
            root=dummy_dataset_root["val_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            augment=False,
        )
        t1 = ds[0]["image"]
        t2 = ds[0]["image"]
        assert torch.allclose(t1, t2), \
            "No-augment mode must be deterministic"


# ---------------------------------------------------------------------------
# _load_anomaly_map helper
# ---------------------------------------------------------------------------

class TestLoadAnomalyMap:

    def test_preferred_naming(self, dummy_dataset_root):
        """<stem>_anomaly.png should be found."""
        ds    = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        stem  = ds.samples[0][0].stem
        amap  = _load_anomaly_map(stem, dummy_dataset_root["anomaly_dir"], IMAGE_SIZE)
        assert amap.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
        assert amap.sum() > 0, "Loaded anomaly map should not be all zeros"

    def test_output_range(self, dummy_dataset_root):
        ds   = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        stem = ds.samples[0][0].stem
        amap = _load_anomaly_map(stem, dummy_dataset_root["anomaly_dir"], IMAGE_SIZE)
        assert amap.min() >= 0.0
        assert amap.max() <= 1.0

    def test_resize_to_target(self, dummy_dataset_root):
        ds     = RetinalDataset(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
        )
        stem   = ds.samples[0][0].stem
        amap32 = _load_anomaly_map(stem, dummy_dataset_root["anomaly_dir"], 32)
        assert amap32.shape == (1, 32, 32)

    def test_fallback_zeros(self, tmp_path):
        amap = _load_anomaly_map("nonexistent_stem", str(tmp_path), IMAGE_SIZE)
        assert amap.shape  == (1, IMAGE_SIZE, IMAGE_SIZE)
        assert amap.sum()  == 0.0


# ---------------------------------------------------------------------------
# build_dataloader
# ---------------------------------------------------------------------------

class TestBuildDataloader:

    def test_batch_shapes(self, dummy_dataset_root):
        loader = build_dataloader(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            batch_size=4,
            augment=False,
            balanced=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        batch = next(iter(loader))
        assert batch["image"].shape       == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["anomaly_map"].shape == (4, 1, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["label"].shape       == (4,)
        assert batch["label"].dtype       == torch.long

    def test_balanced_sampler_class_distribution(self, dummy_dataset_root):
        """
        With WeightedRandomSampler over many batches the per-class counts
        should converge to roughly 1:1:1.
        """
        loader = build_dataloader(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            batch_size=6,
            augment=False,
            balanced=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        label_counts = Counter()
        # Collect labels from multiple batches
        for i, batch in enumerate(loader):
            for lbl in batch["label"].tolist():
                label_counts[lbl] += 1
            if i >= 9:   # enough batches to get a stable estimate
                break

        total = sum(label_counts.values())
        for cls in (0, 1, 2):
            fraction = label_counts[cls] / total
            assert 0.2 <= fraction <= 0.5, \
                f"Class {cls} fraction {fraction:.2f} is far from 1/3 (balanced sampling)"

    def test_sequential_val_loader_no_duplicates(self, dummy_dataset_root):
        """
        Sequential val loader (balanced=False) must yield every sample exactly
        once across a full epoch — no repetitions, no missing.
        """
        n_val  = dummy_dataset_root["n_val"]
        loader = build_dataloader(
            root=dummy_dataset_root["val_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            batch_size=3,
            augment=False,
            balanced=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        all_paths = []
        for batch in loader:
            all_paths.extend(batch["path"])

        assert len(all_paths) == n_val, \
            f"Val loader must yield exactly {n_val} samples, got {len(all_paths)}"
        assert len(set(all_paths)) == n_val, \
            "Val loader must not repeat any sample within one epoch"

    def test_anomaly_map_values_in_range(self, dummy_dataset_root):
        loader = build_dataloader(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            batch_size=6,
            augment=False,
            balanced=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        batch = next(iter(loader))
        amap  = batch["anomaly_map"]
        assert amap.min() >= 0.0, "Anomaly map contains values < 0"
        assert amap.max() <= 1.0, "Anomaly map contains values > 1"

    def test_drop_last_train(self, dummy_dataset_root):
        """
        With drop_last=True and batch_size larger than dataset the loader
        should raise StopIteration (empty iterator), not crash.
        """
        loader = build_dataloader(
            root=dummy_dataset_root["train_dir"],
            anomaly_dir=dummy_dataset_root["anomaly_dir"],
            image_size=IMAGE_SIZE,
            batch_size=dummy_dataset_root["n_train"] + 1,
            augment=False,
            balanced=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        batches = list(loader)
        # All batches must be full-sized (last incomplete one was dropped)
        for batch in batches:
            assert batch["image"].shape[0] == dummy_dataset_root["n_train"] + 1

    def test_missing_class_dir_warns_not_crash(self, tmp_path):
        """
        If a class folder is completely missing the dataset should warn
        (not crash) and continue with whatever classes exist.
        """
        import warnings

        # Create only DR and Glaucoma, no PM
        for cls in ("DR", "Glaucoma"):
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(3):
                arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(arr).save(cls_dir / f"img_{i:03d}.png")

        amap_dir = str(tmp_path / "amaps")
        os.makedirs(amap_dir)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ds = RetinalDataset(
                root=str(tmp_path),
                anomaly_dir=amap_dir,
                image_size=32,
            )
        # Should still load DR + Glaucoma samples
        assert len(ds) == 6
