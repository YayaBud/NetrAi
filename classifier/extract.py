"""
NetrAi Classifier — Feature Extraction
=========================================
After SegFormer training is complete, this script:

  1. Loads the best trained NetrAiEncoder checkpoint.
  2. Runs every image in the dataset through it (frozen, eval mode).
  3. For each image, loads its cached RETFound embedding from disk.
  4. Concatenates → 1793-D vector.
  5. Saves:
       features/train_features.npy   (N_train, 1793)
       features/train_labels.npy     (N_train,)
       features/train_stems.json     list of image stems (for debugging)
       features/val_features.npy     (N_val,   1793)
       features/val_labels.npy       (N_val,)
       features/val_stems.json

These .npy files are the direct input to XGBoost training.

Run:
    python -m classifier extract --config classifier/config.yaml
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model    import NetrAiEncoder
from .data     import build_dataloader, RetinalDataset
from .retfound import load_cached_embeddings_batch, make_cache_key
from .utils    import (
    load_config, load_checkpoint,
    get_device, get_amp_dtype,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Feature extraction logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model:         NetrAiEncoder,
    loader:        DataLoader,
    retfound_dir:  str,
    device:        torch.device,
    amp_dtype:     torch.dtype,
    split_name:    str = "split",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extracts 769-D SegFormer vectors and concatenates 1024-D RETFound
    embeddings to produce final 1793-D feature matrix.

    Returns:
        features  np.ndarray  (N, 1793)
        labels    np.ndarray  (N,)
        stems     list[str]   image stems (for traceability)
    """
    model.eval()

    all_feats  = []
    all_labels = []
    all_stems  = []

    warned_missing = set()

    for batch in tqdm(loader, desc=f"  Extracting [{split_name}]", unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        amap   = batch["anomaly_map"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        paths  = batch["path"]

        stems = [Path(p).stem for p in paths]

        # Build collision-safe cache keys: "train_DR_img_001" etc.
        cache_keys = [make_cache_key(p, split=split_name) for p in paths]

        # --- SegFormer → 769-D ---
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            vector_769, mu, _ = model(images, amap)
        seg_np = vector_769.cpu().float().numpy()  # (B, 769)

        # --- RETFound cache → 1024-D ---
        retfound_np, found_flags = load_cached_embeddings_batch(cache_keys, retfound_dir)

        for i, (stem, found) in enumerate(zip(stems, found_flags)):
            if not found and stem not in warned_missing:
                warned_missing.add(stem)
                print(f"  [WARN] RETFound cache missing for: {stem} (key={cache_keys[i]}) — using zeros")

        # --- Concatenate → 1793-D ---
        combined = np.concatenate([seg_np, retfound_np], axis=1)  # (B, 1793)

        all_feats.append(combined)
        all_labels.append(labels)
        all_stems.extend(stems)

    features = np.vstack(all_feats).astype(np.float32)  # (N, 1793)
    labels   = np.concatenate(all_labels).astype(np.int32)

    return features, labels, all_stems


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_features(
    features:   np.ndarray,
    labels:     np.ndarray,
    stems:      list[str],
    output_dir: str,
    prefix:     str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    feat_path  = os.path.join(output_dir, f"{prefix}_features.npy")
    label_path = os.path.join(output_dir, f"{prefix}_labels.npy")
    stem_path  = os.path.join(output_dir, f"{prefix}_stems.json")

    np.save(feat_path,  features)
    np.save(label_path, labels)
    with open(stem_path, "w") as f:
        json.dump(stems, f, indent=2)

    class_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(
        f"  Saved {prefix}: {features.shape}  labels={class_counts}  "
        f"→ {feat_path}"
    )


def load_features(
    output_dir: str,
    prefix:     str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = np.load(os.path.join(output_dir, f"{prefix}_features.npy"))
    labels   = np.load(os.path.join(output_dir, f"{prefix}_labels.npy"))
    with open(os.path.join(output_dir, f"{prefix}_stems.json")) as f:
        stems = json.load(f)
    return features, labels, stems


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Extract 1793-D feature vectors")
    parser.add_argument("--config",     default="classifier/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SegFormer checkpoint (default: best.pt)")
    parser.add_argument("--splits",     nargs="+", default=["train", "val"],
                        help="Which splits to process")
    args = parser.parse_args(args)

    cfg    = load_config(args.config)
    logger = setup_logging(cfg['paths']['checkpoint_dir'], name="extract")
    device = get_device()
    amp_dt = get_amp_dtype(device)

    # ---- Load model ----
    mc = cfg['model']
    model = NetrAiEncoder(
        backbone_name     = mc['backbone'],
        decoder_embed_dim = mc['decoder_embed_dim'],
        path_a_dim        = mc['path_a_dim'],
        path_b_dim        = mc['path_b_dim'],
        alpha_init        = mc['alpha_init'],
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(
        cfg['paths']['checkpoint_dir'], "best.pt"
    )
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    logger.info(f"  Checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Freeze model — purely a feature extractor from here
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Extract each split ----
    p  = cfg['paths']
    dc = cfg['data']
    tc = cfg['training']

    for split in args.splits:
        split_dir = os.path.join(p['data_dir'], split)
        if not os.path.isdir(split_dir):
            logger.warning(f"Split directory not found, skipping: {split_dir}")
            continue

        loader = build_dataloader(
            root        = split_dir,
            anomaly_dir = p['anomaly_maps_dir'],
            image_size  = dc['image_size'],
            batch_size  = tc['batch_size'],
            augment     = False,      # always inference-mode for extraction
            balanced    = False,      # sequential — every sample exactly once
            num_workers = dc['num_workers'],
            pin_memory  = dc['pin_memory'],
            drop_last   = False,      # we need every sample
        )

        features, labels, stems = extract_features(
            model        = model,
            loader       = loader,
            retfound_dir = p['retfound_cache_dir'],
            device       = device,
            amp_dtype    = amp_dt,
            split_name   = split,
        )

        save_features(features, labels, stems, p['features_dir'], prefix=split)

    logger.info("Feature extraction complete ✓")


if __name__ == "__main__":
    main()
