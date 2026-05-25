"""
NetrAi Classifier — Feature Extraction (v2)
=============================================
Phase 2: extract frozen 256-D vectors for XGBoost.

Steps:
  1. Load the best trained NetrAiEncoder checkpoint.
  2. Freeze all weights. Set to eval mode (VIBs become deterministic μ).
  3. For each image:
       - Load 6-channel stacked tensor
       - Load pre-cached RETFound 1024-D embedding
       - Forward pass → z_fused (256-D)
  4. Save:
       features/train_features.npy   (N_train, 256)
       features/train_labels.npy     (N_train, 3)   multi-hot float
       features/train_labels_int.npy (N_train,)     integer class index
       features/train_stems.json     image stems (for traceability)
       features/val_features.npy     (N_val,   256)
       features/val_labels.npy       (N_val,   3)
       features/val_labels_int.npy   (N_val,)
       features/val_stems.json

These .npy files are the direct input to XGBoost training (Phase 2).

Run:
    python -m classifier extract --config classifier/config.yaml
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model  import NetrAiEncoder
from .data   import build_dataloader
from .utils  import (
    load_config, load_checkpoint,
    get_device, get_amp_dtype,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model:      NetrAiEncoder,
    loader:     DataLoader,
    device:     torch.device,
    amp_dtype:  torch.dtype,
    split_name: str = "split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Extracts 256-D fused vectors (z_fused = z1 ⊕ z2) in deterministic
    eval mode (VIBs output μ only, no noise).

    Returns:
        features    np.ndarray  (N, 256)  float32
        labels_vec  np.ndarray  (N, 3)    float32 multi-hot
        labels_int  np.ndarray  (N,)      int32   class index
        stems       list[str]   image stems
    """
    model.eval()   # VIBs → deterministic mode (z = μ, ε = 0)

    all_feats    = []
    all_lbls_vec = []
    all_lbls_int = []
    all_stems    = []

    for batch in tqdm(loader, desc=f"  Extracting [{split_name}]", unit="batch"):
        six_ch       = batch["six_ch"].to(device,  non_blocking=True)
        retfound_emb = batch["retfound_emb"].to(device, non_blocking=True)
        label_vec    = batch["label_vec"].cpu().numpy()    # (B, 3)
        label_int    = batch["label"].cpu().numpy()        # (B,)
        paths        = batch["path"]

        stems = [Path(p).stem for p in paths]

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            z_fused, z1, mu1, lv1, mu2, lv2 = model(six_ch, retfound_emb)

        # z_fused is deterministic μ in eval mode — stable for XGBoost
        feats = z_fused.cpu().float().numpy()   # (B, 256)

        all_feats.append(feats)
        all_lbls_vec.append(label_vec)
        all_lbls_int.append(label_int)
        all_stems.extend(stems)

    features   = np.vstack(all_feats).astype(np.float32)      # (N, 256)
    labels_vec = np.vstack(all_lbls_vec).astype(np.float32)   # (N, 3)
    labels_int = np.concatenate(all_lbls_int).astype(np.int32) # (N,)

    return features, labels_vec, labels_int, all_stems


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_features(
    features:   np.ndarray,
    labels_vec: np.ndarray,
    labels_int: np.ndarray,
    stems:      list[str],
    output_dir: str,
    prefix:     str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"{prefix}_features.npy"),   features)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"),     labels_vec)
    np.save(os.path.join(output_dir, f"{prefix}_labels_int.npy"), labels_int)
    with open(os.path.join(output_dir, f"{prefix}_stems.json"), "w") as f:
        json.dump(stems, f, indent=2)

    class_counts = {int(c): int((labels_int == c).sum()) for c in np.unique(labels_int)}
    print(
        f"  Saved {prefix}: features={features.shape}  labels={class_counts}  "
        f"→ {output_dir}"
    )


def load_features(
    output_dir: str,
    prefix:     str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features   = np.load(os.path.join(output_dir, f"{prefix}_features.npy"))
    labels_vec = np.load(os.path.join(output_dir, f"{prefix}_labels.npy"))
    labels_int = np.load(os.path.join(output_dir, f"{prefix}_labels_int.npy"))
    with open(os.path.join(output_dir, f"{prefix}_stems.json")) as f:
        stems = json.load(f)
    return features, labels_vec, labels_int, stems


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Phase 2: Extract 256-D feature vectors")
    parser.add_argument("--config",     default="classifier/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SegFormer checkpoint (default: best.pt)")
    parser.add_argument("--splits",     nargs="+", default=["train", "val"])
    args = parser.parse_args(args)

    cfg    = load_config(args.config)
    logger = setup_logging(cfg['paths']['checkpoint_dir'], name="extract")
    device = get_device()
    amp_dt = get_amp_dtype(device)

    # ---- Load model ----
    mc = cfg['model']
    model = NetrAiEncoder(
        backbone_name = mc['backbone'],
        head_out_dim  = mc['head_out_dim'],
        vib_hidden    = mc['vib_hidden'],
        vib_out_dim   = mc['vib_out_dim'],
        dropout       = mc.get('dropout', 0.3),
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(
        cfg['paths']['checkpoint_dir'], "best.pt"
    )
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    logger.info(f"  Checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Freeze all — purely feature extractor from here
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
            root               = split_dir,
            anomaly_dir        = p['anomaly_maps_dir'],
            retfound_cache_dir = p.get('retfound_cache_dir'),
            split              = split,
            image_size         = dc['image_size'],
            batch_size         = tc['batch_size'],
            augment            = False,
            balanced           = False,
            num_workers        = dc['num_workers'],
            pin_memory         = dc['pin_memory'],
            drop_last          = False,
        )

        features, labels_vec, labels_int, stems = extract_features(
            model      = model,
            loader     = loader,
            device     = device,
            amp_dtype  = amp_dt,
            split_name = split,
        )

        save_features(
            features   = features,
            labels_vec = labels_vec,
            labels_int = labels_int,
            stems      = stems,
            output_dir = p['features_dir'],
            prefix     = split,
        )

    logger.info("Feature extraction complete ✓")
    logger.info(f"  Feature dim: 256-D  (128-D VIB1 ⊕ 128-D VIB2)")


if __name__ == "__main__":
    main()
