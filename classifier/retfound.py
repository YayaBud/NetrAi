"""
NetrAi Classifier — RETFound Embedding Cache
=============================================
RETFound (Ma et al., 2023) is a ViT-Large/16 foundation model pretrained
on 1.6M retinal images using Masked Autoencoder self-supervision.

Pre-computation strategy:
  1. Load RETFound once, freeze it completely.
  2. Forward every image in the dataset → extract 1024-D [CLS] token.
  3. Save each embedding as  <image_stem>.pt  in the cache directory.
  4. Unload RETFound from VRAM — it is NEVER touched again during training.

At XGBoost training / inference:
  - Load the matching .pt file from disk.
  - Concatenate with the 769-D SegFormer vector → 1793-D XGBoost input.

Supported weight sources:
  A) Local .pth file (RETFound_cfp_weights.pth) — recommended
  B) HuggingFace ViT-L/16 ImageNet-21k  — fallback (domain gap exists)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# RETFound model wrapper
# ---------------------------------------------------------------------------

class RETFoundExtractor(nn.Module):
    """
    Wraps RETFound (or ViT-L/16 fallback) for 1024-D embedding extraction.

    The [CLS] token from the last transformer block is used as the
    global image representation.

    Args:
        weights_path  Path to local RETFound .pth checkpoint, or None.
        device        torch.device
    """

    RETFOUND_INPUT_SIZE = 224   # RETFound was trained at 224×224

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model(weights_path)
        self.model   = self.model.to(self.device).eval()

        # Freeze completely — no gradients, ever
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.isfile(weights_path):
            return self._load_retfound(weights_path)
        else:
            if weights_path:
                print(f"  [RETFound] weights not found at {weights_path}, using HF fallback.")
            else:
                print("  [RETFound] no weights path provided, using HF ViT-L/16 fallback.")
            return self._load_vit_fallback()

    def _load_retfound(self, path: str) -> nn.Module:
        """
        Load RETFound weights into a ViT-L/16 backbone.
        RETFound uses the timm MAE architecture.
        """
        try:
            import timm
            # ViT-Large/16 — matches RETFound architecture
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,        # remove classification head
                global_pool="token",  # use [CLS] token
            )
            checkpoint = torch.load(path, map_location="cpu")
            # RETFound checkpoints store model under 'model' key
            state_dict = checkpoint.get("model", checkpoint)
            # Strip MAE decoder keys if present
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("decoder")
            }
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"  [RETFound] loaded from {path}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
            return model

        except ImportError:
            print("  [RETFound] timm not installed — falling back to HuggingFace ViT-L/16.")
            return self._load_vit_fallback()

    def _load_vit_fallback(self) -> nn.Module:
        """HuggingFace ViT-L/16 as domain-gap fallback."""
        from transformers import ViTModel
        vit = ViTModel.from_pretrained(
            "google/vit-large-patch16-224-in21k",
            add_pooling_layer=False,
        )

        # Wrap with a forward that mirrors the timm [CLS] interface
        class _ViTWrapper(nn.Module):
            def __init__(self, vit):
                super().__init__()
                self.vit = vit

            def forward(self, x):
                out = self.vit(pixel_values=x)
                return out.last_hidden_state[:, 0]  # [CLS] token → (B, 1024)

        return _ViTWrapper(vit)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224) — RETFound input size
        Returns: (B, 1024) embeddings
        """
        return self.model(images)


# ---------------------------------------------------------------------------
# Transform for RETFound (224×224, ImageNet stats)
# ---------------------------------------------------------------------------

RETFOUND_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Cache key helpers — prevents name collisions across splits/classes
# ---------------------------------------------------------------------------

def _path_to_cache_key(image_path: str, root_dirs: list[str]) -> str:
    """
    Converts an absolute image path into a collision-safe cache key.

    Given root_dirs = ["data/classifier/train", "data/classifier/val"]
    and   image_path = "data/classifier/train/DR/image_001.png"
    Returns: "train_DR_image_001"

    Falls back to just the stem if no root matches (shouldn't happen).
    """
    p = Path(image_path)
    for root in root_dirs:
        root_p = Path(root)
        try:
            rel = p.relative_to(root_p)
            # rel = "DR/image_001.png" → "DR_image_001"
            # Prepend the root's last dir name (e.g. "train" or "val")
            prefix = root_p.name  # "train" or "val"
            key = f"{prefix}_{rel.with_suffix('').as_posix().replace('/', '_')}"
            return key
        except ValueError:
            continue
    return p.stem  # fallback


def make_cache_key(image_path: str, split: str = "") -> str:
    """
    Build a cache key from an image path, for use during extraction/inference.

    image_path: full path like "data/classifier/train/DR/img_001.png"
    split:      "train" or "val" — the split name for the prefix

    Returns: "train_DR_img_001" (matches what precompute_retfound_cache saves)
    """
    p = Path(image_path)
    # The parent directory is the class name (DR / Glaucoma / PM)
    class_name = p.parent.name
    stem = p.stem
    if split:
        return f"{split}_{class_name}_{stem}"
    return f"{class_name}_{stem}"


# ---------------------------------------------------------------------------
# Dataset for cache pre-computation (image paths only, no labels needed)
# ---------------------------------------------------------------------------

class _ImagePathDataset(Dataset):
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, root_dirs: list[str], transform):
        self.samples   = []
        self.transform = transform
        for root in root_dirs:
            for dirpath, _, fnames in os.walk(root):
                for fname in sorted(fnames):
                    p = Path(dirpath) / fname
                    if p.suffix.lower() in self.SUPPORTED_EXTS:
                        self.samples.append(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img  = Image.open(path).convert("RGB")
        return self.transform(img), str(path)


# ---------------------------------------------------------------------------
# Pre-computation entry point
# ---------------------------------------------------------------------------

def precompute_retfound_cache(
    image_dirs:    list[str],
    cache_dir:     str,
    weights_path:  Optional[str] = None,
    batch_size:    int = 32,
    num_workers:   int = 4,
    device:        Optional[torch.device] = None,
    overwrite:     bool = False,
) -> None:
    """
    Runs the entire dataset through frozen RETFound and saves per-image
    embeddings as  <cache_dir>/<cache_key>.pt  (float32, shape [1024]).

    Cache key is derived from the image's relative path to avoid collisions
    when the same filename appears in different split/class folders.
    E.g. train/DR/img_001.png → cache key "train_DR_img_001"

    Args:
        image_dirs   List of root directories to walk recursively.
        cache_dir    Output directory for .pt embedding files.
        weights_path Path to RETFound .pth checkpoint, or None.
        batch_size   Images per GPU batch.
        num_workers  DataLoader workers.
        device       CUDA device (auto-detected if None).
        overwrite    If False, skip images whose cache already exists.
    """
    os.makedirs(cache_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    extractor = RETFoundExtractor(weights_path=weights_path, device=device)
    print(f"  [RETFound] caching embeddings → {cache_dir}")

    ds     = _ImagePathDataset(image_dirs, transform=RETFOUND_TRANSFORM)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    skipped = 0
    written = 0

    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="  RETFound cache", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            embs = extractor(imgs)   # (B, 1024)

            for emb, path in zip(embs, paths):
                cache_key  = _path_to_cache_key(path, image_dirs)
                cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

                if os.path.isfile(cache_path) and not overwrite:
                    skipped += 1
                    continue

                torch.save(emb.cpu().float(), cache_path)
                written += 1

    print(f"  [RETFound] done — written={written}  skipped={skipped}")

    # Explicitly unload RETFound from VRAM
    del extractor
    torch.cuda.empty_cache()
    print("  [RETFound] model unloaded from VRAM ✓")


# ---------------------------------------------------------------------------
# Load a single cached embedding at inference / XGBoost training time
# ---------------------------------------------------------------------------

def load_cached_embedding(
    cache_key: str,
    cache_dir: str,
) -> Optional[torch.Tensor]:
    """
    Returns a (1024,) float tensor, or None if not found.

    cache_key: collision-safe key, e.g. "train_DR_img_001"
               (use make_cache_key() to build this from a path)
    """
    path = os.path.join(cache_dir, f"{cache_key}.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")


def load_cached_embeddings_batch(
    cache_keys: list[str],
    cache_dir:  str,
) -> tuple[np.ndarray, list[bool]]:
    """
    Load a batch of embeddings by cache key.

    Returns:
        embs  np.ndarray  (N, 1024)  — zeros for missing entries
        found list[bool]  — True where embedding was found
    """
    embs  = np.zeros((len(cache_keys), 1024), dtype=np.float32)
    found = []
    for i, key in enumerate(cache_keys):
        emb = load_cached_embedding(key, cache_dir)
        if emb is not None:
            embs[i] = emb.numpy()
            found.append(True)
        else:
            found.append(False)
    return embs, found


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Pre-compute RETFound embedding cache")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    p = cfg['paths']
    train_dir = os.path.join(p['data_dir'], "train")
    val_dir   = os.path.join(p['data_dir'], "val")

    precompute_retfound_cache(
        image_dirs   = [train_dir, val_dir],
        cache_dir    = p['retfound_cache_dir'],
        weights_path = p.get('retfound_weights'),
        batch_size   = cfg['retfound']['cache_batch_size'],
        overwrite    = args.overwrite,
    )
