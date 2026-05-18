import os
from glob import glob
import torch 
import pandas as pd
from PIL import Image,UnidentifiedImageError
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

def make_transform(crop_size, is_train):
    # Swapped to BILINEAR to prevent bicubic ringing artifacts in the L1 residual
    base = [transforms.Resize((crop_size+64, crop_size+64),
                               interpolation=transforms.InterpolationMode.BILINEAR)]
    if is_train:
        aug = [
            transforms.RandomResizedCrop(crop_size, scale=(0.8,1.0), ratio=(0.9,1.1),
                                          interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), # Bumped to 0.5 because 256px tiles are orientation-agnostic
            transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR), # Added to handle vessel angles
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ]
    else:
        aug = [transforms.CenterCrop(crop_size)]
        
    return transforms.Compose(base + aug + [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3), # Verify RETFound is okay with this!
    ])

class RetinaDataset(Dataset):
    """
    RetinaDataset: A robust data-loading engine designed for large-scale, heterogeneous 
    retinal image datasets (EyePACS, DDR, etc.).
    
    Key Features:
    - Multi-source entry: Parses .csv manifests, .txt path lists, or direct directory crawling.
    - Path resolution: Automatically handles relative vs. absolute paths to maintain portability.
    - Fault Tolerance: Implements a retry-loop to skip corrupted image files without crashing 
      high-VRAM training epochs.
    - Global Conditioning: Returns image paths alongside tensors to facilitate key-based 
      caching for the RETFound conditioner.
    """
    def __init__(self, source, crop_size=512, is_train=True, bad_files_txt=None):
        all_images = []
        src = str(source)
        if src.endswith(".csv") and os.path.isfile(src):
            df = pd.read_csv(src)
            if 'path' in df.columns:
                col = 'path'
            elif 'image' in df.columns:
                col = 'image'
            else:
                raise ValueError(f"CSV {src} must have a 'path' or 'image' column.")
            csv_dir   = os.path.dirname(os.path.dirname(os.path.abspath(src)))
            raw_paths = df[col].dropna().tolist()
            all_images = [
                p if os.path.isabs(p) else os.path.join(csv_dir, p)
                for p in raw_paths
            ]
            extra = ""
            if 'source' in df.columns:
                counts = df['source'].value_counts().to_dict()
                extra  = " | " + " ".join(f"{k}:{v}" for k, v in counts.items())
            print(f"Loaded {len(all_images)} images from {os.path.basename(src)}{extra}")
        elif src.endswith(".txt") and os.path.isfile(src):
            with open(src) as f:
                all_images = [l.strip() for l in f if l.strip()]
            print(f"Loaded {len(all_images)} image paths from {os.path.basename(src)}")
        else:
            for ext in ("*.jpeg","*.jpg","*.png","*.PNG","*.JPG","*.JPEG"):
                all_images.extend(glob(os.path.join(src,"**",ext), recursive=True))
            print(f"Found {len(all_images)} images in {src}")

        if bad_files_txt and os.path.exists(bad_files_txt):
            with open(bad_files_txt) as f:
                bad = set(l.strip() for l in f)
            before = len(all_images)
            all_images = [p for p in all_images if p not in bad]
            print(f"Filtered {before-len(all_images)} bad files — {len(all_images)} remaining")

        if not all_images:
            raise RuntimeError(f"No images found from source: {src}")

        self.images    = all_images
        self.transform = make_transform(crop_size, is_train)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                img = Image.open(self.images[idx]).convert("RGB")
                return self.transform(img), self.images[idx]
            except (UnidentifiedImageError, OSError):
                idx = (idx+1) % len(self.images)
        raise RuntimeError("Too many consecutive bad images")

# -----------------------------------------------------------------------------
# COLLATE FUNCTION
# -----------------------------------------------------------------------------
def collate_fn(batch):
    # Keeps image tensors stacked, but preserves file paths as a list for caching/DDR eval
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

