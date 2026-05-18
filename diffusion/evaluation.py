import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi

from .diffusion import multidiffusion_reconstruct_full, make_retinal_mask
from glob import glob

# -----------------------------------------------------------------------------
# 1. METRICS + DDR EVALUATION
# -----------------------------------------------------------------------------

def compute_ssim(img1, img2):
    def rgb2gray(img):
        return 0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]
    g1 = rgb2gray(img1).astype(np.float64)
    g2 = rgb2gray(img2).astype(np.float64)
    C1, C2   = 0.01**2, 0.03**2
    mu1, mu2 = g1.mean(), g2.mean()
    s1       = ((g1-mu1)**2).mean()
    s2       = ((g2-mu2)**2).mean()
    s12      = ((g1-mu1)*(g2-mu2)).mean()
    return float(((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2)))


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else float(10*np.log10(1.0/mse))

def postprocess_residual(orig_np, recon_np, retinal_mask_np=None):
    """
    Vessel-aware residual post-processing for retinal anomaly detection.

    Strategy:
      1. Baseline-correct the residual (median subtraction)
      2. Frangi filter on the GREEN channel with multi-scale sigmas
      3. SOFT suppression — smoothly attenuate vessel regions
         without clipping nearby anomalies
      4. Also run Frangi on the residual itself to catch vessel-shaped
         artifacts the model introduced during reconstruction
    """
    # -- Step 1: Raw residual with baseline correction --
    residual = np.abs(orig_np - recon_np).mean(axis=2)
    if retinal_mask_np is not None:
        residual = residual * retinal_mask_np

    # Baseline correction: subtract the median of non-zero pixels
    # This removes the diffuse "haze" from imperfect reconstruction
    median = np.median(residual[residual > 0]) if (residual > 0).any() else 0.0
    residual = (residual - median).clip(0)

    # Light smoothing to reduce pixel noise without killing small lesions
    residual = gaussian_filter(residual, sigma=0.5)

    # -- Step 2: Vessel detection on GREEN channel (multi-scale) --
    green = orig_np[:, :, 1]

    # Multi-scale Frangi covering capillaries (σ=2) through arcades (σ=9)
    # black_ridges=True because vessels are darker than background in fundus
    vessel_response = frangi(green, sigmas=np.arange(2, 9, 1),
                              black_ridges=True)

    # -- Step 3: Vessel detection on the RESIDUAL itself --
    # Catches vessel-shaped reconstruction artifacts regardless of input anatomy
    residual_vessel = frangi(residual, sigmas=np.arange(1, 6, 1),
                              black_ridges=False)  # ridges are bright in residual

    # Combine both vessel maps (take the max response)
    combined_vessel = np.maximum(vessel_response, residual_vessel)

    # -- Step 4: SOFT suppression --
    # Exponential decay gives smooth falloff, preserving anomalies
    # adjacent to (but not on) vessels.
    #   1.0 = not a vessel → keep full residual
    #   0.0 = strong vessel → fully suppress
    if combined_vessel.max() > 1e-8:
        vessel_norm = combined_vessel / combined_vessel.max()
    else:
        vessel_norm = combined_vessel

    # Tunable suppression strength: higher gamma = more aggressive vessel removal
    gamma = 1.5
    suppression_weight = np.exp(-gamma * vessel_norm)

    # Apply soft suppression
    cleaned = residual * suppression_weight

    # -- Step 5: Final masking & normalization --
    if retinal_mask_np is not None:
        cleaned = cleaned * retinal_mask_np

    rmax = cleaned.max()
    if rmax > 1e-8:
        cleaned = cleaned / rmax

    return cleaned.astype(np.float32)

DDR_LESION_TYPES = ["MA", "HE", "EX", "SE"]

def load_combined_ddr_mask(stem, ddr_masks_dir, size=512):
    combined  = np.zeros((size, size), dtype=np.uint8)
    found_any = False
    for ltype in DDR_LESION_TYPES:
        for ext in (".tif", ".png", ".jpg"):
            mask_path = os.path.join(ddr_masks_dir, ltype, stem + ext)
            if os.path.exists(mask_path):
                try:
                    mask_pil = Image.open(mask_path).convert("L")
                    mask_np  = np.array(mask_pil.resize((size, size), Image.NEAREST))
                    combined = np.maximum(combined, (mask_np > 127).astype(np.uint8))
                    found_any = True
                except Exception:
                    pass
                break
    return combined if found_any else None


@torch.no_grad()
def compute_ddr_metrics(model, cached_cond, ddim_scheduler,
                         alphas_cumprod, device, amp_dtype, device_type,
                         ddr_images_dir, ddr_masks_dir,
                         simplex_freq=8, simplex_octaves=4,
                         max_images=200, T_start=400, n_steps=15,
                         max_lcw=0.4,
                         max_seconds=None, progress=True):
    """
    DDR pixel-level AUROC/AP/Dice using vessel-suppressed (clean) residual
    as the anomaly score. Frangi filtering on both the green channel and the
    residual itself suppresses vessel signal before scoring.
    """
    if not os.path.isdir(ddr_images_dir) or not os.path.isdir(ddr_masks_dir):
        print(f"  DDR eval skipped — dirs not found")
        return None

    img_paths = sorted(glob(os.path.join(ddr_images_dir, "*.jpg")) +
                       glob(os.path.join(ddr_images_dir, "*.jpeg")) +
                       glob(os.path.join(ddr_images_dir, "*.png")))
    pairs = []
    for ip in img_paths:
        stem    = Path(ip).stem
        mask_np = load_combined_ddr_mask(stem, ddr_masks_dir, size=512)
        if mask_np is not None and mask_np.sum() > 0:
            pairs.append((ip, mask_np))

    if not pairs:
        print("  DDR eval skipped — no paired image/mask files found")
        return None

    total_pairs = len(pairs)
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]

    ddim_scheduler.set_timesteps(n_steps)
    ts_all = ddim_scheduler.timesteps
    start_idx = (ts_all - T_start).abs().argmin().item()
    ddr_step_count = max(1, len(ts_all[start_idx:]))

    if len(pairs) < total_pairs:
        print(f"  DDR eval: using {len(pairs)}/{total_pairs} paired images "
              f"(combined MA+HE+EX+SE masks)", flush=True)
    else:
        print(f"  DDR eval: {len(pairs)} paired images "
              f"(combined MA+HE+EX+SE masks)", flush=True)
    budget_msg = f"{max_seconds/60:.1f} min" if max_seconds else "none"
    print(f"  DDR eval budget: {ddr_step_count} DDIM steps/image | "
          f"max runtime: {budget_msg}", flush=True)

    to_tensor  = transforms.ToTensor()
    resize_512 = transforms.Resize((512, 512),
                                    interpolation=transforms.InterpolationMode.BICUBIC)
    normalize  = transforms.Normalize([0.5]*3, [0.5]*3)
    all_preds, all_gts = [], []

    start_time = time.time()
    stop_reason = None
    pbar = tqdm(total=len(pairs) * ddr_step_count,
                desc="  DDR eval",
                unit="step",
                leave=False,
                dynamic_ncols=True,
                disable=not progress)
    try:
        for idx, (img_path, mask_np) in enumerate(pairs, start=1):
            # -- THE DEFRAG VALVE: Prevents VRAM fragmentation on full datasets --
            if idx % 50 == 0:
                torch.cuda.empty_cache()
            if max_seconds is not None and time.time() - start_time >= max_seconds:
                stop_reason = f"time budget reached after {max_seconds/60:.1f} min"
                break

            image_name = os.path.basename(img_path)
            pbar.set_description(f"  DDR {idx}/{len(pairs)} {image_name[:32]}")

            def _progress_callback(step_idx, step_total):
                pbar.update(1)
                elapsed = time.time() - start_time
                pbar.set_postfix(valid=len(all_preds),
                                  elapsed=f"{elapsed/60:.1f}m")
                if (max_seconds is not None and elapsed >= max_seconds and
                        step_idx < step_total):
                    raise TimeoutError(
                        f"DDR eval time budget reached after {max_seconds/60:.1f} min")

            try:
                img_pil = Image.open(img_path).convert("RGB")
                img_t   = normalize(to_tensor(resize_512(img_pil))).unsqueeze(0).to(device)

                recon_512, residual = multidiffusion_reconstruct_full(
                    img_t, img_path, model, cached_cond, ddim_scheduler,
                    alphas_cumprod, device, amp_dtype, device_type,
                    simplex_freq, simplex_octaves, T_start, n_steps,
                    max_lcw=max_lcw,
                    progress_callback=_progress_callback)

                # residual already retinal-masked, shape (1,1,512,512)
                # -- NEW: Injecting the Vessel-Aware Post-Processing --
                
                # 1. Convert tensors back to [0, 1] numpy arrays for Frangi
                orig_np = ((img_t.squeeze().permute(1,2,0).cpu().float().numpy() + 1) / 2).clip(0, 1)
                recon_np = ((recon_512.squeeze().permute(1,2,0).cpu().float().numpy() + 1) / 2).clip(0, 1)
                
                # 2. Get the 2D retinal mask
                rmask_t = make_retinal_mask(img_t)
                rmask_np = rmask_t.squeeze().cpu().float().numpy()
                if rmask_np.ndim == 3: 
                    rmask_np = rmask_np.squeeze(0)  # Ensure it's 2D (H,W)
                
                # 3. Apply the Sniper (Frangi vessel soft-suppression)
                # Note: postprocess_residual already handles the [0, 1] normalization!
                clean_np = postprocess_residual(orig_np, recon_np, rmask_np)
                
                # 4. Use the cleaned map for AUROC instead of the raw residual
                pred_np = clean_np
                # --------------------------------------------------------

                all_preds.append(pred_np.flatten())
                all_gts.append(mask_np.flatten())
            except TimeoutError as e:
                stop_reason = str(e)
                break
            except (UnidentifiedImageError, OSError) as e:
                print(f"  DDR: skipped {image_name}: {e}", flush=True)
                continue
    finally:
        pbar.close()

    if stop_reason:
        print(f"  DDR eval stopped early: {stop_reason}", flush=True)

    if not all_preds:
        print("  DDR eval: no valid pairs processed")
        return None

    preds = np.concatenate(all_preds)
    gts   = np.concatenate(all_gts)

    try:
        pixel_auroc = float(roc_auc_score(gts, preds))
        pixel_ap    = float(average_precision_score(gts, preds))
    except Exception as e:
        print(f"  DDR AUROC/AP failed: {e}")
        return None

    best_dice, best_thresh = 0.0, 0.5
    for thresh in np.linspace(0.1, 0.9, 20):
        pred_bin = (preds > thresh).astype(np.uint8)
        tp = (pred_bin * gts).sum()
        fp = (pred_bin * (1-gts)).sum()
        fn = ((1-pred_bin) * gts).sum()
        denom = 2*tp + fp + fn
        dice  = float(2*tp / denom) if denom > 0 else 0.0
        if dice > best_dice:
            best_dice, best_thresh = dice, thresh

    return {
        'ddr_auroc':  pixel_auroc,
        'ddr_ap':     pixel_ap,
        'ddr_dice':   best_dice,
        'ddr_thresh': best_thresh,
        'n_images':   len(all_preds),
    }


@torch.no_grad()
def compute_val_metrics(model, cached_cond, ddim_scheduler,
                         val_loader, alphas_cumprod,
                         device, amp_dtype, device_type,
                         max_batches=2, simplex_freq=8, simplex_octaves=4,
                         T_start=400, n_steps=15,
                         max_lcw=0.4):
    """
    fix #18: SSIM/PSNR over ALL images in batch (not just index 0).
    SDEdit-based AUROC removed — DDR is the only meaningful anomaly metric.
    """
    model.eval()
    ssim_scores, psnr_scores = [], []

    for batch_idx, (batch, paths) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        B     = batch.shape[0]

        for i in range(B):
            recon_512, _ = multidiffusion_reconstruct_full(
                batch[i:i+1], paths[i], model, cached_cond, ddim_scheduler,
                alphas_cumprod, device, amp_dtype, device_type,
                simplex_freq, simplex_octaves, T_start, n_steps,
                max_lcw=max_lcw)
            orig_np  = ((batch[i].permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
            recon_np = ((recon_512.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
            ssim_scores.append(compute_ssim(orig_np, recon_np))
            psnr_scores.append(compute_psnr(orig_np, recon_np))

    return {
        'ssim': float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        'psnr': float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        # kept zero so dashboard schema is unchanged; DDR is the anomaly metric
        'pixel_auroc': 0.0,
        'pixel_ap':    0.0,
    }

