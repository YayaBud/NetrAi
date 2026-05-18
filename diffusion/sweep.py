import os
import csv
import time
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from .data       import RetinaDataset, collate_fn
from .diffusion  import (make_retinal_mask, full_reconstruct_and_residual)
from .evaluation import (postprocess_residual, compute_ssim, compute_psnr)


@torch.no_grad()
def run_sweep(model, cached_cond, ddim_scheduler, alphas_cumprod,
              device, amp_dtype, device_type, sweep_csv, sweep_out_dir,
              crop_size, t_starts, ddim_steps_list, lcw_values,
              simplex_freq=8, simplex_octaves=4):
    """Run tiled inference sweeps and save per-combo outputs plus panels."""
    os.makedirs(sweep_out_dir, exist_ok=True)
    sweep_dataset = RetinaDataset(sweep_csv, crop_size, is_train=False)
    sweep_loader = DataLoader(sweep_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
    combos = [(t_start, ddim_steps, lcw)
              for t_start in t_starts
              for ddim_steps in ddim_steps_list
              for lcw in lcw_values]
    if not combos:
        raise RuntimeError("Sweep has no parameter combinations")

    metrics_path = os.path.join(sweep_out_dir, "sweep_metrics.csv")
    model.eval(); cached_cond.eval()

    def _tensor_to_rgb_np(tensor):
        arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy()
        return ((arr + 1) / 2).clip(0, 1)

    def _save_panels(image_stem, panel_recons):
        for lcw in lcw_values:
            lcw_label = str(lcw)
            n_rows, n_cols = len(t_starts), len(ddim_steps_list)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(max(n_cols, 1) * 3,
                                              max(n_rows, 1) * 3))
            axes = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)
            fig.suptitle(f"{image_stem} | max_lcw={lcw_label}", fontsize=10)
            for r, t_start in enumerate(t_starts):
                for c, ddim_steps in enumerate(ddim_steps_list):
                    ax = axes[r, c]
                    recon_np = panel_recons.get(lcw, {}).get((t_start, ddim_steps))
                    if recon_np is not None:
                        ax.imshow(recon_np)
                    else:
                        ax.text(0.5, 0.5, "missing", ha="center", va="center")
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"S{ddim_steps}", fontsize=8)
                    if c == 0:
                        ax.set_ylabel(f"T{t_start}", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(sweep_out_dir,
                                     f"{image_stem}_L{lcw_label}_panel.png"),
                        dpi=96, bbox_inches="tight")
            plt.close(fig)

    with open(metrics_path, "w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["image", "T_start", "ddim_steps", "max_lcw", "ssim",
                         "psnr", "resid_mean", "resid_max", "recon_seconds"])

        image_pbar = tqdm(sweep_loader, desc="Sweep images", unit="img")
        for img, paths in image_pbar:
            img = img.to(device, non_blocking=True)
            img_path = paths[0]
            img_stem = Path(img_path).stem
            orig_np = _tensor_to_rgb_np(img)
            panel_recons = {lcw: {} for lcw in lcw_values}

            for t_start, ddim_steps, lcw in tqdm(
                    combos, desc=f"  {img_stem}", unit="combo", leave=False):
                lcw_label = str(lcw)
                combo_name = f"{img_stem}_T{t_start}_S{ddim_steps}_L{lcw_label}"
                start_time = time.time()
                recon_512, residual = full_reconstruct_and_residual(
                    img, img_path, model, cached_cond, ddim_scheduler,
                    alphas_cumprod, device, amp_dtype, device_type,
                    simplex_freq=simplex_freq,
                    simplex_octaves=simplex_octaves,
                    T_start=t_start,
                    n_steps=ddim_steps,
                    max_lcw=lcw)
                recon_seconds = time.time() - start_time

                # -- SNIPER INJECTION: Use Vessel Suppression for Sweep Heatmaps --
                recon_np = _tensor_to_rgb_np(recon_512)
                
                # 1. Get 2D Retinal Mask (required for the Sniper)
                rmask_t = make_retinal_mask(img)
                rmask_np = rmask_t.squeeze().cpu().float().numpy()
                if rmask_np.ndim == 3: rmask_np = rmask_np.squeeze(0)

                # 2. Run the Sniper (Frangi Vessel Suppression)
                # This ensures the sweep results show LESIONS, not VESSELS
                clean_np = postprocess_residual(orig_np, recon_np, rmask_np)
                
                # 3. Assign for saving (resid_np is what the rest of the function uses)
                resid_np = clean_np 
                
                resid_mean = float(resid_np.mean())
                resid_max = float(resid_np.max())
                ssim = compute_ssim(orig_np, recon_np)
                psnr = compute_psnr(orig_np, recon_np)

                Image.fromarray((recon_np * 255).astype(np.uint8)).save(
                    os.path.join(sweep_out_dir, f"{combo_name}.png"))
                plt.imsave(os.path.join(sweep_out_dir, f"{combo_name}_resid.png"),
                           resid_np, cmap="hot", vmin=0, vmax=resid_max + 1e-8)
                writer.writerow([img_path, t_start, ddim_steps, lcw,
                                 f"{ssim:.6f}", f"{psnr:.4f}",
                                 f"{resid_mean:.6f}", f"{resid_max:.6f}",
                                 f"{recon_seconds:.3f}"])
                metrics_file.flush()
                panel_recons[lcw][(t_start, ddim_steps)] = recon_np

                del recon_512, residual
                if device == "cuda":
                    torch.cuda.empty_cache()

            _save_panels(img_stem, panel_recons)
            cached_cond._vit_cache.clear()

    print(f"Sweep complete. Results written to {sweep_out_dir}")
