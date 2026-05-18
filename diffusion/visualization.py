import os
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .diffusion  import (make_retinal_mask, TILE_STRIDE,
                         full_reconstruct_and_residual)
from .evaluation import postprocess_residual


@torch.no_grad()
def save_visualizations(epoch, model, cached_cond,
                          ddim_scheduler, vis_images, vis_paths,
                          alphas_cumprod, device, amp_dtype, device_type,
                          checkpoint_dir, anomaly_maps_raw_dir, n_vis,
                          simplex_freq=8, simplex_octaves=4,
                          T_start=400, n_steps=15,
                          precomputed=None):
    model.eval()
    rows = []

    for i in range(n_vis):
        img = vis_images[i:i+1]

        if precomputed and i in precomputed:
            recon_512, ms_resid = precomputed[i]
        else:
            recon_512, ms_resid = full_reconstruct_and_residual(
                img, vis_paths[i], model, cached_cond, ddim_scheduler,
                alphas_cumprod, device, amp_dtype, device_type,
                simplex_freq, simplex_octaves, T_start, n_steps)

        orig_np  = ((img.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        recon_np = ((recon_512.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        rmask_np = make_retinal_mask(img).squeeze(0).permute(1,2,0).cpu().float().numpy()
        recon_np = recon_np * rmask_np
        diff_np  = ((orig_np - recon_np) * rmask_np).mean(axis=2)
        ms_np    = ms_resid.squeeze().cpu().float().numpy()

        rmask_2d = rmask_np.squeeze(-1)
        clean_np = postprocess_residual(orig_np, recon_np, rmask_2d)

        rows.append((orig_np, recon_np, diff_np, ms_np, clean_np))

        plt.imsave(os.path.join(anomaly_maps_raw_dir, f'raw_{epoch:04d}_{i}.png'),
                   clean_np, cmap='hot')

    n_cols = 5
    fig, axes = plt.subplots(n_vis, n_cols, figsize=(n_cols*3, n_vis*3))
    fig.suptitle(f'Epoch {epoch} — MultiDiffusion Reconstruction (stride={TILE_STRIDE})',
                 fontsize=10)
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'Recon (masked)', 'Signed Diff', 'MultiScale',
                  'Clean Residual']

    for i, (orig, recon, diff, ms, clean) in enumerate(rows):
        axes[i,0].imshow(orig);       axes[i,0].axis('off')
        axes[i,1].imshow(recon);      axes[i,1].axis('off')
        im2 = axes[i,2].imshow(diff, cmap='RdBu', vmin=-0.15, vmax=0.15)
        axes[i,2].axis('off'); plt.colorbar(im2, ax=axes[i,2], fraction=0.046)
        im3 = axes[i,3].imshow(ms, cmap='hot', vmin=0, vmax=ms.max()+1e-8)
        axes[i,3].axis('off'); plt.colorbar(im3, ax=axes[i,3], fraction=0.046)
        im4 = axes[i,4].imshow(clean, cmap='hot', vmin=0, vmax=1)
        axes[i,4].axis('off'); plt.colorbar(im4, ax=axes[i,4], fraction=0.046)

        if i == 0:
            for ax, t in zip(axes[i], col_titles):
                ax.set_title(t, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'recon_epoch_{epoch:04d}.png'),
                dpi=72, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def save_anomaly_maps(epoch, vis_images, vis_paths, checkpoint_dir, precomputed):
    out_dir = os.path.join(checkpoint_dir, "anomaly_maps")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(vis_images)):
        img = vis_images[i:i+1]
        recon_512, ms_resid = precomputed[i]

        orig_np  = ((img.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        recon_np = ((recon_512.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        rmask_np = make_retinal_mask(img).squeeze(0).permute(1,2,0).cpu().float().numpy()
        recon_np = recon_np * rmask_np
        ms_np    = ms_resid.squeeze().cpu().float().numpy()
        rmask_2d = rmask_np.squeeze(-1)
        clean_np = postprocess_residual(orig_np, recon_np, rmask_2d)

        overlay_np  = orig_np.copy()
        heat_colors = plt.cm.hot(clean_np)[..., :3]
        alpha       = np.clip(clean_np * 2.5, 0, 0.6)
        for c in range(3):
            overlay_np[..., c] = (overlay_np[..., c] * (1 - alpha)
                                   + heat_colors[..., c] * alpha)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle(f'Epoch {epoch:04d}  |  img {i}', color='white', fontsize=11)

        axes[0].imshow(orig_np);  axes[0].set_title('Original',       color='white', fontsize=9)
        axes[1].imshow(recon_np); axes[1].set_title('Reconstruction',  color='white', fontsize=9)

        im2 = axes[2].imshow(ms_np,    cmap='inferno', vmin=0, vmax=ms_np.max()+1e-8)
        axes[2].set_title('Raw Residual',   color='white', fontsize=9)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

        im3 = axes[3].imshow(clean_np, cmap='inferno', vmin=0, vmax=1)
        axes[3].set_title('Clean Residual', color='white', fontsize=9)
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

        axes[4].imshow(overlay_np)
        axes[4].set_title('Clean Overlay', color='white', fontsize=9)

        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('#0a0a0a')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f'epoch_{epoch:04d}_img_{i}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()


def save_metrics_dashboard(checkpoint_dir, metrics_history, ddr_history):
    if len(metrics_history) < 2:
        return

    ep  = [m['epoch']       for m in metrics_history]
    aur = [m['pixel_auroc'] for m in metrics_history]
    ap  = [m['pixel_ap']    for m in metrics_history]
    ssm = [m['ssim']        for m in metrics_history]
    psn = [m['psnr']        for m in metrics_history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(ep, aur, 'b-o', label='AUROC', markersize=4)
    axes[0].plot(ep, ap,  'r-o', label='AP',    markersize=4)
    axes[0].axhline(0.7, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].axhline(0.5, color='red',  linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].set_title('Anomaly Detection (val — DISABLED w/o SDEdit)')
    axes[0].legend(fontsize=7); axes[0].grid(True)
    axes[0].set_ylim(0, 1)

    axes[1].plot(ep, ssm, 'g-o', markersize=4)
    axes[1].axhline(0.85, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].set_title('SSIM'); axes[1].grid(True)

    axes[2].plot(ep, psn, 'm-o', markersize=4)
    axes[2].axhline(25.0, color='purple', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[2].set_title('PSNR (dB)'); axes[2].grid(True)

    if ddr_history:
        dep   = [d['epoch']     for d in ddr_history]
        daur  = [d['ddr_auroc'] for d in ddr_history]
        ddice = [d['ddr_dice']  for d in ddr_history]
        axes[3].plot(dep, daur,  'b-s', label='DDR AUROC', markersize=5)
        axes[3].plot(dep, ddice, 'r-s', label='DDR Dice',  markersize=5)
        axes[3].set_title('DDR Real-Lesion Eval (residual-based)')
        axes[3].legend(fontsize=7)
        axes[3].grid(True); axes[3].set_ylim(0, 1)
    else:
        axes[3].text(0.5, 0.5, 'DDR eval\nnot yet run',
                     ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('DDR Real-Lesion Eval')

    plt.suptitle('512px v4 — MultiDiffusion | Metrics', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'metrics_dashboard.png'),
                dpi=72, bbox_inches='tight')
    plt.close()
