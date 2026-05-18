import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.amp import autocast

# -----------------------------------------------------------------------------
# 1. NOISE gaussian but named as simplex
# -----------------------------------------------------------------------------

def generate_simplex_noise(shape, device, frequency=8, octaves=4):
    """Standard Gaussian noise. Signature kept for call-site compatibility."""
    return torch.randn(*shape, device=device)


def add_simplex_noise(x, timesteps, alphas_cumprod, frequency=8, octaves=4):
    """Standard DDPM forward process with Gaussian noise."""
    device = x.device
    ac     = alphas_cumprod[timesteps].float().view(-1, 1, 1, 1)
    noise  = torch.randn_like(x.float())
    return ac.sqrt() * x.float() + (1 - ac).sqrt() * noise, noise


def simplex_ddim_step(x_t, pred_noise, t, t_prev, alphas_cumprod,
                       device, frequency=8, octaves=4, eta=0.0):
    """Standard DDIM reverse step with Gaussian stochastics (eta > 0)."""
    ac_t    = alphas_cumprod[t.long()].float().view(-1, 1, 1, 1)
    ac_prev = alphas_cumprod[t_prev.long()].float().view(-1, 1, 1, 1) \
              if t_prev.min() >= 0 else torch.ones_like(ac_t)

    pred_x0 = (x_t.float() - (1 - ac_t).sqrt() * pred_noise.float()) / (ac_t.sqrt() + 1e-8)
    pred_x0 = pred_x0.clamp(-1, 1)

    sigma = eta * ((1 - ac_prev) / (1 - ac_t).clamp(min=1e-8) *
                   (1 - ac_t / ac_prev.clamp(min=1e-8))).clamp(min=0).sqrt()

    coeff_x0  = ac_prev.sqrt()
    coeff_dir = (1 - ac_prev - sigma ** 2).clamp(min=0).sqrt()

    x_prev = coeff_x0 * pred_x0 + coeff_dir * pred_noise.float()

    if eta > 0:
        x_prev = x_prev + sigma * torch.randn_like(x_t.float())

    return x_prev.clamp(-1, 1)

# -----------------------------------------------------------------------------
# 2. MULTIDIFFUSION TILING
# -----------------------------------------------------------------------------

TILE_SIZE = 256
FULL_SIZE = 512
TILE_STRIDE = 128
NUM_TRAIN_TILES = 2 

def get_tile_views(H, W,tile_size=TILE_SIZE, stride=TILE_STRIDE):
    views = []
    for h_start in range(0, H - tile_size + 1,stride):
        for w_start in range(0, W - tile_size + 1,stride):
            views.append((h_start,h_start+tile_size,
            w_start,w_start+tile_size))
    return views

TILE_VIEWS = get_tile_views(FULL_SIZE, FULL_SIZE, TILE_SIZE, TILE_STRIDE)


def make_linear_weight(tile_size=TILE_SIZE, device='cpu'):
    """
    Creates a 2D pyramid weight that peaks at 1.0 in the center and hits 0 at
    the edges. Perfect for 50% overlap tiling.
    """
    coords = torch.linspace(-1, 1, tile_size, device=device)
    w_1d = (1.0 - torch.abs(coords)).clamp(min=1e-3)
    yy, xx = torch.meshgrid(w_1d, w_1d, indexing='ij')
    w = yy * xx
    return w.unsqueeze(0).unsqueeze(0)

# O11: cache linear weight per device
_LINEAR_WEIGHT_CACHE = {}
def get_linear_weight(tile_size=TILE_SIZE, device='cpu'):
    key = (tile_size, str(device))
    if key not in _LINEAR_WEIGHT_CACHE:
        _LINEAR_WEIGHT_CACHE[key] = make_linear_weight(tile_size, device)
    return _LINEAR_WEIGHT_CACHE[key]

# -----------------------------------------------------------------------------
# 3. RETINAL MASK
# -----------------------------------------------------------------------------

def make_retinal_mask(images, margin=0.0):
    """
    Dynamic Elliptical Retinal Mask
    Auto-fits to the actual bounds of the illuminated fundus FOV.
    """
    B, C, H, W = images.shape
    device = images.device
    
    # 1. Revert to [0, 1] RGB space
    img_01 = (images.float() + 1.0) / 2.0
    
    # 2. Pixel-by-pixel intensity check (The Foreground)
    fg = 1.0 - (img_01.mean(dim=1, keepdim=True) < 0.05).float()
    
    # 3. Dynamic Auto-fitting Ellipse
    masks = []
    for i in range(B):
        single_fg = fg[i, 0] # Grab the 2D mask for this specific image in the batch
        
        # Find the coordinates of all illuminated pixels
        non_zero_indices = torch.nonzero(single_fg)
        
        # Fallback if image is completely black (corrupted)
        if non_zero_indices.numel() == 0:
            masks.append(torch.zeros_like(single_fg))
            continue
            
        # Find the exact geographic bounds of the retina
        min_y, min_x = torch.min(non_zero_indices, dim=0)[0]
        max_y, max_x = torch.max(non_zero_indices, dim=0)[0]
        
        # Calculate the exact center
        cy = (max_y + min_y) / 2.0
        cx = (max_x + min_x) / 2.0
        
        # Calculate the independent horizontal (rx) and vertical (ry) radii,
        # and shrink them by your 5% safety margin to avoid edge artifacts.
        ry = ((max_y - min_y) / 2.0) * (1.0 - margin)
        rx = ((max_x - min_x) / 2.0) * (1.0 - margin)
        
        # Prevent division-by-zero crashes on extremely weird images
        ry = max(ry, 1.0)
        rx = max(rx, 1.0)
        
        # Generate the coordinate grid
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W, device=device).float() - cx
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        
        # The Ellipse Equation: (x^2 / a^2) + (y^2 / b^2) <= 1
        ellipse = ((yy**2 / ry**2) + (xx**2 / rx**2)) <= 1.0
        masks.append(ellipse.float())
        
    # Stack the batch back together
    ellipse_mask = torch.stack(masks).unsqueeze(1)
    
    # Intersection: Must be illuminated AND inside the safety ellipse
    return ellipse_mask * fg

# -----------------------------------------------------------------------------
# 4. MULTIDIFFUSION DDIM INFERENCE
# -----------------------------------------------------------------------------

@torch.no_grad()
def multidiffusion_reconstruct(img_512, cond, model, ddim_scheduler,
                                alphas_cumprod, device, amp_dtype, device_type,
                                simplex_freq=8, simplex_octaves=4,
                                T_start=400, n_steps=15,
                                cached_cond=None, max_lcw=0.4,
                                progress_callback=None):
    """
    Reconstruct a 512px image using MultiDiffusion per-step fusion.
    fix #23: batched ViT for tile conds (one forward pass for all 9 tiles).
    """
    linear_w    = get_linear_weight(TILE_SIZE, device=device)
    global_cond = cond  # (1, 1, 768)

    ddim_scheduler.set_timesteps(n_steps)
    ts_all    = ddim_scheduler.timesteps
    start_idx = (ts_all - T_start).abs().argmin().item()
    ts_used   = ts_all[start_idx:]

    t_start_vec = torch.tensor([ts_used[0].item()], device=device, dtype=torch.long)
    x_t, _ = add_simplex_noise(img_512, t_start_vec, alphas_cumprod,
                                simplex_freq, simplex_octaves)

    # fix #23: single batched ViT forward for all tile conds
    if cached_cond is not None:
        pure_local_conds = cached_cond.get_tile_conds_batched(img_512, TILE_VIEWS)
    else:
        pure_local_conds = [global_cond] * len(TILE_VIEWS)

    # O7: precompute count (constant across all timesteps)
    count = torch.zeros(1, 1, FULL_SIZE, FULL_SIZE, device=device)
    for (h0, h1, w0, w1) in TILE_VIEWS:
        count[:, :, h0:h1, w0:w1] += linear_w

    for idx, t in enumerate(ts_used):
        t_vec    = torch.tensor([t.item()], device=device, dtype=torch.long)
        t_prev   = ts_used[idx+1] if idx + 1 < len(ts_used) else torch.tensor(-1)
        t_prev_v = torch.tensor([t_prev.item() if hasattr(t_prev, 'item') else t_prev],
                                  device=device, dtype=torch.long)
        t_ratio = t.item() / 1000.0
        dynamic_lcw = max_lcw * (1.0 - t_ratio)

        value = torch.zeros_like(x_t)

        for tile_idx, (h0, h1, w0, w1) in enumerate(TILE_VIEWS):
            tile      = x_t[:, :, h0:h1, w0:w1]
            local_c   = pure_local_conds[tile_idx]
            tile_cond = dynamic_lcw * local_c + (1.0 - dynamic_lcw) * global_cond

            with autocast(device_type=device_type, dtype=amp_dtype):
                pred_noise = model(tile.to(amp_dtype), t_vec,
                                   encoder_hidden_states=tile_cond).sample

            tile_denoised = simplex_ddim_step(
                tile, pred_noise.float(), t_vec, t_prev_v,
                alphas_cumprod, device, simplex_freq, simplex_octaves)

            value[:, :, h0:h1, w0:w1] += tile_denoised * linear_w

        x_t = value / (count + 1e-8)
        if progress_callback is not None:
            progress_callback(idx + 1, len(ts_used))

    return x_t.clamp(-1, 1)


@torch.no_grad()
def multidiffusion_reconstruct_full(img_512, path, model, cached_cond,
                                     ddim_scheduler, alphas_cumprod,
                                     device, amp_dtype, device_type,
                                     simplex_freq=8, simplex_octaves=4,
                                     T_start=400, n_steps=15,
                                     max_lcw=0.4,
                                     progress_callback=None):
    """Full reconstruction pipeline. Returns: (recon_512, residual_512)"""
    cond = cached_cond.get_full_image_cond(img_512, path)
    recon_512 = multidiffusion_reconstruct(
        img_512, cond, model, ddim_scheduler,
        alphas_cumprod, device, amp_dtype, device_type,
        simplex_freq, simplex_octaves, T_start, n_steps,
        cached_cond=cached_cond,
        max_lcw=max_lcw,
        progress_callback=progress_callback)
    rmask    = make_retinal_mask(img_512)
    residual = (img_512.float() - recon_512.float()).abs().mean(dim=1, keepdim=True)
    residual = residual * rmask
    return recon_512, residual
# -----------------------------------------------------------------------------
# 5. MULTI-SCALE RECONSTRUCTION
# -----------------------------------------------------------------------------

MULTISCALE_SIZES   = [256, 128, 64]
MULTISCALE_WEIGHTS = [0.5, 0.3, 0.2]


@torch.no_grad()
def multiscale_residual(img_512, cond, model, ddim_scheduler,
                         alphas_cumprod, device, amp_dtype, device_type,
                         simplex_freq=8, simplex_octaves=4, T_start=400, n_steps=15):
    ensemble = torch.zeros(1, 1, FULL_SIZE, FULL_SIZE, device=device)
    rmask = make_retinal_mask(img_512)

    # Unweighted maximum: each scale competes at full confidence.
    # Weights were needed for += (dampen blurry scales) but with max they
    # just starve coarser scales — a hemorrhage at 64px capped at 0.2
    # would lose to 256px background noise at 0.25.
    for scale in MULTISCALE_SIZES:
        img_s = F.interpolate(img_512, size=(scale, scale),
                               mode='bilinear', align_corners=False)
        t_start_vec = torch.tensor([T_start], device=device, dtype=torch.long)
        x_s, _ = add_simplex_noise(img_s, t_start_vec, alphas_cumprod,
                                    simplex_freq, simplex_octaves)
        ddim_scheduler.set_timesteps(n_steps)
        ts_all    = ddim_scheduler.timesteps
        start_idx = (ts_all - T_start).abs().argmin().item()
        ts_used_s = ts_all[start_idx:]
        for s_idx, t in enumerate(ts_used_s):
            t_vec    = torch.tensor([t.item()], device=device, dtype=torch.long)
            t_prev   = ts_used_s[s_idx+1] if s_idx + 1 < len(ts_used_s) else torch.tensor(-1)
            t_prev_v = torch.tensor([t_prev.item() if hasattr(t_prev, 'item') else t_prev],
                                      device=device, dtype=torch.long)
            with autocast(device_type=device_type, dtype=amp_dtype):
                pn = model(x_s.to(amp_dtype), t_vec,
                           encoder_hidden_states=cond).sample
            x_s = simplex_ddim_step(
                x_s, pn.float(), t_vec, t_prev_v,
                alphas_cumprod, device, simplex_freq, simplex_octaves)
        recon_s  = x_s.clamp(-1, 1)
        img_up   = F.interpolate(img_s,   size=(FULL_SIZE, FULL_SIZE),
                                  mode='bilinear', align_corners=False)
        recon_up = F.interpolate(recon_s, size=(FULL_SIZE, FULL_SIZE),
                                  mode='bilinear', align_corners=False)
        resid = (img_up.float() - recon_up.float()).abs().mean(dim=1, keepdim=True)
        # Pure unweighted maximum — let the network's natural confidence speak
        ensemble = torch.maximum(ensemble, resid)

    return ensemble * rmask


@torch.no_grad()
def full_reconstruct_and_residual(img_512, path, model, cached_cond,
                                    ddim_scheduler, alphas_cumprod,
                                    device, amp_dtype, device_type,
                                    simplex_freq=8, simplex_octaves=4,
                                    T_start=400, n_steps=15,
                                    max_lcw=0.4):
    """Returns: (recon_512, multiscale_residual_512)"""
    cond = cached_cond.get_full_image_cond(img_512, path)
    recon_512 = multidiffusion_reconstruct(
        img_512, cond, model, ddim_scheduler,
        alphas_cumprod, device, amp_dtype, device_type,
        simplex_freq, simplex_octaves, T_start, n_steps,
        cached_cond=cached_cond,
        max_lcw=max_lcw)
    ms_residual = multiscale_residual(
        img_512, cond, model, ddim_scheduler,
        alphas_cumprod, device, amp_dtype, device_type,
        simplex_freq, simplex_octaves, T_start, n_steps)
    return recon_512, ms_residual
