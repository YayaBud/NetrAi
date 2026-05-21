import torch

import os
import csv
import math
import random
import yaml
import torch
import torch._dynamo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from .data       import RetinaDataset, collate_fn
from .models     import RETFoundConditioner, CachedConditioner
from .diffusion  import (add_simplex_noise, make_retinal_mask,
                         TILE_VIEWS, TILE_STRIDE, NUM_TRAIN_TILES,
                         full_reconstruct_and_residual)
from .losses     import diffusion_loss
from .evaluation import compute_val_metrics, compute_ddr_metrics
from .utils      import (strip_compile_prefix, repair_csv_header,
                         append_csv_row, load_loss_history,
                         setup_terminal_logging, save_lcw_curve)
from .visualization import (save_visualizations, save_anomaly_maps,
                             save_metrics_dashboard)
from .sweep      import run_sweep

def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    p = cfg['paths']
    t = cfg['training']
    d = cfg['diffusion']
    e = cfg['eval']
    s = cfg['sweep']

    DATA_TRAIN          = p['data_train']
    DATA_VAL            = p['data_val']
    BAD_FILES_TXT       = p['bad_files_txt']
    CHECKPOINT_DIR      = p['checkpoint_dir']
    PRETRAINED_256_CKPT = p['pretrained_256']
    DDR_IMAGES_DIR      = p['ddr_images_dir']
    DDR_MASKS_DIR       = p['ddr_masks_dir']
    RETFOUND_WEIGHTS    = p.get('retfound_weights')

    CROP_SIZE           = t['crop_size']
    EPOCHS              = t['epochs']
    BATCH_SIZE          = t['batch_size']
    ACCUM_STEPS         = t['accum_steps']
    WARMUP_EPOCHS       = t['warmup_epochs']
    NUM_WORKERS_TRAIN   = t['num_workers_train']
    NUM_WORKERS_VAL     = t['num_workers_val']
    PREFETCH_FACTOR     = t['prefetch_factor']
    LR_UNET             = t['lr_unet']
    LR_CONDITIONER      = t['lr_conditioner']

    SNR_GAMMA           = t['snr_gamma']

    SIMPLEX_FREQ        = d['simplex_freq']
    SIMPLEX_OCTAVES     = d['simplex_octaves']
    DDIM_STEPS          = d['ddim_steps']
    DDIM_T_START        = d['ddim_t_start']
    MAX_TRAIN_LCW       = d['max_train_lcw']

    VIS_EVERY           = e['vis_every']
    EVAL_EVERY          = e['eval_every']
    NUM_VIS             = e['num_vis']
    DDR_EVAL_EVERY      = e['ddr_eval_every']
    DDR_MAX_IMAGES      = e['ddr_max_images']
    DDR_MAX_SECONDS     = e['ddr_max_seconds']
    LCW_PLOT_EVERY      = e['lcw_plot_every']

    SWEEP_MODE      = s['enabled']
    SWEEP_CSV       = s['csv']
    SWEEP_OUT_DIR   = s['out_dir']
    SWEEP_T_STARTS  = s['t_starts']
    SWEEP_DDIM_STEPS = s['ddim_steps']
    SWEEP_LCW       = s['lcw_values']

    # -- Environment setup -----------------------------------------------------
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True,max_split_size_mb:256")
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32       = True
    # torch.backends.cudnn.benchmark        = True

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    amp_dtype   = (torch.bfloat16
                   if (device == "cuda" and torch.cuda.is_bf16_supported())
                   else torch.float16)

    ddr_img_str = DDR_MAX_IMAGES if DDR_MAX_IMAGES else "all"
    ddr_sec_str = f"{DDR_MAX_SECONDS/60:.0f} min" if DDR_MAX_SECONDS else "none"

    print(f"Device: {device} | AMP: {amp_dtype}")
    print(f"512px v4 | MultiDiffusion (stride={TILE_STRIDE}) | DDIM{DDIM_STEPS}")
    print(f"Training: {NUM_TRAIN_TILES} random tiles/step | "
          f"Inference: {len(TILE_VIEWS)} overlapping tiles")
    print(f"DDR eval: every {DDR_EVAL_EVERY} epochs | max {ddr_img_str} imgs | "
          f"max {ddr_sec_str}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ANOMALY_MAPS_RAW = os.path.join(CHECKPOINT_DIR, "anomaly_maps_raw")
    os.makedirs(ANOMALY_MAPS_RAW, exist_ok=True)
    TERMINAL_LOG = os.path.join(CHECKPOINT_DIR, "train_terminal.log")
    setup_terminal_logging(TERMINAL_LOG)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_schedule="squaredcos_cap_v2")
    ddim_scheduler  = DDIMScheduler(num_train_timesteps=1000,
                                    beta_schedule="squaredcos_cap_v2")
    ddim_scheduler.set_timesteps(DDIM_STEPS)

    # fix #28: alphas_cumprod on GPU once
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    print("Building UNet...")
    model = UNet2DConditionModel(
        sample_size=256,
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D","CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D","CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D","UpBlock2D"),
        cross_attention_dim=768,
    ).to(device)
    model.enable_gradient_checkpointing()

    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")

    conditioner = RETFoundConditioner(cross_attention_dim=768,
                                       retfound_weights=RETFOUND_WEIGHTS).to(device)
    cached_cond = CachedConditioner(conditioner)

    last_ckpt = os.path.join(CHECKPOINT_DIR, "last.pt")

    # -- Weight loading (BEFORE torch.compile) --------------------------------
    pretrained_from_256 = False

    if os.path.exists(last_ckpt):
        # Resume path takes precedence — actual load done below after compile.
        # We just print here; the real load_state_dict happens in resume block.
        print(f"Found {last_ckpt} — will resume from it after compile")
    elif os.path.exists(PRETRAINED_256_CKPT):
        print(f"Loading 256px checkpoint: {PRETRAINED_256_CKPT}")
        try:
            ckpt_256 = torch.load(PRETRAINED_256_CKPT, map_location=device, weights_only=True)
        except Exception:
            ckpt_256 = torch.load(PRETRAINED_256_CKPT, map_location=device, weights_only=False)
        # compile-prefix safety
        model.load_state_dict(strip_compile_prefix(ckpt_256['model']))
        conditioner.proj.load_state_dict(strip_compile_prefix(ckpt_256['conditioner_proj']))
        pretrained_from_256 = True
        print("  256px checkpoint loaded")
    else:
        print("No pretrained checkpoint found — training from scratch")

    # -- torch.compile AFTER weight load --------------------------------------
    # fix: raise dynamo cache limit — default of 8 is too low for diffusers
    # cross-attention (9 layers × train/eval switches × xformers processor
    # identity checks all burn cache slots fast).
    torch._dynamo.config.cache_size_limit = 64

    USE_COMPILE = False
    # try:
    #     model = torch.compile(model, mode="default")
    #     USE_COMPILE = True
    #     print("torch.compile enabled (default mode)")
    # except Exception as e:
    #     print(f"torch.compile skipped: {e}")

    # fix: xformers + compile = attention processor identity thrash → cache misses.
    # Only enable xformers when compile is NOT active.
    if is_xformers_available() and not USE_COMPILE:
        try:
            model.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        except Exception as e:
            print(f"xformers skipped: {e}")
    elif is_xformers_available() and USE_COMPILE:
        print("xformers skipped — compile active (prevents attention processor cache thrash)")

    # -- Optimizer + base_lrs --------------------------------------------------
    # fix #4: bake 30% scaling INTO base_lrs so warmup ramps from the right base
    if pretrained_from_256:
        base_lrs = [LR_UNET * 0.3, LR_CONDITIONER * 0.3]
        WARMUP_EPOCHS = 1
        print(f"  Pretrained-256 base LRs: UNet={base_lrs[0]:.1e} "
              f"cond={base_lrs[1]:.1e} (warmup={WARMUP_EPOCHS})")
    else:
        base_lrs = [LR_UNET, LR_CONDITIONER]

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(),             'lr': base_lrs[0]},
        {'params': conditioner.proj.parameters(),  'lr': base_lrs[1]},
    ], weight_decay=1e-4)
    # O8: cache param list for clip_grad_norm_ (avoid rebuilding every step)
    all_params = list(model.parameters()) + list(conditioner.proj.parameters())

    lr_scheduler = CosineAnnealingLR(optimizer,
                                      T_max=max(EPOCHS-WARMUP_EPOCHS, 1),
                                      eta_min=1e-7)
    scaler = GradScaler(device, enabled=(amp_dtype == torch.float16))

    # -- Datasets --------------------------------------------------------------
    train_dataset = RetinaDataset(DATA_TRAIN, CROP_SIZE, is_train=True,
                                   bad_files_txt=BAD_FILES_TXT)
    val_dataset   = RetinaDataset(DATA_VAL,   CROP_SIZE, is_train=False)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError("Empty train dataset — check DATA_TRAIN path")
    if len(val_dataset) == 0:
        raise RuntimeError("Empty val dataset — check DATA_VAL path")

    gpu_gen = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    pin     = (device == "cuda") and (gpu_gen < 12)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS_TRAIN, pin_memory=pin,
                               collate_fn=collate_fn, drop_last=True,
                               persistent_workers=(NUM_WORKERS_TRAIN > 0),
                               prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS_TRAIN > 0 else None))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS_VAL, pin_memory=pin,
                               collate_fn=collate_fn,
                               persistent_workers=(NUM_WORKERS_VAL > 0),
                               prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS_VAL > 0 else None))

    # -- Resume ----------------------------------------------------------------
    start_epoch   = 0
    best_val_loss = float("inf")
    best_auroc    = 0.0
    train_losses, val_losses, metrics_history, ddr_history = [], [], [], []
    lcw_curve_x, lcw_curve_y = [], []

    best_loss_ckpt  = os.path.join(CHECKPOINT_DIR, "best_loss.pt")
    best_auroc_ckpt = os.path.join(CHECKPOINT_DIR, "best_auroc.pt")
    loss_csv        = os.path.join(CHECKPOINT_DIR, "loss.csv")
    metrics_csv     = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    ddr_csv         = os.path.join(CHECKPOINT_DIR, "ddr_metrics.csv")
    epoch_metrics_csv = os.path.join(CHECKPOINT_DIR, "epoch_metrics.csv")

    if os.path.exists(last_ckpt):
        print("Resuming from last.pt...")
        try:
            ckpt_r = torch.load(last_ckpt, map_location=device, weights_only=True)
        except Exception:
            ckpt_r = torch.load(last_ckpt, map_location=device, weights_only=False)
        # compile-prefix safe load (works for both compiled and uncompiled saves)
        m_state = strip_compile_prefix(ckpt_r['model'])
        # If model is currently compiled, load into the inner module
        target = model._orig_mod if hasattr(model, '_orig_mod') else model
        target.load_state_dict(m_state)
        conditioner.proj.load_state_dict(strip_compile_prefix(ckpt_r['conditioner_proj']))
        if 'seg_head' in ckpt_r:
            print("  skipping legacy seg_head weights (v4 residual-only)")
        if 'ss2d' in ckpt_r:
            print("  (skipping legacy ss2d weights from earlier v2)")
        optimizer_state_loaded = False
        try:
            optimizer.load_state_dict(ckpt_r['optimizer'])
            optimizer_state_loaded = True
        except ValueError as e:
            print(f"  optimizer state skipped (v4 removed seg_head params): {e}")
        scaler.load_state_dict(ckpt_r['scaler'])
        # fix #5: do NOT clobber optimizer LRs with base_lrs after load_state_dict
        start_epoch   = ckpt_r['epoch'] + 1
        best_val_loss = ckpt_r.get('best_val_loss', float('inf'))
        best_auroc    = ckpt_r.get('best_auroc', 0.0)
        if 'scheduler' in ckpt_r and start_epoch >= WARMUP_EPOCHS and optimizer_state_loaded:
            lr_scheduler.load_state_dict(ckpt_r['scheduler'])
            lr_scheduler.T_max = max(EPOCHS - WARMUP_EPOCHS, 1)
        loss_history = load_loss_history(loss_csv)
        if loss_history:
            for row in loss_history:
                if row.get("train_loss") is not None:
                    train_losses.append(float(row["train_loss"]))
                if row.get("val_loss") is not None:
                    val_losses.append(float(row["val_loss"]))
                if row.get("lcw") is not None:
                    lcw_curve_x.append(float(row["epoch"]))
                    lcw_curve_y.append(float(row["lcw"]))
        # Restore metrics_history from metrics.csv so dashboards continue
        if os.path.exists(metrics_csv):
            with open(metrics_csv) as f:
                for row in csv.DictReader(f):
                    metrics_history.append({
                        'epoch':       int(row['epoch']),
                        'ssim':        float(row.get('ssim', 0)),
                        'psnr':        float(row.get('psnr', 0)),
                        'pixel_auroc': float(row.get('pixel_auroc', 0)),
                        'pixel_ap':    float(row.get('pixel_ap', 0)),
                    })
            print(f"  Restored {len(metrics_history)} metrics history entries")
        # Restore ddr_history from ddr_metrics.csv
        if os.path.exists(ddr_csv):
            with open(ddr_csv) as f:
                for row in csv.DictReader(f):
                    ddr_history.append({
                        'epoch':      int(row['epoch']),
                        'ddr_auroc':  float(row.get('ddr_auroc', 0)),
                        'ddr_ap':     float(row.get('ddr_ap', 0)),
                        'ddr_dice':   float(row.get('ddr_dice', 0)),
                        'ddr_thresh': float(row.get('ddr_thresh', 0.5)),
                        'n_images':   int(row.get('n_images', 0)),
                    })
            print(f"  Restored {len(ddr_history)} DDR history entries")
        print(f"Resumed from epoch {start_epoch}")

    for csv_path, header in [
        (loss_csv,    ["epoch","train_loss","val_loss","snr","ms","val_snr","val_ms","lr","lcw","seg_weight"]),
        (metrics_csv, ["epoch","ssim","psnr","pixel_auroc","pixel_ap"]),
        (ddr_csv,     ["epoch","ddr_auroc","ddr_ap","ddr_dice","ddr_thresh","n_images"]),
        (epoch_metrics_csv, ["epoch","train_loss","val_loss","snr","ms","val_snr","val_ms","lr","lcw","seg_weight","ssim","psnr","pixel_auroc","pixel_ap","ddr_auroc","ddr_ap","ddr_dice","ddr_thresh","n_images"]),
    ]:
        if start_epoch == 0:
            with open(csv_path,"w",newline="") as f:
                csv.writer(f).writerow(header)
        else:
            repair_csv_header(csv_path, header)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    raw_vis    = next(iter(val_loader))
    vis_images = raw_vis[0][:min(NUM_VIS, len(raw_vis[0]))].to(device)
    vis_paths  = raw_vis[1][:min(NUM_VIS, len(raw_vis[1]))]
    n_vis      = len(vis_images)

    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"VRAM before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    if not SWEEP_MODE:
        print(f"\nStarting {EPOCHS} epochs | 512px v4\n")

    # -- Training loop ---------------------------------------------------------
    if SWEEP_MODE:
        print(f"\nSWEEP_MODE=1 | writing results to {SWEEP_OUT_DIR}\n")
        run_sweep(model, cached_cond, ddim_scheduler, alphas_cumprod,
                  device, amp_dtype, device_type,
                  SWEEP_CSV, SWEEP_OUT_DIR, CROP_SIZE,
                  SWEEP_T_STARTS, SWEEP_DDIM_STEPS, SWEEP_LCW,
                  SIMPLEX_FREQ, SIMPLEX_OCTAVES)
        return

    total_train_steps = max(EPOCHS * len(train_loader), 1)

    for epoch in range(start_epoch, EPOCHS):

        in_warmup = epoch < WARMUP_EPOCHS
        if in_warmup:
            wf = (epoch+1) / WARMUP_EPOCHS
            for pg, lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = lr * wf  # base_lrs already has 30% scaling baked in

        sw = 0.0

        model.train(); cached_cond.train()
        # fix #6: cached_cond.train() routes through RETFoundConditioner.train()
        #         which keeps self.vit in eval

        train_loss = comp_snr = comp_ms = 0.0
        epoch_lcw = 0.0  # track effective LCW for this epoch
        optimizer.zero_grad(set_to_none=True)  # O5: faster than zeroing

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [v4]")

        for step, (batch, paths) in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)
            B     = batch.shape[0]

            sampled_views = random.sample(TILE_VIEWS, min(NUM_TRAIN_TILES, len(TILE_VIEWS)))
            n_tiles = len(sampled_views)

            # fix #8: hoist cond outside tile loop
            current_step = epoch * len(train_loader) + step
            progress = current_step / total_train_steps
            curve_warmup = (1.0 - math.cos(math.pi * progress)) / 2.0

            global_cond = cached_cond.get_full_image_cond(batch, paths)

            # O4: compute retinal mask once for full image, slice per tile
            full_retinal_mask = make_retinal_mask(batch)

            # O1: batch all tile ViT extractions into single forward pass
            all_tiles = torch.cat([batch[:, :, h0:h1, w0:w1]
                                   for (h0, h1, w0, w1) in sampled_views])
            with torch.no_grad():
                all_local_feats = conditioner.extract_features(all_tiles)
            all_local_conds = conditioner.proj(all_local_feats.float()).unsqueeze(1)
            del all_tiles, all_local_feats

            total_d_loss = torch.tensor(0.0, device=device)
            step_lcw_sum = 0.0
            for tile_i, (h0, h1, w0, w1) in enumerate(sampled_views):
                tile = batch[:, :, h0:h1, w0:w1]

                retinal_mask = full_retinal_mask[:, :, h0:h1, w0:w1]
                timesteps    = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device).long()
                noisy, noise = add_simplex_noise(tile, timesteps, alphas_cumprod,
                                                  SIMPLEX_FREQ, SIMPLEX_OCTAVES)

                t_ratio = timesteps.float() / noise_scheduler.config.num_train_timesteps
                dynamic_lcw = MAX_TRAIN_LCW * curve_warmup * (1.0 - t_ratio)
                dynamic_lcw = dynamic_lcw.view(B, 1, 1)

                local_cond = all_local_conds[tile_i * B:(tile_i + 1) * B]
                blended_cond = dynamic_lcw * local_cond + (1.0 - dynamic_lcw) * global_cond

                with autocast(device_type=device_type, dtype=amp_dtype):
                    pred_noise = model(noisy.to(amp_dtype), timesteps,
                                       encoder_hidden_states=blended_cond).sample

                ac_t    = alphas_cumprod[timesteps].float().view(-1,1,1,1)
                pred_x0 = ((noisy.detach().float() - (1-ac_t).sqrt()*pred_noise.float())
                            / (ac_t.sqrt()+1e-8)).clamp(-1,1)

                d_loss, comp = diffusion_loss(
                    pred_noise, noise, pred_x0, tile,
                    retinal_mask, alphas_cumprod, timesteps, SNR_GAMMA)

                total_d_loss = total_d_loss + d_loss / n_tiles
                comp_snr    += (comp['snr'] / n_tiles).item()
                comp_ms     += (comp['ms']  / n_tiles).item()
                step_lcw_sum += dynamic_lcw.mean().item()
                del d_loss, pred_noise, noisy, noise, pred_x0, tile
                del local_cond, blended_cond
            del all_local_conds, full_retinal_mask

            group_start = (step // ACCUM_STEPS) * ACCUM_STEPS
            effective_accum = min(ACCUM_STEPS, len(train_loader) - group_start)
            combined = total_d_loss / effective_accum
            scaler.scale(combined).backward()

            if (step+1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # O8
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # O5

            train_loss += total_d_loss.item()
            epoch_lcw = step_lcw_sum / max(n_tiles, 1)  # actual average LCW this step
            lcw_curve_x.append(epoch + ((step + 1) / max(len(train_loader), 1)))
            lcw_curve_y.append(epoch_lcw)

            if step % 20 == 0:
                avg_loss = train_loss / (step + 1)
                avg_snr  = comp_snr / (step + 1)
                avg_ms   = comp_ms / (step + 1)
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    snr=f"{avg_snr:.4f}",
                    ms=f"{avg_ms:.4f}",
                    lcw=f"{epoch_lcw:.3f}",
                    tiles=f"{n_tiles}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            if LCW_PLOT_EVERY <= 1 or step % LCW_PLOT_EVERY == 0:
                save_lcw_curve(CHECKPOINT_DIR, lcw_curve_x, lcw_curve_y)

        # Flush remaining accumulation
        if len(train_loader) % ACCUM_STEPS != 0:
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # O8
                scaler.step(optimizer); scaler.update()
            except Exception as e:
                print(f"  (final flush skipped: {type(e).__name__})")
            optimizer.zero_grad(set_to_none=True)  # O5

        train_loss /= len(train_loader)
        avg_snr_epoch = comp_snr / max(len(train_loader), 1)
        avg_ms_epoch  = comp_ms / max(len(train_loader), 1)

        # -- Validate ----------------------------------------------------------
        model.eval(); cached_cond.eval()
        val_loss_acc = torch.tensor(0.0, device=device)
        val_snr_acc  = 0.0
        val_ms_acc   = 0.0
        val_tile_count = 0
        val_progress = min(1.0, ((epoch + 1) * len(train_loader)) / total_train_steps)
        val_curve_warmup = (1.0 - math.cos(math.pi * val_progress)) / 2.0

        with torch.inference_mode():
            for batch_v, paths_v in val_loader:
                batch_v = batch_v.to(device, non_blocking=True)
                B_v     = batch_v.shape[0]
                step_loss = torch.tensor(0.0, device=device)
                for bi in range(B_v):
                    # fix #9: hoist cond outside tile loop
                    cond_v = cached_cond.get_full_image_cond(batch_v[bi:bi+1], paths_v[bi])
                    val_views = random.sample(TILE_VIEWS, min(2, len(TILE_VIEWS)))
                    n_val_tiles = len(val_views)
                    for (h0, h1, w0, w1) in val_views:
                        tile = batch_v[bi:bi+1, :, h0:h1, w0:w1]
                        ts   = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (1,), device=device).long()
                        noisy, noise = add_simplex_noise(
                            tile, ts, alphas_cumprod, SIMPLEX_FREQ, SIMPLEX_OCTAVES)
                        t_ratio_v = ts.float() / noise_scheduler.config.num_train_timesteps
                        dynamic_lcw_v = MAX_TRAIN_LCW * val_curve_warmup * (1.0 - t_ratio_v)
                        dynamic_lcw_v = dynamic_lcw_v.view(1, 1, 1)
                        local_feats_v = conditioner.extract_features(tile)
                        local_cond_v = conditioner.proj(local_feats_v.float()).unsqueeze(1)
                        blended_cond_v = (dynamic_lcw_v * local_cond_v +
                                          (1.0 - dynamic_lcw_v) * cond_v)
                        retinal_mask_v = make_retinal_mask(tile)
                        with autocast(device_type=device_type, dtype=amp_dtype):
                            pn = model(noisy.to(amp_dtype), ts,
                                       encoder_hidden_states=blended_cond_v).sample
                        ac_t_v   = alphas_cumprod[ts].float().view(-1,1,1,1)
                        pred_x0_v = ((noisy.detach().float() - (1-ac_t_v).sqrt()*pn.float())
                                     / (ac_t_v.sqrt()+1e-8)).clamp(-1,1)
                        v_loss, v_comp = diffusion_loss(
                            pn, noise, pred_x0_v, tile,
                            retinal_mask_v, alphas_cumprod, ts, SNR_GAMMA)
                        # fix #25: tensor accumulation, no per-tile .item()
                        step_loss = step_loss + v_loss / (B_v * n_val_tiles)
                        val_snr_acc += v_comp['snr'].item()
                        val_ms_acc  += v_comp['ms'].item()
                        val_tile_count += 1
                val_loss_acc = val_loss_acc + step_loss

        val_loss     = val_loss_acc.item() / len(val_loader)
        val_snr_avg  = val_snr_acc / max(val_tile_count, 1)
        val_ms_avg   = val_ms_acc  / max(val_tile_count, 1)
        current_lr   = optimizer.param_groups[0]['lr']

        # -- Epoch summary (comprehensive) -------------------------------------
        print(f"\n{'='*80}")
        print(f"  Epoch {epoch:3d}/{EPOCHS}  |  LR: {current_lr:.1e}  |  LCW: {epoch_lcw:.3f}")
        print(f"  --------------------------------------------------------------")
        print(f"  train_loss: {train_loss:.5f}  |  val_loss: {val_loss:.5f}")
        print(f"  snr (train): {avg_snr_epoch:.5f}  |  snr (val): {val_snr_avg:.5f}")
        print(f"  ms  (train): {avg_ms_epoch:.5f}  |  ms  (val): {val_ms_avg:.5f}")
        print(f"{'='*80}")

        if not in_warmup:
            lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        append_csv_row(loss_csv, [
            epoch, train_loss, val_loss,
            f"{avg_snr_epoch:.6f}", f"{avg_ms_epoch:.6f}",
            f"{val_snr_avg:.6f}", f"{val_ms_avg:.6f}",
            current_lr, f"{epoch_lcw:.4f}", sw,
        ], "loss row")

        # -- Metrics (SSIM/PSNR over val) --------------------------------------
        metrics = None
        save_best_auroc = False
        if epoch % EVAL_EVERY == 0:
            print("  Computing val metrics (SSIM/PSNR)...")
            metrics = compute_val_metrics(
                model, cached_cond, ddim_scheduler,
                val_loader, alphas_cumprod, device, amp_dtype, device_type,
                max_batches=3, simplex_freq=SIMPLEX_FREQ,
                simplex_octaves=SIMPLEX_OCTAVES,
                T_start=DDIM_T_START, n_steps=DDIM_STEPS,
                max_lcw=MAX_TRAIN_LCW)
            metrics['epoch'] = epoch
            metrics_history.append(metrics)
            print(f"  +- Val Metrics ---------------------------------")
            print(f"  |  ssim:  {metrics['ssim']:.4f}")
            print(f"  |  psnr:  {metrics['psnr']:.2f} dB")
            print(f"  +----------------------------------------------")
            append_csv_row(metrics_csv, [epoch, f"{metrics['ssim']:.6f}",
                f"{metrics['psnr']:.4f}", f"{metrics['pixel_auroc']:.6f}",
                f"{metrics['pixel_ap']:.6f}"], "metrics row")

        # -- DDR eval (residual-based, the real anomaly metric) ---------------
        ddr_metrics = None
        is_final_epoch = (epoch == EPOCHS - 1)
        if DDR_IMAGES_DIR and ((epoch + 1) % DDR_EVAL_EVERY == 0 or is_final_epoch):
            print("  DDR real-lesion evaluation (residual-based)...")
            torch.cuda.empty_cache()
            ddr_metrics = compute_ddr_metrics(
                model, cached_cond, ddim_scheduler,
                alphas_cumprod, device, amp_dtype, device_type,
                DDR_IMAGES_DIR, DDR_MASKS_DIR,
                simplex_freq=SIMPLEX_FREQ, simplex_octaves=SIMPLEX_OCTAVES,
                max_images=DDR_MAX_IMAGES, T_start=DDIM_T_START,
                n_steps=DDIM_STEPS, max_lcw=MAX_TRAIN_LCW,
                max_seconds=DDR_MAX_SECONDS)
            if ddr_metrics:
                ddr_metrics['epoch'] = epoch
                ddr_history.append(ddr_metrics)
                print(f"  +- DDR Real-Lesion Eval [{ddr_metrics['n_images']} imgs] --")
                print(f"  |  ddr_auroc: {ddr_metrics['ddr_auroc']:.4f}")
                print(f"  |  ddr_ap:    {ddr_metrics['ddr_ap']:.4f}")
                print(f"  |  ddr_dice:  {ddr_metrics['ddr_dice']:.4f}  @thresh={ddr_metrics['ddr_thresh']:.2f}")
                print(f"  +----------------------------------------------")
                append_csv_row(ddr_csv, [
                    epoch,
                    f"{ddr_metrics['ddr_auroc']:.6f}",
                    f"{ddr_metrics['ddr_ap']:.6f}",
                    f"{ddr_metrics['ddr_dice']:.6f}",
                    f"{ddr_metrics['ddr_thresh']:.3f}",
                    ddr_metrics['n_images'],
                ], "DDR row")
                if ddr_metrics['ddr_auroc'] > best_auroc:
                    best_auroc = ddr_metrics['ddr_auroc']
                    save_best_auroc = True

        # -- Unified epoch history --------------------------------------------
        append_csv_row(epoch_metrics_csv, [
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{avg_snr_epoch:.6f}", f"{avg_ms_epoch:.6f}",
            f"{val_snr_avg:.6f}", f"{val_ms_avg:.6f}",
            current_lr, f"{epoch_lcw:.4f}", sw,
            f"{metrics['ssim']:.6f}" if metrics else "",
            f"{metrics['psnr']:.4f}" if metrics else "",
            f"{metrics['pixel_auroc']:.6f}" if metrics else "",
            f"{metrics['pixel_ap']:.6f}" if metrics else "",
            f"{ddr_metrics['ddr_auroc']:.6f}" if ddr_metrics else "",
            f"{ddr_metrics['ddr_ap']:.6f}" if ddr_metrics else "",
            f"{ddr_metrics['ddr_dice']:.6f}" if ddr_metrics else "",
            f"{ddr_metrics['ddr_thresh']:.3f}" if ddr_metrics else "",
            ddr_metrics['n_images'] if ddr_metrics else "",
        ], "epoch history row")

        # -- Checkpoints -------------------------------------------------------
        ckpt_data = {
            'epoch':            epoch,
            'model':            model.state_dict(),
            'conditioner_proj': conditioner.proj.state_dict(),
            'optimizer':        optimizer.state_dict(),
            'scaler':           scaler.state_dict(),
            'scheduler':        lr_scheduler.state_dict(),
            'best_val_loss':    best_val_loss,
            'best_auroc':       best_auroc,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss; ckpt_data['best_val_loss'] = best_val_loss
            torch.save(ckpt_data, best_loss_ckpt)
            print(f"  best_loss.pt saved (val={best_val_loss:.6f})")
        if save_best_auroc:
            ckpt_data['best_auroc'] = best_auroc
            torch.save(ckpt_data, best_auroc_ckpt)
            print(f"  best_auroc.pt saved (DDR AUROC={best_auroc:.4f})")
        torch.save(ckpt_data, last_ckpt)
        # periodic checkpoint every 2 epochs
        if epoch % 2 == 0:
            periodic_ckpt = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:04d}.pt")
            torch.save(ckpt_data, periodic_ckpt)
            print(f"  epoch_{epoch:04d}.pt saved")

        # -- Visualize ---------------------------------------------------------
        if epoch % VIS_EVERY == 0:
            torch.cuda.empty_cache()
            model.eval()

            precomputed = {}
            for i in range(n_vis):
                img = vis_images[i:i+1]
                recon_512, ms_resid = full_reconstruct_and_residual(
                    img, vis_paths[i], model, cached_cond, ddim_scheduler,
                    alphas_cumprod, device, amp_dtype, device_type,
                    SIMPLEX_FREQ, SIMPLEX_OCTAVES, DDIM_T_START, DDIM_STEPS,
                    max_lcw=MAX_TRAIN_LCW)
                precomputed[i] = (recon_512, ms_resid)

            save_visualizations(
                epoch, model, cached_cond,
                ddim_scheduler, vis_images, vis_paths,
                alphas_cumprod, device, amp_dtype, device_type,
                CHECKPOINT_DIR, ANOMALY_MAPS_RAW,
                n_vis, SIMPLEX_FREQ, SIMPLEX_OCTAVES,
                DDIM_T_START, DDIM_STEPS,
                precomputed=precomputed)

            save_anomaly_maps(epoch, vis_images, vis_paths, CHECKPOINT_DIR, precomputed)

            torch.cuda.empty_cache()

        # -- Loss curve --------------------------------------------------------
        plt.figure(figsize=(10,4))
        plt.plot(train_losses, label='Train', alpha=0.8)
        plt.plot(val_losses,   label='Val',   alpha=0.8)
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('512px v4 — MultiDiffusion | Loss Curve')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve.png'))
        plt.close()

        save_lcw_curve(CHECKPOINT_DIR, lcw_curve_x, lcw_curve_y)

        save_metrics_dashboard(CHECKPOINT_DIR, metrics_history, ddr_history)

        cached_cond._vit_cache.clear()

    print(f"\nDone. Best val loss: {best_val_loss:.6f} | Best DDR AUROC: {best_auroc:.4f}")

    # -- Sweep (runs after training when SWEEP_MODE=1) ------------------------
    if SWEEP_MODE:
        print("\nSWEEP_MODE=1 detected -- running parameter sweep...")
        best_ckpt = best_loss_ckpt if os.path.exists(best_loss_ckpt) else last_ckpt
        print(f"  Loading {os.path.basename(best_ckpt)} for sweep inference")
        try:
            sw_ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        except Exception:
            sw_ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        target = model._orig_mod if hasattr(model, '_orig_mod') else model
        target.load_state_dict(strip_compile_prefix(sw_ckpt['model']))
        conditioner.proj.load_state_dict(strip_compile_prefix(sw_ckpt['conditioner_proj']))
        model.eval(); cached_cond.eval()
        run_sweep(
            model, cached_cond, ddim_scheduler, alphas_cumprod,
            device, amp_dtype, device_type,
            SWEEP_CSV, SWEEP_OUT_DIR, CROP_SIZE,
            SWEEP_T_STARTS, SWEEP_DDIM_STEPS, SWEEP_LCW,
            SIMPLEX_FREQ, SIMPLEX_OCTAVES,
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)


