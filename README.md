# NetrAI — Retinal Anomaly Detection via Diffusion-Based Reconstruction

> A diffusion model system for unsupervised retinal lesion detection using SDEdit-style reconstruction error as an anomaly score. Conditions on frozen RETFound (ViT-Large) embeddings and performs inference via MultiDiffusion tiling over full 512px fundus images.

---

## Table of Contents

- [Overview](#overview)
- [Training Dataset](#training-dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Module Reference — Diffusion](#module-reference--diffusion)
  - [train.py](#trainpy)
  - [models.py](#modelspy)
  - [diffusion.py](#diffusionpy)
  - [data.py](#datapy)
  - [losses.py](#lossespy)
  - [evaluation.py](#evaluationpy)
  - [visualization.py](#visualizationpy)
  - [sweep.py](#sweeppy)
  - [utils.py](#utilspy)
  - [config.yaml](#configyaml)
- [Key Design Decisions — Diffusion](#key-design-decisions--diffusion)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Requirements](#requirements)
- [Usage — Diffusion](#usage--diffusion)
- [Classifier Pipeline](#classifier-pipeline)
  - [Classifier Architecture](#classifier-architecture)
  - [Classifier Project Structure](#classifier-project-structure)
  - [Classifier Setup](#classifier-setup)
  - [Classifier Training Pipeline](#classifier-training-pipeline)
  - [Classifier Configuration Reference](#classifier-configuration-reference)
  - [Running Tests](#running-tests)
  - [Key Design Decisions — Classifier](#key-design-decisions--classifier)

---

## Overview

NetrAI frames retinal anomaly detection as a **reconstruction problem**. A DDPM/DDIM diffusion UNet is trained exclusively on healthy retinal images. At inference, a test image is partially noised (SDEdit, `T_start < 1000`) and reconstructed. The residual between the original and reconstruction is the anomaly map — lesions the model never saw during training produce high residual signal.

The primary evaluation metric is **pixel-level AUROC on the DDR and iDRiD datasets** (757 annotated fundus images with MA/HE/EX/SE lesion masks), measured via vessel-suppressed residual maps (Frangi filter post-processing).

---

## Training Dataset

The model is trained exclusively on **22,104 healthy fundus images** drawn from 6 public sources:

| Source | Count | % of Total |
|---|---|---|
| EyePACS | 13,500 | 61.1% |
| DDR | 5,241 | 23.7% |
| APTOS | 1,625 | 7.4% |
| REFUGE2 | 1,080 | 4.9% |
| MESSIDOR-2 | 630 | 2.9% |
| STARE | 28 | 0.1% |
| **Total** | **22,104** | **100%** |

Only grade-0 (no diabetic retinopathy) images are used from graded datasets (EyePACS, DDR, APTOS, MESSIDOR-2). The model never sees lesion-bearing images during training — anomaly detection at inference relies entirely on reconstruction error against this healthy prior.

---

## Architecture

```
Input (512×512 fundus image)
        |
        ▼
RETFoundConditioner ──► Frozen ViT-Large (224px, ImageNet-normalized)
        |                       |
        |               1024-d class token
        |                       |
        |               proj MLP (1024+768→768, GELU + LayerNorm)
        |                       |
        |               cross_attention_dim=768 conditioning vector
        |
        ▼
UNet2DConditionModel (diffusers)
  ├── sample_size:        256  (trained on 256×256 tiles)
  ├── in/out channels:    3
  ├── block_out_channels: (128, 256, 512, 512)
  ├── down blocks:        DownBlock2D → 3× CrossAttnDownBlock2D
  ├── up blocks:          3× CrossAttnUpBlock2D → UpBlock2D
  └── cross_attention_dim: 768
        |
        ▼ (inference only)
MultiDiffusion Tiling (9 overlapping 256×256 tiles @ stride=128, over 512×512)
        |
        ▼
Reconstruction (512×512)
        |
        ▼
Residual Map ──► Retinal Ellipse Mask ──► Frangi Vessel Suppression
        |
        ▼
Anomaly Score (pixel-level, used for DDR AUROC)
```

---

## Project Structure

```
NetrAI/
├── config.yaml           # All hyperparameters and paths (single source of truth)
│
├── train.py              # Main training loop, model setup, checkpoint I/O
├── models.py             # RETFoundConditioner, CachedConditioner
├── diffusion.py          # Noise schedule, DDIM steps, MultiDiffusion tiling, residual
├── data.py               # RetinaDataset (CSV/TXT/dir), transforms, collate_fn
├── losses.py             # SNR-weighted loss, L1+Focal Frequency hybrid loss
├── evaluation.py         # DDR AUROC/AP/Dice, SSIM/PSNR, vessel-aware post-processing
├── visualization.py      # Reconstruction panels, anomaly map overlays, metrics dashboard
├── sweep.py              # Hyperparameter sweep: T_start × DDIM_STEPS × LCW grid
└── utils.py              # CSV helpers, checkpoint utilities, terminal logging, LCW curve

classifier/               # Disease classification pipeline (see Classifier section)
├── config.yaml
├── model.py
├── losses.py
├── data.py
├── retfound.py
├── train.py
├── extract.py
├── xgboost_clf.py
├── inference.py
├── utils.py
├── __init__.py
├── __main__.py
└── tests/
```

---

## Module Reference — Diffusion

### `train.py`

The orchestration hub. Reads `config.yaml`, builds all components, and runs the training loop.

**Key responsibilities:**
- Parses all config sections (`paths`, `training`, `diffusion`, `eval`, `sweep`) into local variables.
- Sets CUDA environment flags: `expandable_segments`, TF32, cuDNN benchmark.
- Builds the UNet (`UNet2DConditionModel` from diffusers) with channels-last memory format and gradient checkpointing enabled.
- Builds `RETFoundConditioner` + `CachedConditioner`.
- Attempts `torch.compile(model, mode="default")` and falls back gracefully.
- Enables xformers **only when compile is inactive** (compile + xformers causes attention processor cache thrash).
- Constructs two-group AdamW optimizer (separate LRs for UNet and conditioner projection MLP).
- Handles three checkpoint loading scenarios:
  - i. Resume from `last.pt` (full state including optimizer/scaler/scheduler)
  - ii. Warm-start from a 256px checkpoint (loads weights, resets LR to 30% base)
  - iii. Train from scratch

**Training loop per epoch:**
- Randomly samples `NUM_TRAIN_TILES` tiles per image during training (not all 9) to keep iteration time reasonable.
- Computes `lcw` (Local Conditioning Weight) via a cosine schedule over the entire training run — zero at step 0, rising to `MAX_TRAIN_LCW` by the final step.
- Mixed-precision forward pass with `autocast` (bfloat16 if supported, else float16).
- `diffusion_loss` = 0.6 × SNR-weighted MSE + 0.4 × L1/Focal-Frequency hybrid.
- Gradient accumulation over `ACCUM_STEPS` micro-batches, then `clip_grad_norm(1.0)`.
- Cosine LR schedule with linear warmup.
- Periodic: validation SSIM/PSNR, DDR AUROC eval, visualization saves, CSV logging.
- Saves: `last.pt`, `best_loss.pt`, `best_auroc.pt`, `loss.csv`, `metrics.csv`, `ddr_metrics.csv`.

**LCW Schedule:**
```
LCW(step) = MAX_TRAIN_LCW × 0.5 × (1 - cos(π × (current_step / total_train_steps)))
```
During inference, LCW scales down linearly with timestep: `dynamic_lcw = max_lcw × (1 - t/1000)`, giving global conditioning dominance at high noise and local tile conditioning at low noise.

---

### `models.py`

#### `RETFoundConditioner`

Wraps a frozen RETFound ViT-Large (via `timm`) with a trainable projection MLP.

- Loads `RETFound_cfp_weights.pth` from a configurable path (falls back to torchvision ViT-L if timm is unavailable).
- ViT parameters are frozen (`requires_grad_(False)`) and kept in `eval()` permanently — `train()` override ensures only the projection MLP ever enters training mode.
- Input preprocessing: bicubic resize to 224×224 → revert from diffusion `[-1,1]` to `[0,1]` → ImageNet normalize.
- Output: `(B, 1, 768)` cross-attention conditioning tensor.
- ImageNet mean/std registered as **buffers** (no per-call tensor allocation, optimization O6).

#### `CachedConditioner`

A stateful wrapper around `RETFoundConditioner` that caches raw ViT features (not proj MLP outputs) in an `OrderedDict` LRU cache (max 500 entries).

- Cache key: `(image_path, "full")` or `(image_path, tile_id)`.
- LRU eviction via `move_to_end` / `popitem(last=False)`.
- `get_full_image_cond(img, path)` — get conditioning for a full 512px image.
- `get_tile_conds_batched(img_512, tile_views)` — single batched ViT forward pass for all 9 tiles (optimization fix #23, eliminates 9 sequential ViT calls during inference).

---

### `diffusion.py`

All diffusion math, MultiDiffusion tiling, and the retinal mask.

**Noise functions** (named "simplex" for call-site compatibility, actually standard Gaussian):
- `generate_simplex_noise` — `torch.randn`, shaped.
- `add_simplex_noise` — standard DDPM forward process: `x_t = √ā·x₀ + √(1-ā)·ε`.
- `simplex_ddim_step` — DDIM reverse step with optional stochasticity (`eta`). Clamps `pred_x0` to `[-1,1]`.

**MultiDiffusion tiling constants:**

| Constant | Value | Notes |
|---|---|---|
| `TILE_SIZE` | 256 | UNet input size |
| `FULL_SIZE` | 512 | Full inference resolution |
| `TILE_STRIDE` | 128 | 50% overlap |
| `NUM_TRAIN_TILES` | 2 | Tiles sampled per image during training |
| `TILE_VIEWS` | 9 tiles | Precomputed `(h0,h1,w0,w1)` coordinates |

- `make_linear_weight` — 2D pyramid weight for tile blending. Peaks at 1.0 in tile center, tapers to near-zero at edges, ensuring smooth seams. Cached per device (`_LINEAR_WEIGHT_CACHE`).
- `make_retinal_mask` — Dynamic elliptical FOV mask:
  1. Convert to `[0,1]`, threshold pixels < 0.05 as background.
  2. Find bounding box of illuminated pixels per image in batch.
  3. Fit an ellipse to the exact FOV bounds (independent `rx`, `ry`).
  4. Returns intersection of illuminated pixels and ellipse — excludes black borders, camera vignette.
- `multidiffusion_reconstruct` — Core inference loop:
  1. Partially noise `img_512` to `T_start`.
  2. Run DDIM denoising with per-tile UNet calls, fusing outputs via pyramid-weighted accumulation divided by precomputed `count`.
  3. `dynamic_lcw` blends tile-specific local conditioning with global conditioning at each timestep.
- `multidiffusion_reconstruct_full` — Adds retinal masking and computes L1 residual after `multidiffusion_reconstruct`.

**Multi-scale residual** (`multiscale_residual`):
- Runs separate reconstructions at 256px, 128px, 64px.
- Upscales all back to 512px.
- Takes element-wise **maximum** across scales (no weighted sum — avoids starving coarse scales).
- Applies retinal mask.

---

### `data.py`

- `make_transform` — Builds torchvision transform pipeline. BILINEAR interpolation throughout (avoids bicubic ringing artifacts in L1 residuals).
- Train augmentations: RandomResizedCrop (scale 0.8–1.0, ratio 0.9–1.1), RandomHorizontalFlip (p=0.5), RandomVerticalFlip (p=0.5), RandomRotation ±15°, ColorJitter (mild).
- Val: CenterCrop only.

`RetinaDataset` — Accepts three source types:
- `.csv` with a `path` or `image` column (handles relative paths using `csv_dir` as base, supports `source` column for per-source counts)
- `.txt` newline-separated path list
- Directory (recursive glob for jpg/jpeg/png)

Features:
- Optional `bad_files_txt` to pre-filter known corrupted files.
- Retry loop (up to 10 attempts) to skip unreadable images at runtime without crashing.
- Returns `(tensor, path)` tuples — paths are used as cache keys in `CachedConditioner`.

`collate_fn` — Stacks tensors, keeps paths as a Python list (not tensor). Required for cache key usage.

---

### `losses.py`

`snr_weighted_loss` (Min-SNR strategy, γ=2.0):
```
SNR(t) = ā_t / (1 - ā_t)
weight(t) = clamp(SNR, max=γ) / (SNR + ε)
loss = mean(weight × MSE(pred_noise, noise))
```
- High noise (`t = 1000`): SNR ≈ 0, weight ≈ 1.0 — maximum penalty on coarse structure errors.
- Low noise (`t = 0`): SNR >> γ, weight drops to γ/SNR — prevents obsessing over tile boundary micro-details.
- γ=2.0 is intentionally aggressive (standard is 5.0) to synergize with the LCW tile fusion — stops the model from memorizing rigid 256px tile edges.

`l1_focal_frequency_loss` — Two-component hybrid:
1. **Spatial L1 (The Anchor):** `mean(|pred_x0 - x0|)` inside retinal mask. Provides stable, localized gradient for gross structure.
2. **Focal Frequency Loss (The Sniper):** FFT on masked inputs → amplitude spectrum difference → self-weighted by `freq_diff.detach()` (harder frequencies get higher weight). Improves high-frequency lesion texture recovery.
Combined: `L = L1 + 0.05 × FFL`

`diffusion_loss` — Top-level combinator:
```
total = 0.6 × snr_weighted_loss + 0.4 × l1_focal_frequency_loss
```

---

### `evaluation.py`

`postprocess_residual` — Vessel-aware anomaly map cleaning pipeline:
1. Raw L1 residual per pixel, median-subtracted (removes diffuse reconstruction haze).
2. Light Gaussian blur (σ=0.5) to kill single-pixel noise.
3. Frangi filter on green channel (σ=2–8, black ridges — vessel detection on input anatomy).
4. Frangi filter on residual itself (σ=1–5, bright ridges — catches vessel-shaped model artifacts).
5. Combined vessel map → soft exponential suppression: `weight = exp(-1.5 × vessel_norm)`.
6. Final retinal mask application + normalization to `[0,1]`.

This preserves lesions adjacent to vessels (exponential decay, not hard clipping) while eliminating the dominant vessel signal that would otherwise dominate AUROC.

`compute_ddr_metrics` — Full DDR evaluation:
- Globs all `.jpg` / `.jpeg` / `.png` in `ddr_images_dir`.
- For each, loads combined binary mask from `ddr_masks_dir/{MA,HE,EX,SE}/{stem}.tif`.
- Runs `multidiffusion_reconstruct_full` → `postprocess_residual`.
- Concatenates all pixel predictions and GT labels.
- Computes pixel-level AUROC, AP, and best-threshold Dice (threshold sweep over 20 points in `[0.1, 0.9]`).
- Supports time-budget (`max_seconds`) and count-budget (`max_images`) cutoffs with graceful early stopping.
- VRAM defrag: `torch.cuda.empty_cache()` every 50 images.

`compute_val_metrics` — SSIM/PSNR on validation batches. SDEdit AUROC removed (DDR is the only meaningful anomaly metric, per code comment).

`compute_ssim` / `compute_psnr` — Pure numpy implementations, grayscale SSIM.

---

### `visualization.py`

`save_visualizations` — Per-epoch 5-panel reconstruction grid:

| Column | Content |
|---|---|
| Original | Raw input |
| Recon (masked) | Reconstruction × retinal mask |
| Signed Diff | `(orig - recon)` in RdBu colormap, ±0.15 |
| MultiScale | Multi-scale ensemble residual (hot) |
| Clean Residual | Vessel-suppressed anomaly map (hot, normalized) |

`save_anomaly_maps` — Per-image dark-theme 5-panel figure with overlay (original + alpha-blended hot anomaly map, α capped at 0.6).

`save_metrics_dashboard` — 4-panel matplotlib figure: AUROC/AP (val), SSIM, PSNR, DDR AUROC/Dice history.

---

### `sweep.py`

`run_sweep` — Grid search over `T_start × DDIM_steps × max_lcw`:
- Iterates all parameter combinations for each sweep image.
- Saves per-combo reconstructions and vessel-suppressed residual heatmaps.
- Applies Frangi Vessel Suppression (`postprocess_residual`) directly to the sweep outputs so the generated hyperparameter heatmaps evaluate true lesions rather than healthy vascular structures.
- Writes `sweep_metrics.csv` with SSIM, PSNR, residual mean/max, and wall-clock seconds per combo.
- Generates panel plots (grid of DDIM_steps × T_start) per image per LCW value.

Activated via `sweep.enabled: true` in config — runs instead of training when set.

---

### `utils.py`

- `strip_compile_prefix` — Strips `_orig_mod.` from state dicts saved under `torch.compile`, enabling cross-compiled/uncompiled checkpoint loading.
- `repair_csv_header` — Rewrites a CSV header in-place if the schema on disk differs from the current expected columns (handles checkpoint resume across schema changes).
- `append_csv_row` — Safe CSV append with exception handling so a disk write failure never crashes training.
- `load_loss_history` — Parses `loss.csv` with schema tolerance (missing columns default to `None`) to reconstruct LCW history curves on resume.
- `_TeeStream` / `setup_terminal_logging` — Redirects `stdout` and `stderr` to both console and a log file simultaneously via `atexit`-registered cleanup.
- `save_lcw_curve` — Plots the LCW schedule as experienced during training (epoch progress vs LCW value, with scatter downsampling for large runs).

---

### `config.yaml`

Single YAML file controlling all behavior. No hardcoded paths in code.

```yaml
paths:
  data_train:       # CSV/TXT/dir — healthy training images
  data_val:         # CSV/TXT/dir — healthy validation images
  bad_files_txt:    # Optional path list to pre-filter corrupted images
  checkpoint_dir:   # Where to save checkpoints + logs
  pretrained_256:   # Optional 256px warm-start checkpoint
  ddr_images_dir:   # DDR evaluation image directory
  ddr_masks_dir:    # DDR evaluation mask directory (MA/HE/EX/SE subdirs)
  idrid_images_dir: # iDRiD evaluation image directory
  idrid_masks_dir:  # iDRiD evaluation mask directory (MA/HE/EX/SE subdirs)
  retfound_weights: # Path to RETFound_cfp_weights.pth

sweep:
  enabled:    false
  csv:                     # Images to sweep over
  out_dir:                 # Where to save sweep outputs
  t_starts:   [200,250,300,350]
  ddim_steps: [50]
  lcw_values: [0.4]

training:
  crop_size:         512   # Training crop (tiles are 256×256 within this)
  epochs:            20
  batch_size:        6
  accum_steps:       6     # Effective batch = batch_size × accum_steps = 36
  warmup_epochs:     3
  num_workers_train: 4
  num_workers_val:   0
  prefetch_factor:   2
  lr_unet:           5e-6
  lr_conditioner:    1e-5
  snr_gamma:         2.0

diffusion:
  simplex_freq:     8      # Unused (Gaussian noise used)
  simplex_octaves:  4      # Unused
  ddim_steps:       50
  ddim_t_start:     300    # Partial noising depth for SDEdit
  max_train_lcw:    0.4    # Peak local conditioning weight

eval:
  vis_every:          1    # Visualize every N epochs
  eval_every:         10   # SSIM/PSNR every N epochs
  num_vis:            1    # Number of images in vis panel
  ddr_eval_every:     10   # DDR AUROC every N epochs
  ddr_max_images:     ~    # null = all paired images
  ddr_max_seconds:    ~    # null = no time cap
  idrid_eval_every:   10   # iDRiD AUROC every N epochs
  idrid_max_images:   ~
  idrid_max_seconds:  ~
  lcw_plot_every:     50   # Save LCW curve every N steps
```

---

## Key Design Decisions — Diffusion

| Decision | Rationale |
|---|---|
| Train on 256px tiles, infer on 512px via MultiDiffusion | UNet fits in VRAM at 256px; MultiDiffusion fuses overlapping tiles for seamless 512px output |
| Frozen RETFound ViT-Large | RETFound captures fundus-specific anatomy; fine-tuning would destroy the generic healthy-retina prior |
| Cache raw ViT features, not proj MLP outputs | Proj MLP trains, so caching its output would cause stale gradients across iterations |
| SNR-γ=2.0 (aggressive) | Paired with LCW: prevents the model from overfitting tile boundary micro-textures at low noise |
| Frangi on both input green channel AND residual | Green channel catches vascular anatomy; residual Frangi catches vessel-shaped reconstruction artifacts the model introduces |
| Element-wise max for multi-scale ensemble | Weighted sum would cap small-scale lesion signal (at 0.2 weight for 64px scale); max lets each scale compete at full confidence |
| LCW rises with cosine schedule over full training run | Prevents tile-specific conditioning from overwhelming global structure conditioning before the UNet has learned coarse anatomy |
| BILINEAR interpolation throughout data pipeline | Avoids bicubic ringing artifacts that contaminate the L1 residual anomaly map |
| xformers disabled when `torch.compile` is active | Compile + xformers causes attention processor identity checks to thrash the dynamo cache (set to 64 limit) |

---

## Training Pipeline

```
Epoch start
|
├── [cosine over full run] LCW schedule → LCW value for this step
|
├── For each batch:
|   ├── Sample NUM_TRAIN_TILES=2 random tile views
|   ├── CachedConditioner → ViT features (LRU cached per path)
|   ├── add_simplex_noise → x_t at random t ∈ [0, 1000]
|   ├── autocast forward → pred_noise
|   ├── diffusion_loss (SNR + FFL hybrid)
|   ├── GradScaler backward
|   └── every ACCUM_STEPS: clip_grad_norm(1.0) → optimizer.step()
|
├── [every VIS_EVERY epochs] save_visualizations
├── [every EVAL_EVERY epochs] compute_val_metrics (SSIM/PSNR)
├── [every DDR_EVAL_EVERY epochs] compute_ddr_metrics → DDR AUROC
|
└── Save: last.pt, best_loss.pt, best_auroc.pt, CSVs, dashboard
```

---

## Inference Pipeline

```
Input: 512×512 fundus image

1. CachedConditioner.get_full_image_cond()  →  global_cond (1,1,768)
2. CachedConditioner.get_tile_conds_batched()  →  9× local_cond (1,1,768)
3. add_simplex_noise(img, T_start=300)  →  x_T
4. DDIM loop (50 steps):
   For each of 9 tiles:
     tile_cond = LCW(t) × local_cond + (1 - LCW(t)) × global_cond
     pred_noise = UNet(tile, t, tile_cond)
     tile_denoised = DDIM_step(tile, pred_noise)
     accumulate: value[h0:h1, w0:w1] += tile_denoised × pyramid_weight
   x_t = value / count
5. recon_512 = x_T (final)
6. residual = |img - recon_512|.mean(channel) × retinal_mask
7. postprocess_residual():
   - Median subtract → Gaussian blur
   - Frangi(green channel) + Frangi(residual) → vessel map
   - anomaly_map = residual × exp(-1.5 × vessel_norm)
8. Output: anomaly_map ∈ [0,1], pixel-level
```

---

## Evaluation

### DDR Dataset
- 757 labeled fundus images with pixel-level annotations for 4 lesion types:
  - **MA** — Microaneurysms
  - **HE** — Hemorrhages
  - **EX** — Hard Exudates
  - **SE** — Soft Exudates (Cotton Wool Spots)
- Masks combined into a single binary map (logical OR across lesion types).
- Pixel-level metrics: **AUROC** (primary), **AP**, **Dice** (best threshold via sweep).
- Expected directory structure:
  ```
  ddr_masks_dir/
  ├── MA/  {stem}.tif
  ├── HE/  {stem}.tif
  ├── EX/  {stem}.tif
  └── SE/  {stem}.tif
  ```

### iDRiD Dataset
- 81 labeled fundus images with pixel-level annotations for 4 lesion types (MA, HE, EX, SE).
- Masks combined into a single binary map (logical OR across lesion types).
- Pixel-level metrics: **AUROC** (primary), **AP**, **Dice** (best threshold via sweep).
- Expected directory structure:
  ```
  idrid_masks_dir/
  ├── MA/  {stem}.tif
  ├── HE/  {stem}.tif
  ├── EX/  {stem}.tif
  └── SE/  {stem}.tif
  ```

---

## Checkpoints

All checkpoints saved to `checkpoint_dir`:

| File | Contents |
|---|---|
| `last.pt` | Full training state: model, conditioner_proj, optimizer, scaler, scheduler, epoch, best metrics |
| `best_loss.pt` | Snapshot at lowest validation loss |
| `best_auroc.pt` | Snapshot at highest DDR AUROC |
| `loss.csv` | Per-epoch: train_loss, val_loss, snr, ms, val_snr, val_ms, lr, lcw |
| `metrics.csv` | Per-eval: SSIM, PSNR, pixel_auroc, pixel_ap |
| `ddr_metrics.csv` | Per-DDR-eval: ddr_auroc, ddr_ap, ddr_dice, ddr_thresh, n_images |
| `idrid_metrics.csv` | Per-iDRiD-eval: idrid_auroc, idrid_ap, idrid_dice, idrid_thresh, n_images |
| `epoch_metrics.csv` | Combined per-epoch summary |
| `train_terminal.log` | Full stdout/stderr mirror |
| `lcw_curve.png` | LCW vs epoch progress plot |
| `metrics_dashboard.png` | 4-panel metrics history figure |
| `recon_epoch_XXXX.png` | Per-epoch reconstruction panels |
| `anomaly_maps_raw/` | Per-image hot-colormap anomaly maps |
| `anomaly_maps/` | Per-image dark-theme overlay panels |

---

## Requirements

```
torch >= 2.0
torchvision
diffusers
timm
scikit-image       # frangi filter
scikit-learn       # roc_auc_score, average_precision_score
scipy              # gaussian_filter
numpy
pandas
Pillow
matplotlib
tqdm
pyyaml
```

Optional:
```
xformers           # Memory-efficient attention (disabled when torch.compile is active)
```

---

## Usage — Diffusion

**Train:**
```bash
python -m diffusion.train --config diffusion/config.yaml
```

*For persistent logging in a detached session:*
```bash
python -m diffusion.train --config diffusion/config.yaml 2>&1 | tee -a checkpoints_512v4/train.log
```

**Sweep mode** (set `sweep.enabled: true` in config):
```bash
python -m diffusion.train --config diffusion/config.yaml
```

**Resume:** Training auto-resumes from `last.pt` if it exists in `checkpoint_dir`. No flag needed.

**Warm-start from 256px checkpoint:** Set `paths.pretrained_256` to your 256px `last.pt`. LR is automatically scaled to 30% base and warmup reduced to 1 epoch.

---

---

# Classifier Pipeline

The `classifier/` module takes the diffusion model's **clean residual anomaly maps** and trains a hybrid deep learning + classical ML pipeline to classify retinal images into three disease categories: **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Pathologic Myopia (PM)**.

> The diffusion model generates the *where* (anomaly maps). The classifier determines the *what* (disease label).

---

## Classifier Architecture

```
Retina Image (512×512)
        │
        ├──[ONE TIME]──▶ RETFound-Large (frozen ViT-L/16)
        │                      │
        │               1024-D .pt cache (per image)
        │               key: {split}_{class}_{stem}.pt
        │
        └──▶ MIT-B3 SegFormer (ImageNet init, trainable)
                    │
             Custom Decode Head
             F_concat  (B × 1024 × 128 × 128)
                    │
             Late Spatial Gate
             F_gated = F + α·(F ⊙ A_scaled)    α = learned scalar
                    │
             Global Avg Pool  →  (B × 1024)
                    │
        ┌───────────┴───────────┐
     Path A                 Path B (VIB)
  Linear(1024→384)       Linear(1024→768)
     384-D                 μ(384) + log_σ²(384)
     raw context         Training: z = μ + σε
                         Inference: z = μ  (deterministic)
        └───────────┬───────────┘
                    │ + scalar: mean(A_scaled)  [1-D]
                    ▼
              769-D Vector
                    │
         Load cached 1024-D RETFound embedding
                    │
              1793-D Vector
                    ▼
               XGBoost (tree_method=hist, GPU)
          DR / Glaucoma / PM + confidence %
```

### Loss Function

No cross-entropy head. SupCon is the **sole** supervisory signal during SegFormer training:

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{SupCon}}_{1.0} + \underbrace{1.0 \cdot \beta(t) \cdot \mathcal{L}_{KL}}_{\text{VIB bottleneck}} + \underbrace{0.1 \cdot \mathcal{L}_{Ortho}}_{\mu\ \text{only}}$$

| Loss | Applied to | Purpose |
|---|---|---|
| **SupCon** | Full 769-D vector | Forces same-disease vectors to cluster, pushes apart different diseases |
| **KL** (`β`-annealed) | Path B μ, log_σ² | VIB bottleneck — discards noisy features, keeps strongest disease signals |
| **Ortho** | Path B **μ only** | Cosine similarity penalty — forces distinct, non-overlapping features per class |

**β-annealing:** β = 0 for first 10 epochs → linearly ramps to 0.001 over next 20 epochs. `lambda_kl = 1.0` is a transparent pass-through — β via `BetaScheduler` is the **sole** bottleneck coefficient (matching Alemi et al. 2017 and Higgins et al. 2017).

**Class balance:** `WeightedRandomSampler` enforces 1:1:1 (DR:Glaucoma:PM) per batch + class-aware SupCon temperature (PM uses τ=0.04 vs 0.07).

---

## Classifier Project Structure

```
classifier/
├── config.yaml          ← All hyperparameters
├── model.py             ← NetrAiEncoder (SegFormer + gate + VIB bottleneck)
├── losses.py            ← SupCon + KL + Ortho + BetaScheduler + NetrAiLoss
├── data.py              ← RetinalDataset + balanced DataLoader
├── retfound.py          ← RETFound pre-computation + collision-safe cache I/O
├── train.py             ← SegFormer training loop (AMP, dual LR, checkpointing)
├── extract.py           ← 769-D → 1793-D feature extraction to .npy
├── xgboost_clf.py       ← XGBoost train / eval / SHAP (tree_method=hist)
├── inference.py         ← Single-image end-to-end diagnosis
├── utils.py             ← Logging, checkpointing, metrics, LR scheduler
├── requirements.txt
├── __init__.py
├── __main__.py          ← CLI dispatcher (5 commands)
└── tests/
    ├── conftest.py      ← Shared pytest fixtures (temp dataset, no GPU needed)
    ├── test_model.py    ← Model shape + gradient contracts
    ├── test_losses.py   ← Loss function unit tests
    └── test_data.py     ← Dataset + DataLoader tests
```

---

## Classifier Setup

```bash
# Install dependencies
pip install -r classifier/requirements.txt

# Verify tests pass (no GPU, no downloads required)
pytest classifier/tests/ -v
```

---

## Classifier Training Pipeline

### Step 0 — Prepare Data

Organise images into class folders:

```
data/classifier/
├── train/
│   ├── DR/          ← .jpg / .png retina images
│   ├── Glaucoma/
│   └── PM/
└── val/
    ├── DR/
    ├── Glaucoma/
    └── PM/
```

Place the diffusion model's **clean residual** anomaly maps in a flat directory:

```
data/anomaly_maps/
├── <image_stem>_anomaly.png   ← preferred naming
└── <image_stem>.png           ← fallback naming
```

> If an anomaly map is missing, the gate defaults to `F_gated = F_concat` (identity — no anomaly guidance). Training continues without the diffusion prior for that sample.

Update paths in `classifier/config.yaml` to match your directory layout.

---

### Step 1 — Cache RETFound Embeddings *(one time)*

```bash
python -m classifier cache-retfound --config classifier/config.yaml
```

Runs every image through frozen RETFound-Large. Saves collision-safe 1024-D `.pt` files:
- `train/DR/img_001.png` → `retfound_cache/train_DR_img_001.pt`
- `val/Glaucoma/img_001.png` → `retfound_cache/val_Glaucoma_img_001.pt`

RETFound is then unloaded from VRAM permanently.

> **RETFound weights**: Download `RETFound_cfp_weights.pth` from the [RETFound repository](https://github.com/rmaphoh/RETFound_MAE) and set `paths.retfound_weights` in `config.yaml`. Falls back to HuggingFace ViT-L/16 ImageNet-21k if not provided (domain gap applies).

---

### Step 2 — Train the SegFormer Encoder

```bash
python -m classifier train --config classifier/config.yaml

# Resume from a checkpoint
python -m classifier train --config classifier/config.yaml \
                           --resume checkpoints/classifier/epoch_0050.pt
```

Trains for `training.epochs` epochs. Best checkpoint by val loss saved as `best.pt`.

**What to watch in the logs:**
- `α` (gate scalar) should stabilise — too high means the gate is dominating
- `β` ramps up after epoch 10 — `l_kl` will start increasing
- `l_ortho` should decrease as class μ vectors become more orthogonal
- `l_supcon` drives everything — if it stalls, check class balance

---

### Step 3 — Extract Feature Vectors

```bash
python -m classifier extract --config classifier/config.yaml
```

Loads `best.pt`, runs every image through frozen encoder, concatenates RETFound embeddings, saves:

```
features/
├── train_features.npy   (N_train, 1793)
├── train_labels.npy     (N_train,)
├── train_stems.json
├── val_features.npy     (N_val, 1793)
├── val_labels.npy       (N_val,)
└── val_stems.json
```

---

### Step 4 — Train XGBoost

```bash
python -m classifier xgboost --config classifier/config.yaml --shap
```

Trains on 1793-D vectors with early stopping. Uses `tree_method=hist` for reliable GPU acceleration. Saves:
- `xgboost_model.pkl` — the trained booster
- `xgboost_results.json` — train/val metrics (accuracy, macro F1, AUC-ROC, confusion matrix)
- `shap_importance.json` — top feature importances (if `--shap`)

**Feature name mapping in SHAP output:**

| Dimension range | Name prefix | Source |
|---|---|---|
| 0 – 383 | `segformer_pathA_XXX` | Path A raw context |
| 384 – 767 | `segformer_vib_XXX` | Path B VIB μ |
| 768 | `global_anomaly_score` | mean(clean residual) |
| 769 – 1792 | `retfound_XXXX` | RETFound [CLS] embedding |

---

### Step 5 — Run Inference

```bash
python -m classifier infer \
    --config  classifier/config.yaml \
    --image   patient_001.jpg \
    --anomaly patient_001_anomaly.png
```

Output:
```
══════════════════════════════════════════════════
  DIAGNOSIS:  DR
  CONFIDENCE:
    DR          92.4%  ████████████████████████████████████████
    Glaucoma     5.9%  ██
    PM           1.7%
  Vector dim: (1793,)
══════════════════════════════════════════════════
```

> If no anomaly map is available, omit `--anomaly`. The gate defaults to identity.

---

## Classifier Configuration Reference

```yaml
training:
  epochs:      100
  batch_size:  16
  lr:          1.0e-4        # head LR; backbone gets lr × 0.1
  weight_decay: 1.0e-4
  warmup_epochs: 5
  grad_clip:   1.0
  amp:         true          # bfloat16 on Ampere+, float16 otherwise

  supcon_weight:  1.0        # SupCon drives everything
  lambda_kl:      1.0        # pass-through — β is the sole VIB knob
  lambda_ortho:   0.1        # Orthogonal penalty weight

  supcon_temperatures:
    0: 0.07                  # DR
    1: 0.07                  # Glaucoma
    2: 0.04                  # PM (minority → sharper gradient)

  beta_warmup_epochs: 10     # β=0 for first N epochs
  beta_anneal_epochs: 20     # linear ramp to beta_target
  beta_target:       0.001   # true empirical bottleneck weight

  save_every: 5
  eval_every: 1

xgboost:
  n_estimators:  1000
  max_depth:     6
  learning_rate: 0.05
  subsample:     0.8
  colsample_bytree: 0.8
  tree_method:   "hist"      # required for reliable GPU tree building
  device:        "cuda"
  early_stopping_rounds: 50
```

---

## Running Tests

```bash
# All tests (no GPU, no downloads)
pytest classifier/tests/ -v

# Individual suites
pytest classifier/tests/test_losses.py -v
pytest classifier/tests/test_model.py  -v
pytest classifier/tests/test_data.py   -v
```

All tests use a temporary dummy dataset and mock encoders — no real images or model downloads required.

---

## Key Design Decisions — Classifier

| Decision | Rationale |
|---|---|
| No CE head during SegFormer training | SupCon alone forces better-separated clusters than CE+SupCon |
| `lambda_kl = 1.0` (pass-through) | β via BetaScheduler is the sole VIB knob — matches Alemi et al. 2017. The old `0.01` caused double-scaling: effective weight was `0.00001` |
| Ortho penalty on μ only, not full 769-D | Path A must remain free to capture subtle early-stage signals |
| VIB inference uses μ, not z | Deterministic embeddings → stable XGBoost decision boundaries |
| Collision-safe RETFound cache keys | `{split}_{class}_{stem}.pt` prevents train/val name collision when datasets share filenames |
| `torch.set_grad_enabled` as context manager | Guarantees global grad state is restored even if validation throws an exception |
| `(features * 0).sum()` for zero losses | Graph-connected zero — prevents `None` gradients when SupCon or Ortho have no valid pairs |
| RETFound cached before training | Never occupies VRAM during training; 1024-D domain context always available |
| XGBoost over MLP | Tabular supremacy, column sampling overfitting resistance, SHAP explainability |
| `tree_method=hist` | Only tree method that reliably uses GPU with `device=cuda` in XGBoost 2.0+ |
| `WeightedRandomSampler` + class-aware τ | Two complementary fixes for class imbalance at hardware and math level |
