# NetrAI — Retinal Intelligence System

> A two-stage retinal analysis system: a **diffusion model** for unsupervised lesion detection, feeding into a **dual-stream classifier** for disease identification (DR / Glaucoma / PM).
>
> Stage 1 (Diffusion) answers *where* — generating anomaly maps via SDEdit reconstruction error.  
> Stage 2 (Classifier) answers *what* — classifying disease from those maps + RETFound domain embeddings.

---

## Table of Contents

**Diffusion Model**
1. [Overview](#1-overview)
2. [Training Dataset](#2-training-dataset)
3. [Diffusion Architecture](#3-diffusion-architecture)
4. [Module Reference — Diffusion](#4-module-reference--diffusion)
5. [Key Design Decisions — Diffusion](#5-key-design-decisions--diffusion)
6. [Diffusion Training Pipeline](#6-diffusion-training-pipeline)
7. [Diffusion Inference Pipeline](#7-diffusion-inference-pipeline)
8. [Evaluation](#8-evaluation)
9. [Checkpoints](#9-checkpoints)
10. [Requirements](#10-requirements)
11. [Usage — Diffusion](#11-usage--diffusion)

**Classifier Pipeline**

12. [Classifier Overview](#12-classifier-overview)
13. [Classifier Architecture — Full Breakdown](#13-classifier-architecture--full-breakdown)
14. [Loss Function Deep Dive](#14-loss-function-deep-dive)
15. [Two-Phase Training Strategy](#15-two-phase-training-strategy)
16. [Why These Architectural Choices](#16-why-these-architectural-choices)
17. [Classifier Project Structure](#17-classifier-project-structure)
18. [Data Layout](#18-data-layout)
19. [Classifier Setup](#19-classifier-setup)
20. [Classifier Pipeline — Step by Step](#20-classifier-pipeline--step-by-step)
21. [Classifier Configuration Reference](#21-classifier-configuration-reference)
22. [Inference](#22-inference)
23. [Key Hyperparameter Decisions](#23-key-hyperparameter-decisions)
24. [Feature Dimensions Reference](#24-feature-dimensions-reference)
25. [Running Tests](#25-running-tests)

---

# Part I — Diffusion Model

---

## 1. Overview

NetrAI frames retinal anomaly detection as a **reconstruction problem**. A DDPM/DDIM diffusion UNet is trained exclusively on healthy retinal images. At inference, a test image is partially noised (SDEdit, `T_start < 1000`) and reconstructed. The residual between the original and reconstruction is the anomaly map — lesions the model never saw during training produce high residual signal.

**The forward process as an eraser:** When a diseased image (e.g., containing a haemorrhage) is fed into the DDPM forward process, the added noise mathematically destroys the disease signal. The UNet is then instructed (via the 768-d RETFound conditioning vector) to reconstruct a *healthy* version of that eye. Because the disease was erased by the noise and the UNet only knows how to draw healthy retinas, it simply fails to redraw the lesion. Subtracting the reconstruction from the original leaves only what the UNet could not account for — the disease.

The primary evaluation metric is **pixel-level AUROC on the DDR and iDRiD datasets** (757 annotated fundus images with MA/HE/EX/SE lesion masks), measured via vessel-suppressed residual maps (Frangi filter post-processing).

---

## 2. Training Dataset

The model is trained exclusively on **22,104 healthy fundus images** from 6 public sources:

| Source | Count | % of Total |
|--------|-------|-----------|
| EyePACS | 13,500 | 61.1% |
| DDR | 5,241 | 23.7% |
| APTOS | 1,625 | 7.4% |
| REFUGE2 | 1,080 | 4.9% |
| MESSIDOR-2 | 630 | 2.9% |
| STARE | 28 | 0.1% |
| **Total** | **22,104** | **100%** |

Only grade-0 (no diabetic retinopathy) images are used from graded datasets. The model never sees lesion-bearing images during training — anomaly detection at inference relies entirely on reconstruction error against this healthy prior.

---

## 3. Diffusion Architecture

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

### Why `block_out_channels: (128, 256, 512, 512)`?

The channel count doubles at each down-block to compensate for shrinking spatial resolution — when the image is half the size, you need twice the channels to preserve information capacity. The progression caps at **512 instead of doubling to 1024** purely for VRAM reasons: a 1024-channel layer would quadruple the parameter count and activation memory, crashing a 24GB GPU at `batch_size=6`. Repeating 512 twice gives the bottleneck an extra "thinking layer" at maximum depth without exceeding memory limits.

### Inside a `CrossAttnUpBlock2D`

Each Up Block is an assembly line that runs its ResNet+Attention pair **twice** before upsampling:

```
For i in [1, 2]:  ← two iterations
    hidden = ResNet(hidden, time_embedding)         # pixel refinement, clock-aware
    hidden = SpatialTransformer(hidden):            # read the 768-d instructions
        → Self-Attention  (each pixel attends to all other pixels in the tile)
        → Cross-Attention (each pixel attends to the 768-d RETFound vector)
hidden = Upsampler(hidden)                         # bilinear 2× spatial upscale
```

- **Self-Attention** ensures spatial coherence — blood vessels flow continuously across the tile.
- **Cross-Attention** injects the healthy-eye blueprint at every spatial position. This is the step where the 768-d conditioning vector *actively guides* what the UNet draws.
- Running the loop **twice** doubles the model's refinement capacity at that resolution without the memory cost of adding a third full block.
        |
        ▼
Reconstruction (512×512)
        |
        ▼
Residual Map ──► Retinal Ellipse Mask ──► Frangi Vessel Suppression
        |
        ▼
Anomaly Score (pixel-level, used for DDR AUROC)
        |
        ▼
     [Feed to Classifier — Part II]
```

---

## 4. Module Reference — Diffusion

### Cross-File Wiring & State

`	ext
train.py (Orchestrator)
  |-- data.py       (collate_fn passes paths alongside tensors for cache keys)
  |-- models.py     (LRU _vit_cache wiped at end of epoch to prevent stale gradients)
  |-- diffusion.py  (Provides TILE_VIEWS, alphas_cumprod, and make_retinal_mask)
  |-- evaluation.py (Vessel suppression via Frangi)
  +-- sweep.py      (Loads best_loss.pt and sweeps LCW/t_start combinations)
`

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
- ImageNet mean/std registered as **buffers** (no per-call tensor allocation).

#### `CachedConditioner`

A stateful wrapper around `RETFoundConditioner` that caches raw ViT features (not proj MLP outputs) in an `OrderedDict` LRU cache (max 500 entries).

- Cache key: `(image_path, "full")` or `(image_path, tile_id)`.
- LRU eviction via `move_to_end` / `popitem(last=False)`.
- `get_full_image_cond(img, path)` — conditioning for a full 512px image.
- `get_tile_conds_batched(img_512, tile_views)` — single batched ViT forward pass for all 9 tiles (eliminates 9 sequential ViT calls during inference).

---

### `diffusion.py`

All diffusion math, MultiDiffusion tiling, and the retinal mask.

**Noise functions:**
- `generate_simplex_noise` — `torch.randn`, shaped.
- `add_simplex_noise` — standard DDPM forward process: `x_t = √ā·x₀ + √(1-ā)·ε`.
- `simplex_ddim_step` — DDIM reverse step with optional stochasticity (`eta`). Clamps `pred_x0` to `[-1,1]`.

**MultiDiffusion tiling constants:**

| Constant | Value | Notes |
|----------|-------|-------|
| `TILE_SIZE` | 256 | UNet input size |
| `FULL_SIZE` | 512 | Full inference resolution |
| `TILE_STRIDE` | 128 | 50% overlap |
| `NUM_TRAIN_TILES` | 2 | Tiles sampled per image during training |
| `TILE_VIEWS` | 9 tiles | Precomputed `(h0,h1,w0,w1)` coordinates |

- `make_linear_weight` — 2D pyramid weight for tile blending. Peaks at 1.0 in tile center, tapers to near-zero at edges, ensuring smooth seams. Cached per device (`_LINEAR_WEIGHT_CACHE`).
- `make_retinal_mask` — Dynamic elliptical FOV mask: threshold < 0.05 as background → fit ellipse to FOV bounds → returns intersection of illuminated pixels and ellipse.
- `multidiffusion_reconstruct` — Core inference loop: partially noise → DDIM denoising with per-tile UNet calls, fusing outputs via pyramid-weighted accumulation, with `dynamic_lcw` blending tile-specific and global conditioning.
- `multiscale_residual` — Runs separate reconstructions at 256px, 128px, 64px → upscale all to 512px → element-wise **maximum** across scales (no weighted sum — avoids starving coarse scales).

---

### `data.py`

- `make_transform` — Builds torchvision transform pipeline. BILINEAR interpolation throughout (avoids bicubic ringing artifacts in L1 residuals).
- Train augmentations: RandomResizedCrop (scale 0.8–1.0), RandomHorizontalFlip, RandomVerticalFlip, RandomRotation ±15°, ColorJitter (mild).
- Val: CenterCrop only.

`RetinaDataset` — Accepts three source types: `.csv` (with `path`/`image` column), `.txt` (newline-separated paths), or directory (recursive glob).

Features:
- Optional `bad_files_txt` to pre-filter known corrupted files.
- Retry loop (up to 10 attempts) to skip unreadable images at runtime without crashing.
- Returns `(tensor, path)` tuples — paths are used as cache keys in `CachedConditioner`.
- `collate_fn` — Stacks tensors, keeps paths as a Python list (not tensor).

---

### `losses.py`

`snr_weighted_loss` (Min-SNR strategy, γ=2.0):
```
SNR(t) = ā_t / (1 - ā_t)
weight(t) = clamp(SNR, max=γ) / (SNR + ε)
loss = mean(weight × MSE(pred_noise, noise))
```
- γ=2.0 is intentionally aggressive (standard is 5.0) to synergize with the LCW tile fusion — stops the model from memorizing rigid 256px tile edges.

`l1_focal_frequency_loss` — Two-component hybrid:
1. **Spatial L1 (The Anchor):** `mean(|pred_x0 - x0|)` inside retinal mask.
2. **Focal Frequency Loss (The Sniper):** FFT on masked inputs → amplitude spectrum difference → self-weighted by `freq_diff.detach()` (harder frequencies get higher weight).  
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

`compute_ddr_metrics` — Full DDR evaluation: glob all images, combine 4-lesion-type masks (MA/HE/EX/SE), run `multidiffusion_reconstruct_full` → `postprocess_residual`, compute pixel-level AUROC/AP/Dice. Supports time-budget and count-budget cutoffs.

---

### `visualization.py`

`save_visualizations` — Per-epoch 5-panel reconstruction grid:

| Column | Content |
|--------|---------|
| Original | Raw input |
| Recon (masked) | Reconstruction × retinal mask |
| Signed Diff | `(orig - recon)` in RdBu colormap, ±0.15 |
| MultiScale | Multi-scale ensemble residual (hot) |
| Clean Residual | Vessel-suppressed anomaly map (hot, normalized) |

`save_anomaly_maps` — Per-image dark-theme 5-panel figure with overlay.

`save_metrics_dashboard` — 4-panel matplotlib figure: AUROC/AP (val), SSIM, PSNR, DDR AUROC/Dice history.

---

### `sweep.py`

`run_sweep` — Grid search over `T_start × DDIM_steps × max_lcw`. Saves per-combo reconstructions and vessel-suppressed residual heatmaps. Writes `sweep_metrics.csv` with SSIM, PSNR, residual mean/max, and wall-clock seconds per combo. Generates panel plots per image per LCW value.

Activated via `sweep.enabled: true` in config — runs instead of training when set.

---

### `utils.py`

- `strip_compile_prefix` — Strips `_orig_mod.` from state dicts saved under `torch.compile`.
- `repair_csv_header` — Rewrites a CSV header in-place if the schema on disk differs from current expected columns.
- `append_csv_row` — Safe CSV append with exception handling.
- `load_loss_history` — Parses `loss.csv` with schema tolerance.
- `_TeeStream` / `setup_terminal_logging` — Redirects stdout and stderr to both console and a log file simultaneously.
- `save_lcw_curve` — Plots the LCW schedule as experienced during training.

---

### `config.yaml` (Diffusion)

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
  idrid_masks_dir:  # iDRiD evaluation mask directory
  retfound_weights: # Path to RETFound_cfp_weights.pth

sweep:
  enabled:    false
  t_starts:   [200, 250, 300, 350]
  ddim_steps: [50]
  lcw_values: [0.4]

training:
  crop_size:         512
  epochs:            20
  batch_size:        6
  accum_steps:       6     # Effective batch = batch_size × accum_steps = 36
  warmup_epochs:     3
  lr_unet:           5e-6
  lr_conditioner:    1e-5
  snr_gamma:         2.0

diffusion:
  ddim_steps:       50
  ddim_t_start:     300   # Partial noising depth for SDEdit
  max_train_lcw:    0.4   # Peak local conditioning weight

eval:
  vis_every:         1
  eval_every:        10
  ddr_eval_every:    10
  ddr_max_images:    ~    # null = all images
  ddr_max_seconds:   ~    # null = no time cap
```

---

## 5. Key Design Decisions — Diffusion

| Decision | Rationale |
|----------|-----------|
| Train on 256px tiles, infer on 512px via MultiDiffusion | UNet fits in VRAM at 256px; MultiDiffusion fuses overlapping tiles for seamless 512px output |
| `T_start=300` not `T_start=1000` | At t=1000 the image is pure noise — coarse retinal structure (optic disc position, vessel layout) is completely destroyed, forcing the UNet to hallucinate anatomy from scratch. At t=300, the noise is strong enough to erase small lesions (MA, HE) but the UNet can still see the global eye shape through the noise, producing a reconstruction that preserves anatomy while removing pathology. |
| `block_out_channels: (128, 256, 512, 512)` — cap at 512 | Doubling to 1024 at the bottleneck would quadruple parameter + activation memory, exceeding 24GB VRAM at the target batch size. Repeating 512 instead gives an extra deep reasoning layer at maximum compression depth with no additional memory cost. |
| Frozen RETFound ViT-Large | RETFound captures fundus-specific anatomy; fine-tuning would destroy the generic healthy-retina prior |
| Cache raw ViT features, not proj MLP outputs | Proj MLP trains, so caching its output would cause stale gradients across iterations |
| SNR-γ=2.0 (aggressive) | Paired with LCW: prevents the model from overfitting tile boundary micro-textures at low noise |
| Frangi on both input green channel AND residual | Green channel catches vascular anatomy; residual Frangi catches vessel-shaped reconstruction artifacts |
| Element-wise max for multi-scale ensemble | Weighted sum would cap small-scale lesion signal; max lets each scale compete at full confidence |
| LCW rises with cosine schedule over full training | Prevents tile-specific conditioning from overwhelming global structure conditioning before the UNet learns coarse anatomy |
| BILINEAR interpolation throughout data pipeline | Avoids bicubic ringing artifacts that contaminate the L1 residual anomaly map |
| xformers disabled when `torch.compile` is active | Compile + xformers causes attention processor identity checks to thrash the dynamo cache |

---

## 6. Diffusion Training Pipeline

### Tensor Flow & LCW (Line-by-Line Highlights)

- **Data:** (B, 3, 512, 512) pixels in [-1,1].
- **Batched ViT (O1):** 2 random (256, 256) tiles are extracted, batched together to (B*2, 3, 256, 256), and passed through RETFound once (	rain.py:402-406) to save 50% compute.
- **Mask:** Computed once per image (B, 1, 512, 512), sliced per tile.
- **Conditioning Blend:** lended = LCW * local_cond + (1-LCW) * global_cond.
- **LCW Warmup:** The LCW dynamically rises via a cosine schedule (	rain.py:421-423) across the training run so the UNet learns global anatomy before focusing on tile-level detail.

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
|   ├── diffusion_loss (0.6 × SNR-weighted + 0.4 × FFL hybrid)
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

## 7. Diffusion Inference Pipeline

```
Input: 512×512 fundus image

1. CachedConditioner.get_full_image_cond()    →  global_cond (1,1,768)
2. CachedConditioner.get_tile_conds_batched() →  9× local_cond (1,1,768)
3. add_simplex_noise(img, T_start=300)        →  x_T
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

The anomaly map is then saved and used as input to the Classifier pipeline.

---

## 8. Evaluation

### DDR Dataset
- 757 labeled fundus images with pixel-level annotations: **MA** (Microaneurysms), **HE** (Hemorrhages), **EX** (Hard Exudates), **SE** (Soft Exudates).
- Masks combined into a single binary map (logical OR across lesion types).
- Pixel-level metrics: **AUROC** (primary), **AP**, **Dice** (best threshold via sweep).
- Required structure: `ddr_masks_dir/{MA,HE,EX,SE}/{stem}.tif`

### iDRiD Dataset
- 81 labeled fundus images with the same 4 lesion type annotations.
- Same evaluation protocol as DDR.
- Required structure: `idrid_masks_dir/{MA,HE,EX,SE}/{stem}.tif`

---

## 9. Checkpoints

All checkpoints saved to `checkpoint_dir`:

| File | Contents |
|------|---------|
| `last.pt` | Full training state: model, conditioner_proj, optimizer, scaler, scheduler, epoch, best metrics |
| `best_loss.pt` | Snapshot at lowest validation loss |
| `best_auroc.pt` | Snapshot at highest DDR AUROC |
| `loss.csv` | Per-epoch: train_loss, val_loss, snr, ms, val_snr, val_ms, lr, lcw |
| `metrics.csv` | Per-eval: SSIM, PSNR, pixel_auroc, pixel_ap |
| `ddr_metrics.csv` | Per-DDR-eval: ddr_auroc, ddr_ap, ddr_dice, ddr_thresh, n_images |
| `train_terminal.log` | Full stdout/stderr mirror |
| `lcw_curve.png` | LCW vs epoch progress plot |
| `metrics_dashboard.png` | 4-panel metrics history figure |
| `recon_epoch_XXXX.png` | Per-epoch reconstruction panels |
| `anomaly_maps/` | Per-image dark-theme overlay panels |

---

## 10. Requirements

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
xgboost            # Phase 2 classifier training
shap               # Feature importance analysis
```

---

## 11. Usage — Diffusion

```bash
# Train
python -m diffusion.train --config diffusion/config.yaml

# With persistent logging in a detached session
python -m diffusion.train --config diffusion/config.yaml 2>&1 | tee -a checkpoints/train.log

# Sweep mode (set sweep.enabled: true in config)
python -m diffusion.train --config diffusion/config.yaml
```

**Resume:** Training auto-resumes from `last.pt` if it exists in `checkpoint_dir`. No flag needed.

**Warm-start from 256px checkpoint:** Set `paths.pretrained_256` to your 256px `last.pt`. LR is automatically scaled to 30% base and warmup reduced to 1 epoch.

---
---

# Part II — Classifier Pipeline

---

## 12. Classifier Overview

The `classifier/` module takes the diffusion model's **clean residual anomaly maps** and classifies retinal images into three disease categories: **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Pathological Myopia (PM)**.

The classifier is a two-phase pipeline:
- **Phase 1** — End-to-end differentiable training: dual-stream encoder (MIT-B3 SegFormer + pre-cached RETFound) with expert branches, dual VIB bottleneck, and BCEWithLogits losses.
- **Phase 2** — Static feature extraction → 3 independent binary XGBoost classifiers.

Labels are **multi-label** (BCEWithLogitsLoss, not Softmax). DR, Glaucoma, and PM are not mutually exclusive — a patient can have all three simultaneously.

---

## 13. Classifier Architecture — Full Breakdown

```
================================================================================
                PHASE 1: END-TO-END DIFFERENTIABLE TRAINING
================================================================================

  [ 6-Channel Stack ]                        [ Pre-Cached RETFound Embedding ]
(3ch RGB + 3ch Clean Residual)                        (1024-D, from disk)
         |                                                      |
         v                                                      v
[ MIT-B3 SEGFORMER ]                                   (loaded by DataLoader
  (6ch input, trains)                                  — never runs live)
         |
  hidden[0]        hidden[2]       hidden[3]
 (128×128×64)    (32×32×320)     (16×16×512)
         |              |               |
    [ DR HEAD ]   [ GLAUC HEAD ]   [ PM HEAD ]
    1x1 → 3x3     1x1 → 3x3      SE-Block (no spatial conv)
    CBAM → Pool    CBAM → Pool    GAP → MLP
         |              |               |
       256-D           256-D          256-D
         └──────────────┴───────────────┘
                        |
                     768-D
                        |
                    [ VIB 1 ]
                768 → 256 (hidden) → 128
                        |
            128-D (z1, μ1, log_σ²1)
             ┌──────────┴──────────┐
             |                     |
  [ Aux Linear Classifier ]        |
     (128 → 3, BCEWithLogits)      |
             |                     |      [ VIB 2 ]
          (L_aux)                  |  1024 → 256 → 128
                                   |      |
                                   |  128-D (z2, μ2, log_σ²2)
                                   |      |
                           z1 (128) ⊕ z2 (128) = 256-D fused
                                       |
                           [ Main Linear Classifier ]
                              (256 → 3, BCEWithLogits)
                                       |
                                    (L_main)

    L_total = L_main + λ_aux · L_aux + β · (KL₁ + KL₂)

================================================================================
                PHASE 2: FEATURE EXTRACTION & META-CLASSIFICATION
================================================================================

  1. FREEZE all weights.  2. SET eval mode (VIBs → deterministic μ, ε=0)
  3. DROP Aux + Main classifiers (training scaffolding, not used in inference)
  4. FORWARD every image → extract 256-D z_fused
  5. SAVE to features/train_features.npy, features/val_features.npy
  6. TRAIN 3 independent binary XGBoost classifiers

        [ 256-D Extracted Vector ]
                    |
        ┌───────────┼───────────┐
        v           v           v
 [XGBoost_DR]  [XGBoost_Glauc]  [XGBoost_PM]
        |           |           |
     P(DR)       P(Glauc)     P(PM)
  (independent sigmoid — NOT softmax — does NOT sum to 1)
```

### Stage 1 — 6-Channel Input Construction

```
RGB Image (3ch, ImageNet-normalised)    +    Clean Residual (3ch, [0,1] replicated)
                                six_ch = cat([image, residual], dim=0)  →  (6, 512, 512)
```

The residual channels 3-5 are **zero-initialised** in the patch_embed projection — the model starts with exactly the pretrained RGB behavior and learns to use the residual channels from scratch. This is safer than random init (which corrupts pretrained weights) or copy init (which gives the residual the same interpretation as RGB, which is wrong).

### Stage 2 — MIT-B3 SegFormer (Stream A)

MIT-B3 hierarchical Mix Transformer with 4 stages. The first patch embedding projection is surgically adapted from `Conv2d(3, 64)` to `Conv2d(6, 64)` with the zero-init strategy above. All other weights stay pretrained.

**Stage output dimensions at 512×512 input:**

| Stage | `hidden_states[]` | Spatial | Channels | Used by |
|-------|-------------------|---------|----------|---------|
| 1 | `[0]` | 128×128 | 64 | DR Head |
| 2 | `[1]` | 64×64 | 128 | — |
| 3 | `[2]` | 32×32 | 320 | Glauc Head |
| 4 | `[3]` | 16×16 | 512 | PM Head |

### Stage 3 — Expert Branches

#### DR Head (Scale `hidden[0]`, 128×128×64)
```
Conv2d(64 → 128, kernel=1) → Conv2d(128 → 128, kernel=3) → BN + GELU
→ CBAM (channel + spatial attention)
→ AdaptiveAvgPool2d(1) → Flatten
→ Linear(128 → 256) + GELU + Dropout(0.3)
Output: (B, 256)
```
**Why Scale 0?** DR is microaneurysms, dot haemorrhages, hard exudates — tiny, spatially precise features. Scale 0 preserves the highest spatial resolution (128px). **Why CBAM?** It simultaneously identifies which channels encode lesion types AND where spatially they appear.

#### Glauc Head (Scale `hidden[2]`, 32×32×320)
```
Conv2d(320 → 256, kernel=1) → Conv2d(256 → 256, kernel=3) → BN + GELU
→ CBAM (channel + spatial attention)
→ AdaptiveAvgPool2d(1) → Flatten
→ Linear(256 → 256) + GELU + Dropout(0.3)
Output: (B, 256)
```
**Why Scale 2?** Glaucoma signature is optic disc CDR enlargement and rim thinning. The optic disc occupies ~1/8 of the image. At Scale 2 (32×32), the disc maps to 4-6 pixels — enough for disc/cup boundary assessment without high-resolution vessel noise.

#### PM Head (Scale `hidden[3]`, 16×16×512)
```
SE-Block (channel-only attention):
    AdaptiveAvgPool2d(1) → Flatten → Linear(512→32) → ReLU → Linear(32→512) → Sigmoid
    Output = Input × scale   [pure channel re-weighting, NO spatial conv]
→ AdaptiveAvgPool2d(1) → Flatten
→ Linear(512 → 256) + GELU + Dropout(0.3)
Output: (B, 256)
```
**Why NO spatial convolutions?** SegFormer Scale 3 uses self-attention — every spatial position already contains context from the whole image. Applying a spatial conv to these features is redundant. SE-Block asks *"which channels fire for global deformation?"* — the correct question for a global structural disease like PM (axial elongation, posterior staphyloma).

### Stage 4 — Dual VIB

```
fused_768 = cat([dr_feat, glauc_feat, pm_feat])  →  (B, 768)

VIB 1  →  768 → Linear(256) + GELU → μ₁ (128), log_σ²₁ (128)
VIB 2  →  1024 → Linear(256) + GELU → μ₂ (128), log_σ²₂ (128)

Training:   z_i = μ_i + exp(0.5 × log_σ²_i) ⊙ ε,   ε ~ N(0, I)
Inference:  z_i = μ_i   (deterministic — XGBoost requires stable split thresholds)

z_fused = cat([z₁, z₂])  →  (B, 256)
```

**Why two VIBs?** A single VIB on the 1792-D concatenation would allow the optimizer to collapse z₁ to N(0,I) and free-ride entirely on the frozen RETFound signal (z₂). Two separate VIBs force each stream to independently justify its own compression.

### Stage 5 — Temporary Phase-1 Classifiers

```python
aux_classifier  = nn.Linear(128, 3)   # on z₁ alone → L_aux
main_classifier = nn.Linear(256, 3)   # on z_fused  → L_main
```

**Discarded after Phase 1.** Without `aux_classifier`, VIB1 can minimize `L_main` perfectly by routing through z₂ and setting z₁ → N(0,I). `L_aux` creates a gradient path that depends **only** on z₁ — VIB1 cannot escape it.

### Stage 6 — XGBoost Meta-Classification

Three independent binary classifiers on the static 256-D vectors:

| Model | Objective | Eval Metric | Saves to |
|-------|-----------|------------|---------|
| `xgb_DR.pkl` | `binary:logistic` | AUC | `xgboost/xgb_DR.pkl` |
| `xgb_Glaucoma.pkl` | `binary:logistic` | AUC | `xgboost/xgb_Glaucoma.pkl` |
| `xgb_PM.pkl` | `binary:logistic` | AUC | `xgboost/xgb_PM.pkl` |

**Why 3 binary classifiers?** `multi:softmax` forces probabilities to sum to 1, mathematically suppressing comorbidities. Three independent sigmoid outputs produce separate confidences that can all be high simultaneously.

---

## 14. Loss Function Deep Dive

```
L_total = L_main + λ_aux · L_aux + β · λ_kl · (KL₁ + KL₂)
```

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| `L_main` | `BCEWithLogitsLoss(main_classifier(z_fused), label_vec)` | 1.0 | Primary disease signal |
| `L_aux` | `BCEWithLogitsLoss(aux_classifier(z₁), label_vec)` | `λ_aux = 0.4` | Forces VIB1 to encode disease independently |
| `KL₁` | `-½ Σ(1 + log_σ²₁ - μ₁² - σ₁²)` | `β × λ_kl` | Compresses custom head stream |
| `KL₂` | `-½ Σ(1 + log_σ²₂ - μ₂² - σ₂²)` | `β × λ_kl` | Compresses RETFound stream |

`label_vec` is a `(B, 3)` float multi-hot vector: `[1,0,0]` for DR-only, `[1,1,0]` for DR+Glaucoma, etc.

### β-Annealing Schedule

```
Epochs [0, 10):    β = 0.0      ← classifiers establish clusters first
Epochs [10, 30):   β = 0 → 0.001  (linear ramp)
Epochs [30, end]:  β = 0.001   ← gentle constant compression
```

Starting at β=0 is critical. At epoch 0, VIBs produce garbage. High β would collapse both VIBs to N(0,I) immediately (zero KL = cheap). The classifiers get noise and learn nothing. Starting at β=0 lets `L_main` and `L_aux` establish disease-separating clusters first, then β compresses those meaningful clusters.

---

## 15. Two-Phase Training Strategy

### Phase 1 — End-to-End Differentiable

| Component | Mode | LR |
|-----------|------|----|
| MIT-B3 backbone (pretrained) | Trains | `lr × 0.1 = 1e-5` |
| DR / Glauc / PM Expert Heads | Trains | `lr = 1e-4` |
| VIB 1 + VIB 2 | Trains | `lr = 1e-4` |
| Aux Classifier + Main Classifier | Trains | `lr = 1e-4` |
| RETFound ViT-Large | **Frozen** | 0 |

10× lower LR for the backbone preserves its pretrained ImageNet representations while the new expert heads learn from scratch at normal speed.

### Phase 2 — Feature Extraction → XGBoost

1. Load best Phase 1 checkpoint (`best.pt`)
2. `model.eval()` → VIBs deterministic (`z = μ`, `ε = 0`)
3. Freeze all weights
4. Forward every image → collect `z_fused` (256-D)
5. Save as NumPy arrays to `features/`
6. Train 3 binary XGBoost classifiers

**Why ε=0 at extraction?** XGBoost builds decision trees that find consistent split thresholds. If z is stochastic, the same image produces a slightly different 256-D vector each time — trees cannot find stable splits. Using μ gives XGBoost a deterministic, reproducible tabular input.

---

## 16. Why These Architectural Choices

| Choice | Alternative | Why This Was Chosen |
|--------|------------|---------------------|
| 6-channel input (RGB + residual) | Late-fusion gate | Bakes diffusion prior into backbone's earliest feature computation. Gate was an add-on; 6ch makes the residual a first-class input. |
| 3 expert heads on different scales | Single decode head, all scales fused | Each disease lives at a different spatial frequency. One head can't optimise all three simultaneously. |
| SE-Block (no spatial conv) for PM | 3×3 conv + spatial attention | SegFormer Scale 3 features already contain global self-attention. Spatial conv would be redundant and add wrong inductive bias. |
| Dual VIB | Single VIB on 1792-D concat | Single VIB allows optimizer to free-ride on frozen RETFound. Dual VIB forces independent compression per stream. |
| Aux classifier on z₁ | Gradient scaling / warmup tricks | Aux loss is path-of-no-escape for VIB1 — it must encode disease signal or its own loss explodes. Clean and principled. |
| 3 binary XGBoost | 1 multiclass XGBoost (softmax) | Diseases are not mutually exclusive. Softmax suppresses valid comorbidity signals. |
| Pre-cached RETFound | RETFound live during training | Saves ~1.2GB VRAM and 3-5× training time. RETFound's weights never change; recomputing every epoch is pure waste. |
| BCEWithLogitsLoss | SupCon + Ortho | BCE provides direct, interpretable disease prediction. For multi-label setup (comorbidities), BCE is the natural choice. |

---

## 17. Classifier Project Structure

```
classifier/
├── config.yaml          ← Master hyperparameter file
├── model.py             ← CBAM, SEBlock, DRHead, GlaucHead, PMHead,
│                           VIB, NetrAiEncoder (dual VIB + temp classifiers)
├── losses.py            ← BCEWithLogitsLoss (main + aux) + dual KL + BetaScheduler
├── data.py              ← RetinalDataset (6ch stack + RETFound cache + multi-hot labels)
│                           + build_dataloader (WeightedRandomSampler)
├── retfound.py          ← RETFoundExtractor + precompute cache + cache I/O
│                           (UNCHANGED — cache format fully compatible)
├── train.py             ← Phase 1 Trainer class (AMP, dual LR groups, checkpoint)
├── extract.py           ← Phase 2: frozen 256-D extraction to .npy
├── xgboost_clf.py       ← BinaryXGBoost × 3 + NetrAiXGBoost wrapper + SHAP
├── inference.py         ← Single-image end-to-end pipeline
├── utils.py             ← Logging, checkpointing, metrics, LR scheduler
├── requirements.txt
├── __init__.py
├── __main__.py          ← CLI dispatcher
└── tests/
    ├── conftest.py      ← Shared pytest fixtures (temp dataset, no GPU needed)
    ├── test_model.py    ← Shape + gradient contracts
    ├── test_losses.py   ← Loss function unit tests
    └── test_data.py     ← Dataset + DataLoader tests
```

---

## 18. Data Layout

```
data/
├── classifier/
│   ├── train/
│   │   ├── DR/          ← .jpg / .png retina images
│   │   ├── Glaucoma/
│   │   └── PM/
│   └── val/
│       ├── DR/
│       ├── Glaucoma/
│       └── PM/
├── anomaly_maps/
│   ├── <image_stem>_anomaly.png   ← preferred naming
│   └── <image_stem>.png           ← fallback naming
└── retfound_cache/                ← generated by cache-retfound step
    ├── train_DR_image_001.pt
    ├── train_Glaucoma_scan_042.pt
    └── ...
```

> **Missing anomaly map**: residual channels default to zeros. Model degrades gracefully.  
> **Missing RETFound cache**: VIB2 receives zeros. Aux classifier on z₁ still forces VIB1 to learn. Model degrades gracefully.

**Class balance:** `WeightedRandomSampler` enforces 1:1:1 (DR:Glaucoma:PM) per batch during training. Each sample weight = `total / (n_classes × class_count)`. Validation uses unbalanced sequential iteration to evaluate on the true class distribution.

---

## 19. Classifier Setup

```bash
# Install dependencies
pip install -r classifier/requirements.txt

# Verify test suite passes (no GPU, no downloads required)
pytest classifier/tests/ -v
```

### VRAM Requirements

| Component | fp32 | bf16 (AMP) |
|-----------|------|-----------|
| MIT-B3 SegFormer | ~180 MB | ~90 MB |
| Expert heads + VIBs | ~50 MB | ~25 MB |
| Activations (batch=8, 512px) | ~8-10 GB | ~4-5 GB |
| RETFound (Phase 1) | **0 MB** (cached) | **0 MB** |
| **Total (Phase 1, batch=8)** | **~10-12 GB** | **~5-6 GB** |

With 20GB VRAM, batch size 8-12 is safe. AMP is enabled by default.

---

## 20. Classifier Pipeline — Step by Step

### Step 0 — Configure Paths

Edit `classifier/config.yaml`:
```yaml
paths:
  data_dir:           "data/classifier"
  anomaly_maps_dir:   "data/anomaly_maps"
  checkpoint_dir:     "checkpoints/classifier"
  features_dir:       "features"
  retfound_cache_dir: "retfound_cache"
  retfound_weights:   "path/to/RETFound_cfp_weights.pth"
```

Download RETFound weights from [RETFound repository](https://github.com/rmaphoh/RETFound_MAE).

---

### Step 1 — Cache RETFound Embeddings *(one-time, ~minutes)*

```bash
python -m classifier cache-retfound --config classifier/config.yaml

# Force recompute
python -m classifier cache-retfound --config classifier/config.yaml --overwrite
```

Runs every image through frozen RETFound-Large. Saves `<split>_<class>_<stem>.pt` per image. RETFound is fully unloaded from VRAM afterwards and never used during training.

---

### Step 2 — Phase 1: Train the SegFormer Encoder

```bash
python -m classifier train --config classifier/config.yaml

# Resume from checkpoint
python -m classifier train --config classifier/config.yaml \
                           --resume checkpoints/classifier/epoch_0030.pt
```

Trains for `training.epochs` epochs (default 60). Best checkpoint by validation loss saved as `best.pt`.

**What to monitor:**

| Metric | Healthy behaviour | Red flag |
|--------|-----------------|----------|
| `l_main` | Decreasing | Plateau early → check data loading |
| `l_aux` | Decreasing | Stays high → VIB1 not learning; check retfound cache |
| `l_kl1`, `l_kl2` | ~0 until epoch 10, then rises slightly | Exploding → reduce `beta_target` |
| `val_loss` | Tracking train_loss with small gap | Diverging → increase `dropout` |
| `β` | 0 for first 10 epochs, then linear ramp | — |

---

### Step 3 — Phase 2: Extract Feature Vectors

```bash
python -m classifier extract --config classifier/config.yaml

# Use specific checkpoint
python -m classifier extract --config classifier/config.yaml \
                             --checkpoint checkpoints/classifier/epoch_0045.pt
```

Produces:
```
features/
├── train_features.npy    (N_train, 256)  float32  — 128 VIB1 ⊕ 128 VIB2
├── train_labels.npy      (N_train, 3)    float32  multi-hot
├── train_labels_int.npy  (N_train,)      int32    class index
├── train_stems.json      list of image stems
├── val_features.npy      (N_val,   256)
├── val_labels.npy        (N_val,   3)
├── val_labels_int.npy    (N_val,)
└── val_stems.json
```

---

### Step 4 — Phase 2: Train XGBoost Classifiers

```bash
python -m classifier xgboost --config classifier/config.yaml

# With SHAP feature importance
python -m classifier xgboost --config classifier/config.yaml --shap
```

Trains 3 independent binary classifiers. Saves:
```
checkpoints/classifier/
├── xgboost/
│   ├── xgb_DR.pkl
│   ├── xgb_Glaucoma.pkl
│   └── xgb_PM.pkl
├── xgboost_results.json     ← per-disease AUC, AP, accuracy
└── shap/
    ├── shap_DR.json
    ├── shap_Glaucoma.json
    └── shap_PM.json
```

**SHAP feature name mapping:**

| Dimension slice | Name prefix | Source |
|----------------|-------------|--------|
| `[0:128]` | `vib1_z_000` … `vib1_z_127` | Custom SegFormer heads (DR+Glauc+PM) |
| `[128:256]` | `vib2_z_000` … `vib2_z_127` | RETFound stream |

---

### Step 5 — Inference

See [Section 22](#22-inference).

---

## 21. Classifier Configuration Reference

```yaml
# classifier/config.yaml — complete annotated reference

paths:
  data_dir:           "data/classifier"
  anomaly_maps_dir:   "data/anomaly_maps"
  checkpoint_dir:     "checkpoints/classifier"
  features_dir:       "features"
  retfound_cache_dir: "retfound_cache"
  retfound_weights:   null          # path to .pth or null (HF fallback)

data:
  image_size: 512
  mean: [0.485, 0.456, 0.406]       # ImageNet — RGB channels only
  std:  [0.229, 0.224, 0.225]       # residual channels kept in [0,1]
  num_workers: 4
  pin_memory: true

model:
  backbone:     "nvidia/mit-b3"
  head_out_dim: 256                 # each expert head output dim
                                    # 3 × 256 = 768 → VIB1 input
  vib_hidden:   256                 # VIB pre-projection hidden dim
  vib_out_dim:  128                 # z₁ and z₂ each; z_fused = 256
  dropout:      0.3

training:
  epochs:        60
  batch_size:    8                  # safe on 20GB VRAM with AMP
  lr:            1.0e-4             # head/VIB LR; backbone gets lr × 0.1
  weight_decay:  1.0e-4
  warmup_epochs: 5                  # LR scheduler linear warmup
  grad_clip:     1.0
  amp:           true               # bfloat16 on Ampere, float16 otherwise

  lambda_aux:    0.4                # L_aux weight (anti-free-riding)
  lambda_kl:     1.0                # KL pass-through (β is the main knob)

  beta_warmup_epochs:  10           # β = 0 for first N epochs
  beta_anneal_epochs:  20           # linear 0 → beta_target
  beta_target:         0.001

  save_every: 5
  eval_every: 1

retfound:
  embed_dim:        1024
  image_size:       224
  cache_batch_size: 32

xgboost:
  n_estimators:          500
  max_depth:             6
  learning_rate:         0.05
  subsample:             0.8
  colsample_bytree:      0.8
  min_child_weight:      3
  gamma:                 0.1
  reg_alpha:             0.1        # L1 regularisation
  reg_lambda:            1.0        # L2 regularisation
  early_stopping_rounds: 50
  seed:                  42
  device:                "cuda"
  tree_method:           "hist"     # required for GPU tree building

classes:
  names: ["DR", "Glaucoma", "PM"]
```

---

## 22. Inference

### CLI — Single Image

```bash
python -m classifier infer \
    --config  classifier/config.yaml \
    --image   patient_001.jpg \
    --anomaly patient_001_anomaly.png
```

Output:
```
════════════════════════════════════════════════════
  DIAGNOSIS:  DR
  PROBABILITIES (independent per disease):
    DR          87.3%  ████████████████████████████████████
    Glaucoma    12.1%  ████
    PM           4.8%  █
  Vector dim: (256,)
════════════════════════════════════════════════════
```

> Probabilities are **independent** — they do NOT sum to 1. DR=87% and Glaucoma=12% simultaneously is valid (comorbidity).

### Without Anomaly Map

```bash
python -m classifier infer --config classifier/config.yaml --image patient_001.jpg
```
Residual channels default to zeros.

### On-the-Fly RETFound (no cache)

```bash
python -m classifier infer --config classifier/config.yaml \
                           --image patient_001.jpg \
                           --load-retfound
```

### Python API

```python
from classifier import NetrAiInference
from classifier.utils import load_config

cfg    = load_config("classifier/config.yaml")
engine = NetrAiInference(cfg)

result = engine.predict("patient.jpg", "patient_anomaly.png")
# {
#     "diagnosis":     "DR",
#     "probabilities": {"DR": 0.873, "Glaucoma": 0.121, "PM": 0.048},
#     "vector_256":    np.ndarray (256,)
# }

# Batch inference
results = engine.predict_batch(
    image_paths   = ["img1.jpg", "img2.jpg"],
    anomaly_paths = ["img1_anom.png", "img2_anom.png"],
)
```

---

## 23. Key Hyperparameter Decisions

| Hyperparameter | Value | Why |
|---------------|-------|-----|
| `batch_size` | 8 | Safe on 20GB VRAM with AMP. |
| `lr` (heads) | 1e-4 | Standard for new layers on pretrained backbone. |
| `lr` (backbone) | 1e-5 | 10× lower to preserve pretrained MIT-B3. |
| `head_out_dim` | 256 | Wide enough for complex features, narrow enough to avoid redundancy per head. |
| `vib_out_dim` | 128 | Compresses 768-D (VIB1) by 6× and 1024-D (VIB2) by 8×. z_fused = 256. |
| `lambda_aux` | 0.4 | 40% weight — forces VIB1 to learn without dominating L_main. |
| `beta_target` | 0.001 | Mild bottleneck. Higher β causes posterior collapse. 0.001 compresses without killing. |
| `beta_warmup` | 10 epochs | Classifiers need ~10 epochs to establish initial clusters before compression. |
| `dropout` | 0.3 | Applied in expert head FC layers. Appropriate for medical imaging with small datasets. |
| `n_estimators` | 500 | Sufficient for 256-D tabular input with early stopping at 50 rounds. |
| `colsample_bytree` | 0.8 | 80% feature subsampling — key regulariser for 256 medical features. |

---

## 24. Feature Dimensions Reference

| Tensor | Source | Shape |
|--------|--------|-------|
| `six_ch[:, 0:3]` | RGB image (ImageNet-normalised) | (B, 3, 512, 512) |
| `six_ch[:, 3:6]` | Clean residual ×3 (diffusion model output) | (B, 3, 512, 512) |
| `hidden[0]` | MIT-B3 Stage 1 | (B, 64, 128, 128) |
| `hidden[2]` | MIT-B3 Stage 3 | (B, 320, 32, 32) |
| `hidden[3]` | MIT-B3 Stage 4 | (B, 512, 16, 16) |
| `dr_feat` | DRHead output | (B, 256) |
| `glauc_feat` | GlaucHead output | (B, 256) |
| `pm_feat` | PMHead output | (B, 256) |
| `fused_768` | `cat([dr, glauc, pm])` | (B, 768) |
| `retfound_emb` | Pre-cached RETFound [CLS] | (B, 1024) |
| `z₁` | VIB1 sample | (B, 128) |
| `z₂` | VIB2 sample | (B, 128) |
| `z_fused` | `cat([z₁, z₂])` | (B, 256) |
| `features.npy[:, 0:128]` | VIB1 μ — custom heads stream | (N, 128) |
| `features.npy[:, 128:256]` | VIB2 μ — RETFound stream | (N, 128) |

---

## 25. Running Tests

```bash
# Full test suite (no GPU, no model downloads required)
pytest classifier/tests/ -v

# Individual modules
pytest classifier/tests/test_model.py  -v
pytest classifier/tests/test_losses.py -v
pytest classifier/tests/test_data.py   -v
```

All tests use a temporary dummy dataset and toy model shapes. No real images, no pretrained downloads, no CUDA required.
