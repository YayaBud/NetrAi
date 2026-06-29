# NetrAD: Domain-Agnostic Retinal Anomaly Detection via Multi-Scale Diffusion

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

NetrAD frames retinal anomaly detection as a **reconstruction problem**. A DDPM/DDIM diffusion UNet is trained exclusively on healthy retinal images. At inference, a test image is partially noised (SDEdit, `T_start < 1000`) and reconstructed. The residual between the original and reconstruction is the anomaly map — lesions the model never saw during training produce high residual signal.

**The forward process as an eraser:** When a diseased image (e.g., containing a haemorrhage) is fed into the DDPM forward process, the added noise mathematically destroys the disease signal. The UNet is then instructed (via the 768-d RETFound conditioning vector) to reconstruct a *healthy* version of that eye. Because the disease was erased by the noise and the UNet only knows how to draw healthy retinas, it simply fails to redraw the lesion. Subtracting the reconstruction from the original leaves only what the UNet could not account for — the disease.

The primary evaluation metric is **pixel-level AUROC on the DDR and iDRiD datasets** (757 annotated fundus images with MA/HE/EX/SE lesion masks), measured via vessel-suppressed residual maps (Frangi filter post-processing).

---

## 2. Training Dataset

The model is trained on **67,318 healthy fundus images** (residing in [`diffusionV2/`](file:///d:/NetrAi/diffusionV2)), with a **~3,366-image healthy validation hold-out** carved at 5% / seed 42 by [`carve_val.py`](file:///d:/NetrAi/diffusionV2/carve_val.py). Images are pooled from 10 public sources — only **disease-screening-negative ("grade-0" / healthy)** subsets are used.

### Per-Source Breakdown

| Source | Images (n) | Tier | Notes |
|--------|-----------|------|-------|
| **eyepacs** | 25,808 | Bulk | Grade-0 subset; dominant volume — defines the healthy manifold |
| **airogs** | 20,000 | Bulk | Glaucoma-screened; large camera/illumination diversity |
| **eddfs** | 15,000 | Bulk | Multi-centre screening cohort |
| **odir** | 2,160 | Mid | Ocular Disease Intelligent Recognition; multi-label, healthy subset only |
| **aptos** | 1,606 | Mid | Grade-0 DR screening; varied camera/pigmentation |
| **messidor** | 1,017 | Mid | Messidor-2 grade-0; three-site camera diversity |
| **g1020** | 724 | Long tail | Glaucoma-screened; optic disc–centered crops |
| **origa** | 482 | Long tail | Glaucoma-screened; high-res disc imaging |
| **refuge2** | 360 | Long tail | Glaucoma challenge; disc-centred, high-quality |
| **palm** | 161 | Long tail | Pathological myopia challenge; rare staphyloma morphology |
| **Total** | **67,318** | — | Train ≈ 63,952 · Val ≈ 3,366 (5% / seed 42) |

### Domain Tier Summary

| Tier | Sources | Share | Role |
|------|---------|-------|------|
| Bulk (~90%) | EyePACS, AIROGS, EDDFS | ~60,808 images | Define the healthy manifold |
| Mid | ODIR, APTOS, Messidor-2 | ~4,783 images | Camera / illumination / pigmentation diversity |
| Long tail | G1020, ORIGA, REFUGE2, PALM | ~1,727 images | Rare-domain coverage; disc-centred morphology |

**Domain imbalance is corrected, not ignored.** A `sqrt`-frequency `WeightedRandomSampler` ([`data.make_domain_sampler`](file:///d:/NetrAi/diffusionV2/data.py)) reweights each sample by `1/√(domain_count)` — softening the head (EyePACS/AIROGS/EDDFS) without over-boosting tiny domains (PALM = 161 would otherwise get a ~400× draw rate and overfit).

> **Label-purity caveat:** "healthy" means *screening-negative for that source's target disease*. Glaucoma-screened sources (AIROGS, ORIGA, REFUGE2, G1020) were never screened for **DR**, so a residual fraction may carry DR lesions and mildly contaminate DR detection. Bulk-volume sources dominate the manifold and dilute this.

The model never sees lesion-bearing **DDR / iDRiD** images during training — those are **held out** for pixel-level evaluation. Anomaly detection relies entirely on reconstruction error against this healthy prior.

---

## 3. Diffusion Architecture

```
Input (768×768 fundus image)
        |
        ▼
RETFoundConditioner ──► Frozen ViT-Large (224px, ImageNet-normalized)
        |                       |
        |               1024-d class token
        |                       |
        |               proj MLP: Linear(1024→768) → GELU → Linear(768→768) → LayerNorm
        |                       |
        |               cross_attention_dim=768 conditioning vector → [B, 1, 768]
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
MultiDiffusion Tiling (16 overlapping 256×256 tiles @ stride=170 ≈ 33%, over 768×768)
  + A-LCW learned gate (per-tile, per-step local/global conditioning mix)

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
Reconstruction (768×768)
        |
        ▼
Residual Map ──► Circular FOV Mask ──► Frangi vessel suppression ──► per-image MAD-z
        |
        ▼
Anomaly Score (pixel-level, used for DDR / iDRiD AUROC)
        |
        ▼
     [Feed to Classifier — Part II]
```

---

## 4. Module Reference — Diffusion

### Cross-File Wiring & State

`	ext
train.py (Orchestrator)
  |-- data.py       (collate_fn passes paths alongside tensors for cache keys;
  |                  make_domain_sampler builds the sqrt-freq WeightedRandomSampler)
  |-- models.py     (RETFoundConditioner + proj + A-LCW LCWGate; Disk/LRU feature cache)
  |-- diffusion.py  (Provides TILE_VIEWS (16), alphas_cumprod, make_retinal_mask, recon)
  |-- losses.py     (Min-SNR MSE + L1/FFL hybrid)
  +-- evaluation.py (DDR/iDRiD metrics; vessel-suppressed MAD-z postprocess)
`

### `train.py`

The orchestration hub. Reads `config.yaml`, builds all components, and runs the training loop.

**Key responsibilities:**
- Parses config sections (`paths`, `training`, `diffusion`, `eval`) into local variables.
- Sets CUDA environment flags: `expandable_segments`, TF32, cuDNN benchmark.
- Builds the UNet (`UNet2DConditionModel`, `sample_size=256`) in channels-last layout. Gradient checkpointing is **off** by default (VRAM headroom confirmed → ~25% throughput gain); xformers memory-efficient attention is **on**.
- Builds `RETFoundConditioner` → `CachedConditioner` (in-memory LRU) → `DiskCachedConditioner` (persistent **pre-projection** `[1,1024]` feature cache).
- `torch.compile` is **off** by default — reduce-overhead cudagraphs collide with the xformers `flash_bwd` custom op under per-tile sequential backward.
- Constructs a **three-group** AdamW optimizer: UNet @ `lr_unet`, proj MLP @ `lr_conditioner`, **A-LCW gate** @ `lr_conditioner` (fused when available, `weight_decay=1e-4`).
- Maintains EMA (decay 0.999) over UNet + proj + A-LCW gate.
- Handles three checkpoint scenarios:
  - i. Resume from `last.pt` (full state: model, proj, gate, optimizer, scaler, scheduler, EMA, best metrics).
  - ii. **Warm-start from a 256px checkpoint** — loads UNet + proj weights, scales base LR to **30%**, forces `warmup_epochs = 1`.
  - iii. Train from scratch.

**Training loop per step (memory-safe two-phase):**
- Samples `NUM_TRAIN_TILES` **random** tile views per image (not all 16) to keep iteration time low.
- **Phase 1** — one batched (no-grad) RETFound forward over all sampled tiles; then, *per tile*, `proj` is applied live (independent autograd graph) and the **A-LCW gate** predicts the local/global mix; `blended = lcw·local + (1−lcw)·global` (global cond detached).
- **Phase 2** — per-tile UNet forward → `diffusion_loss` → **immediate backward**, freeing that tile's activations before the next. Peak VRAM = **one** tile's forward+backward, *independent of tile count*.
- `diffusion_loss` = 0.6 × Min-SNR MSE + 0.4 × (L1 + FFL) hybrid, under `autocast` (bf16 if supported).
- Gradient accumulation over `ACCUM_STEPS` micro-batches → `clip_grad_norm(1.0)` → `optimizer.step()`.
- `CosineAnnealingLR` (`T_max = epochs − warmup`, `eta_min=1e-7`) after linear warmup.
- Periodic: val SSIM/PSNR, **DDR/iDRiD AUROC** eval (OOM-guarded — skips and continues training on CUDA OOM), visualization saves, CSV logging.
- Saves: `last.pt`, `best_loss.pt` (lowest val loss), `best_auroc.pt` (highest DDR AUROC), `loss.csv`, `metrics.csv`, `ddr_metrics.csv`.

**A-LCW (Adaptive Local Conditioning Weight).** The old fixed cosine LCW schedule is **replaced** by a learned gate (`models.LCWGate`): a small MLP that, per tile and per denoising step, predicts how much to trust the tile's *local* conditioning vs the whole-image *global* conditioning:

```
lcw       = sigmoid( MLP([ global_cond, local_cond, t_ratio ]) )  ∈ (0, 1)
tile_cond = lcw · local_cond + (1 − lcw) · global_cond
```

`max_train_lcw` in config is **no longer used in training** — it survives only as an inference fallback for old checkpoints that lack `lcw_gate` weights.

---

### `models.py`

#### `LCWGate` (A-LCW)

Small MLP gate — the core of **Adaptive Local Conditioning Weight**. Input `[global_cond, local_cond, t_ratio]`; output a per-tile, per-step scalar in `(0,1)` broadcast over the conditioning channels: `lcw = σ(MLP(...))`. Trained live (own LR group, EMA, checkpoint). Replaces the old hand-tuned cosine LCW schedule.

#### `RETFoundConditioner`

Frozen RETFound ViT-Large backbone + a small **trainable** projection head and the A-LCW gate.

- Loads `RETFound_cfp_weights.pth` (falls back to torchvision ViT-L if unavailable). ViT params frozen (`requires_grad_(False)`), permanently in `eval()`; the `train()` override keeps only `proj` + `lcw_gate` trainable.
- `proj`: `Linear(1024→768) → GELU → Linear(768→768) → LayerNorm`, output `(B, 1, 768)`.
- Preprocessing: resize to 224×224 → `[-1,1]`→`[0,1]` → ImageNet normalize (mean/std as buffers).
- `extract_features()` returns the raw `[B, 1024]` CLS token; **projection is applied live each step** so `proj` receives gradients from the per-tile local path.

#### `CachedConditioner` → `DiskCachedConditioner`

Two-layer feature cache around the conditioner:
- **`CachedConditioner`** — in-memory `OrderedDict` LRU of raw ViT features (keyed by image path); `get_tile_conds_batched()` runs one batched ViT forward for all 16 tiles.
- **`DiskCachedConditioner`** — persistent on-disk cache storing the **raw, pre-projection** `[1, 1024]` CLS token per image. Projection is applied on every call, decoupling the cache from the evolving `proj` weights. (Old post-projection caches froze a random epoch-0 projection; those `[…, 768]` files are rejected and recomputed.)

---

### `diffusion.py`

All diffusion math, MultiDiffusion tiling, and the retinal mask.

**Noise functions** (names are legacy — noise is standard **Gaussian**; simplex was tested and rejected, so `simplex_freq`/`simplex_octaves` are no-ops):
- `add_simplex_noise` — standard DDPM forward process: `x_t = √ā·x₀ + √(1-ā)·ε`.
- `simplex_ddim_step` — DDIM reverse step with optional stochasticity (`eta`). Clamps `pred_x0` to `[-1,1]`.

**MultiDiffusion tiling constants:**

| Constant | Value | Notes |
|----------|-------|-------|
| `TILE_SIZE` | 256 | UNet input size |
| `FULL_SIZE` | **768** | Full inference resolution |
| `TILE_STRIDE` | **170** | ~33% overlap — minimum that hides seams → sharper residual |
| `NUM_TRAIN_TILES` | 2 | Random tiles sampled per image **during training** |
| `TILE_VIEWS` | **16** tiles | 4×4 grid via `_even_starts` → `[0,171,341,512]` per axis |

- `_even_starts` / `get_tile_views` — evenly spaced tile starts from `round(span/stride)+1` (avoids the degenerate 2px-apart edge pair the old fixed-stride logic produced).
- `get_linear_weight` / `make_cosine_weight` — **cosine²** blend window (`sin²(x)+1e-3`). The wide high-weight plateau lets 33% overlap hide seams; the `1e-3` floor prevents zero-weight starvation at boundaries.
- `make_retinal_mask` — **circular** FOV crop (`r = min(H,W)//2`) intersected with an intensity foreground mask (mean < 0.05 = background). Removes tiling border artifacts; assumes the fundus is centered (valid after direct Resize).
- `multidiffusion_reconstruct` — core inference loop: partially noise → DDIM denoising with per-tile UNet calls, fusing outputs via cosine²-weighted accumulation, with the **A-LCW gate** blending tile-specific and global conditioning per step.
- `full_reconstruct_and_residual` / `multiscale_residual` — multi-scale reconstruction → upscale to 768px → element-wise **maximum** across scales (max, not weighted sum — avoids starving coarse scales).

---

### `data.py`

- `make_transform` — torchvision pipeline, **BILINEAR** throughout (avoids bicubic ringing in the L1 residual). Augmentations are deliberately conservative for reconstruction-UAD (cranking colour would normalise lesion colour and hide exudates/haemorrhages):
  - `Resize(crop_size)` — direct resize, **no** RandomResizedCrop (crop jitter breaks the path-keyed conditioning and the centered FOV mask).
  - `RandomHorizontalFlip` — L/R-eye is anatomically valid; laterality invariance.
  - `RandomAffine(±10°, scale 0.92–1.08)` — rotation + **±8% FOV/scale jitter** for the unseen DDR/iDRiD magnifications.
  - `ColorJitter(0.2, 0.2, 0.15, 0.04)` — mild illumination / sensor / tint variation.
  - **No** VerticalFlip (fundus is never acquired upside-down). Val: resize only.
- `RetinaDataset` — accepts `.csv` (`path`/`image` column, optional `source` column), `.txt`, or a directory (recursive glob). Tracks `self.sources` parallel to `self.images`. Optional `bad_files_txt` pre-filter; a 10-try retry loop skips unreadable images at runtime. Returns `(tensor, path)` (path = cache key).
- `make_domain_sampler` — builds the `sqrt`-frequency `WeightedRandomSampler` over the `source` column (weight ∝ `1/√(domain_count)`); returns `None` (→ uniform shuffle) if there is no source column or only one domain.
- `collate_fn` — stacks tensors, keeps paths as a Python list.

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

`postprocess_residual` — vessel-aware anomaly-map cleaning pipeline:
1. Raw L1 residual per pixel, retinal-mask gated, **median-subtracted** (removes diffuse imperfect-reconstruction haze).
2. Light Gaussian blur (σ=0.5).
3. Frangi on the **green channel** (σ=2–8, black ridges) → real vessel anatomy.
4. Frangi on the **residual** (σ=1–5, bright ridges) → vessel-shaped reconstruction artifacts, **gated** by a dilated tolerance band off the green-channel vessel map — so ridge-shaped residual *away* from vessels (flame haemorrhages, NVD/NVE) is treated as lesion and survives.
5. Combined vessel map → soft exponential suppression: `weight = exp(-1.5 × vessel_norm)`.
6. **Per-image MAD-z score** (`(x − median) / (1.4826·MAD)`, clipped at 0) for cross-image calibration, then re-masked.

`compute_ddr_metrics` — pixel-level **DDR / iDRiD** evaluation: union the 4 lesion-type masks (MA/HE/EX/SE), reconstruct via MultiDiffusion → `postprocess_residual`, compute **AUROC / AP / Dice**. Supports count- and time-budget cutoffs; called from the training loop under an OOM guard.

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
  data_train:       # CSV/TXT/dir — healthy training images (e.g. train_keep_768.csv)
  data_val:         # CSV — healthy validation hold-out (carve_val.py output)
  bad_files_txt:    # Optional path list to pre-filter corrupted images
  checkpoint_dir:   # Where to save checkpoints + logs (e.g. checkpoints_768v5)
  pretrained_256:   # 256px warm-start checkpoint (best_loss.pt)
  feat_cache_dir:   # Persistent pre-projection RETFound feature cache
  ddr_images_dir:   # DDR / iDRiD eval image directory
  ddr_masks_dir:    # Eval mask directory (MA/HE/EX/SE subdirs)
  retfound_weights: # Path to RETFound_cfp_weights.pth

training:
  seed:                   42
  crop_size:              768
  epochs:                 15      # warm-start fine-tune; detection sweet spot ~epoch 8–15
  batch_size:             6
  accum_steps:            6       # effective batch = 6 × 6 = 36
  warmup_epochs:          3       # forced to 1 when warm-starting from 256px
  num_train_tiles:        2       # random tiles/step (of 16)
  lr_unet:                5.0e-6  # ×0.3 on warm-start
  lr_conditioner:         1.0e-5  # proj + A-LCW gate; ×0.3 on warm-start
  snr_gamma:              2.0
  gradient_checkpointing: false   # off → ~25% throughput (VRAM headroom confirmed)
  channels_last:          true    # NHWC Tensor Core layout
  compile:                false   # off — cudagraphs collide with xformers flash_bwd

diffusion:
  ddim_steps:       50
  ddim_t_start:     350   # SDEdit noising depth (kept < 400 to spare microaneurysm-scale anatomy)
  max_train_lcw:    0.4   # inference fallback only — training uses the A-LCW gate

eval:
  vis_every:        1
  eval_every:       10
  ddr_eval_every:   5     # over 15 epochs → eval at epoch 4 / 9 / 14
  ddr_max_images:   ~     # null = full set
  dice_threshold:   ~     # null = oracle sweep (dev only); set + lock a float before reporting
```

---

## 5. Key Design Decisions — Diffusion

| Decision | Rationale |
|----------|-----------|
| Train on 256px tiles, infer on **768px** via MultiDiffusion | UNet fits in VRAM at 256px; MultiDiffusion fuses 16 overlapping tiles for a seamless 768px reconstruction. |
| `T_start=350` not `1000` | At t=1000 the image is pure noise — coarse anatomy (disc, vessel layout) is destroyed and the UNet must hallucinate. At t=350 the noise erases small lesions (MA/HE) while the global eye shape survives, so the reconstruction keeps anatomy and drops pathology. Kept < 400 to spare microaneurysm-scale structure. |
| `block_out_channels: (128,256,512,512)` — cap at 512 | Doubling to 1024 at the bottleneck quadruples params + activation memory. Repeating 512 adds an extra deep reasoning layer at max compression depth for no extra memory. |
| Warm-start from a 256px checkpoint (LR ×0.3, warmup 1) | This is a fine-tune, not a scratch run; the 256px `best_loss` is a competent recon base. Lower LR + short warmup anneal cleanly onto the 768px manifold. |
| Frozen RETFound ViT-Large | RETFound captures fundus-specific anatomy; fine-tuning would destroy the generic healthy-retina prior. |
| Cache **pre-projection** raw ViT features (on disk) | `proj` trains, so caching its output would freeze a random epoch-0 projection. Caching the raw `[1,1024]` CLS and applying `proj` live keeps the global cond current. |
| **A-LCW learned gate** (vs fixed cosine LCW) | A per-tile, per-step MLP gate learns the local/global conditioning mix instead of a hand-tuned schedule — each tile/timestep decides how much to trust local detail. |
| Per-tile **sequential backward** (phase-2) | Forward+backward one tile at a time frees activations between tiles → peak VRAM = a single tile, independent of tile count. Enables 768px on 40GB. |
| `sqrt`-frequency domain sampler | Corrects the 90%-head domain imbalance without over-boosting tiny domains (PALM = 161 out of 67,318). |
| SNR-γ=2.0 (aggressive) | Balances against the L1/FFL term at low noise; prevents overfitting tile-boundary micro-textures. |
| Residual-Frangi **gated by a real-vessel band** | Suppresses vessel-shaped artifacts only where they coincide with anatomy — ridge-shaped residual away from vessels (flame haemorrhages, NVD/NVE) survives as lesion. |
| Element-wise **max** for the multi-scale ensemble | Weighted sum caps small-scale lesion signal; max lets each scale compete at full confidence. |
| **BILINEAR** interpolation throughout | Avoids bicubic ringing that contaminates the L1 residual. |
| `compile` off, xformers on | reduce-overhead cudagraphs collide with the xformers `flash_bwd` custom op under per-tile sequential backward; xformers alone gives most of the attention speedup. |

---

## 6. Diffusion Training Pipeline

### Tensor Flow & A-LCW (Highlights)

- **Data:** `(B, 3, 768, 768)` pixels in `[-1,1]`.
- **Random tiles:** `NUM_TRAIN_TILES=2` random `(256,256)` tile views/step, batched through RETFound **once** (no-grad) for all tiles.
- **Mask:** circular FOV mask computed once per image `(B,1,768,768)`, sliced per tile.
- **Per-tile proj + gate (live):** `proj` is applied per tile (independent graph) -> `local_cond`; the **A-LCW gate** predicts `lcw` and blends `blended = lcw*local + (1-lcw)*global` (global detached).
- **Sequential backward:** each tile's UNet forward -> loss -> backward immediately, freeing activations before the next tile.

```
Epoch start
|
├── For each batch:
|   ├── Sample NUM_TRAIN_TILES=2 random tile views (of 16)
|   ├── Disk/LRU cache → raw [1,1024] ViT features (proj applied live)
|   ├── add_simplex_noise (Gaussian) → x_t at random t ∈ [0, 1000]
|   ├── Phase 1: per-tile proj + A-LCW gate → blended_cond
|   ├── Phase 2 (per tile): autocast UNet → pred_noise
|   │            └ diffusion_loss (0.6·Min-SNR MSE + 0.4·(L1+FFL)) → backward
|   └── every ACCUM_STEPS: clip_grad_norm(1.0) → optimizer.step() → EMA update
|
├── [every VIS_EVERY epochs]      save_visualizations
├── [every EVAL_EVERY epochs]     compute_val_metrics (SSIM/PSNR)
├── [every DDR_EVAL_EVERY epochs] compute_ddr_metrics → DDR/iDRiD AUROC (OOM-guarded)
|
└── Save: last.pt, best_loss.pt, best_auroc.pt, CSVs, dashboard
```

---

## 7. Diffusion Inference Pipeline

```
Input: 768×768 fundus image

1. get_full_image_cond()              →  global_cond (1,1,768)
2. get_tile_conds_batched()           →  16× local_cond (1,1,768)
3. add_simplex_noise(img, T_start=350) →  x_T
4. DDIM loop (50 steps):
   For each of 16 tiles:
     lcw       = LCWGate(global_cond, local_cond, t_ratio)   # learned, per tile/step
     tile_cond = lcw × local_cond + (1 - lcw) × global_cond
     pred_noise    = UNet(tile, t, tile_cond)
     tile_denoised = DDIM_step(tile, pred_noise)
     accumulate: value[h0:h1, w0:w1] += tile_denoised × cosine²_weight
   x_t = value / count
5. recon_768 = x_0 (final)
6. residual = |img - recon_768|.mean(channel) × circular_mask
7. postprocess_residual():
   - Median subtract → Gaussian blur
   - Frangi(green) + vessel-band-gated Frangi(residual) → vessel map
   - residual × exp(-1.5 × vessel_norm) → per-image MAD-z
8. Output: pixel-level anomaly map (MAD-z scored)
```

The anomaly map is then saved and used as input to the Classifier pipeline.

---

## 8. Evaluation

Both eval sets are **held out** — never seen during training. Scoring uses the vessel-suppressed, per-image **MAD-z** anomaly map (`postprocess_residual`).

### DDR Dataset
- 757 labeled fundus images with pixel-level annotations: **MA** (Microaneurysms), **HE** (Hemorrhages), **EX** (Hard Exudates), **SE** (Soft Exudates).
- Masks combined into a single binary map (logical OR across lesion types).
- Pixel-level metrics: **AUROC** (primary), **AP**, **Dice** (threshold from a locked val sweep; the oracle sweep is dev-only and must not be reported).
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

> **Model selection:** pick **`best_auroc.pt`** for deployment, not `best_loss.pt`. Reconstruction-UAD *detection* peaks (epoch ~8–15) **before** reconstruction loss fully converges; past the peak the model over-reconstructs lesions and detection drops. `best_auroc.pt` captures that peak (evaluated at the DDR-eval epochs).

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
# 0. One-time: carve a 5% healthy val hold-out from the train CSV
python carve_val.py

# 1. One-time: pre-compute the RETFound feature cache (pre-projection [1,1024])
python cache_features.py --config config.yaml

# 2. Train
python train.py --config config.yaml

# With persistent logging
python train.py --config config.yaml 2>&1 | tee -a checkpoints_768v5/train.log
```

**Resume:** training auto-resumes from `last.pt` if it exists in `checkpoint_dir`. No flag needed.

**Warm-start from a 256px checkpoint:** set `paths.pretrained_256` to a 256px `best_loss.pt`. Base LR is scaled to 30% and warmup forced to 1 epoch automatically.

**Model selection:** pick **`best_auroc.pt`** for deployment — reconstruction-UAD *detection* peaks (epoch ~8–15) **before** reconstruction loss converges; `best_loss.pt` over-reconstructs lesions and detection drops.

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
