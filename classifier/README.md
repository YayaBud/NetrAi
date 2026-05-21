# NetrAi Classifier

Retinal disease classification pipeline for **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Pathologic Myopia (PM)**.

---

## Architecture Overview

```
Retina Image (512×512)
        │
        ├──[ONE TIME]──▶ RETFound-Large (frozen ViT-L/16)
        │                      │
        │               1024-D .pt cache (per image)
        │
        └──▶ MIT-B3 SegFormer (ImageNet init, being trained)
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
               XGBoost
          DR / Glaucoma / PM + confidence %
```

### Loss Function (SegFormer training only — no CE head)

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{SupCon}}_{1.0} + \underbrace{0.01 \cdot \mathcal{L}_{KL}(\beta)}_{VIB} + \underbrace{0.1 \cdot \mathcal{L}_{Ortho}}_{\mu\ \text{only}}$$

| Loss | Applied to | Purpose |
|---|---|---|
| **SupCon** | Full 769-D vector | Forces same-disease vectors to cluster, pushes apart different diseases |
| **KL** | Path B μ, log_σ² | VIB bottleneck — discards noisy features, keeps strongest disease signals |
| **Ortho** | Path B **μ only** | Cosine similarity penalty — forces distinct, non-overlapping features per class |

**β-annealing**: β = 0 for first 10 epochs → linearly ramps to 0.001 over next 20 epochs.  
**Class balance**: `WeightedRandomSampler` enforces 1:1:1 (DR:Glaucoma:PM) per batch + class-aware SupCon temperature (PM uses τ=0.04 vs 0.07).

---

## Project Structure

```
classifier/
├── config.yaml          ← All hyperparameters
├── model.py             ← NetrAiEncoder (SegFormer + gate + VIB bottleneck)
├── losses.py            ← SupCon + KL + Ortho + BetaScheduler + NetrAiLoss
├── data.py              ← RetinalDataset + balanced DataLoader
├── retfound.py          ← RETFound pre-computation + cache I/O
├── train.py             ← SegFormer training loop
├── extract.py           ← 769-D → 1793-D feature extraction to .npy
├── xgboost_clf.py       ← XGBoost train / eval / SHAP
├── inference.py         ← Single-image end-to-end diagnosis
├── utils.py             ← Logging, checkpointing, metrics, LR scheduler
├── requirements.txt
├── __init__.py
├── __main__.py          ← CLI dispatcher
└── tests/
    ├── conftest.py      ← Shared pytest fixtures (temp dataset)
    ├── test_model.py    ← Model shape + gradient contracts
    ├── test_losses.py   ← Loss function unit tests
    └── test_data.py     ← Dataset + DataLoader tests
```

---

## Setup

```bash
# Install dependencies
pip install -r classifier/requirements.txt

# Verify tests pass (no GPU required)
pytest classifier/tests/ -v
```

---

## Training Pipeline — Step by Step

### Step 0 — Prepare your data

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

Place your diffusion model's **clean residual** anomaly maps in a flat directory:

```
data/anomaly_maps/
├── <image_stem>_anomaly.png   ← preferred naming
└── <image_stem>.png           ← fallback naming
```

> **Note**: If an anomaly map is missing for an image, the gate defaults to `F_gated = F_concat` (identity — no anomaly guidance). The model still trains but without diffusion prior for that sample.

Update paths in `classifier/config.yaml` to match your directory layout.

---

### Step 1 — Cache RETFound Embeddings *(one time, ~minutes)*

```bash
python -m classifier cache-retfound --config classifier/config.yaml
```

Runs every image through frozen RETFound-Large, saves 1024-D `.pt` files to `retfound_cache/`. RETFound is then unloaded from VRAM permanently.

> **RETFound weights**: Download `RETFound_cfp_weights.pth` from the [RETFound repository](https://github.com/rmaphoh/RETFound_MAE) and set `paths.retfound_weights` in `config.yaml`. If not provided, falls back to HuggingFace ViT-L/16 ImageNet-21k (domain gap applies).

---

### Step 2 — Train the SegFormer Encoder

```bash
python -m classifier train --config classifier/config.yaml

# Resume from a checkpoint
python -m classifier train --config classifier/config.yaml \
                           --resume checkpoints/classifier/epoch_0050.pt
```

Trains for `training.epochs` epochs. Checkpoints saved every `training.save_every` epochs to `checkpoints/classifier/`. Best checkpoint by val loss saved as `best.pt`.

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
# Train with SHAP feature importance analysis
python -m classifier xgboost --config classifier/config.yaml --shap
```

Trains on the 1793-D vectors with early stopping. Saves:
- `checkpoints/classifier/xgboost_model.pkl` — the trained booster
- `checkpoints/classifier/xgboost_results.json` — train/val metrics
- `checkpoints/classifier/shap_importance.json` — top feature importances (if `--shap`)

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

> If no anomaly map is available, omit `--anomaly`. The gate defaults to identity (no diffusion prior).

---

## Configuration Reference

Key settings in `classifier/config.yaml`:

```yaml
training:
  epochs:      100
  batch_size:  16
  lr:          1.0e-4        # head LR; backbone gets lr × 0.1
  
  supcon_weight:  1.0        # SupCon drives everything
  lambda_kl:      0.01       # VIB KL weight
  lambda_ortho:   0.1        # Orthogonal penalty weight
  
  supcon_temperatures:
    0: 0.07                  # DR
    1: 0.07                  # Glaucoma
    2: 0.04                  # PM (minority → sharper gradient)
  
  beta_warmup_epochs: 10     # β=0 for first N epochs
  beta_anneal_epochs: 20     # linear ramp to beta_target
  beta_target:       0.001

xgboost:
  n_estimators:  1000
  max_depth:     6
  learning_rate: 0.05
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

## Key Design Decisions

| Decision | Rationale |
|---|---|
| No CE head during SegFormer training | SupCon alone forces better-separated clusters than CE+SupCon |
| Ortho penalty on μ only, not full 769-D | Path A must remain free to capture subtle early-stage signals |
| VIB inference uses μ, not z | Deterministic embeddings → stable XGBoost decision boundaries |
| RETFound cached before training | Never occupies VRAM during training; 1024-D domain context always available |
| XGBoost over MLP | Tabular supremacy, column sampling overfitting resistance, SHAP explainability |
| WeightedRandomSampler + class-aware τ | Two complementary fixes for class imbalance at hardware and math level |
