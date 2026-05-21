"""
NetrAi Classifier — Utility Functions
=======================================
Shared helpers for:
  - Config loading
  - Checkpointing (save / load)
  - Logging (console + file)
  - Metrics (AverageMeter, classification report)
  - VRAM / device helpers
"""

import os
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Optional, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, name: str = "netr_ai") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks a running mean of a scalar value (loss, accuracy, etc.)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsLogger:
    """Accumulates per-epoch metrics and writes them to a JSON file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history: list[dict] = []
        # Resume from existing log
        if os.path.isfile(log_path):
            with open(log_path) as f:
                self.history = json.load(f)

    def log(self, metrics: dict):
        self.history.append(metrics)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def latest(self) -> Optional[dict]:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(state, best_path)
        return best_path
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Loads a checkpoint into `model` (and optionally `optimizer`).
    Returns the checkpoint dict (contains 'epoch', 'metrics', etc.).
    """
    device = device or torch.device("cpu")
    ckpt   = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

CLASS_NAMES = ["DR", "Glaucoma", "PM"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,  # (N, 3) softmax probabilities
) -> dict:
    """
    Returns a dict with:
      - per-class precision / recall / F1
      - macro-averaged metrics
      - confusion matrix (as list of lists)
      - AUC-ROC (macro OvR) if y_prob provided
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy":         report["accuracy"],
        "macro_f1":         report["macro avg"]["f1-score"],
        "macro_precision":  report["macro avg"]["precision"],
        "macro_recall":     report["macro avg"]["recall"],
        "confusion_matrix": cm,
        "per_class":        {
            cls: {
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1":        report[cls]["f1-score"],
                "support":   report[cls]["support"],
            }
            for cls in CLASS_NAMES
        },
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["macro_auc"] = float(auc)
        except Exception:
            metrics["macro_auc"] = None

    return metrics


def print_metrics(metrics: dict, logger: Optional[logging.Logger] = None) -> None:
    lines = [
        f"  Accuracy:      {metrics['accuracy']:.4f}",
        f"  Macro F1:      {metrics['macro_f1']:.4f}",
        f"  Macro AUC:     {metrics.get('macro_auc', 'N/A')}",
        f"  Macro Prec:    {metrics['macro_precision']:.4f}",
        f"  Macro Recall:  {metrics['macro_recall']:.4f}",
    ]
    for cls, vals in metrics.get("per_class", {}).items():
        lines.append(
            f"  [{cls}]  P={vals['precision']:.3f}  "
            f"R={vals['recall']:.3f}  F1={vals['f1']:.3f}"
        )
    cm = metrics.get("confusion_matrix")
    if cm:
        lines.append(f"  Confusion matrix (DR / Glaucoma / PM):")
        for row in cm:
            lines.append(f"    {row}")

    out = "\n".join(lines)
    if logger:
        logger.info(out)
    else:
        print(out)


# ---------------------------------------------------------------------------
# Learning-rate warmup scheduler
# ---------------------------------------------------------------------------

class LinearWarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `eta_min`.
    """

    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        eta_min:       float = 1e-6,
        last_epoch:    int   = -1,
    ):
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup:
            scale = (ep + 1) / max(self.warmup, 1)
        else:
            import math
            prog  = (ep - self.warmup) / max(self.total - self.warmup, 1)
            scale = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + math.cos(math.pi * prog))
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"  Device: {props.name}  VRAM={props.total_memory // 1024**2} MB")
    else:
        dev = torch.device("cpu")
        print("  Device: CPU")
    return dev


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on Ampere+ (sm_80+), float16 otherwise."""
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability()
        return torch.bfloat16 if cc[0] >= 8 else torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> str:
        s = int(time.time() - self._start)
        return f"{s // 3600:02d}h {(s % 3600) // 60:02d}m {s % 60:02d}s"

    def reset(self):
        self._start = time.time()
