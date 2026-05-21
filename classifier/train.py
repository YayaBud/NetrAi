"""
NetrAi Classifier — SegFormer Training Loop
=============================================
Trains NetrAiEncoder using:
  L_total = L_SupCon(1.0) + λ_kl · L_KL(β) + λ_ortho · L_Ortho

No cross-entropy head — SupCon is the sole supervisory signal.
XGBoost handles final classification after this training phase.

Run:
    python -m classifier train --config classifier/config.yaml
"""

import os
import argparse
import yaml
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .model   import NetrAiEncoder
from .losses  import NetrAiLoss
from .data    import build_dataloader
from .utils   import (
    load_config, save_config,
    setup_logging,
    AverageMeter, MetricsLogger,
    save_checkpoint, load_checkpoint,
    LinearWarmupCosineScheduler,
    get_device, get_amp_dtype,
    Timer,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the full SegFormer training loop.

    Usage:
        trainer = Trainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger

        self.device   = get_device()
        self.amp_type = get_amp_dtype(self.device)
        self.use_amp  = cfg['training']['amp'] and self.device.type == "cuda"

        # Directories
        self.ckpt_dir = cfg['paths']['checkpoint_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Save a copy of config next to the checkpoints
        save_config(cfg, os.path.join(self.ckpt_dir, "config.yaml"))

        self._build_model()
        self._build_data()
        self._build_optimiser()
        self._build_loss()

        self.scaler        = GradScaler('cuda', enabled=self.use_amp)
        self.metrics_log   = MetricsLogger(
            os.path.join(self.ckpt_dir, "metrics.json")
        )
        self.start_epoch   = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        mc = self.cfg['model']
        self.model = NetrAiEncoder(
            backbone_name     = mc['backbone'],
            decoder_embed_dim = mc['decoder_embed_dim'],
            path_a_dim        = mc['path_a_dim'],
            path_b_dim        = mc['path_b_dim'],
            alpha_init        = mc['alpha_init'],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model: NetrAiEncoder  params={n_params / 1e6:.1f}M")
        self.logger.info(f"  Learned gate α init = {mc['alpha_init']}")

    def _build_data(self):
        p  = self.cfg['paths']
        dc = self.cfg['data']
        tc = self.cfg['training']

        self.train_loader = build_dataloader(
            root        = os.path.join(p['data_dir'], "train"),
            anomaly_dir = p['anomaly_maps_dir'],
            image_size  = dc['image_size'],
            batch_size  = tc['batch_size'],
            augment     = True,
            balanced    = True,   # WeightedRandomSampler — 1:1:1 per batch
            num_workers = dc['num_workers'],
            pin_memory  = dc['pin_memory'],
            drop_last   = True,
        )
        self.val_loader = build_dataloader(
            root        = os.path.join(p['data_dir'], "val"),
            anomaly_dir = p['anomaly_maps_dir'],
            image_size  = dc['image_size'],
            batch_size  = tc['batch_size'],
            augment     = False,
            balanced    = False,  # sequential — every sample exactly once
            num_workers = dc['num_workers'],
            pin_memory  = dc['pin_memory'],
            drop_last   = False,
        )

    def _build_optimiser(self):
        tc = self.cfg['training']
        # Use different LR for backbone vs. custom heads
        encoder_params = list(self.model.encoder.parameters())
        head_params    = (
            list(self.model.decoder.parameters())
            + list(self.model.gate.parameters())
            + list(self.model.bottleneck.parameters())
        )
        self.optimiser = torch.optim.AdamW([
            {"params": encoder_params, "lr": tc['lr'] * 0.1},  # 10× lower for pretrained backbone
            {"params": head_params,    "lr": tc['lr']},
        ], weight_decay=tc['weight_decay'])

        self.scheduler = LinearWarmupCosineScheduler(
            optimizer     = self.optimiser,
            warmup_epochs = tc['warmup_epochs'],
            total_epochs  = tc['epochs'],
        )

    def _build_loss(self):
        self.criterion = NetrAiLoss(self.cfg)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------

    def resume(self, ckpt_path: str):
        self.logger.info(f"Resuming from {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, self.model, self.optimiser, self.device)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.logger.info(f"  Resumed at epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, epoch: int, train: bool) -> dict:
        self.model.train(train)

        meters = {k: AverageMeter(k)
                  for k in ("loss", "l_supcon", "l_kl", "l_ortho")}
        tc = self.cfg['training']

        with torch.set_grad_enabled(train):
            for batch in loader:
                images     = batch["image"].to(self.device, non_blocking=True)
                amap       = batch["anomaly_map"].to(self.device, non_blocking=True)
                labels     = batch["label"].to(self.device, non_blocking=True)

                with autocast('cuda', dtype=self.amp_type, enabled=self.use_amp):
                    vector_769, mu, log_var = self.model(images, amap)
                    loss, breakdown        = self.criterion(
                        vector_769, mu, log_var, labels, epoch
                    )

                if train:
                    self.optimiser.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimiser)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc['grad_clip']
                    )
                    self.scaler.step(self.optimiser)
                    self.scaler.update()

                B = images.size(0)
                for k in meters:
                    meters[k].update(breakdown[k], n=B)

        return {k: m.avg for k, m in meters.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        tc    = self.cfg['training']
        timer = Timer()
        self.logger.info(
            f"Training for {tc['epochs']} epochs  "
            f"AMP={'on' if self.use_amp else 'off'}  "
            f"device={self.device}"
        )

        for epoch in range(self.start_epoch, tc['epochs']):
            # ---- Train ----
            train_metrics = self._run_epoch(self.train_loader, epoch, train=True)
            self.scheduler.step()

            # ---- Validate ----
            val_metrics = {}
            if (epoch + 1) % tc['eval_every'] == 0:
                val_metrics = self._run_epoch(self.val_loader, epoch, train=False)

            # ---- Logging ----
            lr = self.optimiser.param_groups[1]['lr']  # head LR
            self.logger.info(
                f"Epoch {epoch+1:03d}/{tc['epochs']} | "
                f"lr={lr:.2e} | "
                f"β={self.criterion.beta_sched.get_beta(epoch):.5f} | "
                f"α={self.model.gate.alpha.item():.4f} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"SupCon={train_metrics['l_supcon']:.4f} "
                f"KL={train_metrics['l_kl']:.4f} "
                f"Ortho={train_metrics['l_ortho']:.4f}"
                + (f" | val_loss={val_metrics.get('loss', 0):.4f}"
                   if val_metrics else "")
                + f" | elapsed={timer.elapsed()}"
            )

            log_entry = {
                "epoch":    epoch + 1,
                "lr":       lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
                "alpha":    float(self.model.gate.alpha.item()),
                "beta":     self.criterion.beta_sched.get_beta(epoch),
            }
            self.metrics_log.log(log_entry)

            # ---- Checkpoint ----
            is_best = False
            val_loss = val_metrics.get("loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True

            if (epoch + 1) % tc['save_every'] == 0 or is_best:
                state = {
                    "epoch":            epoch,
                    "model_state":      self.model.state_dict(),
                    "optimizer_state":  self.optimiser.state_dict(),
                    "scheduler_state":  self.scheduler.state_dict(),
                    "best_val_loss":    self.best_val_loss,
                    "config":           self.cfg,
                }
                fname = f"epoch_{epoch+1:04d}.pt"
                path  = save_checkpoint(
                    state, self.ckpt_dir, filename=fname, is_best=is_best
                )
                if is_best:
                    self.logger.info(f"  ★ New best val_loss={self.best_val_loss:.4f} → {path}")

        self.logger.info(f"Training complete. Best val_loss={self.best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point  (python -m classifier train ...)
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Train NetrAi SegFormer encoder")
    parser.add_argument("--config",  default="classifier/config.yaml")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-dir", default=None, help="Override log directory")
    args = parser.parse_args(args)

    cfg    = load_config(args.config)
    logdir = args.log_dir or cfg['paths']['checkpoint_dir']
    logger = setup_logging(logdir, name="train")

    trainer = Trainer(cfg, logger)
    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
