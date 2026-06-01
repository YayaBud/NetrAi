"""
NetrAi Classifier — SegFormer Training Loop (v2)
==================================================
Phase 1 differentiable training.

Loss:
    L_total = L_main + λ_aux · L_aux + β · (KL₁ + KL₂)

    L_main   BCEWithLogitsLoss — main_classifier(z_fused) vs label_vec
    L_aux    BCEWithLogitsLoss — aux_classifier(z1) vs label_vec
    KL₁ + KL₂ — dual VIB bottleneck penalty with β-annealing

After training: aux_classifier and main_classifier are discarded.
The frozen encoder extracts 256-D vectors for XGBoost (Phase 2).

Run:
    python -m classifier train --config classifier/config.yaml
"""

import os
import argparse
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
    Encapsulates Phase 1 training loop.

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

        self.ckpt_dir = cfg['paths']['checkpoint_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)
        save_config(cfg, os.path.join(self.ckpt_dir, "config.yaml"))

        self._build_model()
        self._build_data()
        self._build_optimiser()
        self._build_loss()

        self.scaler      = GradScaler('cuda', enabled=self.use_amp)
        self.metrics_log = MetricsLogger(
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
            backbone_name = mc['backbone'],
            head_out_dim  = mc['head_out_dim'],
            vib_hidden    = mc['vib_hidden'],
            vib_out_dim   = mc['vib_out_dim'],
            dropout       = mc.get('dropout', 0.3),
        ).to(self.device)

        # Optionally freeze early backbone stages to prevent overfitting on small data.
        # Must be called BEFORE _build_optimiser so frozen params are excluded.
        n_freeze = self.cfg['training'].get('freeze_backbone_stages', 0)
        if n_freeze > 0:
            self._freeze_backbone_stages(n_freeze)

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"Model: NetrAiEncoder  trainable={n_train/1e6:.1f}M / {n_total/1e6:.1f}M total"
            + (f"  (MIT-B3 stages 0-{n_freeze-1} frozen)" if n_freeze else "")
        )

    def _freeze_backbone_stages(self, n_stages: int) -> None:
        """
        Freeze the first n_stages of the MIT-B3 SegFormer encoder.

        MIT-B3 structure inside self.model.encoder.encoder (SegformerEncoder):
            patch_embeddings  — ModuleList[4], one overlap-patch-embed per stage
            block             — ModuleList[4], transformer blocks per stage
            layer_norm        — ModuleList[4], LN after each stage

        Rule: patch_embeddings[0].proj stays TRAINABLE even when stage 0 is
        frozen, because it was surgically modified to accept 6 channels and
        channels 3-5 (residual) are zero-init and need to learn from scratch.
        Freezing it would permanently silence the anomaly-map input.
        """
        enc = self.model.encoder.encoder   # SegformerEncoder
        n_stages = min(n_stages, 4)        # MIT-B3 has 4 stages

        for i in range(n_stages):
            # Freeze transformer blocks
            for param in enc.block[i].parameters():
                param.requires_grad_(False)
            # Freeze per-stage layer norm
            for param in enc.layer_norm[i].parameters():
                param.requires_grad_(False)
            # Freeze patch embedding for stages 1+ only.
            # Stage 0 patch_embed was modified for 6-ch — must stay trainable.
            if i > 0:
                for param in enc.patch_embeddings[i].parameters():
                    param.requires_grad_(False)

        self.logger.info(
            f"  Froze MIT-B3 stages 0..{n_stages-1} "
            f"(patch_embeddings[0].proj kept trainable for 6-ch residual input)"
        )

    def _build_data(self):
        p  = self.cfg['paths']
        dc = self.cfg['data']
        tc = self.cfg['training']

        shared = dict(
            anomaly_dir        = p['anomaly_maps_dir'],
            retfound_cache_dir = p.get('retfound_cache_dir'),
            image_size         = dc['image_size'],
            batch_size         = tc['batch_size'],
            num_workers        = dc['num_workers'],
            pin_memory         = dc['pin_memory'],
        )

        self.train_loader = build_dataloader(
            root     = os.path.join(p['data_dir'], "train"),
            split    = "train",
            augment  = True,
            balanced = True,
            drop_last = True,
            **shared,
        )
        self.val_loader = build_dataloader(
            root     = os.path.join(p['data_dir'], "val"),
            split    = "val",
            augment  = False,
            balanced = False,
            drop_last = False,
            **shared,
        )

    def _build_optimiser(self):
        tc = self.cfg['training']

        # Filter to trainable-only params — frozen stages must be excluded here.
        # If frozen params are added to AdamW, it wastes optimizer state memory
        # and can interfere with gradient scaling under AMP.
        encoder_params = [
            p for p in self.model.encoder.parameters() if p.requires_grad
        ]

        # Everything else → higher LR (new layers, learn from scratch)
        new_params = (
            list(self.model.dr_head.parameters())
            + list(self.model.glauc_head.parameters())
            + list(self.model.pm_head.parameters())
            + list(self.model.vib1.parameters())
            + list(self.model.vib2.parameters())
            + list(self.model.aux_classifier.parameters())
            + list(self.model.main_classifier.parameters())
        )
        # Expert heads and VIBs are always new — no requires_grad filter needed.
        # (They are never frozen.)

        self.optimiser = torch.optim.AdamW([
            {"params": encoder_params, "lr": tc['lr'] * 0.1},   # 10× lower for backbone
            {"params": new_params,     "lr": tc['lr']},          # full LR for heads+VIBs
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
        self.start_epoch   = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.logger.info(f"  Resumed at epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # Training / validation step
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, epoch: int, train: bool) -> dict:
        self.model.train(train)

        meters = {k: AverageMeter(k)
                  for k in ("loss", "l_main", "l_aux", "l_kl1", "l_kl2", "l_kl")}
        tc = self.cfg['training']

        with torch.set_grad_enabled(train):
            for batch in loader:
                six_ch       = batch["six_ch"].to(self.device,  non_blocking=True)
                retfound_emb = batch["retfound_emb"].to(self.device, non_blocking=True)
                label_vec    = batch["label_vec"].to(self.device, non_blocking=True)

                with autocast('cuda', dtype=self.amp_type, enabled=self.use_amp):
                    z_fused, z1, mu1, log_var1, mu2, log_var2 = self.model(
                        six_ch, retfound_emb
                    )
                    aux_logits  = self.model.aux_classifier(z1)
                    main_logits = self.model.main_classifier(z_fused)

                    loss, breakdown = self.criterion(
                        main_logits, aux_logits,
                        mu1, log_var1, mu2, log_var2,
                        label_vec, epoch,
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

                B = six_ch.size(0)
                for k in meters:
                    if k in breakdown:
                        meters[k].update(breakdown[k], n=B)

        return {k: m.avg for k, m in meters.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        tc      = self.cfg['training']
        timer   = Timer()
        patience = tc.get('early_stop_patience', 0)   # 0 = disabled
        patience_ctr = 0

        self.logger.info(
            f"Phase 1 training | max_epochs={tc['epochs']} | "
            f"early_stop_patience={patience or 'off'} | "
            f"AMP={'on' if self.use_amp else 'off'} | device={self.device}"
        )

        for epoch in range(self.start_epoch, tc['epochs']):

            # ── Train ──────────────────────────────────────────────────────
            train_metrics = self._run_epoch(self.train_loader, epoch, train=True)
            self.scheduler.step()

            # ── Validate ───────────────────────────────────────────────────
            val_metrics = {}
            if (epoch + 1) % tc['eval_every'] == 0:
                val_metrics = self._run_epoch(self.val_loader, epoch, train=False)

            # ── Logging ────────────────────────────────────────────────────
            lr = self.optimiser.param_groups[1]['lr']   # head LR
            self.logger.info(
                f"Epoch {epoch+1:03d}/{tc['epochs']} | "
                f"lr={lr:.2e} | "
                f"β={self.criterion.beta_sched.get_beta(epoch):.5f} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"main={train_metrics['l_main']:.4f} "
                f"aux={train_metrics['l_aux']:.4f} "
                f"kl1={train_metrics['l_kl1']:.4f} "
                f"kl2={train_metrics['l_kl2']:.4f}"
                + (f" | val_loss={val_metrics.get('loss', 0):.4f}  "
                   f"patience={patience_ctr}/{patience}"
                   if val_metrics and patience else
                   f" | val_loss={val_metrics.get('loss', 0):.4f}"
                   if val_metrics else "")
                + f" | {timer.elapsed()}"
            )

            log_entry = {
                "epoch":          epoch + 1,
                "lr":             lr,
                "patience_ctr":   patience_ctr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
                "beta":           self.criterion.beta_sched.get_beta(epoch),
            }
            self.metrics_log.log(log_entry)

            # ── Checkpoint + early stopping ────────────────────────────────
            is_best  = False
            val_loss = val_metrics.get("loss", float("inf"))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best  = True
                patience_ctr = 0          # improvement → reset counter
            elif val_metrics:             # only count stagnation on eval epochs
                patience_ctr += 1

            if (epoch + 1) % tc['save_every'] == 0 or is_best:
                state = {
                    "epoch":           epoch,
                    "model_state":     self.model.state_dict(),
                    "optimizer_state": self.optimiser.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_val_loss":   self.best_val_loss,
                    "config":          self.cfg,
                }
                fname = f"epoch_{epoch+1:04d}.pt"
                path  = save_checkpoint(
                    state, self.ckpt_dir, filename=fname, is_best=is_best
                )
                if is_best:
                    self.logger.info(
                        f"  ★ New best val_loss={self.best_val_loss:.4f} → {path}"
                    )

            # ── Early stopping check ───────────────────────────────────────
            if patience > 0 and patience_ctr >= patience:
                self.logger.info(
                    f"\n  ⏹  Early stopping triggered at epoch {epoch+1} — "
                    f"val_loss did not improve for {patience} consecutive eval epochs.\n"
                    f"  Best val_loss={self.best_val_loss:.4f} (saved as best.pt)"
                )
                break

        self.logger.info(
            f"Phase 1 complete after {epoch+1} epoch(s). "
            f"Best val_loss={self.best_val_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Phase 1: Train NetrAi SegFormer encoder")
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
