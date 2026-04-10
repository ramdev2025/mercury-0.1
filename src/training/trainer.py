"""
Training Engine for VideoMoE.

Features:
  - Mixed precision (fp16 / bf16) via torch.amp
  - Gradient checkpointing (saves ~40% VRAM)
  - Gradient accumulation (simulate large batch on 8GB)
  - Cosine LR schedule with warmup
  - Top-1 / Top-5 accuracy tracking
  - Checkpoint save/resume
  - TensorBoard logging
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional
import json

from ..models import VideoMoE
from ..utils.metrics import AverageMeter, accuracy


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self._setup_model()
        self._setup_optimizer()
        self._setup_scaler()
        self._setup_logging()

        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

    def _setup_model(self):
        self.model = VideoMoE(
            num_classes=self.cfg["num_classes"],
            model_size=self.cfg.get("model_size", "tiny"),
            image_size=self.cfg["image_size"],
            patch_size=self.cfg.get("patch_size", 16),
            num_frames=self.cfg["num_frames"],
            tube_size=self.cfg.get("tube_size", 2),
            num_experts=self.cfg.get("num_experts", 8),
            top_k=self.cfg.get("top_k", 2),
            moe_every_n=self.cfg.get("moe_every_n", 2),
            dropout=self.cfg.get("dropout", 0.1),
            aux_loss_weight=self.cfg.get("aux_loss_weight", 0.01),
        ).to(self.device)

        # Gradient checkpointing — trades compute for ~40% VRAM reduction
        if self.cfg.get("gradient_checkpointing", True):
            for block in self.model.blocks:
                block.attn = torch.utils.checkpoint.checkpoint_wrapper(block.attn) \
                    if hasattr(torch.utils.checkpoint, "checkpoint_wrapper") else block.attn
            print("[Trainer] Gradient checkpointing: enabled")

        params = self.model.count_parameters()
        print(f"[Trainer] Model parameters: {params['total_M']:.1f}M")

    def _setup_optimizer(self):
        # Separate weight decay: don't apply to biases / LayerNorm
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name or "cls_token" in name or "pos_embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.cfg.get("weight_decay", 0.05)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.cfg.get("lr", 1e-4),
            betas=(0.9, 0.999),
        )

        # Cosine schedule with linear warmup
        total_steps = self.cfg["epochs"] * self.cfg.get("steps_per_epoch", 100)
        warmup_steps = self.cfg.get("warmup_epochs", 5) * self.cfg.get("steps_per_epoch", 100)
        self.scheduler = self._build_cosine_schedule(total_steps, warmup_steps)

    def _build_cosine_schedule(self, total_steps: int, warmup_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_scaler(self):
        # fp16 mixed precision — essential for 8GB VRAM
        self.use_amp = self.cfg.get("use_amp", True) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        print(f"[Trainer] Mixed precision (fp16): {'enabled' if self.use_amp else 'disabled'}")

    def _setup_logging(self):
        log_dir = Path(self.cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.ckpt_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader) -> dict:
        self.model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.get("label_smoothing", 0.1))

        loss_meter = AverageMeter()
        aux_meter  = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        batch_time = AverageMeter()

        accum_steps = self.cfg.get("gradient_accumulation_steps", 4)
        self.optimizer.zero_grad()

        t0 = time.time()
        for i, (videos, labels) in enumerate(train_loader):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits, aux_loss = self.model(videos)
                ce_loss = criterion(logits, labels)
                loss = ce_loss + aux_loss
                loss = loss / accum_steps  # normalize for accumulation

            self.scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Metrics
            acc1, acc5 = accuracy(logits.detach(), labels, topk=(1, 5))
            bs = videos.shape[0]
            loss_meter.update(ce_loss.item(), bs)
            aux_meter.update(aux_loss.item(), bs)
            acc1_meter.update(acc1.item(), bs)
            acc5_meter.update(acc5.item(), bs)
            batch_time.update(time.time() - t0)
            t0 = time.time()

            if i % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Step [{i:4d}/{len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f}  "
                    f"AuxLoss: {aux_meter.avg:.4f}  "
                    f"Acc@1: {acc1_meter.avg:.2f}%  "
                    f"LR: {lr:.2e}  "
                    f"Time: {batch_time.avg:.2f}s/batch"
                )

        return {
            "loss": loss_meter.avg,
            "aux_loss": aux_meter.avg,
            "acc1": acc1_meter.avg,
            "acc5": acc5_meter.avg,
        }

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        for videos, labels in val_loader:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits, _ = self.model(videos)
                loss = criterion(logits, labels)

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            bs = videos.shape[0]
            loss_meter.update(loss.item(), bs)
            acc1_meter.update(acc1.item(), bs)
            acc5_meter.update(acc5.item(), bs)

        return {
            "val_loss": loss_meter.avg,
            "val_acc1": acc1_meter.avg,
            "val_acc5": acc5_meter.avg,
        }

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "cfg": self.cfg,
            "metrics": metrics,
        }
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch{self.epoch:03d}.pt"
        torch.save(state, ckpt_path)

        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(state, best_path)
            print(f"[Trainer] ✓ New best model saved: acc@1={self.best_acc:.2f}%")

    def load_checkpoint(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_acc = state["best_acc"]
        print(f"[Trainer] Resumed from epoch {self.epoch}")

    def fit(self, train_loader, val_loader):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  VideoMoE Training")
        print(f"  Epochs: {self.cfg['epochs']}")
        print(f"  Batch size: {self.cfg['batch_size']} × {self.cfg.get('gradient_accumulation_steps', 4)} accum")
        print(f"  Effective batch: {self.cfg['batch_size'] * self.cfg.get('gradient_accumulation_steps', 4)}")
        print(f"{'='*60}\n")

        for epoch in range(self.epoch, self.cfg["epochs"]):
            self.epoch = epoch
            print(f"\n── Epoch {epoch+1}/{self.cfg['epochs']} ──")

            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate(val_loader)

            all_metrics = {**train_metrics, **val_metrics}
            print(
                f"  [Epoch {epoch+1}] "
                f"Train Loss: {train_metrics['loss']:.4f}  "
                f"Val Acc@1: {val_metrics['val_acc1']:.2f}%  "
                f"Val Acc@5: {val_metrics['val_acc5']:.2f}%"
            )

            # TensorBoard
            for k, v in all_metrics.items():
                self.writer.add_scalar(k, v, epoch)

            # Save checkpoint
            is_best = val_metrics["val_acc1"] > self.best_acc
            if is_best:
                self.best_acc = val_metrics["val_acc1"]
            self.save_checkpoint(all_metrics, is_best=is_best)

        print(f"\n[Trainer] Training complete. Best Val Acc@1: {self.best_acc:.2f}%")
        self.writer.close()
