#!/usr/bin/env python3
"""
Train VideoMoE-Tiny on UCF-101.

Usage:
    python scripts/train.py --config configs/train_tiny.yaml

    # Resume from checkpoint:
    python scripts/train.py --config configs/train_tiny.yaml \
        --resume checkpoints/checkpoint_epoch010.pt

    # Override config values:
    python scripts/train.py --config configs/train_tiny.yaml \
        --batch_size 2 --epochs 30
"""

import sys
import os
import argparse
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer
from src.data.ucf101 import build_dataloaders
from src.utils.vram import print_vram, estimate_model_vram, clear_vram
from src.models import VideoMoE


def parse_args():
    parser = argparse.ArgumentParser(description="Train VideoMoE")
    parser.add_argument("--config", type=str, default="configs/train_tiny.yaml")
    parser.add_argument("--resume", type=str, default=None)
    # Allow config overrides from CLI
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--model_size",   type=str,   default=None)
    parser.add_argument("--num_frames",   type=int,   default=None)
    parser.add_argument("--data_root",    type=str,   default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    for key in ["batch_size", "epochs", "lr", "model_size", "num_frames", "data_root"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
            print(f"[Config] Override: {key} = {val}")

    print("\n[Config]")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    # Pre-flight VRAM check
    if torch.cuda.is_available():
        model_check = VideoMoE(
            num_classes=cfg["num_classes"],
            model_size=cfg.get("model_size", "tiny"),
            num_frames=cfg["num_frames"],
        )
        vram_est = estimate_model_vram(
            model_check,
            input_shape=(3, cfg["num_frames"], cfg["image_size"], cfg["image_size"]),
            batch_size=cfg["batch_size"],
        )
        del model_check
        clear_vram()

        print(f"[VRAM Estimate]")
        print(f"  Parameters:  {vram_est['param_mb']} MB")
        print(f"  Activations: {vram_est['activation_mb']} MB")
        print(f"  Total:       {vram_est['total_mb']} MB")
        print(f"  Fits 8GB:    {'✓ YES' if vram_est['fits_8gb'] else '✗ NO — reduce batch_size or num_frames'}")
        print()

        if not vram_est["fits_8gb"]:
            print("[WARN] VRAM estimate exceeds 7.5GB. Reduce batch_size or enable gradient_checkpointing.")

    # Build data
    print("[Data] Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)

    # Update steps_per_epoch in cfg for scheduler
    cfg["steps_per_epoch"] = len(train_loader)

    # Build trainer
    trainer = Trainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print_vram("Before training")

    # Train!
    trainer.fit(train_loader, val_loader)

    print_vram("After training")


if __name__ == "__main__":
    main()
