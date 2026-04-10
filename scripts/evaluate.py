#!/usr/bin/env python3
"""
Evaluate a trained MercuryMoE checkpoint.

Usage:
    # Evaluate on UCF-101 test set
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

    # Run inference on a single video
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \
        --video path/to/video.mp4

    # Analyze MoE expert utilization
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \
        --analyze_experts
"""

import sys
import os
import argparse
import yaml
import torch
import json
from pathlib import Path
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MercuryMoE
from src.data.ucf101 import UCF101Dataset, build_dataloaders
from src.utils.metrics import AverageMeter, accuracy
from src.utils.vram import print_vram


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> tuple:
    """Load model and config from a checkpoint file."""
    state = torch.load(ckpt_path, map_location=device)
    cfg   = state["cfg"]

    model = MercuryMoE(
        num_classes=cfg["num_classes"],
        model_size=cfg.get("model_size", "tiny"),
        image_size=cfg["image_size"],
        num_frames=cfg["num_frames"],
        num_experts=cfg.get("num_experts", 8),
        top_k=cfg.get("top_k", 2),
    ).to(device)

    model.load_state_dict(state["model_state"])
    model.eval()

    print(f"[Eval] Loaded checkpoint from epoch {state['epoch']}")
    print(f"[Eval] Best val acc@1: {state.get('best_acc', 'N/A')}")
    return model, cfg


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader, device: torch.device, use_amp: bool = True) -> dict:
    """Full evaluation loop."""
    criterion = torch.nn.CrossEntropyLoss()
    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()

    model.eval()
    for videos, labels in val_loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits, _ = model(videos)
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        bs = videos.shape[0]
        loss_m.update(loss.item(), bs)
        acc1_m.update(acc1.item(), bs)
        acc5_m.update(acc5.item(), bs)

    return {
        "val_loss": round(loss_m.avg, 4),
        "val_acc1": round(acc1_m.avg, 2),
        "val_acc5": round(acc5_m.avg, 2),
    }


def infer_video(model: torch.nn.Module, video_path: str, cfg: dict, device: torch.device) -> dict:
    """Run inference on a single video file."""
    from src.data.ucf101 import UCF101Dataset, VideoAugment
    import decord
    import numpy as np

    augment = VideoAugment(image_size=cfg["image_size"], is_train=False)

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    num_frames = cfg["num_frames"]

    frame_ids = [int(i * total / num_frames) for i in range(num_frames)]
    frames = vr.get_batch(frame_ids).asnumpy()
    frame_list = [frames[i] for i in range(len(frames))]

    video_tensor = augment(frame_list).unsqueeze(0).to(device)  # [1, C, T, H, W]

    model.eval()
    with torch.no_grad(), autocast():
        logits, _ = model(video_tensor)
        probs = torch.softmax(logits, dim=-1)

    top5_probs, top5_idx = probs[0].topk(5)

    return {
        "top5_predictions": [
            {"class_idx": idx.item(), "confidence": prob.item()}
            for idx, prob in zip(top5_idx, top5_probs)
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--video",           type=str, default=None)
    parser.add_argument("--analyze_experts", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)

    print_vram("After model load")

    if args.video:
        print(f"\n[Infer] Running inference on: {args.video}")
        result = infer_video(model, args.video, cfg, device)
        print("\nTop-5 predictions:")
        for i, pred in enumerate(result["top5_predictions"]):
            print(f"  {i+1}. Class {pred['class_idx']:3d}  — {pred['confidence']*100:.1f}%")
        return

    # Full evaluation
    print("\n[Eval] Building val dataloader...")
    _, val_loader = build_dataloaders(cfg)

    print("[Eval] Running evaluation...")
    metrics = evaluate(model, val_loader, device, use_amp=cfg.get("use_amp", True))

    print("\n" + "="*40)
    print("  Evaluation Results")
    print("="*40)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("="*40)

    # Save results
    results_path = Path("logs") / "eval_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Results saved to {results_path}")


if __name__ == "__main__":
    main()
