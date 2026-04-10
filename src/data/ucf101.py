"""
UCF-101 Dataset Loader.

Downloads and preprocesses UCF-101 for MercuryMoE training.
UCF-101: 101 action classes, ~13,000 clips, 25 FPS, variable length.

Usage:
    from src.data.ucf101 import UCF101Dataset, build_dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
"""

import os
import random
import decord
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, List, Optional
import json


# ─── Normalization constants (ImageNet mean/std — standard for video models) ──
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class VideoAugment:
    """
    Spatial + temporal augmentations for training.
    Applied per-clip, consistent across frames.
    """

    def __init__(self, image_size: int = 224, is_train: bool = True):
        if is_train:
            self.spatial = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.spatial = transforms.Compose([
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __call__(self, frames: List) -> torch.Tensor:
        """
        Args:
            frames: list of PIL images
        Returns:
            tensor: [C, T, H, W]
        """
        from PIL import Image
        # Apply same random state to all frames for consistency
        seed = random.randint(0, 2**31)
        out = []
        for frame in frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            random.seed(seed)
            torch.manual_seed(seed)
            out.append(self.spatial(frame))
        return torch.stack(out, dim=1)  # [C, T, H, W]


class UCF101Dataset(Dataset):
    """
    UCF-101 video dataset.

    Directory structure expected:
        data/
          ucf101/
            videos/
              ApplyEyeMakeup/
                v_ApplyEyeMakeup_g01_c01.avi
                ...
              ...
            splits/
              trainlist01.txt
              testlist01.txt
            classInd.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "val" | "test"
        split_id: int = 1,             # UCF has 3 train/test splits
        num_frames: int = 8,
        image_size: int = 224,
        frame_stride: int = 4,         # sample every N frames
        max_clips_per_video: int = 1,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.frame_stride = frame_stride

        self.augment = VideoAugment(image_size, is_train=(split == "train"))

        # Load class index
        self.class_to_idx = self._load_class_index()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load file list
        self.samples = self._load_split(split, split_id)

        print(f"[UCF101] {split} split: {len(self.samples)} clips, {len(self.class_to_idx)} classes")

    def _load_class_index(self) -> dict:
        class_file = self.root / "classInd.txt"
        class_to_idx = {}
        if class_file.exists():
            with open(class_file) as f:
                for line in f:
                    idx, name = line.strip().split()
                    class_to_idx[name] = int(idx) - 1  # 0-indexed
        else:
            # Auto-discover from folder structure
            video_dir = self.root / "videos"
            classes = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])
            class_to_idx = {c: i for i, c in enumerate(classes)}
        return class_to_idx

    def _load_split(self, split: str, split_id: int) -> List[Tuple[Path, int]]:
        split_dir = self.root / "splits"
        if split == "train":
            split_file = split_dir / f"trainlist0{split_id}.txt"
        else:
            split_file = split_dir / f"testlist0{split_id}.txt"

        samples = []
        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    rel_path = parts[0]
                    video_path = self.root / "videos" / rel_path
                    class_name = rel_path.split("/")[0]
                    label = self.class_to_idx.get(class_name, 0)
                    if video_path.exists():
                        samples.append((video_path, label))
        else:
            # Fallback: auto-scan directory
            video_dir = self.root / "videos"
            for class_dir in sorted(video_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = self.class_to_idx.get(class_dir.name, 0)
                for video_file in class_dir.glob("*.avi"):
                    samples.append((video_file, label))
            # 80/20 train/val split if no split files
            random.seed(42)
            random.shuffle(samples)
            n = int(len(samples) * 0.8)
            samples = samples[:n] if split == "train" else samples[n:]

        return samples

    def _load_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """Load and sample frames from video using decord."""
        try:
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            total_frames = len(vr)

            # Sample `num_frames` frames uniformly
            if total_frames <= self.num_frames:
                frame_ids = list(range(total_frames))
                # Repeat last frame to fill
                while len(frame_ids) < self.num_frames:
                    frame_ids.append(frame_ids[-1])
            else:
                # Uniformly sample with stride
                start = random.randint(0, max(0, total_frames - self.num_frames * self.frame_stride))
                frame_ids = [start + i * self.frame_stride for i in range(self.num_frames)]
                frame_ids = [min(f, total_frames - 1) for f in frame_ids]

            frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
            return frames
        except Exception as e:
            print(f"[WARN] Failed to load {video_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)

        if frames is None:
            # Return black video on error
            return torch.zeros(3, self.num_frames, self.image_size, self.image_size), label

        # [T, H, W, C] → list of numpy arrays
        frame_list = [frames[i] for i in range(len(frames))]
        video_tensor = self.augment(frame_list)  # [C, T, H, W]

        return video_tensor, label


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config dict.

    Args:
        cfg: config dict with keys:
            data_root, num_frames, image_size, batch_size,
            num_workers, frame_stride
    """
    train_dataset = UCF101Dataset(
        root=cfg["data_root"],
        split="train",
        split_id=cfg.get("split_id", 1),
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        frame_stride=cfg.get("frame_stride", 4),
    )
    val_dataset = UCF101Dataset(
        root=cfg["data_root"],
        split="val",
        split_id=cfg.get("split_id", 1),
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        frame_stride=cfg.get("frame_stride", 4),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader
