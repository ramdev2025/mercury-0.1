#!/usr/bin/env python3
"""
Download and prepare UCF-101 dataset.

Usage:
    python scripts/download_data.py --data_dir data/ucf101

Downloads:
  - UCF-101 videos (~7.2 GB)
  - Train/test split files
  - Class index file

After download:
  data/ucf101/
    videos/      ← extracted video folders
    splits/      ← train/test split txt files
    classInd.txt ← class name → index mapping
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


UCF101_URL     = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF101_SPLITS  = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

# Mirror (use if main URL is slow)
MIRROR_URL = "https://storage.googleapis.com/thumos14_files/UCF101_videos.rar"


def download_with_progress(url: str, dest: Path, label: str = ""):
    """Download a file with a progress bar."""
    print(f"Downloading {label or url}")
    print(f"  → {dest}")

    def reporthook(block, block_size, total_size):
        if total_size > 0:
            pct = min(100, block * block_size * 100 / total_size)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def extract_archive(archive_path: Path, dest: Path):
    print(f"Extracting {archive_path.name}...")
    dest.mkdir(parents=True, exist_ok=True)

    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif suffix in [".tar", ".gz", ".bz2"]:
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)
    elif suffix == ".rar":
        # Try using unrar or 7z
        ret = os.system(f"unrar x '{archive_path}' '{dest}/' -y")
        if ret != 0:
            ret = os.system(f"7z x '{archive_path}' -o'{dest}/' -y")
        if ret != 0:
            print("[ERROR] Could not extract .rar file.")
            print("  Install: sudo apt install unrar  OR  sudo apt install p7zip-full")
            print(f"  Then manually run: unrar x {archive_path} {dest}/")
            sys.exit(1)
    print(f"  Extracted to {dest}")


def create_mock_dataset(data_dir: Path, num_classes: int = 5, clips_per_class: int = 3):
    """
    Create a tiny mock UCF-101 dataset with synthetic videos for testing
    (no download needed). Uses OpenCV to create random-pixel videos.
    """
    print(f"\n[Mock] Creating synthetic dataset with {num_classes} classes, {clips_per_class} clips each...")
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[ERROR] opencv-python not installed. Run: pip install opencv-python")
        return

    class_names = [
        "Archery", "BabyGolf", "Basketball", "Biking", "Boxing",
        "CliffDiving", "Cricket", "Diving", "Fencing", "Golf",
    ][:num_classes]

    videos_dir = data_dir / "videos"
    splits_dir = data_dir / "splits"
    videos_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Write classInd.txt
    with open(data_dir / "classInd.txt", "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i+1} {name}\n")

    train_lines = []
    test_lines  = []

    for c_idx, class_name in enumerate(class_names):
        class_dir = videos_dir / class_name
        class_dir.mkdir(exist_ok=True)

        for clip_idx in range(clips_per_class):
            filename = f"v_{class_name}_g01_c{clip_idx+1:02d}.avi"
            filepath = class_dir / filename

            # Write a 2-second 10fps synthetic video
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(filepath), fourcc, 10.0, (224, 224))
            for _ in range(20):
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            rel_path = f"{class_name}/{filename}"
            if clip_idx < clips_per_class - 1:
                train_lines.append(f"{rel_path} {c_idx+1}")
            else:
                test_lines.append(f"{rel_path}")

    with open(splits_dir / "trainlist01.txt", "w") as f:
        f.write("\n".join(train_lines))
    with open(splits_dir / "testlist01.txt", "w") as f:
        f.write("\n".join(test_lines))

    print(f"[Mock] Dataset created at {data_dir}")
    print(f"  Classes: {num_classes}")
    print(f"  Train clips: {len(train_lines)}")
    print(f"  Test clips:  {len(test_lines)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ucf101")
    parser.add_argument("--mock", action="store_true",
                        help="Create a small mock dataset for testing (no download)")
    parser.add_argument("--mock_classes", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        create_mock_dataset(data_dir, num_classes=args.mock_classes)
        return

    # Real download
    print("=" * 60)
    print("  UCF-101 Dataset Download")
    print("=" * 60)
    print(f"  Destination: {data_dir.resolve()}")
    print(f"  Size: ~7.2 GB (videos) + ~1 MB (splits)")
    print()

    proceed = input("Proceed? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Aborted.")
        return

    # Download splits (small, always get these first)
    splits_zip = data_dir / "splits.zip"
    if not splits_zip.exists():
        download_with_progress(UCF101_SPLITS, splits_zip, label="Train/Test Splits")
    extract_archive(splits_zip, data_dir / "splits")

    # Download videos
    video_rar = data_dir / "UCF101.rar"
    if not video_rar.exists():
        download_with_progress(UCF101_URL, video_rar, label="UCF-101 Videos")
    extract_archive(video_rar, data_dir / "videos")

    print("\nDataset ready!")
    print(f"  Path: {data_dir.resolve()}")
    print("  Next: python scripts/train.py --config configs/train_tiny.yaml")


if __name__ == "__main__":
    main()
