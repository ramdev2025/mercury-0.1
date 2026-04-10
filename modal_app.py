#!/usr/bin/env python3
"""
VideoMoE on Modal — L4 GPU Environment

Usage:
    # Deploy and run training:
    modal run modal_app.py::main

    # Run with custom config:
    modal run modal_app.py::main --config configs/train_tiny.yaml

    # Run evaluation:
    modal run modal_app.py::main --evaluate-checkpoint checkpoints/best_model.pt

    # Open a shell for debugging:
    modal shell modal_app.py
"""

import modal
from pathlib import Path

# ── App Definition ─────────────────────────────────────────────
app = modal.App("video-moe")

# ── Local directory for mounting code ──────────────────────────
LOCAL_DIR = Path(__file__).parent

# ── Secrets ────────────────────────────────────────────────────
# Load secrets from Modal (API keys, etc.)
# Create with: modal secret create videomoe-secrets MODAL_API_KEY=<your_key>
try:
    secrets = [modal.Secret.from_name("videomoe-secrets")]
except Exception:
    # Fallback if secret doesn't exist yet
    secrets = []

# ── GPU Image Setup ────────────────────────────────────────────
# Use NVIDIA CUDA 12.1 base image (compatible with L4 GPUs)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "unrar",
        "p7zip-full",
    )
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "decord>=0.6.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "ipykernel>=6.25.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        # AnimateDiff & Generation
        "diffusers>=0.24.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "einops>=0.7.0",
        "xformers>=0.0.23",
        # Audio Processing
        "librosa>=0.10.0",
        "scipy>=1.11.0",
    )
    .workdir("/workspace")
    .copy_local_dir(LOCAL_DIR, "/workspace")
    .run_commands("pip install -e .")
)

# ── Volume for persistent storage ──────────────────────────────
# Mount volumes for data, checkpoints, and logs
data_volume = modal.Volume.from_name("video-moe-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("video-moe-checkpoints", create_if_missing=True)
logs_volume = modal.Volume.from_name("video-moe-logs", create_if_missing=True)
outputs_volume = modal.Volume.from_name("video-moe-outputs", create_if_missing=True)
music_volume = modal.Volume.from_name("video-moe-music", create_if_missing=True)


# ── Training Function ──────────────────────────────────────────
@app.function(
    gpu=modal.gpu.L4(count=1),
    image=image,
    secrets=secrets,
    volumes={
        "/workspace/data": data_volume,
        "/workspace/checkpoints": checkpoints_volume,
        "/workspace/logs": logs_volume,
    },
    timeout=86400,  # 24 hours max runtime
)
def train(config_path: str = "configs/train_tiny.yaml"):
    """Run training on L4 GPU."""
    import subprocess
    import sys

    print(f"🚀 Starting training with config: {config_path}")
    print(f"📦 Working directory: /workspace")

    # Run training script
    result = subprocess.run(
        ["python", "scripts/train.py", "--config", config_path],
        cwd="/workspace",
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print("✅ Training completed successfully!")

    # Commit volumes to persist data
    data_volume.commit()
    checkpoints_volume.commit()
    logs_volume.commit()


# ── Video Generation Function ──────────────────────────────────
@app.function(
    gpu=modal.gpu.L4(count=1),
    image=image,
    secrets=secrets,
    volumes={
        "/workspace/checkpoints": checkpoints_volume,
        "/workspace/outputs": outputs_volume,
        "/workspace/music": music_volume,
    },
    timeout=3600,  # 1 hour max for generation
)
def generate(
    prompt: str,
    audio_path: str = "",
    duration: int = 8,
    fps: int = 24,
    resolution: str = "512x512",
    motion_strength: float = 0.7,
    audio_sync_intensity: float = 0.5,
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly",
    seed: int = -1,
):
    """Generate video from text prompt with optional audio sync."""
    import subprocess
    import sys
    
    print(f"🎬 Generating video: '{prompt}'")
    print(f"🎵 Audio: {audio_path if audio_path else 'None'}")
    print(f"⏱️  Duration: {duration}s @ {fps}fps")
    
    cmd = [
        "python", "scripts/generate.py",
        "--prompt", prompt,
        "--duration", str(duration),
        "--fps", str(fps),
        "--resolution", resolution,
        "--motion-strength", str(motion_strength),
        "--audio-sync-intensity", str(audio_sync_intensity),
        "--negative-prompt", negative_prompt,
    ]
    
    if audio_path:
        cmd.extend(["--audio", audio_path])
    
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    
    result = subprocess.run(
        cmd,
        cwd="/workspace",
        capture_output=False,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ Generation failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print("✅ Video generation completed!")
    outputs_volume.commit()


# ── Evaluation Function ────────────────────────────────────────
@app.function(
    gpu=modal.gpu.L4(count=1),
    image=image,
    secrets=secrets,
    volumes={
        "/workspace/data": data_volume,
        "/workspace/checkpoints": checkpoints_volume,
        "/workspace/logs": logs_volume,
    },
    timeout=3600,
)
def evaluate(checkpoint_path: str, config_path: str = "configs/train_tiny.yaml"):
    """Run evaluation on L4 GPU."""
    import subprocess
    import sys

    print(f"🔍 Starting evaluation with checkpoint: {checkpoint_path}")

    result = subprocess.run(
        ["python", "scripts/evaluate.py", "--config", config_path, "--checkpoint", checkpoint_path],
        cwd="/workspace",
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Evaluation failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print("✅ Evaluation completed successfully!")


# ── Shell for Interactive Debugging ────────────────────────────
@app.function(
    gpu=modal.gpu.L4(count=1),
    image=image,
    secrets=secrets,
    volumes={
        "/workspace/data": data_volume,
        "/workspace/checkpoints": checkpoints_volume,
        "/workspace/logs": logs_volume,
    },
)
def shell():
    """Open an interactive shell for debugging."""
    import pty
    import subprocess

    print("🐚 Opening interactive shell...")
    subprocess.run(["bash"], cwd="/workspace")


# ── Local Entry Point ──────────────────────────────────────────
@app.local_entrypoint()
def main(
    config: str = "configs/train_tiny.yaml",
    evaluate_checkpoint: str = "",
    prompt: str = "",
    audio: str = "",
    duration: int = 8,
    fps: int = 24,
):
    """Main entry point for modal run."""
    if evaluate_checkpoint:
        evaluate.remote(evaluate_checkpoint, config)
    elif prompt:
        generate.remote(
            prompt=prompt,
            audio_path=audio,
            duration=duration,
            fps=fps,
        )
    else:
        train.remote(config)
