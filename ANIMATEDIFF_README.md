# MercuryMoE + AnimateDiff - Music Video Generator 🎬🎵

## Overview

This project combines **MercuryMoE** (sparse MoE architecture) with **AnimateDiff** for generating animated music videos and short scene films.

### What You Get

- **Text-to-Video Generation**: Generate video clips from text prompts
- **Music Synchronization**: Sync video motion to audio beats/rhythm
- **MoE Efficiency**: Sparse mixture of experts for faster inference
- **Modal L4 GPU Ready**: Deploy on Modal.com with 24GB VRAM

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Music Video Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Text Prompt ──► Stable Diffusion UNet ──► Latent Frames    │
│       ▲                │                                      │
│       │                ▼                                      │
│  Audio Input ──► Audio Encoder ──► Motion Embeddings         │
│                       │                                       │
│                       ▼                                       │
│              AnimateDiff Temporal Layers                      │
│                   (MoE Enhanced)                              │
│                       │                                       │
│                       ▼                                       │
│              VAE Decoder ──► Video Output                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Stable Diffusion Backbone**: Pre-trained SD 1.5 or SDXL
2. **AnimateDiff Motion Modules**: Temporal attention layers for smooth motion
3. **MoE FFN Layers**: Sparse experts for efficient computation
4. **Audio Encoder**: Extract beat/rhythm features from music
5. **Cross-Modal Attention**: Sync video motion to audio

---

## Quick Start

### 1. Setup Modal Environment

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Create secrets (paste your API keys when prompted)
modal secret create mercury-moe-secrets \
    MODAL_API_KEY=<your_key> \
    HF_TOKEN=<your_huggingface_token>
```

### 2. Upload Music & Reference Data

```bash
# Upload your music files
python upload_data.py --data_dir music/ --type audio

# Upload reference images/videos (optional)
python upload_data.py --data_dir references/ --type images
```

### 3. Generate Music Video

```bash
# Basic generation from text prompt
modal run modal_app.py::generate \
    --prompt "cyberpunk city at night, neon lights, flying cars" \
    --audio music/track.mp3 \
    --duration 8

# With custom parameters
modal run modal_app.py::generate \
    --prompt "anime style forest, magical particles, flowing water" \
    --audio music/track.mp3 \
    --duration 16 \
    --fps 24 \
    --motion_strength 0.8 \
    --audio_sync_intensity 0.7
```

### 4. Advanced Features

```bash
# Generate with style reference image
modal run modal_app.py::generate \
    --prompt "warrior princess in battle" \
    --audio music/epic.mp3 \
    --style_image references/style.png \
    --duration 12

# Batch generate multiple scenes
modal run modal_app.py::batch_generate \
    --scenes-json scenes.json \
    --audio music/album.mp3

# Extend existing video
modal run modal_app.py::extend_video \
    --input_video checkpoints/scene1.mp4 \
    --prompt "continue the chase scene through city" \
    --extension_seconds 4
```

---

## Configuration

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | required | Text description of the scene |
| `--audio` | optional | Path to music file (MP3/WAV) |
| `--duration` | 8 | Video duration in seconds |
| `--fps` | 24 | Frames per second |
| `--resolution` | 512x512 | Video resolution |
| `--motion_strength` | 0.7 | Motion intensity (0.0-1.0) |
| `--audio_sync_intensity` | 0.5 | Audio-reactive motion (0.0-1.0) |
| `--negative_prompt` | "" | Things to avoid in generation |
| `--seed` | random | Random seed for reproducibility |

### Model Selection

Choose your base model in `configs/generation.yaml`:

```yaml
base_model: "runwayml/stable-diffusion-v1-5"  # or SDXL
motion_module: "guoyww/animatediff/mm_sd_v15_v2.ckpt"
use_moe: true
num_experts: 8
top_k: 2
```

---

## Project Structure

```
workspace/
├── modal_app.py              # Modal deployment (train/generate/shell)
├── src/
│   ├── models/
│   │   ├── video_moe.py      # MercuryMoE classifier (original)
│   │   ├── moe.py            # MoE layers
│   │   ├── animatediff.py    # AnimateDiff temporal modules ⭐ NEW
│   │   ├── audio_sync.py     # Audio synchronization ⭐ NEW
│   │   └── generator.py      # Full generation pipeline ⭐ NEW
│   ├── data/
│   │   ├── ucf101.py         # Classification dataset
│   │   └── audio_loader.py   # Audio processing ⭐ NEW
│   └── utils/
│       ├── vram.py           # VRAM monitoring
│       └── video_io.py       # Video encoding/decoding ⭐ NEW
├── configs/
│   ├── train_tiny.yaml       # Classification training
│   └── generation.yaml       # Video generation config ⭐ NEW
├── scripts/
│   ├── train.py              # Train classifier
│   ├── evaluate.py           # Evaluate classifier
│   └── generate.py           # Generate videos ⭐ NEW
├── music/                    # Your music files (uploaded to volume)
├── references/               # Style reference images
└── outputs/                  # Generated videos (persistent volume)
```

---

## Use Cases

### 🎵 Music Videos
```bash
modal run modal_app.py::generate \
    --prompt "silhouette dancer, colorful light trails, abstract background" \
    --audio music/edm_track.mp3 \
    --audio_sync_intensity 0.9 \
    --duration 30
```

### 🎬 Short Films
```bash
modal run modal_app.py::generate \
    --prompt "detective walking through rainy noir street at night" \
    --duration 12 \
    --motion_strength 0.6 \
    --negative_prompt "cartoon, anime, low quality"
```

### 🎨 Animated Scenes
```bash
modal run modal_app.py::generate \
    --prompt "studio ghibli style meadow, wind blowing grass, clouds moving" \
    --style_image references/ghibli_style.png \
    --duration 8 \
    --fps 24
```

---

## Performance on L4 GPU

| Resolution | Duration | VRAM Usage | Generation Time |
|------------|----------|------------|-----------------|
| 512×512 | 8s (192 frames) | ~18 GB | ~2 minutes |
| 512×512 | 16s (384 frames) | ~20 GB | ~4 minutes |
| 768×768 | 8s (192 frames) | ~22 GB | ~3 minutes |

**Tips for L4 (24GB VRAM)**:
- Use `--resolution 512x512` for longer videos
- Enable `--use_fp16` for memory savings
- Split long songs into 30s segments

---

## Training Custom Motion Models

You can fine-tune AnimateDiff on your own video datasets:

```bash
modal run modal_app.py::train_motion \
    --data_dir /workspace/data/custom_videos \
    --base_model "guoyww/animatediff/mm_sd_v15_v2.ckpt" \
    --epochs 100 \
    --batch_size 2
```

---

## API Keys & Secrets

Required secrets in Modal:

```bash
modal secret create mercury-moe-secrets \
    MODAL_API_KEY="<your_modal_key>" \
    HF_TOKEN="<your_huggingface_token>" \
    WANDB_API_KEY="<optional_wandb_key>"
```

- **HF_TOKEN**: Download AnimateDiff weights from HuggingFace
- **WANDB_API_KEY**: Optional training metrics tracking

---

## Troubleshooting

### OOM (Out of Memory)
- Reduce `--duration` or `--resolution`
- Lower `--fps` (try 12 or 16 instead of 24)
- Use `--use_fp16` flag

### Slow Generation
- L4 GPUs are shared; consider dedicated instance
- Reduce number of inference steps (--steps 20 instead of 50)
- Use smaller motion module

### Audio Sync Issues
- Increase `--audio_sync_intensity`
- Ensure audio has clear beat structure
- Try different audio encoder in config

---

## References

- [AnimateDiff](https://github.com/guoyww/AnimateDiff) - Motion module for SD
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Base image model
- [AudioLDM](https://github.com/haoheliu/AudioLDM) - Audio understanding
- [MercuryMoE](./README.md) - Original MoE architecture

---

## Next Steps

1. ✅ Generate your first music video
2. 🎨 Experiment with different styles and prompts
3. 🎵 Test audio synchronization with various genres
4. 🔧 Fine-tune on your custom dataset
5. 🚀 Deploy as API endpoint with `modal serve`

Happy creating! 🎬✨
