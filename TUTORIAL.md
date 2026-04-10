# 🎬 MercuryMoE Music Video Generator - Complete Setup Guide

A complete guide to setting up and deploying your **MoE-powered AI video generation system** for music videos and short films on **Modal.com** using **L4 GPUs**.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Local Environment Setup](#step-1-local-environment-setup)
3. [Step 2: Modal Account & Authentication](#step-2-modal-account--authentication)
4. [Step 3: Configure Secrets](#step-3-configure-secrets)
5. [Step 4: Download Training Data](#step-4-download-training-data)
6. [Step 5: Preprocess & Ingest Data](#step-5-preprocess--ingest-data)
7. [Step 6: Train the Model](#step-6-train-the-model)
8. [Step 7: Generate Music Videos](#step-7-generate-music-videos)
9. [Step 8: Monitoring & Debugging](#step-8-monitoring--debugging)
10. [Troubleshooting](#troubleshooting)
11. [Cost Optimization Tips](#cost-optimization-tips)

---

## Prerequisites

Before starting, ensure you have:

- ✅ A **Modal.com account** (sign up at [modal.com](https://modal.com))
- ✅ A **Hugging Face account** (for model weights)
- ✅ Basic familiarity with terminal/command line
- ✅ ~$10-20 in Modal credits for testing (L4 GPUs cost ~$0.0003/sec)

---

## Step 1: Local Environment Setup

### 1.1 Install Python Dependencies

```bash
# Navigate to your project directory
cd /workspace

# Install Modal CLI and core dependencies
pip install modal moviepy openai-whisper transformers accelerate pandas torch torchvision torchaudio

# Verify installation
modal --version
```

### 1.2 Project Structure Overview

Your project should look like this:

```
/workspace/
├── modal_app.py              # Main Modal application
├── src/
│   ├── models/
│   │   ├── animatediff.py    # AnimateDiff + MoE architecture
│   │   └── audio_sync.py     # Audio synchronization
│   └── data/
│       └── preprocess.py     # Data preprocessing pipeline
├── configs/
│   ├── generation.yaml       # Generation config
│   └── data_config.yaml      # Data config
├── scripts/
│   ├── generate.py           # Generation script
│   └── ingest_data.py        # Data ingestion script
├── data/
│   └── raw/                  # Raw video files (auto-created)
├── music/                    # Audio tracks for generation
├── references/               # Style reference images
└── outputs/                  # Generated videos
```

---

## Step 2: Modal Account & Authentication

### 2.1 Login to Modal

```bash
# Authenticate with Modal
modal token new
```

This will:
1. Open a browser window
2. Ask you to log in to Modal.com
3. Provide an authentication token
4. Automatically configure your local environment

### 2.2 Verify Connection

```bash
# List your volumes (should show existing or empty list)
modal volume ls

# Check available GPU types
modal app list
```

---

## Step 3: Configure Secrets

Secrets are required for API access to Hugging Face and Modal services.

### 3.1 Get Your Hugging Face Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Name it `mercury-moe`
4. Select **"Read"** permissions
5. Copy the token (starts with `hf_...`)

### 3.2 Create Modal Secret

**Option A: One-Line Command (Recommended)**

```bash
modal secret create mercury-moe-secrets \
    MODAL_API_KEY="<paste_your_modal_api_key>" \
    HF_TOKEN="<paste_your_huggingface_token>"
```

**Option B: Interactive Mode**

```bash
modal secret create mercury-moe-secrets
# Follow prompts to add MODAL_API_KEY and HF_TOKEN
```

**Option C: Via Dashboard**

1. Visit [modal.com/secrets](https://modal.com/secrets)
2. Click **"Create Secret"**
3. Name: `mercury-moe-secrets`
4. Add keys:
   - `MODAL_API_KEY` = your Modal API key
   - `HF_TOKEN` = your Hugging Face token
5. Click **Create**

### 3.3 Verify Secret

```bash
modal secret list
# Should show: mercury-moe-secrets
```

---

## Step 4: Download Training Data

Download music videos from YouTube directly to your Modal volume.

### 4.1 Download a Single Video

```bash
modal run modal_app.py::download \
    --youtube-url "https://youtu.be/mrV8kK5t0V8?si=aMMFQenekQJ8kL-8" \
    --output-filename "music_video_01.mp4"
```

### 4.2 Download Multiple Videos

Create a file `video_list.txt`:

```
https://youtu.be/video1_id
https://youtu.be/video2_id
https://youtu.be/video3_id
```

Then run:

```bash
modal run modal_app.py::download-batch \
    --input-file video_list.txt \
    --output-dir data/raw
```

### 4.3 Upload Local Files

If you have local video files:

```bash
python upload_data.py --data_dir path/to/local/videos --volume training-data-vol
```

### 4.4 Verify Uploaded Data

```bash
# List files in your volume
modal volume ls training-data-vol data/raw
```

---

## Step 5: Preprocess & Ingest Data

Process raw videos into training-ready format with transcripts and embeddings.

### 5.1 Run Data Ingestion

```bash
modal run modal_app.py::ingest_dataset \
    --data-dir data/raw \
    --output-vol training-data-vol \
    --max-workers 4
```

**What this does:**
- Extracts audio from videos
- Generates transcripts using Whisper
- Creates text embeddings
- Splits videos into clips (configurable duration)
- Organizes data structure for training

### 5.2 Monitor Progress

Watch the logs in real-time:

```bash
modal app logs <app-name>
```

Or view in the dashboard at [modal.com/apps](https://modal.com/apps)

### 5.3 Verify Processed Data

```bash
modal volume ls training-data-vol data/processed
# Should show: clips/, transcripts/, embeddings/
```

---

## Step 6: Train the Model

Train your AnimateDiff + MoE model on the processed dataset.

### 6.1 Start Training

```bash
modal run modal_app.py::train \
    --data-vol training-data-vol \
    --epochs 100 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --checkpoint-interval 10
```

**Key Parameters:**
- `--epochs`: Number of training epochs (start with 50-100)
- `--batch-size`: Samples per batch (L4 GPU: 4-8 recommended)
- `--learning-rate`: Learning rate (1e-5 to 5e-5)
- `--checkpoint-interval`: Save checkpoint every N epochs

### 6.2 Resume from Checkpoint

If training is interrupted:

```bash
modal run modal_app.py::train \
    --data-vol training-data-vol \
    --resume-from checkpoints/latest.pt
```

### 6.3 Monitor Training

**Via Terminal:**
```bash
modal app logs <train-app-name>
```

**Via Dashboard:**
- Visit [modal.com/apps](https://modal.com/apps)
- Click on your running app
- View logs, GPU utilization, and checkpoints

### 6.4 Expected Training Time

| Epochs | Batch Size | Duration | Estimated Cost |
|--------|------------|----------|----------------|
| 50     | 4          | ~2 hours | ~$2.50         |
| 100    | 4          | ~4 hours | ~$5.00         |
| 200    | 8          | ~6 hours | ~$7.50         |

*Costs based on L4 GPU at $0.0003/sec*

---

## Step 7: Generate Music Videos

Generate music videos using your trained model.

### 7.1 Basic Generation

```bash
modal run modal_app.py::generate \
    --prompt "cyberpunk city at night, neon lights, flying cars" \
    --duration 8 \
    --fps 24 \
    --resolution 512
```

### 7.2 Generate with Audio Sync

```bash
# First, upload your music file
python upload_data.py --file music/my_track.mp3 --volume music-vol

# Then generate with audio
modal run modal_app.py::generate \
    --prompt "dancers in futuristic club, laser lights, energetic motion" \
    --audio music/my_track.mp3 \
    --duration 16 \
    --sync-to-beat true \
    --fps 24 \
    --resolution 512
```

### 7.3 Advanced Generation Options

```bash
modal run modal_app.py::generate \
    --prompt "underwater fantasy scene, bioluminescent creatures, slow motion" \
    --negative-prompt "blurry, distorted, low quality" \
    --duration 12 \
    --fps 30 \
    --resolution 768 \
    --guidance-scale 7.5 \
    --seed 42 \
    --reference-image references/style.png \
    --motion-strength 0.8
```

### 7.4 Batch Generation

Create a JSON file `prompts.json`:

```json
[
  {
    "prompt": "sunset over mountains, cinematic lighting",
    "duration": 8,
    "audio": "music/track1.mp3"
  },
  {
    "prompt": "neon Tokyo streets at night",
    "duration": 12,
    "audio": "music/track2.mp3"
  }
]
```

Run batch generation:

```bash
modal run modal_app.py::generate-batch \
    --prompts-file prompts.json \
    --output-dir outputs/batch_001
```

### 7.5 Download Generated Videos

```bash
# List generated videos
modal volume ls outputs-vol outputs/

# Download specific video
modal volume get outputs-vol outputs/video_001.mp4 ./local_outputs/
```

Or download via dashboard at [modal.com/volumes](https://modal.com/volumes)

---

## Step 8: Monitoring & Debugging

### 8.1 View Application Logs

```bash
# Real-time logs
modal app logs <app-name>

# Last 100 lines
modal app logs <app-name> -n 100

# Filter by level
modal app logs <app-name> --level ERROR
```

### 8.2 Interactive Shell

Debug directly in the Modal environment:

```bash
modal shell modal_app.py
```

Inside the shell:
```python
# Test model loading
from src.models.animatediff import MercuryMoE
model = MercuryMoE.from_pretrained("checkpoints/latest.pt")

# Test data loading
import torch
data = torch.load("/vol/data/processed/sample.pt")
```

### 8.3 Check GPU Utilization

```bash
# While app is running
modal app status <app-name>
```

Or view in dashboard for real-time GPU metrics.

### 8.4 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch size or resolution |
| Slow Download | Use Modal volume instead of local |
| Missing Weights | Verify HF_TOKEN in secrets |
| Audio Desync | Increase `--sync-threshold` parameter |

---

## Troubleshooting

### Problem: "Authentication Failed"

**Solution:**
```bash
modal token new
# Re-authenticate
```

### Problem: "Secret not found"

**Solution:**
```bash
modal secret list
# Verify 'mercury-moe-secrets' exists
modal secret create mercury-moe-secrets MODAL_API_KEY=... HF_TOKEN=...
```

### Problem: "CUDA Out of Memory"

**Solution:**
Reduce batch size or resolution:
```bash
modal run modal_app.py::train --batch-size 2 --resolution 512
```

### Problem: "Video Download Failed"

**Solution:**
Try alternative downloader or upload manually:
```bash
python upload_data.py --file local_video.mp4 --volume training-data-vol
```

### Problem: "Model Weights Not Found"

**Solution:**
Ensure HF_TOKEN has read permissions and model ID is correct in config.

---

## Cost Optimization Tips

### 1. Use Spot Instances (When Available)
```python
# In modal_app.py, add:
@app.function(gpu="L4", timeout=3600, concurrency_limit=1)
```

### 2. Optimize Batch Sizes
- Training: Start with batch_size=2, increase if VRAM allows
- Generation: Use batch processing for multiple prompts

### 3. Set Appropriate Timeouts
```python
# Don't use excessive timeouts
@app.function(timeout=1800)  # 30 minutes max for generation
```

### 4. Clean Up Volumes Regularly
```bash
# Remove old checkpoints
modal volume rm training-data-vol checkpoints/old_*.pt

# Delete unused volumes
modal volume delete unused-volume
```

### 5. Monitor Spending
- Set up billing alerts at [modal.com/billing](https://modal.com/billing)
- Check usage dashboard regularly

### 6. Use Lower Resolution for Testing
```bash
# Test with 512x512 before 768x768
modal run modal_app.py::generate --resolution 512
```

---

## Quick Reference Commands

```bash
# Authentication
modal token new

# Secrets
modal secret create mercury-moe-secrets MODAL_API_KEY=... HF_TOKEN=...
modal secret list

# Data
modal run modal_app.py::download --youtube-url "URL" --output-filename "video.mp4"
modal run modal_app.py::ingest_dataset --data-dir data/raw --output-vol training-data-vol

# Training
modal run modal_app.py::train --epochs 100 --batch-size 4

# Generation
modal run modal_app.py::generate --prompt "your prompt" --duration 8 --audio music/track.mp3

# Monitoring
modal app logs <app-name>
modal shell modal_app.py

# File Management
modal volume ls training-data-vol
modal volume get outputs-vol outputs/video.mp4 ./local/
```

---

## Next Steps

🎉 **You're all set!** Here's what to do next:

1. **Start Small**: Download one video and test the pipeline
2. **Iterate**: Generate short clips (4-8s) to verify quality
3. **Scale Up**: Once satisfied, train longer and generate full music videos
4. **Experiment**: Try different prompts, styles, and audio tracks

### Useful Resources

- [Modal Documentation](https://modal.com/docs)
- [AnimateDiff GitHub](https://github.com/guoyww/AnimateDiff)
- [Hugging Face Models](https://huggingface.co/models)
- [Community Discord](https://discord.gg/modal)

---

## Support

Need help? 

- 📖 Check the [Modal Docs](https://modal.com/docs)
- 💬 Join the Modal Discord community
- 🐛 Report issues on GitHub
- 📧 Contact support@modal.com

---

**Happy Creating! 🎵🎬✨**

Your MoE-powered music video generator is ready to bring your creative visions to life!
