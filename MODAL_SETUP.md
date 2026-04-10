# VideoMoE on Modal — L4 GPU Setup Guide

This guide explains how to run the VideoMoE project on Modal.com using L4 GPUs.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install the Modal client:
   ```bash
   pip install modal
   modal token new
   ```

## Quick Start

### 1. Deploy and Run Training

Run training with the default config (tiny model on UCF-101):

```bash
modal run modal_app.py
```

Or specify a custom config:

```bash
modal run modal_app.py --config configs/train_tiny.yaml
```

### 2. Run Evaluation

Evaluate a trained checkpoint:

```bash
modal run modal_app.py --evaluate-checkpoint checkpoints/best_model.pt
```

### 3. Interactive Shell

Open an interactive shell for debugging:

```bash
modal shell modal_app.py
```

## Architecture Overview

### GPU Configuration
- **GPU Type**: NVIDIA L4 (24GB VRAM)
- **CUDA Version**: 12.1
- **Python Version**: 3.11

### Persistent Storage (Volumes)

The app uses three Modal Volumes for persistent storage:

| Volume Name | Mount Path | Purpose |
|-------------|------------|---------|
| `video-moe-data` | `/workspace/data` | Dataset (UCF-101) |
| `video-moe-checkpoints` | `/workspace/checkpoints` | Model checkpoints |
| `video-moe-logs` | `/workspace/logs` | TensorBoard logs, metrics |

### First-Time Setup: Upload Data

Before running training, you need to upload your dataset to the Modal volume:

```python
# upload_data.py
import modal

data_volume = modal.Volume.from_name("video-moe-data", create_if_missing=True)

# Upload UCF-101 dataset (adjust path as needed)
data_volume.add_local_dir("data/ucf101", remote_path="/ucf101")
data_volume.commit()
```

Then run:
```bash
python upload_data.py
```

## Configuration

### Training Config

Edit `configs/train_tiny.yaml` to adjust:
- `batch_size`: Per-GPU batch size
- `num_frames`: Frames per video clip
- `gradient_accumulation_steps`: For effective batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate

### L4 GPU Optimization

The L4 GPU has 24GB VRAM, which allows for larger batches than the RTX 4060 config:

```yaml
# Recommended for L4 (24GB)
batch_size: 8                  # Double the RTX 4060 setting
gradient_accumulation_steps: 4  # Effective batch = 8 × 4 = 32
num_frames: 16                 # More frames for better temporal modeling
```

## Monitoring

### View Logs

Training logs are streamed to your terminal. You can also view them in the Modal dashboard.

### TensorBoard

Access TensorBoard logs:

```bash
# Download logs from volume
modal volume get video-moe-logs logs/ ./local_logs/

# Run TensorBoard locally
tensorboard --logdir ./local_logs/
```

## Advanced Usage

### Custom Training Script

Modify `modal_app.py` to add custom training logic or pass additional arguments.

### Multi-GPU Training

For distributed training across multiple GPUs:

```python
@app.function(
    gpu=modal.gpu.L4(count=4),  # 4x L4 GPUs
    ...
)
def train_distributed(...):
    ...
```

### Scheduled Jobs

Use Modal's scheduled functions for periodic training runs.

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
1. Reduce `batch_size` in config
2. Reduce `num_frames`
3. Enable `gradient_checkpointing: true`

### Volume Issues

Check volume contents:
```bash
modal volume ls video-moe-data
```

Clear and recreate volumes:
```bash
modal volume rm video-moe-data
modal volume create video-moe-data
```

### Debug Mode

Add print statements or use `modal shell` to inspect the environment:
```bash
modal shell modal_app.py
# Then inside the shell:
ls -la /workspace/data
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

## Cost Estimation

L4 GPU pricing (check modal.com for current rates):
- ~$0.50/hour per L4 GPU
- Training for 50 epochs on UCF-101 tiny model: ~2-4 hours
- Estimated cost: $1-2 per full training run

## Next Steps

1. Upload your dataset to the Modal volume
2. Run a short test training (5 epochs) to verify setup
3. Launch full training run
4. Monitor progress via logs/TensorBoard
5. Evaluate best checkpoint

---

**Questions?** Check the [Modal documentation](https://modal.com/docs) or reach out to the team.
