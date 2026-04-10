# VideoMoE-Tiny 🎬

> A Video Transformer with **Sparse Mixture of Experts (MoE)** layers, built from scratch  
> and tuned to train on **8GB VRAM** (RTX 4060).

---

## What Is This?

**VideoMoE-Tiny** is a lightweight video understanding model that:
- Tokenizes video clips into space-time patches (Tubelet Embedding — like VideoMAE)
- Processes them through a Transformer where **FFN layers are replaced with sparse MoE**
- Routes each token to the Top-2 of 8 specialized experts
- Trains on UCF-101 action recognition (101 classes)

### Why MoE?

| | Dense FFN | Sparse MoE |
|---|---|---|
| **Params** | All active every forward pass | 8 experts, only 2 active per token |
| **VRAM** | High | ~Same peak (experts are small) |
| **Capacity** | Limited | Higher (more specialized sub-networks) |
| **Training** | Simple | Needs load-balancing loss |

MoE lets you grow model capacity without linearly growing compute.

---

## Hardware Requirements

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | RTX 4060 8GB VRAM | fp16 + grad checkpoint |
| RAM | 32GB DDR5 | ✓ Comfortable for 4+ workers |
| Storage | ~10GB free | Videos + checkpoints |
| CPU | Ryzen 7 7000 | ✓ Fast DataLoader workers |

---

## Architecture

```
Input Video [B, 3, 8, 224, 224]
      │
      ▼ TubeletEmbedding (3D Conv, tube=2×16×16)
Tokens [B, 784, 384]
      │
      ▼ Prepend [CLS]
Tokens [B, 785, 384]
      │
      ▼ × 12 Transformer Blocks
   ┌─────────────────────────────┐
   │ Pre-LN                      │
   │ Multi-Head Attention (6h)   │
   │ Pre-LN                      │
   │ MoE FFN (8 experts, top-2)  │  ← every other block
   │    or Dense FFN              │  ← remaining blocks
   └─────────────────────────────┘
      │
      ▼ [CLS] token → LayerNorm
      ▼ Classification Head
Logits [B, 101]
```

**~80M parameters** | **~5.5–6.5 GB peak VRAM** at batch_size=4

---

## Quick Start

### 1. Install dependencies

```bash
# Clone / unzip this project
cd video-moe

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch (CUDA 12.1 for RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other deps
pip install -r requirements.txt
```

### 2a. Quick test with mock data (no download needed)

```bash
# Create 5-class synthetic dataset for testing
python scripts/download_data.py --mock --mock_classes 5

# Edit config to use mock data and fewer classes
# In configs/train_tiny.yaml: set num_classes: 5

# Test run (1 epoch)
python scripts/train.py --config configs/train_tiny.yaml --epochs 1
```

### 2b. Full UCF-101 training

```bash
# Download UCF-101 (~7.2 GB)
python scripts/download_data.py --data_dir data/ucf101

# Train!
python scripts/train.py --config configs/train_tiny.yaml
```

### 3. Monitor training

```bash
# Open TensorBoard (in separate terminal)
tensorboard --logdir logs/
# Browse to http://localhost:6006
```

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### 5. Inference on your own video

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --video /path/to/myvideo.mp4
```

---

## Project Structure

```
video-moe/
├── src/
│   ├── models/
│   │   ├── moe.py          ← MoE layer, sparse router, expert FFN
│   │   ├── tokenizer.py    ← Tubelet embedding, CLS token
│   │   └── video_moe.py    ← Full VideoMoE model
│   ├── data/
│   │   └── ucf101.py       ← UCF-101 dataset loader + augmentations
│   ├── training/
│   │   └── trainer.py      ← Training engine (fp16, grad accum, LR schedule)
│   └── utils/
│       ├── metrics.py      ← Accuracy, AverageMeter, expert utilization
│       └── vram.py         ← VRAM monitor, estimate, profiler
├── configs/
│   └── train_tiny.yaml     ← Default config (RTX 4060 optimized)
├── scripts/
│   ├── train.py            ← Main training entrypoint
│   ├── evaluate.py         ← Eval + single-video inference
│   └── download_data.py    ← UCF-101 download + mock generator
├── checkpoints/            ← Saved model weights
├── logs/                   ← TensorBoard logs
└── requirements.txt
```

---

## Config Reference

Key settings in `configs/train_tiny.yaml`:

| Key | Default | Effect |
|-----|---------|--------|
| `model_size` | `tiny` | tiny=80M, small=150M, base=310M (needs 16GB) |
| `batch_size` | `4` | Per-GPU batch (safe for 8GB) |
| `gradient_accumulation_steps` | `8` | Effective batch = 4×8 = 32 |
| `gradient_checkpointing` | `true` | Saves ~40% VRAM |
| `use_amp` | `true` | fp16 mixed precision |
| `num_experts` | `8` | MoE experts |
| `top_k` | `2` | Active experts per token |
| `num_frames` | `8` | Frames per video clip |
| `aux_loss_weight` | `0.01` | Load-balancing loss scale |

---

## VRAM Breakdown

| Component | Memory |
|-----------|--------|
| Model parameters (fp16) | ~160 MB |
| Optimizer states (AdamW) | ~480 MB |
| Activations (batch=4, T=8) | ~3.5 GB |
| Grad checkpointing savings | −1.5 GB |
| **Peak estimate** | **~5.5–6.5 GB** ✓ |

If you hit OOM:
1. Reduce `batch_size` to `2`
2. Increase `gradient_accumulation_steps` to `16`
3. Reduce `num_frames` to `4`

---

## Expected Results

On UCF-101 (split 1), training 50 epochs:

| Model | Top-1 Acc | Top-5 Acc | Epochs |
|-------|-----------|-----------|--------|
| VideoMoE-Tiny (scratch) | ~65–72% | ~88–92% | 50 |
| VideoMoE-Tiny (pretrained backbone) | ~80%+ | ~95%+ | 20 |

> **Tip**: Initialize the TubeletEmbedding + Transformer with ViT-Small ImageNet weights  
> for a big accuracy boost with minimal code changes.

---

## Next Steps / Extensions

- [ ] Load pre-trained ViT-S/16 weights into the Transformer backbone
- [ ] Add temporal self-attention (Divided Space-Time Attention from TimeSformer)
- [ ] Implement Expert Choice Routing (Google, 2022) for better load balancing
- [ ] Add video prediction head (next-frame prediction as auxiliary task)
- [ ] Fine-tune on your own dataset (change `num_classes` + `data_root`)
- [ ] Export to ONNX for inference

---

## References

- **VideoMAE** (Tong et al., 2022) — Tubelet embedding & masked pretraining
- **ViViT** (Arnab et al., 2021) — Video Vision Transformer
- **Switch Transformers** (Fedus et al., 2022) — Sparse MoE routing
- **TimeSformer** (Bertasius et al., 2021) — Divided space-time attention
- **Expert Choice** (Zhou et al., 2022) — Improved MoE load balancing
