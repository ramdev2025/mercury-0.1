"""
Mercury-MoE-Tiny — Full Video Transformer with sparse MoE FFN layers.

Architecture overview:
  ┌─────────────────────────────────────────┐
  │  Input Video [B, C, T, H, W]            │
  │       ↓ TubeletEmbedding                │
  │  Patch Tokens [B, N, D]                 │
  │       ↓ [CLS] prepend                   │
  │  [B, N+1, D]                            │
  │       ↓ Transformer Blocks × L          │
  │  ┌─────────────────────────────────┐    │
  │  │ LayerNorm                       │    │
  │  │ Multi-Head Self-Attention        │    │
  │  │ LayerNorm                       │    │
  │  │ MoE FFN (8 experts, top-2)      │    │
  │  └─────────────────────────────────┘    │
  │       ↓ [CLS] token                     │
  │  Classification head                    │
  │  [B, num_classes]                       │
  └─────────────────────────────────────────┘

Default: ~80M parameters, fp16, 8GB VRAM-safe with batch_size=4 + grad_checkpoint
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .tokenizer import TubeletEmbedding, CLSTokenWrapper
from .moe import MoELayer


class MoETransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with sparse MoE FFN.

    Replaces the standard dense FFN with a MoE layer.
    Every block uses MoE by default; pass use_moe=False for dense blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_moe: bool = True,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_moe:
            self.ffn = MoELayer(
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                expert_hidden_dim=dim * 4,
                dropout=dropout,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )

        self.drop_path = nn.Identity()  # can replace with DropPath for regularization

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: [B, N, D]
            aux_loss: scalar (0 if dense block)
        """
        # Self-attention (pre-LN)
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = residual + self.drop_path(attn_out)

        # FFN (pre-LN)
        residual = x
        x_norm = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
            aux_loss = torch.tensor(0.0, device=x.device)
        x = residual + self.drop_path(ffn_out)

        return x, aux_loss


class MercuryMoE(nn.Module):
    """
    Mercury-MoE-Tiny: Full video action recognition model.

    Model sizes (approximate):
        tiny   — dim=384, heads=6,  layers=12 → ~80M params
        small  — dim=512, heads=8,  layers=12 → ~150M params
        base   — dim=768, heads=12, layers=12 → ~310M params  (needs 16GB VRAM)

    For RTX 4060 8GB: use 'tiny' with fp16 + gradient checkpointing.
    """

    CONFIGS = {
        "tiny":  dict(dim=384, num_heads=6,  num_layers=12),
        "small": dict(dim=512, num_heads=8,  num_layers=12),
        "base":  dict(dim=768, num_heads=12, num_layers=12),
    }

    def __init__(
        self,
        num_classes: int = 101,          # UCF-101 default
        model_size: str = "tiny",
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        tube_size: int = 2,
        num_experts: int = 8,
        top_k: int = 2,
        moe_every_n: int = 2,           # use MoE every N blocks (others are dense)
        dropout: float = 0.1,
        in_channels: int = 3,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        cfg = self.CONFIGS[model_size]
        dim        = cfg["dim"]
        num_heads  = cfg["num_heads"]
        num_layers = cfg["num_layers"]

        self.aux_loss_weight = aux_loss_weight
        self.num_layers = num_layers

        # --- Video tokenizer ---
        self.tokenizer = TubeletEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tube_size=tube_size,
            in_channels=in_channels,
            embed_dim=dim,
        )
        self.cls_wrapper = CLSTokenWrapper(dim)
        self.embed_dropout = nn.Dropout(dropout)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                use_moe=(i % moe_every_n == 0),  # alternate dense/MoE
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T, H, W] — normalized video tensor
        Returns:
            logits:   [B, num_classes]
            aux_loss: scalar — total MoE load balancing loss
        """
        # Tokenize
        tokens = self.tokenizer(x)            # [B, N, D]
        tokens = self.cls_wrapper(tokens)     # [B, N+1, D]
        tokens = self.embed_dropout(tokens)

        # Transformer
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            tokens, aux_loss = block(tokens)
            total_aux_loss = total_aux_loss + aux_loss

        tokens = self.norm(tokens)

        # CLS token → classification
        cls_token = tokens[:, 0]              # [B, D]
        logits = self.head(cls_token)         # [B, num_classes]

        return logits, total_aux_loss * self.aux_loss_weight

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "total_M": total / 1e6}


# Backward compatibility alias
VideoMoE = MercuryMoE
