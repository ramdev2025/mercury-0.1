"""
Sparse Mixture of Experts (MoE) layer with Top-K routing and load balancing.
Designed for 8GB VRAM — lightweight, efficient, fully trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExpertFFN(nn.Module):
    """Single expert feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseRouter(nn.Module):
    """
    Top-K sparse gating router with auxiliary load-balancing loss.
    Reference: Switch Transformers (Fedus et al., 2022)
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch*seq_len, dim]
        Returns:
            weights: [batch*seq_len, top_k] — softmax weights for selected experts
            indices: [batch*seq_len, top_k] — selected expert indices
            aux_loss: scalar — load balancing loss
        """
        logits = self.gate(x)  # [N, num_experts]
        probs = F.softmax(logits, dim=-1)

        # Top-K selection
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize

        # Auxiliary load balancing loss (encourages uniform expert usage)
        # L_aux = num_experts * sum(f_i * p_i)
        # f_i = fraction of tokens routed to expert i
        # p_i = average router probability for expert i
        token_count = x.shape[0]
        expert_mask = F.one_hot(indices, self.num_experts).float()  # [N, top_k, E]
        expert_mask_flat = expert_mask.sum(dim=1)  # [N, E]
        f_i = expert_mask_flat.mean(dim=0)  # [E] — fraction of tokens
        p_i = probs.mean(dim=0)  # [E] — mean router probability
        aux_loss = self.num_experts * (f_i * p_i).sum()

        return weights, indices, aux_loss


class MoELayer(nn.Module):
    """
    Sparse MoE FFN layer. Replaces a standard FFN in a Transformer block.

    Architecture:
        N experts (ExpertFFN)
        1 sparse router (Top-K)
        Load balancing auxiliary loss
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        expert_hidden_dim = expert_hidden_dim or dim * 4

        self.router = SparseRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(dim, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            out: [batch, seq_len, dim]
            aux_loss: scalar load balancing loss
        """
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        weights, indices, aux_loss = self.router(x_flat)

        out = torch.zeros_like(x_flat)

        # Dispatch tokens to selected experts
        for k in range(self.top_k):
            expert_idx = indices[:, k]        # [N]
            expert_w   = weights[:, k]        # [N]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    token_slice = x_flat[mask]
                    expert_out = self.experts[e](token_slice)
                    out[mask] += expert_w[mask].unsqueeze(-1) * expert_out

        return out.view(B, S, D), aux_loss
