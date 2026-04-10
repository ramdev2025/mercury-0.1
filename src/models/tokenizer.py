"""
Video Tokenizer — Tubelet Embedding.

Converts a video tensor [B, C, T, H, W] into a sequence of patch tokens
[B, num_patches, dim], using 3D convolution over space-time tubes.

Reference: VideoMAE (Tong et al., 2022), ViViT (Arnab et al., 2021)
"""

import torch
import torch.nn as nn
import math


class TubeletEmbedding(nn.Module):
    """
    3D patch/tubelet embedding for video.

    Each "tubelet" spans (tube_t, patch_h, patch_w) voxels.
    A 3D conv maps each tube to an embedding vector.

    Default config:
        tube_t=2, patch_h=16, patch_w=16 → for 8-frame 224x224 video:
        num_patches = (8/2) * (224/16) * (224/16) = 4 * 14 * 14 = 784 tokens
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        tube_size: int = 2,        # temporal stride of each tube
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert num_frames % tube_size == 0,  "num_frames must be divisible by tube_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tube_size = tube_size
        self.embed_dim = embed_dim

        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.grid_t = num_frames  // tube_size
        self.num_patches = self.grid_t * self.grid_h * self.grid_w

        # 3D convolution: maps (tube_t, patch_h, patch_w) → embed_dim
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size),
        )

        # Positional embedding (learnable, 3D-aware)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] — raw video frames, values in [0, 1]
        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"
        assert H == W == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"

        # [B, embed_dim, grid_t, grid_h, grid_w]
        x = self.proj(x)

        # Flatten spatial-temporal grid → sequence
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        x = x + self.pos_embed
        return x


class CLSTokenWrapper(nn.Module):
    """
    Prepends a learnable [CLS] token to the patch sequence.
    Used for classification tasks.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N+1, D]
        """
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)
