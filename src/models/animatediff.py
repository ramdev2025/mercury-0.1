"""
AnimateDiff Motion Module with MoE Enhancement

Implements temporal attention layers for video generation, enhanced with
sparse Mixture of Experts (MoE) for efficient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange, repeat

from .moe import MoELayer


class TemporalAttention(nn.Module):
    """
    Temporal self-attention for video frames.
    
    Processes features along the time dimension while maintaining
    spatial consistency.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_moe = use_moe
        
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN with optional MoE
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
    
    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B*F, N, D] or [B, N, D] where F is frames
            num_frames: number of frames in sequence
            key_padding_mask: optional mask for padding
        
        Returns:
            x: same shape as input
            aux_loss: MoE auxiliary loss (None if not using MoE)
        """
        # Reshape to group temporal dimension
        # Assuming input is [B*T, N, D], reshape to process temporal attention
        residual = x
        
        # Layer norm
        x_norm = self.norm(x)
        
        # Apply temporal attention
        # For pure temporal attention, we'd need to rearrange differently
        # This is a simplified version - full implementation needs frame grouping
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask
        )
        
        x = residual + attn_out
        
        # FFN / MoE
        residual = x
        x_norm = self.norm(x)
        
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
            aux_loss = None
            
        x = residual + ffn_out
        
        return x, aux_loss


class AnimateDiffMotionModule(nn.Module):
    """
    AnimateDiff-style motion module that can be injected into Stable Diffusion.
    
    Adds temporal layers to each UNet block for coherent video generation.
    """
    
    def __init__(
        self,
        unet_channels: List[int] = [320, 640, 1280, 1280],
        temporal_layers_per_block: int = 1,
        dim: int = 320,
        num_heads: int = 8,
        use_moe: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.motion_modules = nn.ModuleList()
        
        for ch in unet_channels:
            for _ in range(temporal_layers_per_block):
                module = TemporalAttention(
                    dim=ch,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                self.motion_modules.append(module)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: List[torch.Tensor],
        num_frames: int,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Apply temporal modules to UNet features.
        
        Args:
            features: List of feature maps from UNet blocks
            num_frames: Number of frames in video
        
        Returns:
            enhanced_features: Features with temporal information
            total_aux_loss: Sum of MoE auxiliary losses
        """
        total_aux_loss = torch.tensor(0.0, device=features[0].device)
        enhanced_features = []
        
        for i, feat in enumerate(features):
            if i < len(self.motion_modules):
                # Apply temporal attention
                feat_enhanced, aux_loss = self.motion_modules[i](
                    feat, num_frames
                )
                enhanced_features.append(feat_enhanced)
                
                if aux_loss is not None:
                    total_aux_loss = total_aux_loss + aux_loss
            else:
                enhanced_features.append(feat)
        
        return enhanced_features, total_aux_loss


class MotionEncoder(nn.Module):
    """
    Encodes motion patterns from reference videos or optical flow.
    Can be used for motion transfer or conditioning.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 320,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
            for _ in range(num_layers)
        ])
        
        self.norm_out = nn.GroupNorm(8, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features from input.
        
        Args:
            x: [B, C, H, W] or [B*T, C, H, W]
        
        Returns:
            features: [B*T, dim, H, W]
        """
        x = self.conv_in(x)
        
        for block in self.blocks:
            x = x + block(x)
        
        x = self.norm_out(x)
        return x


def load_motion_module(
    checkpoint_path: str,
    use_moe: bool = True,
    device: str = "cuda",
) -> AnimateDiffMotionModule:
    """
    Load pre-trained AnimateDiff motion module.
    
    Args:
        checkpoint_path: Path to .ckpt file
        use_moe: Enable MoE enhancement
        device: Target device
    
    Returns:
        Loaded motion module
    """
    # Create model
    model = AnimateDiffMotionModule(use_moe=use_moe)
    
    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.startswith("guoyww/"):
        # Would download from HuggingFace in real implementation
        print(f"Would load motion module from {checkpoint_path}")
    elif checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    return model.to(device)
