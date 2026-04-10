"""
VRAM Profiler — helps you stay within 8GB on the RTX 4060.

Usage:
    from src.utils.vram import VRAMMonitor, estimate_model_vram

    with VRAMMonitor() as mon:
        output = model(input)
    print(mon.peak_mb)
"""

import torch
import gc
from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass


@dataclass
class VRAMStats:
    allocated_mb: float
    reserved_mb:  float
    peak_mb:      float
    free_mb:      float
    total_mb:     float
    util_pct:     float


class VRAMMonitor:
    """Context manager that tracks peak VRAM usage during a block."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.peak_mb = 0.0
        self.baseline_mb = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            self.baseline_mb = torch.cuda.memory_allocated(self.device_id) / 1e6
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(self.device_id) / 1e6
            self.peak_mb = peak - self.baseline_mb

    def __repr__(self) -> str:
        return f"VRAMMonitor(peak={self.peak_mb:.1f} MB)"


def get_vram_stats(device_id: int = 0) -> Optional[VRAMStats]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device_id)
    total    = props.total_memory / 1e6
    alloc    = torch.cuda.memory_allocated(device_id) / 1e6
    reserved = torch.cuda.memory_reserved(device_id) / 1e6
    peak     = torch.cuda.max_memory_allocated(device_id) / 1e6
    free     = (total - reserved)
    return VRAMStats(
        allocated_mb=alloc,
        reserved_mb=reserved,
        peak_mb=peak,
        free_mb=free,
        total_mb=total,
        util_pct=reserved / total * 100,
    )


def print_vram(label: str = "", device_id: int = 0):
    s = get_vram_stats(device_id)
    if s is None:
        print("CUDA not available")
        return
    tag = f"[{label}] " if label else ""
    print(
        f"{tag}VRAM: {s.allocated_mb:.0f}/{s.total_mb:.0f} MB allocated "
        f"({s.util_pct:.1f}% reserved)  |  Peak: {s.peak_mb:.0f} MB  |  Free: {s.free_mb:.0f} MB"
    )


def estimate_model_vram(
    model: torch.nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Rough VRAM estimate for a forward pass.

    Args:
        model: PyTorch model
        input_shape: (C, T, H, W) without batch
        batch_size: batch size to test
        dtype: fp16 for mixed precision
    Returns:
        dict with param_mb, activation_mb, total_mb
    """
    bytes_per_el = 2 if dtype == torch.float16 else 4

    # Parameter memory
    param_mb = sum(p.numel() * bytes_per_el for p in model.parameters()) / 1e6

    # Rough activation estimate: 2x param size for forward + backward
    activation_mb = param_mb * 2.0

    # Input tensor
    C, T, H, W = input_shape
    input_mb = batch_size * C * T * H * W * bytes_per_el / 1e6

    total_mb = param_mb + activation_mb + input_mb

    return {
        "param_mb": round(param_mb, 1),
        "activation_mb": round(activation_mb, 1),
        "input_mb": round(input_mb, 2),
        "total_mb": round(total_mb, 1),
        "fits_8gb": total_mb < 7500,
    }


def clear_vram():
    """Free unused VRAM — call between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[VRAM] Cache cleared")
