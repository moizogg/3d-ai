"""GPU runtime helpers for foundation geometry (T4-aware)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("propertyscan.geometry.runtime")


def torch_module() -> Any | None:
    try:
        import torch  # type: ignore

        return torch
    except ImportError:
        return None


def resolve_torch_device(*, prefer_cuda: bool = True, allow_cpu: bool = False) -> str:
    """Return ``cuda`` or ``cpu``; raise ValueError if CUDA required but missing."""
    torch = torch_module()
    if torch is None:
        raise ValueError("PyTorch is not installed. Install torch with CUDA for Phase 4 geometry.")
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise ValueError(
        "CUDA is required for foundation geometry (allow_cpu_geometry=false). "
        "Use Colab T4 / a CUDA GPU, or geometry.engine=mock for plumbing tests."
    )


def empty_cache() -> None:
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def peak_vram_gb() -> float | None:
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return round(float(torch.cuda.max_memory_allocated(0)) / (1024**3), 3)
    except Exception:
        return None


def reset_peak_vram() -> None:
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


def choose_pair_graph(n_images: int, configured: str) -> str:
    """Pick pair graph for VRAM safety.

    complete: all pairs (best quality, heavy)
    swin-5 / swin-3: sliding window (T4-friendly for large N)
    """
    if configured in ("swin-5", "swin-3", "complete"):
        if configured == "complete" and n_images > 40:
            logger.info(
                "Overriding pair_graph complete→swin-5 for %d images (VRAM safety)",
                n_images,
            )
            return "swin-5"
        return configured
    return "swin-5" if n_images > 25 else "complete"
