"""Shared image loading / metrics for reliable frame validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path, max_side: int | None = None) -> tuple[np.ndarray, int, int]:
    """Load RGB float32 image; optionally downscale long side for speed."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        if max_side is not None and max(w0, h0) > max_side:
            scale = max_side / float(max(w0, h0))
            nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr, w0, h0


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def tenengrad(gray: np.ndarray) -> float:
    """Tenengrad sharpness (Sobel energy). More stable than raw Laplacian var indoors."""
    g = gray.astype(np.float32)
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    padded = np.pad(g, 1, mode="edge")
    gx = (
        kx[0, 0] * padded[:-2, :-2]
        + kx[0, 1] * padded[:-2, 1:-1]
        + kx[0, 2] * padded[:-2, 2:]
        + kx[1, 0] * padded[1:-1, :-2]
        + kx[1, 1] * padded[1:-1, 1:-1]
        + kx[1, 2] * padded[1:-1, 2:]
        + kx[2, 0] * padded[2:, :-2]
        + kx[2, 1] * padded[2:, 1:-1]
        + kx[2, 2] * padded[2:, 2:]
    )
    gy = (
        ky[0, 0] * padded[:-2, :-2]
        + ky[0, 1] * padded[:-2, 1:-1]
        + ky[0, 2] * padded[:-2, 2:]
        + ky[1, 0] * padded[1:-1, :-2]
        + ky[1, 1] * padded[1:-1, 1:-1]
        + ky[1, 2] * padded[1:-1, 2:]
        + ky[2, 0] * padded[2:, :-2]
        + ky[2, 1] * padded[2:, 1:-1]
        + ky[2, 2] * padded[2:, 2:]
    )
    return float(np.mean(gx * gx + gy * gy))


def edge_density(gray: np.ndarray, thr: float | None = None) -> float:
    """Fraction of pixels with strong gradient (texture proxy)."""
    g = gray.astype(np.float32)
    # simple gradient magnitude
    dx = np.diff(g, axis=1, prepend=g[:, :1])
    dy = np.diff(g, axis=0, prepend=g[:1, :])
    mag = np.sqrt(dx * dx + dy * dy)
    if thr is None:
        thr = float(np.percentile(mag, 75)) * 0.5 + 5.0
    return float(np.mean(mag > thr))


def clip_fractions(
    gray: np.ndarray, low: float = 8.0, high: float = 247.0
) -> tuple[float, float]:
    """Fraction of near-black and near-white pixels."""
    total = max(gray.size, 1)
    low_pct = float(np.mean(gray <= low))
    high_pct = float(np.mean(gray >= high))
    return low_pct, high_pct


def percentile_ranks(values: list[float]) -> list[float]:
    """Return percentile rank 0–100 for each value within the list."""
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    # average ranks for ties
    n = len(arr)
    ranks[order] = np.linspace(0.0, 100.0, n)
    # For ties, recompute properly
    # Simple approach: rankdata average
    sorter = np.argsort(arr)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(n)
    sorted_arr = arr[sorter]
    # average ties
    result = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        avg_rank = (i + j) / 2.0
        pct = (avg_rank / max(n - 1, 1)) * 100.0 if n > 1 else 50.0
        for k in range(i, j + 1):
            result[sorter[k]] = pct
        i = j + 1
    return [float(x) for x in result]
