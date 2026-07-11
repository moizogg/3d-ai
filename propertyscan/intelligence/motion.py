"""Camera motion estimates between consecutive frames (redundancy + smear).

Uses OpenCV optical flow when available; falls back to gray mean-abs-diff.
This is intentionally NOT dHash — indoor walls share similar hashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from propertyscan.intelligence.image_ops import load_rgb, to_gray

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def estimate_motion(
    path_a: Path,
    path_b: Path,
    *,
    max_side: int = 320,
) -> float:
    """Estimate motion magnitude between two frames (higher = more camera/scene motion).

    Returns:
        A non-negative scalar. Typical ranges (downscaled ~320px):
          - < 0.5  near-stationary
          - 0.8–3  slow walk / slight pan
          - > 3    faster motion
    """
    rgb_a, _, _ = load_rgb(path_a, max_side=max_side)
    rgb_b, _, _ = load_rgb(path_b, max_side=max_side)
    ga = to_gray(rgb_a)
    gb = to_gray(rgb_b)

    # Align shapes if needed
    h = min(ga.shape[0], gb.shape[0])
    w = min(ga.shape[1], gb.shape[1])
    ga = ga[:h, :w]
    gb = gb[:h, :w]

    if _HAS_CV2:
        return _flow_motion(ga, gb)
    return _mad_motion(ga, gb)


def _flow_motion(ga: np.ndarray, gb: np.ndarray) -> float:
    a = np.clip(ga, 0, 255).astype(np.uint8)
    b = np.clip(gb, 0, 255).astype(np.uint8)
    # Farneback dense flow — robust enough and no feature dependence on texture
    flow = cv2.calcOpticalFlowFarneback(
        a,
        b,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    # Use median to ignore tiny noise / sensor grain
    return float(np.median(mag))


def _mad_motion(ga: np.ndarray, gb: np.ndarray) -> float:
    """Mean absolute difference fallback (no OpenCV).

    Scaled so values are roughly comparable to flow medians on slow walks.
    """
    mad = float(np.mean(np.abs(ga - gb)))
    # MAD on 0–255 gray; slow walk often 3–15, freeze ~0–2
    return mad / 4.0


def motion_series(
    paths: list[Path],
    *,
    max_side: int = 320,
) -> list[float]:
    """Per-frame motion vs previous frame; frame 0 is 0.0."""
    if not paths:
        return []
    out = [0.0]
    for i in range(1, len(paths)):
        try:
            m = estimate_motion(paths[i - 1], paths[i], max_side=max_side)
        except Exception:
            m = 0.0
        out.append(m)
    return out
