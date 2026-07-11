"""Reliable per-frame quality analysis (replaces legacy Laplacian hard-reject).

Design:
  1. Hard-reject ONLY: unreadable, near-total black/white clip, clear motion smear.
  2. Soft-score everything else using clip-relative sharpness + texture awareness.
  3. White walls / low texture are NOT labeled blurry (legacy failure mode).
"""

from __future__ import annotations

from pathlib import Path

from propertyscan.core.config import EngineConfig, FrameIntelligenceSection
from propertyscan.domain.frames import FrameMetadata, FrameStatus
from propertyscan.intelligence.image_ops import (
    clip_fractions,
    edge_density,
    load_rgb,
    percentile_ranks,
    tenengrad,
    to_gray,
)
from propertyscan.intelligence.motion import motion_series


def _measure_frame(path: Path, index: int, max_side: int) -> FrameMetadata | None:
    try:
        rgb, w0, h0 = load_rgb(path, max_side=max_side)
    except Exception as exc:
        return FrameMetadata(
            filename=path.name,
            filepath=path,
            index=index,
            status=FrameStatus.UNREADABLE,
            reject_reason=f"unreadable: {exc}",
        )

    gray = to_gray(rgb)
    sharp = tenengrad(gray)
    tex = edge_density(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    clip_lo, clip_hi = clip_fractions(gray)

    return FrameMetadata(
        filename=path.name,
        filepath=path,
        width=w0,
        height=h0,
        index=index,
        sharpness_raw=round(sharp, 4),
        blur_score=round(sharp, 4),
        texture_score=round(tex * 100.0, 3),
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        clip_low_pct=round(clip_lo, 4),
        clip_high_pct=round(clip_hi, 4),
        status=FrameStatus.CANDIDATE,
    )


def _soft_quality(
    frame: FrameMetadata,
    sharpness_pct: float,
    fi: FrameIntelligenceSection,
) -> tuple[float, dict[str, float], list[str]]:
    """Compute soft quality 0–100 without hard-rejecting low-texture walls."""
    notes: list[str] = []
    tex_frac = frame.texture_score / 100.0

    # Sharpness component: clip-relative percentile
    sharp_comp = sharpness_pct

    # Texture-aware: if very low texture, do not punish "low sharpness"
    # (white wall is sharp enough for MASt3R even with low Tenengrad).
    if tex_frac < fi.low_texture_edge_density:
        sharp_comp = max(sharp_comp, 55.0)
        notes.append("low_texture_protected")

    # Exposure soft score (prefer not extreme means; clipping already hard-gated)
    mid = 127.5
    exposure = max(0.0, 100.0 - abs(frame.brightness - mid) / mid * 80.0)

    # Mild contrast soft score — never a hard reject
    contrast_comp = min(100.0, frame.contrast / 40.0 * 100.0)

    quality = 0.55 * sharp_comp + 0.25 * exposure + 0.20 * contrast_comp
    factors = {
        "sharpness_percentile": round(sharpness_pct, 2),
        "sharpness_component": round(sharp_comp, 2),
        "exposure": round(exposure, 2),
        "contrast": round(min(100.0, contrast_comp), 2),
        "texture": round(frame.texture_score, 2),
    }
    return round(float(quality), 2), factors, notes


def analyze_frames(paths: list[Path], config: EngineConfig) -> list[FrameMetadata]:
    """Analyze all candidates with reliable_v2 policy.

    Passes:
      1. Per-frame measurements
      2. Clip-relative sharpness percentiles
      3. Motion series vs previous frame
      4. Hard rejects + soft ranks
    """
    fi = config.frame_intelligence
    max_side = fi.motion_max_side

    # Pass 1 — measure
    frames: list[FrameMetadata] = []
    for i, p in enumerate(paths):
        m = _measure_frame(Path(p), i, max_side=max_side)
        if m is not None:
            frames.append(m)

    # Pass 2 — relative sharpness among readable frames
    readable_idx = [
        i
        for i, f in enumerate(frames)
        if f.status != FrameStatus.UNREADABLE
    ]
    sharp_vals = [frames[i].sharpness_raw for i in readable_idx]
    ranks = percentile_ranks(sharp_vals)
    rank_by_i = {readable_idx[k]: ranks[k] for k in range(len(readable_idx))}

    # Pass 3 — motion
    readable_paths = [frames[i].filepath for i in readable_idx]
    motions = motion_series(readable_paths, max_side=max_side)
    motion_by_i = {readable_idx[k]: motions[k] for k in range(len(readable_idx))}

    # Pass 4 — decide status
    out: list[FrameMetadata] = []
    for i, frame in enumerate(frames):
        if frame.status == FrameStatus.UNREADABLE:
            out.append(frame)
            continue

        sharp_pct = rank_by_i.get(i, 50.0)
        motion = motion_by_i.get(i, 0.0)
        tex_frac = frame.texture_score / 100.0

        # --- HARD REJECTS ONLY ---
        if frame.clip_low_pct >= fi.clip_black_pct:
            out.append(
                frame.model_copy(
                    update={
                        "status": FrameStatus.CLIPPED_BLACK,
                        "reject_reason": (
                            f"near-black clip {frame.clip_low_pct:.1%} >= {fi.clip_black_pct:.0%}"
                        ),
                        "sharpness_percentile": round(sharp_pct, 2),
                        "motion_from_prev": round(motion, 4),
                        "quality_score": 0.0,
                        "confidence_score": 0.0,
                        "rank_score": 0.0,
                    }
                )
            )
            continue

        if frame.clip_high_pct >= fi.clip_white_pct:
            out.append(
                frame.model_copy(
                    update={
                        "status": FrameStatus.CLIPPED_WHITE,
                        "reject_reason": (
                            f"near-white clip {frame.clip_high_pct:.1%} >= {fi.clip_white_pct:.0%}"
                        ),
                        "sharpness_percentile": round(sharp_pct, 2),
                        "motion_from_prev": round(motion, 4),
                        "quality_score": 0.0,
                        "confidence_score": 0.0,
                        "rank_score": 0.0,
                    }
                )
            )
            continue

        # Motion smear: poor relative sharpness AND textured scene AND high motion.
        # Low-texture white walls NEVER enter this branch via texture gate.
        is_smear = (
            sharp_pct <= fi.motion_smear_sharpness_percentile
            and tex_frac >= fi.motion_smear_min_texture
            and motion >= fi.motion_smear_min_flow
        )
        if is_smear:
            out.append(
                frame.model_copy(
                    update={
                        "status": FrameStatus.MOTION_SMEAR,
                        "reject_reason": (
                            f"motion smear: sharp_pct={sharp_pct:.1f}, "
                            f"texture={tex_frac:.3f}, flow={motion:.2f}"
                        ),
                        "sharpness_percentile": round(sharp_pct, 2),
                        "motion_from_prev": round(motion, 4),
                        "quality_score": 5.0,
                        "confidence_score": 5.0,
                        "rank_score": 5.0,
                        "is_stationary": False,
                    }
                )
            )
            continue

        # --- SOFT SCORE ---
        quality, factors, notes = _soft_quality(frame, sharp_pct, fi)
        factors["motion_from_prev"] = round(motion, 4)
        status = FrameStatus.CANDIDATE
        reject_reason = None
        if quality < fi.low_rank_threshold:
            status = FrameStatus.LOW_RANK
            reject_reason = (
                f"soft quality {quality:.1f} < {fi.low_rank_threshold} "
                "(still selectable if needed)"
            )
            notes.append("low_rank_soft")

        # Rank score used by keyframe greed: quality + small motion bonus
        rank = quality + min(15.0, motion * 2.0)

        out.append(
            frame.model_copy(
                update={
                    "status": status,
                    "reject_reason": reject_reason,
                    "sharpness_percentile": round(sharp_pct, 2),
                    "motion_from_prev": round(motion, 4),
                    "is_stationary": motion < fi.min_motion_to_keep,
                    "quality_score": quality,
                    "confidence_score": quality,
                    "rank_score": round(rank, 2),
                    "confidence_factors": factors,
                    "notes": notes,
                }
            )
        )

    return out


def analyze_frame(
    path: Path,
    *,
    index: int = 0,
    config: EngineConfig,
) -> FrameMetadata:
    """Analyze a single frame (no clip-relative peers → mid percentiles).

    Prefer ``analyze_frames`` for production so relative sharpness works.
    """
    results = analyze_frames([path], config)
    return results[0]


# Back-compat exports used by older tests
def laplacian_variance(gray) -> float:  # type: ignore[no-untyped-def]
    """Deprecated. Use tenengrad. Kept only so imports don't break mid-refactor."""
    from propertyscan.intelligence.image_ops import tenengrad as _t

    return _t(gray)
