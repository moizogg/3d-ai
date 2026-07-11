"""Scene classification heuristics → SceneDescriptor for geometry routing."""

from __future__ import annotations

from propertyscan.domain.frames import FrameSet, FrameStatus
from propertyscan.domain.geometry import SceneDescriptor


def classify_scene(frame_set: FrameSet) -> SceneDescriptor:
    """Derive SceneDescriptor from reliable validation metrics."""
    all_frames = frame_set.frames
    accepted = frame_set.get_accepted()
    pool = accepted if accepted else [
        f for f in all_frames if f.status in (FrameStatus.CANDIDATE, FrameStatus.LOW_RANK)
    ]

    if not pool:
        return SceneDescriptor(
            scene_type="unknown",
            texture_score=0.0,
            blur_ratio=0.0,
            frame_count=0,
            mean_confidence=0.0,
            tags=["empty"],
        )

    mean_tex = sum(f.texture_score for f in pool) / len(pool)
    mean_conf = sum(f.quality_score or f.confidence_score for f in pool) / len(pool)
    mean_bright = sum(f.brightness for f in pool) / len(pool)
    mean_contrast = sum(f.contrast for f in pool) / len(pool)
    mean_sharp_pct = sum(f.sharpness_percentile for f in pool) / len(pool)

    smear_n = sum(1 for f in all_frames if f.status == FrameStatus.MOTION_SMEAR)
    smear_ratio = smear_n / max(len(all_frames), 1)

    is_low_light = mean_bright < 60.0
    is_reflective = mean_bright > 160.0 and mean_contrast < 25.0
    low_texture = mean_tex < 8.0  # texture_score is 0–100

    tags: list[str] = ["residential_indoor", "validation_reliable_v2"]
    if is_low_light:
        tags.append("low_light")
    if is_reflective:
        tags.append("reflective")
    if low_texture:
        tags.append("low_texture")
    if smear_ratio > 0.15:
        tags.append("motion_smear_common")

    scene_type = "residential_indoor"
    if is_reflective and low_texture:
        scene_type = "reflective_interior"
    elif low_texture:
        scene_type = "low_texture_interior"
    elif is_low_light:
        scene_type = "low_light_interior"

    return SceneDescriptor(
        scene_type=scene_type,
        texture_score=round(mean_tex, 2),
        blur_ratio=round(smear_ratio, 4),
        is_reflective=is_reflective,
        is_low_light=is_low_light,
        frame_count=len(accepted) if accepted else len(pool),
        mean_confidence=round(mean_conf, 2),
        tags=tags,
        extra={
            "mean_brightness": round(mean_bright, 2),
            "mean_contrast": round(mean_contrast, 2),
            "mean_sharpness_percentile": round(mean_sharp_pct, 2),
            "candidate_count": len(all_frames),
            "validation_mode": frame_set.validation_mode,
        },
    )
