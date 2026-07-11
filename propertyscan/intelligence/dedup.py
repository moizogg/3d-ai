"""Motion-based redundancy filter (replaces legacy aggressive dHash duplicates).

A frame is REDUNDANT only when the camera barely moved since the last kept frame.
Similar-looking walls while walking are NOT duplicates.

Important: optical flow / MAD is unreliable on low-texture white walls — for those
frames we do NOT cull by motion (keyframe budget handles thinning instead).
"""

from __future__ import annotations

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameMetadata, FrameStatus


def hamming_hex(a: str, b: str) -> int:
    """Legacy helper kept for tests; not used by reliable validation."""
    if not a or not b:
        return 64
    n = max(len(a), len(b))
    a, b = a.zfill(n), b.zfill(n)
    try:
        return int((int(a, 16) ^ int(b, 16)).bit_count())
    except ValueError:
        return 64


def mark_duplicates(
    frames: list[FrameMetadata],
    config: EngineConfig,
) -> list[FrameMetadata]:
    """Mark near-stationary frames as REDUNDANT using motion, not dHash.

    Policy:
      - Walk frames in temporal order.
      - First selectable frame always kept.
      - REDUNDANT only if cumulative motion since last kept < min_motion_to_keep
        AND the frame has enough texture for motion estimates to be meaningful.
      - Low-texture frames are never motion-culled (avoids white-wall false freezes).
      - Hard-rejected frames are left unchanged.
    """
    fi = config.frame_intelligence
    min_motion = fi.min_motion_to_keep
    low_tex = fi.low_texture_edge_density * 100.0  # texture_score is 0–100 scale
    result: list[FrameMetadata] = []
    last_kept = False
    cumulative_motion_since_kept = 0.0

    for frame in frames:
        if frame.is_hard_rejected():
            result.append(frame)
            continue

        if frame.status not in (
            FrameStatus.CANDIDATE,
            FrameStatus.LOW_RANK,
            FrameStatus.ACCEPTED,
        ):
            result.append(frame)
            continue

        motion = float(frame.motion_from_prev or 0.0)
        is_low_texture = frame.texture_score < max(low_tex, 5.0)

        if not last_kept:
            last_kept = True
            cumulative_motion_since_kept = 0.0
            result.append(frame)
            continue

        cumulative_motion_since_kept += motion

        # Low texture: flow is meaningless — never mark redundant here.
        # Keyframe stage will thin via max_keyframes.
        if is_low_texture:
            last_kept = True
            cumulative_motion_since_kept = 0.0
            result.append(
                frame.model_copy(
                    update={
                        "notes": list(frame.notes) + ["low_texture_skip_motion_cull"],
                        "is_stationary": False,
                    }
                )
            )
            continue

        if cumulative_motion_since_kept < min_motion:
            result.append(
                frame.model_copy(
                    update={
                        "status": FrameStatus.REDUNDANT,
                        "reject_reason": (
                            f"near-stationary since last kept "
                            f"(cum_motion={cumulative_motion_since_kept:.3f} "
                            f"< {min_motion})"
                        ),
                        "is_stationary": True,
                        "notes": list(frame.notes) + ["motion_redundant"],
                    }
                )
            )
            continue

        # Enough motion — keep
        last_kept = True
        cumulative_motion_since_kept = 0.0
        result.append(
            frame.model_copy(
                update={
                    "is_stationary": False,
                    "reject_reason": frame.reject_reason
                    if frame.status == FrameStatus.LOW_RANK
                    else None,
                }
            )
        )

    return result
