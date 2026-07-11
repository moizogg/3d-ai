"""Keyframe selection — diversity (motion) + quality ranking.

Does not mass-reject via dHash. Picks a budget of geometrically useful frames.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.core.exceptions import ValidationError
from propertyscan.domain.frames import FrameMetadata, FrameSet, FrameStatus


def select_keyframes(
    frames: list[FrameMetadata],
    config: EngineConfig,
    *,
    selected_dir: Path | None = None,
) -> FrameSet:
    """Select keyframes with greedy temporal diversity + quality.

    Algorithm:
      1. Pool = CANDIDATE + LOW_RANK (hard rejects & REDUNDANT excluded).
      2. Walk in time order; always take first pool frame.
      3. Take next frame if cumulative motion since last pick >= min_motion_to_keep
         OR quality is substantially better and mild motion exists.
      4. If under min_frames after pass, fill remaining pool by rank_score.
      5. Cap at max_keyframes preferring higher rank_score while preserving order.
    """
    min_frames = config.capture.min_frames
    max_kf = config.frame_intelligence.max_keyframes
    min_motion = config.frame_intelligence.min_motion_to_keep

    pool = [f for f in frames if f.is_selectable() or f.status == FrameStatus.CANDIDATE]
    # is_selectable covers CANDIDATE/LOW_RANK; also allow any not hard-rejected/redundant
    pool = [
        f
        for f in frames
        if f.status
        in (FrameStatus.CANDIDATE, FrameStatus.LOW_RANK, FrameStatus.ACCEPTED)
    ]
    pool = sorted(pool, key=lambda f: f.index)

    if not pool:
        raise ValidationError(
            "No selectable frames after reliable validation.",
            suggestion=(
                "All frames were hard-rejected (clip/unreadable/smear) or redundant. "
                "Check capture exposure and that the camera was moving."
            ),
        )

    chosen: list[FrameMetadata] = []
    last_idx_pos = -1
    cum_motion = 0.0

    for frame in pool:
        if not chosen:
            chosen.append(frame)
            last_idx_pos = frame.index
            cum_motion = 0.0
            continue

        motion = float(frame.motion_from_prev or 0.0)
        # Approximate motion from last chosen: sum motions along the way is stored
        # per-frame vs previous candidate; use motion_from_prev as lower bound.
        # Better: use index distance * soft motion
        gap = max(frame.index - last_idx_pos, 1)
        step_motion = motion if motion > 0 else 0.0
        # If frames were consecutive candidates, motion_from_prev is exact step
        effective = step_motion * max(1.0, min(gap, 3) * 0.5)

        take = False
        if effective >= min_motion or motion >= min_motion:
            take = True
        elif gap >= max(2, config.frame_intelligence.min_index_gap_fallback + 1):
            # Time-based fallback when motion estimate is weak but time advanced
            if frame.rank_score >= (chosen[-1].rank_score - 5):
                take = True

        if take:
            chosen.append(frame)
            last_idx_pos = frame.index

        if len(chosen) >= max_kf:
            break

    # Fill if under min_frames: add best remaining by rank_score (even REDUNDANT
    # is not in pool — but all CANDIDATE/LOW_RANK are).
    if len(chosen) < min_frames:
        chosen_paths = {c.filepath.resolve() for c in chosen}
        remaining = [
            f for f in pool if f.filepath.resolve() not in chosen_paths
        ]
        remaining.sort(key=lambda f: (-f.rank_score, f.index))
        for f in remaining:
            if len(chosen) >= min_frames and len(chosen) >= max_kf:
                break
            if len(chosen) >= max_kf:
                break
            chosen.append(f)
            if len(chosen) >= min_frames and len(chosen) >= min(min_frames, max_kf):
                # keep filling until min_frames
                pass
        # Ensure we reach min_frames if pool allows
        for f in remaining:
            if len(chosen) >= min_frames:
                break
            if f.filepath.resolve() in {c.filepath.resolve() for c in chosen}:
                continue
            chosen.append(f)
        chosen.sort(key=lambda f: f.index)

    # Cap by rank if still over max (prefer quality while keeping temporal spread)
    if len(chosen) > max_kf:
        # Keep first, last, and top ranked middle
        first, last = chosen[0], chosen[-1]
        mid = [c for c in chosen[1:-1]]
        mid.sort(key=lambda f: (-f.rank_score, f.index))
        keep_mid = mid[: max(0, max_kf - 2)]
        chosen = sorted([first, last] + keep_mid, key=lambda f: f.index)
        # de-dupe
        seen: set[Path] = set()
        uniq: list[FrameMetadata] = []
        for c in chosen:
            rp = c.filepath.resolve()
            if rp not in seen:
                seen.add(rp)
                uniq.append(c)
        chosen = uniq[:max_kf]

    if len(chosen) < min_frames:
        raise ValidationError(
            f"Only {len(chosen)} keyframes available; need at least {min_frames}.",
            suggestion=(
                "Capture a longer walkthrough with continuous camera motion, "
                "or lower capture.min_frames for experiments."
            ),
            details={
                "available": len(chosen),
                "min_frames": min_frames,
                "pool_size": len(pool),
            },
        )

    chosen_paths = {f.filepath.resolve() for f in chosen}
    notes = [
        "validation_mode=reliable_v2",
        "hard_reject=clip|unreadable|motion_smear_only",
        "redundancy=motion_based_not_dhash",
    ]

    if selected_dir is not None:
        selected_dir = Path(selected_dir)
        selected_dir.mkdir(parents=True, exist_ok=True)
        rewritten: list[FrameMetadata] = []
        for i, frame in enumerate(sorted(chosen, key=lambda f: f.index)):
            dest = selected_dir / f"keyframe_{i:04d}{frame.filepath.suffix.lower()}"
            if frame.filepath.resolve() != dest.resolve():
                shutil.copy2(frame.filepath, dest)
            rewritten.append(
                frame.model_copy(
                    update={
                        "filepath": dest,
                        "filename": dest.name,
                        "index": i,
                        "status": FrameStatus.ACCEPTED,
                        "reject_reason": None,
                    }
                )
            )
        audit: list[FrameMetadata] = []
        for frame in frames:
            if frame.filepath.resolve() in chosen_paths:
                continue
            if frame.status in (
                FrameStatus.CANDIDATE,
                FrameStatus.LOW_RANK,
                FrameStatus.ACCEPTED,
            ):
                audit.append(
                    frame.model_copy(
                        update={
                            "status": FrameStatus.REJECTED,
                            "reject_reason": "not selected as keyframe (budget/diversity)",
                        }
                    )
                )
            else:
                audit.append(frame)
        fs = FrameSet(
            source_type="keyframes",
            source_path=selected_dir,
            frames=list(rewritten) + audit,
            notes=notes,
            validation_mode="reliable_v2",
        )
        fs.recompute_counts()
        return fs

    final: list[FrameMetadata] = []
    for frame in frames:
        if frame.filepath.resolve() in chosen_paths:
            match = next(
                c for c in chosen if c.filepath.resolve() == frame.filepath.resolve()
            )
            final.append(
                match.model_copy(
                    update={"status": FrameStatus.ACCEPTED, "reject_reason": None}
                )
            )
        elif frame.status in (
            FrameStatus.CANDIDATE,
            FrameStatus.LOW_RANK,
            FrameStatus.ACCEPTED,
        ):
            final.append(
                frame.model_copy(
                    update={
                        "status": FrameStatus.REJECTED,
                        "reject_reason": "not selected as keyframe (budget/diversity)",
                    }
                )
            )
        else:
            final.append(frame)

    fs = FrameSet(
        source_type="keyframes",
        source_path=chosen[0].filepath.parent if chosen else Path("."),
        frames=final,
        notes=notes,
        validation_mode="reliable_v2",
    )
    fs.recompute_counts()
    return fs
