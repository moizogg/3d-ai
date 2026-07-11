"""Frame intelligence orchestration (modules; stages wrap this)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from propertyscan.capture.detect import validate_and_load
from propertyscan.core.config import EngineConfig
from propertyscan.core.context import RunContext
from propertyscan.domain.capture import CaptureManifest
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import SceneDescriptor
from propertyscan.intelligence.classify import classify_scene
from propertyscan.intelligence.dedup import mark_duplicates
from propertyscan.intelligence.keyframes import select_keyframes
from propertyscan.intelligence.quality import analyze_frames

logger = logging.getLogger("propertyscan.intelligence.pipeline")


def run_frame_intelligence(
    input_path: Path,
    ctx: RunContext,
) -> tuple[CaptureManifest, FrameSet, SceneDescriptor, Path]:
    """Full Phase 2 pipeline: validate → decode → quality → dedup → keyframes → classify.

    Returns:
        (manifest, frame_set, scene_descriptor, report_path)
    """
    config: EngineConfig = ctx.config
    input_path = Path(input_path)

    # 1. Validate + load manifest
    adapter, manifest = validate_and_load(input_path, config)
    ctx.set("capture_manifest", manifest)
    logger.info(
        "capture kind=%s source=%s files=%s",
        manifest.kind.value,
        manifest.source_path,
        manifest.file_count,
    )

    # 2. Materialize candidate frames
    candidates_root = ctx.artifact_dir("frames")
    paths = adapter.materialize_frames(manifest, candidates_root, config)
    if not paths:
        from propertyscan.core.exceptions import ValidationError

        raise ValidationError(
            "No candidate frames produced from capture.",
            suggestion="Check that the video/folder contains readable media.",
        )
    ctx.set("candidate_paths", paths)
    logger.info("materialized %d candidate frames", len(paths))

    # 3. Quality
    frames = analyze_frames(paths, config)

    # 4. Dedup
    frames = mark_duplicates(frames, config)

    # 5. Keyframes
    selected_dir = ctx.artifact_dir("frames", "selected")
    quarantine_dir = ctx.artifact_dir("frames", "quarantine")
    frame_set = select_keyframes(frames, config, selected_dir=selected_dir)

    # Write quarantine manifest (paths of non-accepted)
    _write_quarantine_manifest(frame_set, quarantine_dir)

    # 6. Classify
    descriptor = classify_scene(frame_set)
    ctx.set("frame_set", frame_set)
    ctx.set("scene_descriptor", descriptor)

    report = _write_report(
        ctx,
        manifest=manifest,
        frame_set=frame_set,
        descriptor=descriptor,
        candidate_count=len(paths),
    )
    logger.info(
        "frame intelligence done: accepted=%d rejected=%d scene=%s report=%s",
        frame_set.accepted_count,
        frame_set.rejected_count,
        descriptor.scene_type,
        report,
    )
    return manifest, frame_set, descriptor, report


def _write_quarantine_manifest(frame_set: FrameSet, quarantine_dir: Path) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    rejected = [
        f.to_dict()
        for f in frame_set.frames
        if f.status.value not in ("accepted",)
    ]
    path = quarantine_dir / "quarantine.json"
    path.write_text(json.dumps(rejected, indent=2, default=str), encoding="utf-8")
    return path


def _write_report(
    ctx: RunContext,
    *,
    manifest: CaptureManifest,
    frame_set: FrameSet,
    descriptor: SceneDescriptor,
    candidate_count: int,
) -> Path:
    reports = ctx.artifact_dir("reports")
    path = reports / "frame_intelligence.json"
    payload: dict[str, Any] = {
        "job_id": ctx.job_id,
        "profile": ctx.config.engine.profile,
        "capture": manifest.to_dict(),
        "candidate_count": candidate_count,
        "frame_set": {
            "accepted_count": frame_set.accepted_count,
            "rejected_count": frame_set.rejected_count,
            "low_confidence_count": frame_set.low_confidence_count,
            "rejection_stats": frame_set.rejection_stats,
            "selected_paths": [str(p) for p in frame_set.get_accepted_paths()],
            "frames": [f.to_dict() for f in frame_set.frames],
        },
        "scene_descriptor": descriptor.to_dict(),
        "validation_mode": "reliable_v2",
        "config": {
            "video_fps": ctx.config.capture.video_fps,
            "max_keyframes": ctx.config.frame_intelligence.max_keyframes,
            "min_frames": ctx.config.capture.min_frames,
            "min_motion_to_keep": ctx.config.frame_intelligence.min_motion_to_keep,
            "low_rank_threshold": ctx.config.frame_intelligence.low_rank_threshold,
            "clip_black_pct": ctx.config.frame_intelligence.clip_black_pct,
            "note": "Legacy blur_threshold/phash_threshold are ignored",
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Also copy summary to output_dir for operator convenience
    out_copy = ctx.output_dir / "frame_intelligence.json"
    out_copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path
