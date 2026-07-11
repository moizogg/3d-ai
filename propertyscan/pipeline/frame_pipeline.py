"""Phase 2 end-to-end frame pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig, load_config
from propertyscan.core.context import RunContext
from propertyscan.core.device import resolve_device
from propertyscan.core.logging import setup_logging
from propertyscan.core.stage import Stage
from propertyscan.intelligence.pipeline import _write_quarantine_manifest, _write_report
from propertyscan.pipeline.stages import (
    ClassifySceneStage,
    DecodeFramesStage,
    DedupStage,
    FrameQualityStage,
    KeyframeStage,
    ValidateCaptureStage,
)

logger = logging.getLogger("propertyscan.pipeline.frames")


def build_frame_stages() -> list[Stage]:
    return [
        ValidateCaptureStage(),
        DecodeFramesStage(),
        FrameQualityStage(),
        DedupStage(),
        KeyframeStage(),
        ClassifySceneStage(),
    ]


def run_frames_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    profile: str | None = None,
    quality: str | None = None,
    config: EngineConfig | None = None,
) -> RunContext:
    """Run stages 1–6 and write frame intelligence artifacts.

    Returns:
        Populated RunContext with frame_set and scene_descriptor.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or load_config(profile=profile, quality=quality)  # type: ignore[arg-type]
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format)

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    work_dir = output_dir / "work" / job_id
    ctx = RunContext(
        config=cfg,
        input_path=input_path,
        output_dir=output_dir,
        work_dir=work_dir,
        job_id=job_id,
        device=resolve_device(prefer_cuda=cfg.device.prefer_cuda),
    )

    stages = build_frame_stages()
    for stage in stages:
        stage.run(ctx)

    # Final reports
    frame_set = ctx.require("frame_set")
    descriptor = ctx.require("scene_descriptor")
    manifest = ctx.require("capture_manifest")
    candidates = ctx.get("candidate_paths") or []
    _write_quarantine_manifest(frame_set, ctx.artifact_dir("frames", "quarantine"))
    report = _write_report(
        ctx,
        manifest=manifest,
        frame_set=frame_set,
        descriptor=descriptor,
        candidate_count=len(candidates),
    )
    ctx.metadata["frame_intelligence"] = {
        "report": str(report),
        "accepted": frame_set.accepted_count,
        "scene_type": descriptor.scene_type,
    }
    ctx.write_metadata()
    ctx.write_provenance()
    return ctx


def frames_summary(ctx: RunContext) -> dict[str, Any]:
    """Build a compact operator summary dict."""
    fs = ctx.get("frame_set")
    desc = ctx.get("scene_descriptor")
    man = ctx.get("capture_manifest")
    return {
        "job_id": ctx.job_id,
        "capture_kind": man.kind.value if man else None,
        "accepted_keyframes": fs.accepted_count if fs else 0,
        "rejected": fs.rejected_count if fs else 0,
        "scene_type": desc.scene_type if desc else None,
        "texture_score": desc.texture_score if desc else None,
        "output_dir": str(ctx.output_dir),
        "stages": [s.to_dict() for s in ctx.stage_history],
    }
