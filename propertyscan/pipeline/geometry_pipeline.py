"""Phase 3 geometry pipeline: stages 7–11 (optionally after frames 1–6)."""

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
from propertyscan.pipeline.frame_pipeline import build_frame_stages, run_frames_pipeline
from propertyscan.pipeline.stages.s07_route import RouteGeometryStage
from propertyscan.pipeline.stages.s08_reconstruct import ReconstructGeometryStage
from propertyscan.pipeline.stages.s09_depth import EstimateDepthStage
from propertyscan.pipeline.stages.s10_fusion import FuseGeometryStage
from propertyscan.pipeline.stages.s11_health import HealthGateStage

logger = logging.getLogger("propertyscan.pipeline.geometry")


def build_geometry_stages() -> list[Stage]:
    return [
        RouteGeometryStage(),
        ReconstructGeometryStage(),
        EstimateDepthStage(),
        FuseGeometryStage(),
        HealthGateStage(),
    ]


def run_geometry_stages(ctx: RunContext) -> RunContext:
    """Run stages 7–11 on an existing context that already has frame_set."""
    for stage in build_geometry_stages():
        stage.run(ctx)
    _write_geometry_report(ctx)
    ctx.write_metadata()
    ctx.write_provenance()
    return ctx


def run_geometry_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    profile: str | None = None,
    quality: str | None = None,
    config: EngineConfig | None = None,
    engine: str | None = None,
    skip_frames: bool = False,
) -> RunContext:
    """Full path: frames (1–6) then geometry (7–11), or geometry-only if context prepared.

    For Phase 3 tests, set ``config.geometry.engine = "mock"`` (or engine=\"mock\").
    """
    cfg = config or load_config(profile=profile, quality=quality)  # type: ignore[arg-type]
    if engine:
        cfg.geometry.engine = engine  # type: ignore[assignment]

    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format)

    if not skip_frames:
        ctx = run_frames_pipeline(
            input_path,
            output_dir,
            config=cfg,
        )
    else:
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        ctx = RunContext(
            config=cfg,
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            work_dir=Path(output_dir) / "work" / job_id,
            job_id=job_id,
            device=resolve_device(prefer_cuda=cfg.device.prefer_cuda),
        )

    return run_geometry_stages(ctx)


def geometry_summary(ctx: RunContext) -> dict[str, Any]:
    geom = ctx.get("geometry_result")
    health = ctx.get("health_report")
    fused = ctx.get("fused_geometry")
    return {
        "job_id": ctx.job_id,
        "provider": geom.provider_name if geom else None,
        "geometry_success": geom.success if geom else False,
        "registered": geom.metrics.registered_cameras if geom else 0,
        "registered_fraction": geom.metrics.registered_fraction if geom else 0.0,
        "health_score": health.score if health else None,
        "health_passed": health.passed if health else None,
        "depth_attached": bool(
            fused and fused.depth and fused.depth.success
        )
        if fused
        else False,
        "engine": ctx.config.geometry.engine,
    }


def _write_geometry_report(ctx: RunContext) -> Path:
    path = ctx.output_dir / "geometry_report.json"
    payload = {
        "job_id": ctx.job_id,
        "summary": geometry_summary(ctx),
        "geometry": ctx.get("geometry_result").to_dict()
        if ctx.get("geometry_result")
        else None,
        "depth": ctx.get("depth_result").to_dict()
        if ctx.get("depth_result")
        else None,
        "fusion": ctx.get("fused_geometry").to_dict()
        if ctx.get("fused_geometry")
        else None,
        "validation": ctx.get("geometry_validation").to_dict()
        if ctx.get("geometry_validation")
        else None,
        "health": ctx.get("health_report").to_dict()
        if ctx.get("health_report")
        else None,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
