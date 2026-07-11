"""Phase 6 export pipeline: train (+ geometry) then inspect → quality → assemble → export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig, load_config
from propertyscan.core.context import RunContext
from propertyscan.core.logging import setup_logging
from propertyscan.core.stage import Stage
from propertyscan.pipeline.stages.s14_inspect import InspectSceneStage
from propertyscan.pipeline.stages.s15_quality import QualityScoreStage
from propertyscan.pipeline.stages.s16_assemble import AssembleSceneStage
from propertyscan.pipeline.stages.s17_export import ExportStage
from propertyscan.pipeline.train_pipeline import run_train_pipeline

logger = logging.getLogger("propertyscan.pipeline.export")


def build_export_stages() -> list[Stage]:
    return [
        InspectSceneStage(),
        QualityScoreStage(),
        AssembleSceneStage(),
        ExportStage(),
    ]


def run_export_stages(ctx: RunContext) -> RunContext:
    for stage in build_export_stages():
        stage.run(ctx)
    _write_final_report(ctx)
    ctx.write_metadata()
    ctx.write_provenance()
    return ctx


def run_export_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    profile: str | None = None,
    quality: str | None = None,
    config: EngineConfig | None = None,
    engine: str | None = None,
    train_backend: str | None = None,
) -> RunContext:
    """Full path through export (frames → geometry → train → inspect → export)."""
    cfg = config or load_config(profile=profile, quality=quality)  # type: ignore[arg-type]
    if engine:
        cfg.geometry.engine = engine  # type: ignore[assignment]
    if train_backend:
        cfg.training.backend = train_backend  # type: ignore[assignment]
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format)

    ctx = run_train_pipeline(
        input_path,
        output_dir,
        config=cfg,
        engine=engine,
        train_backend=train_backend,
        quality=quality,
        profile=profile,
    )
    return run_export_stages(ctx)


def export_summary(ctx: RunContext) -> dict[str, Any]:
    q = ctx.get("quality_report")
    insp = ctx.get("inspection_report")
    exports = ctx.get("exports") or {}
    scene = ctx.get("property_scene")
    return {
        "job_id": ctx.job_id,
        "quality_overall": q.overall if q else None,
        "quality_status": q.status if q else None,
        "gaussians_before": insp.total_gaussians_before if insp else None,
        "gaussians_after": insp.total_gaussians_after if insp else None,
        "exports": exports,
        "scene_id": scene.metadata.scene_id if scene else None,
    }


def _write_final_report(ctx: RunContext) -> Path:
    path = ctx.output_dir / "final_report.json"
    payload = {
        "job_id": ctx.job_id,
        "summary": export_summary(ctx),
        "quality": ctx.get("quality_report").to_dict()
        if ctx.get("quality_report")
        else None,
        "inspection": ctx.get("inspection_report").to_dict()
        if ctx.get("inspection_report")
        else None,
        "exports": ctx.get("exports"),
        "property_scene": str(ctx.output_dir / "property_scene.json"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
