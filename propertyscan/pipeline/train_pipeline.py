"""Phase 5 training pipeline: geometry (7–11) + dataset + train (12–13)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig, load_config
from propertyscan.core.context import RunContext
from propertyscan.core.logging import setup_logging
from propertyscan.core.stage import Stage
from propertyscan.pipeline.geometry_pipeline import run_geometry_pipeline
from propertyscan.pipeline.stages.s12_dataset import BuildDatasetStage
from propertyscan.pipeline.stages.s13_train import TrainGaussiansStage

logger = logging.getLogger("propertyscan.pipeline.train")


def build_train_stages() -> list[Stage]:
    return [BuildDatasetStage(), TrainGaussiansStage()]


def run_train_stages(ctx: RunContext) -> RunContext:
    for stage in build_train_stages():
        stage.run(ctx)
    _write_train_report(ctx)
    ctx.write_metadata()
    ctx.write_provenance()
    return ctx


def run_train_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    profile: str | None = None,
    quality: str | None = None,
    config: EngineConfig | None = None,
    engine: str | None = None,
    train_backend: str | None = None,
) -> RunContext:
    """Frames → geometry → dataset → train.

    For CI without GPU/ns-train: ``engine=mock`` and ``train_backend=mock``.
    """
    cfg = config or load_config(profile=profile, quality=quality)  # type: ignore[arg-type]
    if engine:
        cfg.geometry.engine = engine  # type: ignore[assignment]
    if train_backend:
        cfg.training.backend = train_backend  # type: ignore[assignment]
    if quality:
        cfg.training.quality = quality  # type: ignore[assignment]

    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format)
    ctx = run_geometry_pipeline(
        input_path,
        output_dir,
        config=cfg,
        engine=engine,
    )
    return run_train_stages(ctx)


def train_summary(ctx: RunContext) -> dict[str, Any]:
    ds = ctx.get("training_dataset")
    tr = ctx.get("train_result")
    gs = ctx.get("gaussian_scene")
    health = ctx.get("health_report")
    return {
        "job_id": ctx.job_id,
        "dataset_frames": ds.frame_count if ds else 0,
        "has_depth": ds.has_depth if ds else False,
        "has_init_cloud": ds.has_init_point_cloud if ds else False,
        "train_backend": tr.backend if tr else None,
        "train_success": tr.success if tr else False,
        "iterations": tr.iterations if tr else 0,
        "ply": str(gs.path) if gs and gs.path else None,
        "health_score": health.score if health else None,
        "geometry_engine": ctx.config.geometry.engine,
    }


def _write_train_report(ctx: RunContext) -> Path:
    path = ctx.output_dir / "train_report.json"
    payload = {
        "job_id": ctx.job_id,
        "summary": train_summary(ctx),
        "dataset": ctx.get("training_dataset").to_dict()
        if ctx.get("training_dataset")
        else None,
        "train_result": ctx.get("train_result").to_dict()
        if ctx.get("train_result")
        else None,
        "gaussian_scene": ctx.get("gaussian_scene").to_dict()
        if ctx.get("gaussian_scene")
        else None,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
