"""Collect comparable metrics from a finished RunContext."""

from __future__ import annotations

from typing import Any

from propertyscan.core.context import RunContext


def collect_run_metrics(ctx: RunContext) -> dict[str, Any]:
    """Flatten key metrics for experiment history / benchmarks.

    Purpose:
        One dict that can be appended to history.jsonl across runs.

    Non-responsibilities:
        Does not compute PSNR/SSIM (optional later with holdout renders).
    """
    metrics: dict[str, Any] = {
        "job_id": ctx.job_id,
        "profile": ctx.config.engine.profile,
        "geometry_engine": ctx.config.geometry.engine,
        "train_backend": ctx.config.training.backend,
        "device": ctx.device.device,
        "device_name": ctx.device.device_name,
        "vram_gb": ctx.device.total_vram_gb,
    }

    fs = ctx.get("frame_set")
    if fs is not None:
        metrics["keyframes"] = fs.accepted_count
        metrics["frame_rejected"] = fs.rejected_count

    geom = ctx.get("geometry_result")
    if geom is not None:
        metrics["geometry_success"] = geom.success
        metrics["geometry_provider"] = geom.provider_name
        metrics["registered_cameras"] = geom.metrics.registered_cameras
        metrics["registered_fraction"] = geom.metrics.registered_fraction
        metrics["point_count"] = geom.metrics.point_count
        metrics["geometry_time_s"] = geom.metrics.execution_time_s
        metrics["peak_vram_gb"] = geom.metrics.peak_vram_gb
        metrics["model_id"] = geom.metrics.model_id

    health = ctx.get("health_report")
    if health is not None:
        metrics["health_score"] = health.score
        metrics["health_passed"] = health.passed
        metrics["needle_probability"] = health.needle_probability
        metrics["floater_probability"] = health.floater_probability

    ds = ctx.get("training_dataset")
    if ds is not None:
        metrics["dataset_frames"] = ds.frame_count
        metrics["has_depth"] = ds.has_depth
        metrics["has_init_cloud"] = ds.has_init_point_cloud

    tr = ctx.get("train_result")
    if tr is not None:
        metrics["train_success"] = tr.success
        metrics["train_iterations"] = tr.iterations
        metrics["train_time_s"] = tr.execution_time_s

    insp = ctx.get("inspection_report")
    if insp is not None:
        metrics["gaussians_before"] = insp.total_gaussians_before
        metrics["gaussians_after"] = insp.total_gaussians_after
        metrics["needles_removed"] = insp.needles_removed
        metrics["floaters_removed"] = insp.floaters_removed
        metrics["size_reduction_pct"] = insp.size_reduction_pct

    q = ctx.get("quality_report")
    if q is not None:
        metrics["quality_overall"] = q.overall
        metrics["quality_status"] = q.status
        metrics["quality_camera"] = q.camera
        metrics["quality_artifacts"] = q.artifacts
        metrics["failure_class"] = q.failure_class

    # Stage durations
    stage_times = {
        s.stage_name: s.duration_s for s in ctx.stage_history if s.duration_s
    }
    if stage_times:
        metrics["stage_times_s"] = stage_times
        metrics["total_stage_time_s"] = round(sum(stage_times.values()), 3)

    return metrics
