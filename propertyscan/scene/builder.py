"""Assemble the canonical PropertyScene from pipeline context pieces."""

from __future__ import annotations

from typing import Any

from propertyscan import __version__
from propertyscan.core.context import RunContext
from propertyscan.domain.scene import PropertyScene, SceneMetadata


def build_property_scene(ctx: RunContext) -> PropertyScene:
    """Build PropertyScene from RunContext state.

    Purpose:
        Single product object for exporters and archives.

    Non-responsibilities:
        Does not modify geometry or re-train.
    """
    geom = ctx.get("geometry_result")
    pose_graph = None
    if geom is not None and geom.pose_graph is not None:
        pose_graph = geom.pose_graph

    gs = ctx.get("gaussian_scene")
    history = [s.to_dict() for s in ctx.stage_history]

    stats: dict[str, Any] = {
        "job_id": ctx.job_id,
        "profile": ctx.config.engine.profile,
        "geometry_engine": ctx.config.geometry.engine,
        "train_backend": ctx.config.training.backend,
        "device": ctx.device.to_dict(),
    }
    ds = ctx.get("training_dataset")
    if ds is not None:
        stats["dataset_frames"] = ds.frame_count
        stats["has_depth"] = ds.has_depth
        stats["has_init_cloud"] = ds.has_init_point_cloud
    if gs is not None and gs.stats is not None:
        stats["gaussian_count"] = gs.stats.count

    scene = PropertyScene(
        metadata=SceneMetadata(
            scene_id=ctx.job_id,
            engine_version=__version__,
            profile=ctx.config.engine.profile,
            source_path=str(ctx.input_path),
            notes=["assembled_by_propertyscan_phase6"],
        ),
        frame_set=ctx.get("frame_set"),
        scene_descriptor=ctx.get("scene_descriptor"),
        geometry=geom,
        pose_graph=pose_graph,
        depth=ctx.get("depth_result"),
        gaussian_scene=gs,
        health=ctx.get("health_report"),
        inspection=ctx.get("inspection_report"),
        quality=ctx.get("quality_report"),
        processing_history=history,
        exports=dict(ctx.get("exports") or {}),
        statistics=stats,
    )
    return scene
