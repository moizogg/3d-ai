"""Standard research-oriented artifact tree for a finished run."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from propertyscan.core.context import RunContext
from propertyscan.research.metrics import collect_run_metrics

logger = logging.getLogger("propertyscan.research.artifacts")

# Research-friendly layout (bible spirit, without COLMAP/ folder)
LAYOUT_DIRS = (
    "Frames",
    "Geometry",
    "Depth",
    "Transforms",
    "Training_Logs",
    "PLY",
    "Quality_Report",
    "Metadata",
)


def write_research_layout(
    ctx: RunContext,
    research_root: Path,
    *,
    scene_id: str | None = None,
) -> Path:
    """Copy key outputs into a stable research tree.

    ::

        research_root/<scene_id>/
          Frames/
          Geometry/
          Depth/
          Transforms/
          Training_Logs/
          PLY/
          Quality_Report/
          Metadata/
          metrics.json
    """
    scene_id = scene_id or ctx.job_id
    root = Path(research_root) / scene_id
    for name in LAYOUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)

    # Frames (selected keyframes)
    fs = ctx.get("frame_set")
    if fs is not None:
        for p in fs.get_accepted_paths():
            if Path(p).is_file():
                _copy(Path(p), root / "Frames" / Path(p).name)

    # Geometry artifacts
    geom = ctx.get("geometry_result")
    if geom is not None:
        (root / "Geometry" / "geometry_result.json").write_text(
            json.dumps(geom.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        for key in ("transforms", "dense_ply"):
            p = geom.artifacts.get(key)
            if p and Path(p).is_file():
                dest_dir = root / ("Transforms" if key == "transforms" else "Geometry")
                _copy(Path(p), dest_dir / Path(p).name)

    # Depth
    depth = ctx.get("depth_result")
    if depth is not None and depth.success:
        for dm in depth.depth_maps:
            if Path(dm.path).is_file():
                _copy(Path(dm.path), root / "Depth" / Path(dm.path).name)

    # Dataset transforms (training package)
    ds = ctx.get("training_dataset")
    if ds is not None and Path(ds.transforms_path).is_file():
        _copy(Path(ds.transforms_path), root / "Transforms" / "dataset_transforms.json")

    # Training
    tr = ctx.get("train_result")
    if tr is not None:
        (root / "Training_Logs" / "train_result.json").write_text(
            json.dumps(tr.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        if tr.train_dir:
            log = Path(tr.train_dir) / "ns_train.log"
            if log.is_file():
                _copy(log, root / "Training_Logs" / "ns_train.log")

    # PLY
    gs = ctx.get("gaussian_scene")
    if gs is not None:
        for label, p in (("raw", gs.path), ("cleaned", gs.cleaned_path)):
            if p and Path(p).is_file():
                _copy(Path(p), root / "PLY" / f"{label}_scene.ply")
    exports = ctx.get("exports") or {}
    for name, p in exports.items():
        if p and Path(p).is_file() and str(p).endswith(".ply"):
            _copy(Path(p), root / "PLY" / Path(p).name)

    # Quality / reports
    for key, filename in (
        ("quality_report", "quality_report.json"),
        ("inspection_report", "inspection_report.json"),
        ("health_report", "health_report.json"),
    ):
        obj = ctx.get(key)
        if obj is not None and hasattr(obj, "to_dict"):
            (root / "Quality_Report" / filename).write_text(
                json.dumps(obj.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

    for name in ("final_report.json", "train_report.json", "geometry_report.json"):
        src = ctx.output_dir / name
        if src.is_file():
            _copy(src, root / "Quality_Report" / name)

    # Metadata
    metrics = collect_run_metrics(ctx)
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    if ctx.provenance is not None:
        (root / "Metadata" / "provenance.json").write_text(
            json.dumps(ctx.provenance.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
    meta = ctx.output_dir / "metadata.json"
    if meta.is_file():
        _copy(meta, root / "Metadata" / "metadata.json")
    scene_json = ctx.output_dir / "property_scene.json"
    if scene_json.is_file():
        _copy(scene_json, root / "Metadata" / "property_scene.json")

    logger.info("Research layout written: %s", root)
    return root


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.copy2(src, dest)
