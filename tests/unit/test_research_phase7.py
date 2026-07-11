"""Phase 7: experiment registry, metrics, research layout, benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from propertyscan.core.config import load_config
from propertyscan.pipeline.export_pipeline import run_export_pipeline
from propertyscan.research.artifacts import write_research_layout
from propertyscan.research.benchmark import BenchmarkRunner, _resolve_capture_path
from propertyscan.research.experiment import ExperimentRecord, ExperimentRegistry
from propertyscan.research.metrics import collect_run_metrics


def _scene_folder(root: Path, name: str, n: int = 8) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (120, 90), (10 + i, 20, 30))
        draw = ImageDraw.Draw(img)
        for x in range(0, 120, 6):
            draw.line([(x + i % 2, 0), (x + i % 2, 90)], fill=(255, 255, 255))
        img.save(d / f"frame_{i:04d}.jpg")
    return d


def test_experiment_registry_append(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "reg")
    rec = ExperimentRecord.create(
        scene_id="s1",
        job_id="j1",
        profile="default",
        geometry_engine="mock",
        train_backend="mock",
        quality_preset="standard",
        metrics={"quality_overall": 90},
        tags=["test"],
    )
    reg.append(rec)
    assert reg.history_path.is_file()
    rows = reg.list_recent()
    assert len(rows) == 1
    assert rows[0]["scene_id"] == "s1"
    assert rows[0]["metrics"]["quality_overall"] == 90


def test_collect_metrics_and_research_layout(tmp_path: Path) -> None:
    src = _scene_folder(tmp_path, "scene_a", n=8)
    cfg = load_config(apply_env=False)
    cfg.geometry.engine = "mock"
    cfg.training.backend = "mock"
    cfg.capture.min_frames = 4
    cfg.frame_intelligence.max_keyframes = 6
    cfg.frame_intelligence.min_motion_to_keep = 0.2
    cfg.training.downscale_factor = 2

    ctx = run_export_pipeline(
        src, tmp_path / "run", config=cfg, engine="mock", train_backend="mock"
    )
    metrics = collect_run_metrics(ctx)
    assert metrics["geometry_success"] is True
    assert "quality_overall" in metrics

    research = write_research_layout(ctx, tmp_path / "research", scene_id="scene_a")
    assert (research / "metrics.json").is_file()
    assert (research / "Metadata" / "provenance.json").is_file()
    assert (research / "Quality_Report").is_dir()
    data = json.loads((research / "metrics.json").read_text(encoding="utf-8"))
    assert data["job_id"] == ctx.job_id


def test_resolve_capture_nested_images(tmp_path: Path) -> None:
    scene = tmp_path / "s"
    images = scene / "images"
    images.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), (i, 0, 0)).save(images / f"{i}.jpg")
    assert _resolve_capture_path(scene) == images


def test_benchmark_runner(tmp_path: Path) -> None:
    data = tmp_path / "bench"
    _scene_folder(data, "01_room", n=8)
    _scene_folder(data, "02_hall", n=8)

    cfg = load_config(apply_env=False)
    cfg.capture.min_frames = 4
    cfg.frame_intelligence.max_keyframes = 6
    cfg.frame_intelligence.min_motion_to_keep = 0.2
    cfg.training.downscale_factor = 2

    runner = BenchmarkRunner(
        data_dir=data,
        output_dir=tmp_path / "out",
        config=cfg,
        engine="mock",
        train_backend="mock",
    )
    results = runner.run_all()
    assert len(results) == 2
    assert all(r.success for r in results)
    assert (tmp_path / "out" / "benchmark_summary.json").is_file()
    assert (tmp_path / "out" / "registry" / "history.jsonl").is_file()
    history = (tmp_path / "out" / "registry" / "history.jsonl").read_text(encoding="utf-8")
    assert history.count("\n") >= 2
