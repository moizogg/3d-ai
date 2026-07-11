"""Benchmark runner — run pipeline on a folder of scenes, log history."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig, load_config
from propertyscan.pipeline.export_pipeline import run_export_pipeline
from propertyscan.research.artifacts import write_research_layout
from propertyscan.research.experiment import ExperimentRecord, ExperimentRegistry
from propertyscan.research.metrics import collect_run_metrics

logger = logging.getLogger("propertyscan.research.benchmark")


@dataclass
class BenchmarkSceneResult:
    scene_id: str
    success: bool
    output_dir: Path
    research_dir: Path | None = None
    experiment_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "success": self.success,
            "output_dir": str(self.output_dir),
            "research_dir": str(self.research_dir) if self.research_dir else None,
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "error": self.error,
        }


class BenchmarkRunner:
    """Run export pipeline on each scene under a benchmarks root.

    Scene discovery:
      - subdirectories of ``data_dir`` that contain images or a video
      - skips hidden dirs and ``history.jsonl`` peers

    For CI: use engine=mock, train_backend=mock.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        output_dir: Path,
        config: EngineConfig | None = None,
        profile: str | None = None,
        engine: str = "mock",
        train_backend: str = "mock",
        quality: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_config(
            profile=profile, quality=quality, apply_env=False  # type: ignore[arg-type]
        )
        self.engine = engine
        self.train_backend = train_backend
        self.quality = quality
        self.profile = profile
        self.registry = ExperimentRegistry(self.output_dir / "registry")

    def discover_scenes(self) -> list[Path]:
        if not self.data_dir.is_dir():
            return []
        scenes: list[Path] = []
        for child in sorted(self.data_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name in {"registry", "history"}:
                continue
            if _looks_like_scene(child):
                scenes.append(child)
        # Also allow flat image folders as a single scene named after parent
        if not scenes and _looks_like_scene(self.data_dir):
            scenes.append(self.data_dir)
        return scenes

    def run_all(self) -> list[BenchmarkSceneResult]:
        scenes = self.discover_scenes()
        results: list[BenchmarkSceneResult] = []
        for scene_path in scenes:
            results.append(self.run_scene(scene_path))
        summary_path = self.output_dir / "benchmark_summary.json"
        summary_path.write_text(
            json.dumps([r.to_dict() for r in results], indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Benchmark complete: %d scenes → %s", len(results), summary_path)
        return results

    def run_scene(self, scene_path: Path) -> BenchmarkSceneResult:
        scene_id = scene_path.name if scene_path.resolve() != self.data_dir.resolve() else "default_scene"
        out = self.output_dir / "runs" / scene_id
        out.mkdir(parents=True, exist_ok=True)
        input_path = _resolve_capture_path(scene_path)
        logger.info("Benchmark scene %s ← %s", scene_id, input_path)
        try:
            ctx = run_export_pipeline(
                input_path,
                out,
                config=self.config,
                engine=self.engine,
                train_backend=self.train_backend,
                quality=self.quality,
                profile=self.profile,
            )
            metrics = collect_run_metrics(ctx)
            research_dir = write_research_layout(
                ctx, self.output_dir / "research", scene_id=scene_id
            )
            models = ctx.provenance.models if ctx.provenance else {}
            rec = ExperimentRecord.create(
                scene_id=scene_id,
                job_id=ctx.job_id,
                profile=ctx.config.engine.profile,
                geometry_engine=ctx.config.geometry.engine,
                train_backend=ctx.config.training.backend,
                quality_preset=ctx.config.training.quality,
                metrics=metrics,
                models=models,
                tags=["benchmark"],
                notes=f"benchmark scene {scene_id}",
                output_dir=str(out),
            )
            self.registry.append(rec)
            return BenchmarkSceneResult(
                scene_id=scene_id,
                success=True,
                output_dir=out,
                research_dir=research_dir,
                experiment_id=rec.experiment_id,
                metrics=metrics,
            )
        except Exception as exc:
            logger.exception("Benchmark scene %s failed", scene_id)
            # Still log failure experiment
            rec = ExperimentRecord.create(
                scene_id=scene_id,
                job_id="failed",
                profile=self.config.engine.profile,
                geometry_engine=self.engine,
                train_backend=self.train_backend,
                quality_preset=self.config.training.quality,
                metrics={},
                tags=["benchmark", "failed"],
                notes=str(exc),
                output_dir=str(out),
            )
            self.registry.append(rec)
            return BenchmarkSceneResult(
                scene_id=scene_id,
                success=False,
                output_dir=out,
                experiment_id=rec.experiment_id,
                error=str(exc),
            )


def _looks_like_scene(path: Path) -> bool:
    try:
        _resolve_capture_path(path)
        return True
    except ValueError:
        return False


def _resolve_capture_path(path: Path) -> Path:
    """Pick video file or image folder inside a scene directory."""
    video_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    image_ext = {".jpg", ".jpeg", ".png", ".webp"}
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"Not a scene path: {path}")

    videos = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in video_ext]
    if videos:
        return sorted(videos)[0]

    images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in image_ext]
    if len(images) >= 2:
        return path

    for sub in ("images", "Frames", "frames"):
        img_dir = path / sub
        if img_dir.is_dir():
            nested = [
                p
                for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_ext
            ]
            if len(nested) >= 2:
                return img_dir

    raise ValueError(f"No video or image set found under {path}")
