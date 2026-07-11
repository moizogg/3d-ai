"""Phase 6: inspector, quality, PropertyScene, export pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from propertyscan.core.config import load_config
from propertyscan.domain.gaussian import GaussianScene, GaussianStats
from propertyscan.domain.geometry import GeometryMetrics, GeometryResult, PoseGraph, CameraPose
from propertyscan.domain.quality import HealthReport, InspectionReport
from propertyscan.optimize.inspector import SceneInspector
from propertyscan.optimize.ply_io import write_ply, PlyData
from propertyscan.quality.scorer import score_scene
from propertyscan.export.ply import PlyExporter
from propertyscan.domain.scene import PropertyScene, SceneMetadata
from propertyscan.pipeline.export_pipeline import export_summary, run_export_pipeline


def _ascii_gaussian_ply(path: Path) -> None:
    """Write a small ASCII PLY with scale/opacity attrs for pruning tests."""
    # 5 points: good, floater (low opacity), needle, huge, tiny
    # properties: x y z opacity scale_0 scale_1 scale_2
    vertices = [
        [0, 0, 0, 0.9, 0.1, 0.1, 0.1],  # keep
        [1, 0, 0, 0.01, 0.1, 0.1, 0.1],  # floater
        [2, 0, 0, 0.9, 1.8, 0.05, 0.05],  # needle (aspect~36, max 1.8)
        [3, 0, 0, 0.9, 6.0, 6.0, 6.0],  # huge (linear scales > 5)
        [4, 0, 0, 0.9, 1e-6, 1e-6, 1e-6],  # tiny
        [0.5, 0.5, 0, 0.8, 0.2, 0.2, 0.2],  # keep
    ]
    data = PlyData(
        properties=["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"],
        property_types=["float"] * 7,
        vertices=vertices,
    )
    write_ply(path, data)


def test_inspector_prunes_artifacts(tmp_path: Path) -> None:
    ply = tmp_path / "raw.ply"
    _ascii_gaussian_ply(ply)
    scene = GaussianScene(path=ply, stats=GaussianStats(count=6), trainer_name="test")
    result = SceneInspector().inspect(scene, output_dir=tmp_path / "out")
    assert result.report.total_gaussians_before == 6
    assert result.report.floaters_removed >= 1
    assert result.report.needles_removed >= 1
    assert result.report.huge_removed >= 1
    assert result.report.tiny_removed >= 1
    assert result.report.total_gaussians_after < 6
    assert result.cleaned_path is not None
    assert result.cleaned_path.is_file()


def test_inspector_xyz_only_passthrough(tmp_path: Path) -> None:
    ply = tmp_path / "xyz.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scene = GaussianScene(path=ply, stats=GaussianStats(count=2))
    result = SceneInspector().inspect(scene, output_dir=tmp_path / "o")
    assert result.report.total_gaussians_after == 2
    assert "xyz_only_no_prune" in result.report.notes


def test_quality_scorer_geometry_dominated() -> None:
    geom = GeometryResult(
        provider_name="mock",
        success=True,
        pose_graph=PoseGraph(
            cameras=[
                CameraPose(
                    image_id="0",
                    image_name="a.jpg",
                    registered=True,
                    c2w=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    confidence=0.9,
                ),
                CameraPose(
                    image_id="1",
                    image_name="b.jpg",
                    registered=True,
                    c2w=[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    confidence=0.9,
                ),
            ],
            registered_count=2,
            total_count=2,
            registered_fraction=1.0,
        ),
        metrics=GeometryMetrics(
            registered_cameras=2,
            total_cameras=2,
            registered_fraction=1.0,
            mean_camera_confidence=0.9,
            point_count=1000,
        ),
    )
    health = HealthReport(score=90, expected_quality="excellent", passed=True)
    inspection = InspectionReport(
        total_gaussians_before=1000,
        total_gaussians_after=900,
        needles_removed=5,
        floaters_removed=10,
    )
    q = score_scene(
        geometry=geom, health=health, inspection=inspection, frame_set=None
    )
    assert q.overall >= 70
    assert q.status in ("excellent", "accepted", "marginal")


def test_ply_exporter(tmp_path: Path) -> None:
    ply = tmp_path / "in.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )
    scene = PropertyScene(
        metadata=SceneMetadata(scene_id="t"),
        gaussian_scene=GaussianScene(path=ply, cleaned_path=ply),
    )
    res = PlyExporter().export(scene, tmp_path / "export")
    assert res.success
    assert res.path is not None
    assert res.path.is_file()


def test_end_to_end_export_pipeline_mock(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    for i in range(10):
        img = Image.new("RGB", (160, 120), (10 + i, 20, 30))
        d = ImageDraw.Draw(img)
        for x in range(0, 160, 8):
            d.line([(x + i % 2, 0), (x + i % 2, 120)], fill=(255, 255, 255))
        img.save(src / f"frame_{i:04d}.jpg")

    cfg = load_config(apply_env=False)
    cfg.geometry.engine = "mock"
    cfg.training.backend = "mock"
    cfg.capture.min_frames = 6
    cfg.frame_intelligence.max_keyframes = 8
    cfg.frame_intelligence.min_motion_to_keep = 0.2
    cfg.training.downscale_factor = 2

    out = tmp_path / "out"
    ctx = run_export_pipeline(
        src, out, config=cfg, engine="mock", train_backend="mock"
    )
    s = export_summary(ctx)
    assert s["quality_overall"] is not None
    assert (out / "scene.ply").is_file() or (out / "export" / "scene.ply").is_file()
    assert (out / "property_scene.json").is_file()
    assert (out / "final_report.json").is_file()
    assert (out / "provenance.json").is_file()
