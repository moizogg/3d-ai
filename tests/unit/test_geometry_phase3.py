"""Phase 3: geometry interfaces, router, fusion, health (mock-backed)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from propertyscan.core.config import load_config
from propertyscan.domain.frames import FrameMetadata, FrameSet, FrameStatus
from propertyscan.domain.geometry import SceneDescriptor
from propertyscan.geometry.depth.mock import MockDepthProvider
from propertyscan.geometry.fusion.fuse import fuse_geometry_and_depth
from propertyscan.geometry.providers.dust3r import DUSt3RProvider
from propertyscan.geometry.providers.mast3r import MASt3RProvider
from propertyscan.geometry.providers.mock import MockGeometryProvider
from propertyscan.geometry.router import GeometryRouter
from propertyscan.geometry.validation.geometric import validate_geometry
from propertyscan.geometry.validation.health import evaluate_pretrain_health
from propertyscan.pipeline.geometry_pipeline import (
    geometry_summary,
    run_geometry_pipeline,
)


def _frame_set(tmp_path: Path, n: int = 8) -> FrameSet:
    frames = []
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.jpg"
        img = Image.new("RGB", (320, 240), (30 + i, 40, 50))
        d = ImageDraw.Draw(img)
        for x in range(0, 320, 12):
            d.line([(x + i % 3, 0), (x + i % 3, 240)], fill=(255, 255, 255))
        img.save(p)
        frames.append(
            FrameMetadata(
                filename=p.name,
                filepath=p,
                width=320,
                height=240,
                index=i,
                status=FrameStatus.ACCEPTED,
                quality_score=80.0,
                confidence_score=80.0,
                rank_score=80.0,
            )
        )
    fs = FrameSet(
        source_type="test",
        source_path=tmp_path,
        frames=frames,
        validation_mode="reliable_v2",
    )
    fs.recompute_counts()
    return fs


def test_mock_provider_success(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    fs = _frame_set(tmp_path)
    res = MockGeometryProvider(cfg).reconstruct(
        fs, output_dir=tmp_path / "geo", config=cfg
    )
    assert res.success
    assert res.metrics.registered_cameras == 8
    assert res.pose_graph is not None
    assert all(c.c2w is not None for c in res.pose_graph.cameras if c.registered)
    assert (tmp_path / "geo" / "transforms.json").is_file()


def test_mast3r_honest_failure_without_weights(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    cfg.device.allow_cpu_geometry = False
    fs = _frame_set(tmp_path, n=4)
    res = MASt3RProvider(cfg).reconstruct(fs, output_dir=tmp_path / "m", config=cfg)
    assert res.success is False
    assert res.error_message
    # No silent success: missing torch/dust3r/mast3r/CUDA → clear error
    msg = res.error_message.lower()
    assert any(
        k in msg
        for k in ("cuda", "pytorch", "torch", "dust3r", "mast3r", "missing", "install")
    )


def test_dust3r_honest_failure(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    fs = _frame_set(tmp_path, n=4)
    res = DUSt3RProvider(cfg).reconstruct(fs, output_dir=tmp_path / "d", config=cfg)
    assert res.success is False
    assert res.error_message


def test_router_auto_prefers_mast3r_for_low_texture() -> None:
    cfg = load_config(apply_env=False)
    cfg.geometry.engine = "auto"
    router = GeometryRouter(cfg, include_mock=True)
    desc = SceneDescriptor(
        scene_type="low_texture_interior",
        texture_score=5.0,
        frame_count=40,
    )
    ranked = router.rank(desc)
    assert ranked[0][1].name == "mast3r"


def test_router_force_mock(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    cfg.geometry.engine = "mock"
    router = GeometryRouter(cfg, include_mock=True)
    fs = _frame_set(tmp_path)
    desc = SceneDescriptor(frame_count=fs.accepted_count)
    res = router.reconstruct(fs, output_dir=tmp_path / "r", descriptor=desc, config=cfg)
    assert res.success
    assert res.provider_name == "mock"


def test_no_colmap_in_router() -> None:
    cfg = load_config(apply_env=False)
    router = GeometryRouter(cfg, include_mock=True)
    names = [p.name for p in router.providers]
    assert "colmap" not in names
    assert "sift" not in names


def test_fusion_attaches_depth(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    fs = _frame_set(tmp_path, n=5)
    geo = MockGeometryProvider(cfg).reconstruct(
        fs, output_dir=tmp_path / "g", config=cfg
    )
    depth = MockDepthProvider(cfg).estimate(fs, output_dir=tmp_path / "dep", config=cfg)
    assert depth.success
    fused = fuse_geometry_and_depth(geo, depth, output_dir=tmp_path / "fus")
    assert fused.geometry.success
    assert fused.depth is not None and fused.depth.success
    assert any("depth_attached" in n for n in fused.notes)


def test_health_passes_for_mock(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    fs = _frame_set(tmp_path, n=8)
    geo = MockGeometryProvider(cfg).reconstruct(
        fs, output_dir=tmp_path / "g", config=cfg
    )
    depth = MockDepthProvider(cfg).estimate(fs, output_dir=tmp_path / "d", config=cfg)
    fused = fuse_geometry_and_depth(geo, depth)
    val = validate_geometry(fused, cfg)
    assert val.passed
    health = evaluate_pretrain_health(
        fused=fused,
        validation=val,
        frame_set=fs,
        descriptor=SceneDescriptor(frame_count=8, texture_score=40),
        config=cfg,
    )
    assert health.passed
    assert health.score >= cfg.health.min_score


def test_end_to_end_geometry_pipeline_mock(tmp_path: Path) -> None:
    src = tmp_path / "input"
    src.mkdir()
    for i in range(10):
        img = Image.new("RGB", (320, 240), (20 + i, 40, 60))
        d = ImageDraw.Draw(img)
        for x in range(0, 320, 10):
            d.line([(x + i % 4, 0), (x + i % 4, 240)], fill=(255, 255, 255))
        img.save(src / f"frame_{i:04d}.jpg")

    cfg = load_config(apply_env=False)
    cfg.geometry.engine = "mock"
    cfg.capture.min_frames = 6
    cfg.frame_intelligence.max_keyframes = 8
    cfg.frame_intelligence.min_motion_to_keep = 0.2

    out = tmp_path / "out"
    ctx = run_geometry_pipeline(src, out, config=cfg, engine="mock")
    s = geometry_summary(ctx)
    assert s["geometry_success"] is True
    assert s["provider"] == "mock"
    assert s["health_passed"] is True
    assert s["depth_attached"] is True
    assert (out / "geometry_report.json").is_file()
