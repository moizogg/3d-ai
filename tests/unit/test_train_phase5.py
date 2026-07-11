"""Phase 5: dataset builder, presets, mock trainer, end-to-end train pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from propertyscan.core.config import load_config
from propertyscan.dataset.builder import DatasetBuildError, DatasetBuilder
from propertyscan.domain.frames import FrameMetadata, FrameSet, FrameStatus
from propertyscan.domain.geometry import PointCloud
from propertyscan.geometry.providers.mock import MockGeometryProvider
from propertyscan.geometry.depth.mock import MockDepthProvider
from propertyscan.geometry.fusion.fuse import fuse_geometry_and_depth
from propertyscan.training.factory import get_trainer
from propertyscan.training.presets import resolve_train_preset
from propertyscan.pipeline.train_pipeline import run_train_pipeline, train_summary


def _keyframes(tmp_path: Path, n: int = 6) -> FrameSet:
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        p = tmp_path / f"keyframe_{i:04d}.jpg"
        img = Image.new("RGB", (160, 120), (20 + i, 40, 50))
        d = ImageDraw.Draw(img)
        for x in range(0, 160, 8):
            d.line([(x + i % 2, 0), (x + i % 2, 120)], fill=(255, 255, 255))
        img.save(p)
        frames.append(
            FrameMetadata(
                filename=p.name,
                filepath=p,
                width=160,
                height=120,
                index=i,
                status=FrameStatus.ACCEPTED,
                quality_score=80.0,
                confidence_score=80.0,
            )
        )
    fs = FrameSet(source_type="test", source_path=tmp_path, frames=frames)
    fs.recompute_counts()
    return fs


def test_dataset_builder_writes_transforms_and_images(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    fs = _keyframes(tmp_path / "kf")
    geo = MockGeometryProvider(cfg).reconstruct(
        fs, output_dir=tmp_path / "geo", config=cfg
    )
    # Attach fake ply for init cloud
    ply = tmp_path / "geo" / "cloud.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")
    geo.point_cloud = PointCloud(source="mock", path=ply, point_count=10)

    depth = MockDepthProvider(cfg).estimate(fs, output_dir=tmp_path / "dep", config=cfg)
    fused = fuse_geometry_and_depth(geo, depth)

    ds = DatasetBuilder().build(
        frame_set=fs,
        geometry=fused,
        output_dir=tmp_path / "dataset",
        config=cfg,
        depth=depth,
    )
    assert ds.frame_count >= 2
    assert ds.transforms_path.is_file()
    assert ds.has_depth is True
    assert ds.has_init_point_cloud is True
    data = json.loads(ds.transforms_path.read_text(encoding="utf-8"))
    assert len(data["frames"]) == ds.frame_count
    assert data.get("ply_file_path")
    assert any(f.get("depth_file_path") for f in data["frames"])
    for fr in data["frames"]:
        assert (ds.root / fr["file_path"]).is_file()


def test_dataset_builder_rejects_failed_geometry(tmp_path: Path) -> None:
    from propertyscan.geometry.result_builder import failure_result

    cfg = load_config(apply_env=False)
    fs = _keyframes(tmp_path, n=3)
    bad = failure_result("x", "nope")
    try:
        DatasetBuilder().build(
            frame_set=fs, geometry=bad, output_dir=tmp_path / "d", config=cfg
        )
        assert False, "expected DatasetBuildError"
    except DatasetBuildError:
        pass


def test_dense_geometry_caps_iterations() -> None:
    cfg = load_config(apply_env=False)
    cfg.training.quality = "high"
    cfg.training.reduce_iters_for_dense_geometry = True
    cfg.training.dense_geometry_max_iters = 8000
    from propertyscan.domain.dataset import TrainingDataset

    ds = TrainingDataset(
        root=Path("."),
        images_dir=Path("images"),
        transforms_path=Path("t.json"),
        frame_count=10,
        has_init_point_cloud=True,
    )
    preset = resolve_train_preset(cfg, ds)
    assert preset.iterations == 8000


def test_mock_trainer_writes_ply(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    cfg.training.backend = "mock"
    fs = _keyframes(tmp_path / "kf")
    geo = MockGeometryProvider(cfg).reconstruct(
        fs, output_dir=tmp_path / "geo", config=cfg
    )
    ds = DatasetBuilder().build(
        frame_set=fs, geometry=geo, output_dir=tmp_path / "ds", config=cfg
    )
    result = get_trainer(cfg).train(ds, output_dir=tmp_path / "train", config=cfg)
    assert result.success
    assert result.scene is not None
    assert result.scene.path is not None
    assert result.scene.path.is_file()


def test_end_to_end_train_pipeline_mock(tmp_path: Path) -> None:
    src = tmp_path / "input"
    src.mkdir()
    for i in range(10):
        img = Image.new("RGB", (200, 150), (15 + i, 30, 45))
        d = ImageDraw.Draw(img)
        for x in range(0, 200, 10):
            d.line([(x + i % 3, 0), (x + i % 3, 150)], fill=(255, 255, 255))
        img.save(src / f"frame_{i:04d}.jpg")

    cfg = load_config(apply_env=False)
    cfg.geometry.engine = "mock"
    cfg.training.backend = "mock"
    cfg.capture.min_frames = 6
    cfg.frame_intelligence.max_keyframes = 8
    cfg.frame_intelligence.min_motion_to_keep = 0.2
    cfg.training.downscale_factor = 2

    out = tmp_path / "out"
    ctx = run_train_pipeline(
        src, out, config=cfg, engine="mock", train_backend="mock"
    )
    s = train_summary(ctx)
    assert s["train_success"] is True
    assert s["dataset_frames"] >= 6
    assert s["train_backend"] == "mock"
    assert (out / "train_report.json").is_file()
    assert (out / "geometry_report.json").is_file()
