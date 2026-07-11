"""Phase 4: foundation infer helpers, deps, pair-graph, depth provider contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from propertyscan.core.config import load_config
from propertyscan.domain.frames import FrameMetadata, FrameSet, FrameStatus
from propertyscan.geometry.deps import foundation_ready, probe_torch
from propertyscan.geometry.foundation_infer import (
    FoundationInferResult,
    run_foundation_reconstruction,
)
from propertyscan.geometry.providers.mast3r import MASt3RProvider
from propertyscan.geometry.runtime import choose_pair_graph


def test_choose_pair_graph_vram_safety() -> None:
    assert choose_pair_graph(10, "complete") == "complete"
    assert choose_pair_graph(50, "complete") == "swin-5"
    assert choose_pair_graph(20, "swin-5") == "swin-5"


def test_foundation_ready_structure() -> None:
    status = foundation_ready(need_mast3r=True, need_dust3r=True)
    assert "torch" in status
    assert "can_run_mast3r" in status
    assert "cuda" in status


def test_run_foundation_needs_two_images(tmp_path: Path) -> None:
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(p)
    res = run_foundation_reconstruction(
        engine="dust3r",
        image_paths=[p],
        output_dir=tmp_path / "out",
        model_id="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        prefer_cuda=True,
        allow_cpu=False,
    )
    assert res.success is False
    assert "2" in (res.error_message or "")


def test_run_foundation_missing_deps_is_honest(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (64, 64), (i * 10, 40, 50)).save(p)
        paths.append(p)
    res = run_foundation_reconstruction(
        engine="mast3r",
        image_paths=paths,
        output_dir=tmp_path / "out",
        model_id="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        prefer_cuda=False,
        allow_cpu=True,
    )
    # Without packages installed → failure, never fake success
    if not res.success:
        assert res.error_message
        assert res.poses == []
    else:
        # If user machine has full stack, still must produce poses
        assert len(res.poses) == 3


def test_mast3r_provider_uses_infer_success(tmp_path: Path) -> None:
    """Wire provider success path without real weights via monkeypatch."""
    cfg = load_config(apply_env=False)
    frames = []
    for i in range(4):
        p = tmp_path / f"frame_{i}.jpg"
        img = Image.new("RGB", (128, 96), (20, 30, 40))
        ImageDraw.Draw(img).rectangle([10, 10, 60, 50], outline=(255, 255, 255))
        img.save(p)
        frames.append(
            FrameMetadata(
                filename=p.name,
                filepath=p,
                width=128,
                height=96,
                index=i,
                status=FrameStatus.ACCEPTED,
            )
        )
    fs = FrameSet(source_type="t", source_path=tmp_path, frames=frames)
    fs.recompute_counts()

    identity = [
        [1.0, 0.0, 0.0, float(i)],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    fake = FoundationInferResult(
        success=True,
        engine="mast3r",
        model_id=cfg.geometry.mast3r_model,
        poses=[identity for _ in range(4)],
        focals=[500.0] * 4,
        confidences=[0.9] * 4,
        point_count=1000,
        pair_graph="swin-5",
        global_align_loss=0.02,
        execution_time_s=1.2,
        peak_vram_gb=4.5,
        device="cuda",
        metadata={"image_count": 4},
    )

    with patch(
        "propertyscan.geometry.providers.mast3r.run_foundation_reconstruction",
        return_value=fake,
    ):
        res = MASt3RProvider(cfg).reconstruct(
            fs, output_dir=tmp_path / "geo", config=cfg
        )
    assert res.success is True
    assert res.metrics.registered_cameras == 4
    assert res.metrics.model_id == cfg.geometry.mast3r_model
    assert (tmp_path / "geo" / "transforms.json").is_file()
    assert res.metrics.peak_vram_gb == 4.5


@pytest.mark.gpu
def test_real_mast3r_gpu_optional(tmp_path: Path) -> None:
    """Optional real GPU smoke — skipped unless CUDA + packages present."""
    status = foundation_ready(need_mast3r=True)
    if not status["cuda"] or not status["can_run_mast3r"]:
        pytest.skip("CUDA + MASt3R/DUSt3R not available")
    cfg = load_config(apply_env=False)
    paths = []
    for i in range(3):
        p = tmp_path / f"g{i}.jpg"
        img = Image.new("RGB", (256, 192), (30 + i * 5, 40, 50))
        d = ImageDraw.Draw(img)
        for x in range(0, 256, 16):
            d.line([(x, 0), (x, 192)], fill=(255, 255, 255))
        img.save(p)
        paths.append(p)
    res = run_foundation_reconstruction(
        engine="mast3r",
        image_paths=paths,
        output_dir=tmp_path / "real",
        model_id=cfg.geometry.mast3r_model,
        pair_graph="complete",
        batch_size=1,
        global_align_iters=50,
        prefer_cuda=True,
        allow_cpu=False,
    )
    # Real GPU may OOM or download fail — only assert honesty
    if res.success:
        assert len(res.poses) == 3
    else:
        assert res.error_message
