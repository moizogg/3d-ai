"""End-to-end Phase 2 frame pipeline with reliable validation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from propertyscan.core.config import load_config
from propertyscan.pipeline.frame_pipeline import frames_summary, run_frames_pipeline


def _make_fixture_folder(root: Path, n: int = 12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (320, 240), (20 + i, 40, 60))
        draw = ImageDraw.Draw(img)
        for x in range(0, 320, 10):
            draw.line([(x + (i % 5), 0), (x + (i % 5), 240)], fill=(255, 255, 255))
        for y in range(0, 240, 10):
            draw.line([(0, y), (320, y)], fill=(220, 180, 40))
        draw.rectangle([10 + i, 10, 80 + i, 80], outline=(255, 0, 0), width=2)
        img.save(root / f"frame_{i:04d}.jpg", quality=95)
    return root


def test_run_frames_pipeline_end_to_end(tmp_path: Path) -> None:
    src = _make_fixture_folder(tmp_path / "input", n=12)
    out = tmp_path / "output"

    cfg = load_config(apply_env=False)
    cfg.capture.min_frames = 6
    cfg.frame_intelligence.max_keyframes = 8
    cfg.frame_intelligence.min_motion_to_keep = 0.3

    ctx = run_frames_pipeline(src, out, config=cfg)
    summary = frames_summary(ctx)

    assert summary["accepted_keyframes"] >= 6
    assert summary["capture_kind"] in ("image_folder", "image_sequence")
    assert summary["scene_type"]

    report = out / "frame_intelligence.json"
    assert report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["validation_mode"] == "reliable_v2"
    assert data["frame_set"]["accepted_count"] >= 6

    selected = list((ctx.work_dir / "artifacts" / "frames" / "selected").glob("keyframe_*"))
    assert len(selected) == summary["accepted_keyframes"]


def test_cli_frames_help() -> None:
    from propertyscan.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["frames", "--input", "x", "--out", "y", "--profile", "colab_t4"]
    )
    assert args.command == "frames"
    assert args.profile == "colab_t4"


def test_white_wall_folder_pipeline_does_not_mass_reject(tmp_path: Path) -> None:
    """End-to-end: low-texture walk must still produce keyframes."""
    src = tmp_path / "walls"
    src.mkdir()
    for i in range(12):
        img = Image.new("RGB", (320, 240), (230, 230, 228))
        d = ImageDraw.Draw(img)
        d.line([(0, 200), (320, 200)], fill=(200, 200, 198), width=2)
        d.rectangle([30 + i, 20, 40 + i, 200], outline=(190, 190, 188), width=1)
        img.save(src / f"frame_{i:04d}.jpg", quality=95)

    cfg = load_config(apply_env=False)
    cfg.capture.min_frames = 6
    cfg.frame_intelligence.max_keyframes = 10
    cfg.frame_intelligence.min_motion_to_keep = 0.2

    ctx = run_frames_pipeline(src, tmp_path / "out", config=cfg)
    assert frames_summary(ctx)["accepted_keyframes"] >= 6
