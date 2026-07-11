"""Tests for reliable frame validation (not legacy Laplacian/dHash hard-reject)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from propertyscan.core.config import load_config
from propertyscan.domain.frames import FrameStatus
from propertyscan.intelligence.classify import classify_scene
from propertyscan.intelligence.dedup import mark_duplicates
from propertyscan.intelligence.image_ops import edge_density, percentile_ranks, tenengrad, to_gray
from propertyscan.intelligence.keyframes import select_keyframes
from propertyscan.intelligence.quality import analyze_frame, analyze_frames


def _white_wall_sharp(path: Path, seed: int = 0) -> None:
    """Low-texture but sharp wall-like image (legacy Laplacian would kill this)."""
    img = Image.new("RGB", (320, 240), (235, 235, 232))
    draw = ImageDraw.Draw(img)
    # Subtle baseboard / door edge — real rooms have sparse edges
    draw.line([(0, 200), (320, 200)], fill=(210, 210, 208), width=2)
    draw.rectangle([40 + seed, 30, 48 + seed, 200], outline=(200, 200, 198), width=1)
    img.save(path, quality=95)


def _rich_texture(path: Path, seed: int = 0) -> None:
    img = Image.new("RGB", (320, 240), (40, 50, 60))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 8):
        draw.line([(x + seed % 3, 0), (x + seed % 3, 240)], fill=(255, 255, 255))
    for y in range(0, 240, 8):
        draw.line([(0, y), (320, y)], fill=(220, 180, 40))
    img.save(path, quality=95)


def _motion_smear_textured(path: Path) -> None:
    """Textured then heavily blurred — true smear proxy."""
    img = Image.new("RGB", (320, 240), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 4):
        draw.line([(x, 0), (x, 240)], fill=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=18))
    img.save(path, quality=60)


def _pure_black(path: Path) -> None:
    Image.new("RGB", (256, 256), (0, 0, 0)).save(path)


def _pure_white(path: Path) -> None:
    Image.new("RGB", (256, 256), (255, 255, 255)).save(path)


def test_tenengrad_higher_for_sharp_texture() -> None:
    sharp = np.zeros((64, 64), dtype=np.float32)
    sharp[:, ::2] = 255
    blur = np.full((64, 64), 128.0, dtype=np.float32)
    assert tenengrad(sharp) > tenengrad(blur)


def test_white_wall_not_hard_rejected(tmp_path: Path) -> None:
    """Critical: low-texture sharp wall must NOT be labeled blurry/hard-rejected."""
    paths = []
    for i in range(10):
        p = tmp_path / f"wall_{i:03d}.jpg"
        _white_wall_sharp(p, seed=i)
        paths.append(p)
    cfg = load_config(apply_env=False)
    frames = analyze_frames(paths, cfg)
    hard = [f for f in frames if f.is_hard_rejected()]
    assert len(hard) == 0, [f.status for f in hard]
    # Most should be selectable candidates
    selectable = [f for f in frames if f.is_selectable()]
    assert len(selectable) >= 8


def test_pure_black_hard_rejected(tmp_path: Path) -> None:
    p = tmp_path / "black.jpg"
    _pure_black(p)
    cfg = load_config(apply_env=False)
    meta = analyze_frame(p, config=cfg)
    assert meta.status == FrameStatus.CLIPPED_BLACK


def test_pure_white_hard_rejected(tmp_path: Path) -> None:
    p = tmp_path / "white.jpg"
    _pure_white(p)
    cfg = load_config(apply_env=False)
    meta = analyze_frame(p, config=cfg)
    assert meta.status == FrameStatus.CLIPPED_WHITE


def test_slow_walk_similar_walls_not_mass_redundant(tmp_path: Path) -> None:
    """Slow walk past similar walls: dHash legacy would mass-flag; motion must not."""
    paths = []
    for i in range(20):
        p = tmp_path / f"frame_{i:04d}.jpg"
        # Slightly shifting geometry = camera moved
        _rich_texture(p, seed=i * 2)
        paths.append(p)
    cfg = load_config(apply_env=False)
    cfg.frame_intelligence.min_motion_to_keep = 0.5
    frames = analyze_frames(paths, cfg)
    frames = mark_duplicates(frames, cfg)
    redundant = [f for f in frames if f.status == FrameStatus.REDUNDANT]
    # Should not reject ~50% as "duplicates"
    assert len(redundant) < len(frames) * 0.35, f"too many redundant: {len(redundant)}"


def test_identical_freeze_is_redundant(tmp_path: Path) -> None:
    """True stationary freeze: same image repeatedly → redundant after first."""
    paths = []
    for i in range(6):
        p = tmp_path / f"freeze_{i:04d}.jpg"
        _rich_texture(p, seed=0)  # identical content
        paths.append(p)
    cfg = load_config(apply_env=False)
    cfg.frame_intelligence.min_motion_to_keep = 0.8
    frames = analyze_frames(paths, cfg)
    frames = mark_duplicates(frames, cfg)
    redundant = [f for f in frames if f.status == FrameStatus.REDUNDANT]
    # After first keep, freezes should be redundant
    assert len(redundant) >= 3


def test_keyframes_selects_budget(tmp_path: Path) -> None:
    paths = []
    for i in range(15):
        p = tmp_path / f"frame_{i:04d}.jpg"
        _rich_texture(p, seed=i)
        paths.append(p)
    cfg = load_config(apply_env=False)
    cfg.capture.min_frames = 5
    cfg.frame_intelligence.max_keyframes = 8
    cfg.frame_intelligence.min_motion_to_keep = 0.3
    frames = analyze_frames(paths, cfg)
    frames = mark_duplicates(frames, cfg)
    fs = select_keyframes(frames, cfg, selected_dir=tmp_path / "sel")
    assert fs.accepted_count >= 5
    assert fs.accepted_count <= 8
    assert fs.validation_mode == "reliable_v2"
    assert len(list((tmp_path / "sel").glob("keyframe_*"))) == fs.accepted_count


def test_keyframes_unlimited_keeps_all_selectable(tmp_path: Path) -> None:
    """max_keyframes <= 0 keeps every non-hard-rejected / non-redundant frame."""
    paths = []
    for i in range(12):
        p = tmp_path / f"frame_{i:04d}.jpg"
        _rich_texture(p, seed=i)
        paths.append(p)
    cfg = load_config(apply_env=False)
    cfg.capture.min_frames = 4
    cfg.frame_intelligence.max_keyframes = 0  # unlimited
    cfg.frame_intelligence.min_motion_to_keep = 0.0  # no motion thinning in dedup path
    frames = analyze_frames(paths, cfg)
    frames = mark_duplicates(frames, cfg)
    selectable = [
        f
        for f in frames
        if f.status
        in (FrameStatus.CANDIDATE, FrameStatus.LOW_RANK, FrameStatus.ACCEPTED)
    ]
    fs = select_keyframes(frames, cfg, selected_dir=tmp_path / "all")
    assert fs.accepted_count == len(selectable)
    assert fs.accepted_count == 12


def test_percentile_ranks_monotone() -> None:
    ranks = percentile_ranks([1.0, 2.0, 3.0, 4.0])
    assert ranks[0] < ranks[-1]


def test_edge_density_low_on_flat() -> None:
    flat = np.full((64, 64), 200.0, dtype=np.float32)
    busy = np.zeros((64, 64), dtype=np.float32)
    busy[:, ::2] = 255
    assert edge_density(busy) > edge_density(flat)


def test_classify_scene(tmp_path: Path) -> None:
    paths = []
    for i in range(8):
        p = tmp_path / f"f{i}.jpg"
        _rich_texture(p, seed=i)
        paths.append(p)
    cfg = load_config(apply_env=False)
    cfg.capture.min_frames = 4
    frames = analyze_frames(paths, cfg)
    frames = mark_duplicates(frames, cfg)
    fs = select_keyframes(frames, cfg, selected_dir=tmp_path / "k")
    desc = classify_scene(fs)
    assert desc.frame_count >= 4
    assert "validation_reliable_v2" in desc.tags
