"""Unit tests for capture detection and adapters."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from propertyscan.capture.detect import detect_capture_kind, get_adapter, validate_and_load
from propertyscan.capture.image_folder import ImageFolderAdapter, list_images
from propertyscan.core.config import EngineConfig, load_config
from propertyscan.core.exceptions import ValidationError
from propertyscan.domain.capture import CaptureKind


def _sharp_image(path: Path, seed: int = 0) -> None:
    img = Image.new("RGB", (320, 240), (40 + seed * 3, 80, 120))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 16):
        draw.line([(x, 0), (x, 240)], fill=(255, 255, 255), width=1)
    for y in range(0, 240, 16):
        draw.line([(0, y), (320, y)], fill=(200, 200, 50), width=1)
    draw.rectangle([20 + seed, 20, 100 + seed, 100], outline=(255, 0, 0), width=3)
    img.save(path, quality=95)


def test_detect_image_folder(tmp_path: Path) -> None:
    for i in range(5):
        _sharp_image(tmp_path / f"shot_{i:03d}.jpg", seed=i)
    kind = detect_capture_kind(tmp_path)
    assert kind in (CaptureKind.IMAGE_FOLDER, CaptureKind.IMAGE_SEQUENCE)


def test_detect_image_sequence(tmp_path: Path) -> None:
    for i in range(5):
        _sharp_image(tmp_path / f"frame_{i:04d}.jpg", seed=i)
    kind = detect_capture_kind(tmp_path)
    assert kind == CaptureKind.IMAGE_SEQUENCE


def test_detect_video_by_extension(tmp_path: Path) -> None:
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"\x00\x00")  # not a real video; detection is extension-based
    assert detect_capture_kind(video) == CaptureKind.VIDEO


def test_missing_path_raises() -> None:
    with pytest.raises(ValidationError):
        detect_capture_kind(Path("does/not/exist_xyz"))


def test_empty_folder_raises(tmp_path: Path) -> None:
    cfg = load_config(apply_env=False)
    with pytest.raises(ValidationError):
        ImageFolderAdapter().load_manifest(tmp_path, cfg)


def test_list_images_and_materialize(tmp_path: Path) -> None:
    for i in range(4):
        _sharp_image(tmp_path / f"img_{i}.jpg", seed=i)
    cfg = load_config(apply_env=False)
    adapter, manifest = validate_and_load(tmp_path, cfg)
    assert manifest.file_count == 4
    work = tmp_path / "work"
    paths = adapter.materialize_frames(manifest, work, cfg)
    assert len(paths) == 4
    assert all(p.exists() for p in paths)


def test_arkit_stub_rejects_materialize(tmp_path: Path) -> None:
    (tmp_path / "transforms.json").write_text("{}", encoding="utf-8")
    cfg = load_config(apply_env=False)
    adapter = get_adapter(tmp_path, cfg)
    assert adapter.kind == CaptureKind.ARKIT
    manifest = adapter.load_manifest(tmp_path, cfg)
    with pytest.raises(ValidationError):
        adapter.materialize_frames(manifest, tmp_path / "w", cfg)
