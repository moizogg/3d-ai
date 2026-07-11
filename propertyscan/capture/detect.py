"""Capture kind detection and adapter registry."""

from __future__ import annotations

from pathlib import Path

from propertyscan.capture.arkit import ARKitCaptureAdapter
from propertyscan.capture.base import CaptureAdapter
from propertyscan.capture.image_folder import ImageFolderAdapter
from propertyscan.capture.image_sequence import ImageSequenceAdapter
from propertyscan.capture.video import VideoCaptureAdapter
from propertyscan.core.config import EngineConfig
from propertyscan.core.exceptions import ValidationError
from propertyscan.domain.capture import CaptureKind, CaptureManifest


def _adapters() -> list[CaptureAdapter]:
    # Order matters: more specific first (sequence/arkit before generic folder).
    return [
        ARKitCaptureAdapter(),
        VideoCaptureAdapter(),
        ImageSequenceAdapter(),
        ImageFolderAdapter(),
    ]


def detect_capture_kind(path: Path, config: EngineConfig | None = None) -> CaptureKind:
    """Detect the capture kind for a filesystem path.

    Raises:
        ValidationError: path missing or unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(
            f"Capture path does not exist: {path}",
            suggestion="Provide a video file or a folder of images.",
        )
    cfg = config or EngineConfig()
    for adapter in _adapters():
        if adapter.can_handle(path):
            return adapter.kind
    raise ValidationError(
        f"Unsupported capture source: {path}",
        suggestion=(
            f"Use a video ({', '.join(cfg.capture.video_extensions)}) "
            f"or an image folder ({', '.join(cfg.capture.image_extensions)})."
        ),
    )


def get_adapter(path: Path, config: EngineConfig | None = None) -> CaptureAdapter:
    """Return the first adapter that can handle ``path``."""
    path = Path(path)
    if not path.exists():
        raise ValidationError(
            f"Capture path does not exist: {path}",
            suggestion="Provide a video file or a folder of images.",
        )
    for adapter in _adapters():
        if adapter.can_handle(path):
            return adapter
    cfg = config or EngineConfig()
    raise ValidationError(
        f"No capture adapter for: {path}",
        suggestion=(
            f"Supported: video {cfg.capture.video_extensions} "
            f"or images {cfg.capture.image_extensions}."
        ),
    )


def validate_and_load(path: Path, config: EngineConfig) -> tuple[CaptureAdapter, CaptureManifest]:
    """Detect adapter and load validated CaptureManifest."""
    adapter = get_adapter(path, config)
    manifest = adapter.load_manifest(Path(path), config)
    return adapter, manifest
