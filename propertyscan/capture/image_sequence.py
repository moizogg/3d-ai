"""Image sequence adapter — numbered frame sequences (frame_0001.jpg, etc.)."""

from __future__ import annotations

import re
from pathlib import Path

from propertyscan.capture.image_folder import ImageFolderAdapter, list_images
from propertyscan.core.config import EngineConfig
from propertyscan.domain.capture import CaptureKind, CaptureManifest

_SEQ_RE = re.compile(
    r"^(frame_|img_|image_)?\d{3,}",
    re.IGNORECASE,
)


class ImageSequenceAdapter(ImageFolderAdapter):
    """Like image folder, but only claims dirs dominated by sequential names."""

    kind = CaptureKind.IMAGE_SEQUENCE

    def can_handle(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "transforms.json").is_file():
            return False
        # Peek with default extensions
        images = list_images(
            path,
            [".jpg", ".jpeg", ".png", ".webp"],
        )
        if len(images) < 3:
            return False
        sequential = sum(1 for p in images if _SEQ_RE.match(p.stem))
        return sequential / len(images) >= 0.7

    def load_manifest(self, path: Path, config: EngineConfig) -> CaptureManifest:
        manifest = super().load_manifest(path, config)
        return CaptureManifest(
            kind=CaptureKind.IMAGE_SEQUENCE,
            source_path=manifest.source_path,
            exists=manifest.exists,
            file_count=manifest.file_count,
            duration_s=manifest.duration_s,
            fps=manifest.fps,
            width=manifest.width,
            height=manifest.height,
            codec=manifest.codec,
            has_depth=manifest.has_depth,
            has_poses=manifest.has_poses,
            metric_scale=manifest.metric_scale,
            warnings=list(manifest.warnings),
            extra={**manifest.extra, "sequence": True},
        )
