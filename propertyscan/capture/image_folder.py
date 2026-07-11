"""Image folder capture adapter."""

from __future__ import annotations

import shutil
from pathlib import Path

from propertyscan.capture.base import CaptureAdapter
from propertyscan.core.config import EngineConfig
from propertyscan.core.exceptions import ValidationError
from propertyscan.domain.capture import CaptureKind, CaptureManifest


def list_images(folder: Path, extensions: list[str]) -> list[Path]:
    """Return sorted image paths in a folder (non-recursive)."""
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return files


class ImageFolderAdapter(CaptureAdapter):
    """Load a flat directory of images as capture source."""

    kind = CaptureKind.IMAGE_FOLDER

    def can_handle(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        # ARKit-style dirs handled by ARKit adapter first.
        if (path / "transforms.json").is_file():
            return False
        # Prefer sequence adapter when names look strictly sequential — still ok
        # as folder if sequence adapter doesn't claim it.
        return True

    def load_manifest(self, path: Path, config: EngineConfig) -> CaptureManifest:
        path = Path(path)
        if not path.is_dir():
            raise ValidationError(
                f"Image folder not found: {path}",
                suggestion="Pass a directory containing .jpg/.png frames.",
            )
        images = list_images(path, config.capture.image_extensions)
        if not images:
            raise ValidationError(
                f"No images found in folder: {path}",
                suggestion=f"Expected extensions: {config.capture.image_extensions}",
            )
        width = height = None
        warnings: list[str] = []
        try:
            from PIL import Image

            with Image.open(images[0]) as im:
                width, height = im.size
        except Exception as exc:
            warnings.append(f"Could not read first image size: {exc}")

        if width and height:
            short = min(width, height)
            if short < config.capture.min_resolution:
                warnings.append(
                    f"Resolution short side {short}px < min_resolution "
                    f"{config.capture.min_resolution}px"
                )

        return CaptureManifest(
            kind=CaptureKind.IMAGE_FOLDER,
            source_path=path.resolve(),
            exists=True,
            file_count=len(images),
            width=width,
            height=height,
            warnings=warnings,
            extra={"image_count": len(images)},
        )

    def materialize_frames(
        self,
        manifest: CaptureManifest,
        work_dir: Path,
        config: EngineConfig,
    ) -> list[Path]:
        """Copy (or symlink) images into work_dir/candidates for a stable layout."""
        src = Path(manifest.source_path)
        images = list_images(src, config.capture.image_extensions)
        out_dir = Path(work_dir) / "candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        result: list[Path] = []
        for i, img in enumerate(images):
            dest = out_dir / f"frame_{i:04d}{img.suffix.lower()}"
            if not dest.exists():
                try:
                    dest.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dest)
            result.append(dest)
        return result
