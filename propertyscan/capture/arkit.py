"""Apple ARKit / transforms.json capture adapter (stub for future).

Phase 2: detect and document only — full pose/depth ingest is later.
"""

from __future__ import annotations

from pathlib import Path

from propertyscan.capture.base import CaptureAdapter
from propertyscan.core.config import EngineConfig
from propertyscan.core.exceptions import ValidationError
from propertyscan.domain.capture import CaptureKind, CaptureManifest


class ARKitCaptureAdapter(CaptureAdapter):
    """Stub adapter for future iPhone ARKit / Record3D / Polycam exports.

    Purpose:
        Reserve the capture kind so architecture is future-proof.

    Limitations (Phase 2):
        Detects ``transforms.json`` presence but does not materialize frames
        with metric poses yet. Raises a clear ValidationError directing users
        to use image_folder/video for now.
    """

    kind = CaptureKind.ARKIT

    def can_handle(self, path: Path) -> bool:
        if path.is_file() and path.name.lower() == "transforms.json":
            return True
        if path.is_dir() and (path / "transforms.json").is_file():
            return True
        return False

    def load_manifest(self, path: Path, config: EngineConfig) -> CaptureManifest:
        path = Path(path)
        root = path if path.is_dir() else path.parent
        return CaptureManifest(
            kind=CaptureKind.ARKIT,
            source_path=root.resolve(),
            exists=True,
            has_poses=True,
            has_depth=(root / "depth").is_dir() or (root / "depths").is_dir(),
            metric_scale=True,
            warnings=[
                "ARKit capture detected but full support is not implemented in Phase 2. "
                "Use a plain image folder or video for now."
            ],
            extra={"transforms": str(root / "transforms.json")},
        )

    def materialize_frames(
        self,
        manifest: CaptureManifest,
        work_dir: Path,
        config: EngineConfig,
    ) -> list[Path]:
        raise ValidationError(
            "ARKit / transforms.json capture is not implemented yet (Phase 2 stub).",
            suggestion=(
                "Export frames as a plain image folder or MP4 walkthrough. "
                "Full AppleARKitProvider arrives in a later phase."
            ),
            details=manifest.to_dict(),
        )
