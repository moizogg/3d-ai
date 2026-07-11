"""CaptureAdapter ABC — one adapter per capture source kind."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.capture import CaptureKind, CaptureManifest


class CaptureAdapter(ABC):
    """Abstract capture source adapter.

    Purpose:
        Detect, validate, and materialize frames for a single capture kind.

    Inputs:
        Filesystem path + EngineConfig.

    Outputs:
        CaptureManifest and list of image paths (after materialize).

    Non-responsibilities:
        Frame quality scoring, geometry, training.
    """

    kind: CaptureKind

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this adapter should handle ``path``."""

    @abstractmethod
    def load_manifest(self, path: Path, config: EngineConfig) -> CaptureManifest:
        """Validate path and return a CaptureManifest (no heavy decode yet)."""

    @abstractmethod
    def materialize_frames(
        self,
        manifest: CaptureManifest,
        work_dir: Path,
        config: EngineConfig,
    ) -> list[Path]:
        """Produce a list of RGB image file paths ready for intelligence."""
