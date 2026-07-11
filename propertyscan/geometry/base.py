"""GeometryProvider ABC — interchangeable foundation geometry backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, SceneDescriptor


class GeometryProvider(ABC):
    """Abstract geometry reconstruction provider.

    Purpose:
        Produce poses + point maps + confidence from a FrameSet without
        coupling the pipeline to a specific research model.

    Inputs:
        FrameSet (selected keyframes), EngineConfig, output directory.

    Outputs:
        GeometryResult (honest success/failure — never fake 100% registration).

    Non-responsibilities:
        Depth-only estimation (see DepthProvider), Gaussian training, export.

    Limitations:
        Phase 3: mocks + stubs. Phase 4: real MASt3R/DUSt3R neural weights.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider id, e.g. ``mast3r``, ``dust3r``, ``mock``."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Category: foundation | stereo | arkit | mock | hybrid."""

    @property
    def requires_cuda(self) -> bool:
        """Whether real inference needs a GPU."""
        return True

    @property
    def is_available(self) -> bool:
        """Whether dependencies/weights appear available (override in subclasses)."""
        return True

    @abstractmethod
    def can_handle(self, descriptor: SceneDescriptor) -> float:
        """Suitability score 0.0–1.0 for routing (higher = better match)."""

    @abstractmethod
    def reconstruct(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
        descriptor: SceneDescriptor | None = None,
    ) -> GeometryResult:
        """Run geometry reconstruction.

        Must return success=False with error_message on failure.
        Must never invent poses when models/weights are missing (unless mock).
        """

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider_type": self.provider_type,
            "requires_cuda": self.requires_cuda,
            "is_available": self.is_available,
        }
