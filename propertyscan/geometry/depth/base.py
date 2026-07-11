"""DepthProvider ABC — interchangeable monocular / sensor depth backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig
from propertyscan.domain.depth import DepthResult
from propertyscan.domain.frames import FrameSet


class DepthProvider(ABC):
    """Abstract dense depth estimator.

    Purpose:
        First-class depth for fusion and future 3DGS supervision.

    Non-responsibilities:
        Pose estimation (GeometryProvider), training, export.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def requires_cuda(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return True

    @abstractmethod
    def estimate(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
    ) -> DepthResult:
        """Estimate per-frame depth maps. Honest failure if unavailable."""

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires_cuda": self.requires_cuda,
            "is_available": self.is_available,
        }
