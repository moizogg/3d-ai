"""TrainerBackend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from propertyscan.core.config import EngineConfig
from propertyscan.domain.dataset import TrainingDataset
from propertyscan.domain.gaussian import GaussianScene


@dataclass
class TrainResult:
    """Outcome of a training backend run."""

    success: bool
    backend: str
    scene: GaussianScene | None = None
    train_dir: Path | None = None
    config_path: Path | None = None
    iterations: int = 0
    execution_time_s: float = 0.0
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "backend": self.backend,
            "scene": self.scene.to_dict() if self.scene else None,
            "train_dir": str(self.train_dir) if self.train_dir else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "iterations": self.iterations,
            "execution_time_s": self.execution_time_s,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class TrainerBackend(ABC):
    """Abstract Gaussian training backend.

    Purpose:
        Optimize Gaussians from a validated TrainingDataset.
        Does not invent camera poses.

    Non-responsibilities:
        Geometry reconstruction, frame intelligence, final web export.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if this backend can run in the current environment."""

    @abstractmethod
    def train(
        self,
        dataset: TrainingDataset,
        *,
        output_dir: Path,
        config: EngineConfig,
    ) -> TrainResult:
        """Run training; return honest failure if tools/weights missing."""
