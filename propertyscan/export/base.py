"""Exporter ABC — translate PropertyScene only."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from propertyscan.domain.scene import PropertyScene


@dataclass
class ExportResult:
    success: bool
    format: str
    path: Path | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "format": self.format,
            "path": str(self.path) if self.path else None,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class Exporter(ABC):
    """Export PropertyScene to a delivery format.

    Non-responsibilities:
        Must not modify reconstruction data or re-train.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        ...

    @abstractmethod
    def export(self, scene: PropertyScene, output_dir: Path) -> ExportResult:
        ...
