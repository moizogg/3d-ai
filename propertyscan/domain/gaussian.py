"""Gaussian scene domain types (training output, not PLY product-of-record)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GaussianStats(BaseModel):
    """Aggregate statistics over a Gaussian scene."""

    count: int = 0
    mean_opacity: float | None = None
    mean_scale: float | None = None
    max_scale: float | None = None
    scale_variance: float | None = None
    estimated_needles: int | None = None
    estimated_floaters: int | None = None
    bounding_box_min: list[float] | None = None
    bounding_box_max: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class GaussianScene(BaseModel):
    """Internal Gaussian representation after training / cleaning.

    Purpose:
        Hold paths and stats for the reconstructed splat field.

    Non-responsibilities:
        Export formats (PLY) are handled by exporters reading PropertyScene.
    """

    path: Path | None = None
    cleaned_path: Path | None = None
    stats: GaussianStats = Field(default_factory=GaussianStats)
    training_iterations: int | None = None
    training_time_s: float | None = None
    trainer_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("path", "cleaned_path", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Path | None:
        return Path(v) if v is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "cleaned_path": str(self.cleaned_path) if self.cleaned_path else None,
            "stats": self.stats.to_dict(),
            "training_iterations": self.training_iterations,
            "training_time_s": self.training_time_s,
            "trainer_name": self.trainer_name,
            "metadata": self.metadata,
        }
