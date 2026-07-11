"""Depth domain types for first-class monocular / sensor depth providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DepthMap(BaseModel):
    """One depth map associated with a frame.

    Purpose:
        Store path and scale semantics for a single view.

    Notes:
        Monocular depth is typically relative. Metric scale is reserved for
        future ARKit / LiDAR providers.
    """

    image_id: str
    image_name: str
    path: Path
    width: int = 0
    height: int = 0
    scale: str = "relative"  # relative | metric
    confidence_path: Path | None = None
    min_depth: float | None = None
    max_depth: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("path", "confidence_path", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Path | None:
        if v is None:
            return None
        return Path(v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "confidence_path": str(self.confidence_path) if self.confidence_path else None,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
        }


class DepthResult(BaseModel):
    """Output of a DepthProvider (e.g. Depth Anything V2).

    Purpose:
        First-class depth product for fusion and 3DGS supervision — not an afterthought.
    """

    provider_name: str
    success: bool
    depth_maps: list[DepthMap] = Field(default_factory=list)
    scale_hint: str = "relative"
    execution_time_s: float = 0.0
    model_id: str | None = None
    peak_vram_gb: float | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "success": self.success,
            "depth_maps": [d.to_dict() for d in self.depth_maps],
            "scale_hint": self.scale_hint,
            "execution_time_s": self.execution_time_s,
            "model_id": self.model_id,
            "peak_vram_gb": self.peak_vram_gb,
            "artifacts": self.artifacts,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "count": len(self.depth_maps),
        }
