"""Training dataset domain — high-quality inputs for Gaussian training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TrainingDataset(BaseModel):
    """Nerfstudio-compatible dataset prepared from fused geometry.

    Purpose:
        Single validated package: images, transforms.json, optional depth + init cloud.

    Non-responsibilities:
        Does not train Gaussians or estimate poses.
    """

    root: Path
    images_dir: Path
    transforms_path: Path
    frame_count: int = 0
    has_depth: bool = False
    has_init_point_cloud: bool = False
    init_ply_path: Path | None = None
    depth_dir: Path | None = None
    downscale_factor: int = 1
    provider_name: str | None = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator(
        "root",
        "images_dir",
        "transforms_path",
        "init_ply_path",
        "depth_dir",
        mode="before",
    )
    @classmethod
    def _pathify(cls, v: Any) -> Path | None:
        if v is None:
            return None
        return Path(v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "images_dir": str(self.images_dir),
            "transforms_path": str(self.transforms_path),
            "frame_count": self.frame_count,
            "has_depth": self.has_depth,
            "has_init_point_cloud": self.has_init_point_cloud,
            "init_ply_path": str(self.init_ply_path) if self.init_ply_path else None,
            "depth_dir": str(self.depth_dir) if self.depth_dir else None,
            "downscale_factor": self.downscale_factor,
            "provider_name": self.provider_name,
            "notes": self.notes,
            "metadata": self.metadata,
        }
