"""Capture source domain types (video, image folder, sequences, future ARKit)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CaptureKind(str, Enum):
    """Supported capture source kinds.

    Future kinds (arkit, lidar, polycam) are declared early so adapters can
    plug in without pipeline rewrites.
    """

    VIDEO = "video"
    IMAGE_FOLDER = "image_folder"
    IMAGE_SEQUENCE = "image_sequence"
    ARKIT = "arkit"  # future: transforms.json + depth + metric scale
    LIDAR = "lidar"  # future
    POLYCAM = "polycam"  # future
    UNKNOWN = "unknown"


class CaptureManifest(BaseModel):
    """Validated description of a user-provided capture.

    Purpose:
        Output of input validation / capture detection before frame decode.

    Inputs:
        Path to a video file, image folder, or structured export.

    Outputs:
        Kind, media metadata, warnings — consumed by decode / intelligence.
    """

    kind: CaptureKind
    source_path: Path
    exists: bool = True
    file_count: int = 0
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    has_depth: bool = False
    has_poses: bool = False
    metric_scale: bool = False
    warnings: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("source_path", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Path:
        return Path(v)

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["source_path"] = str(self.source_path)
        data["kind"] = self.kind.value
        return data
