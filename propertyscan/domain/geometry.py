"""Geometry domain types — the heart of the AI-first reconstruction engine.

These types are provider-agnostic: MASt3R, DUSt3R, future ARKit / VGGT all
produce the same structures. COLMAP is intentionally not modeled as primary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import numpy as np


class CameraPose(BaseModel):
    """Single camera in the reconstructed pose graph.

    Purpose:
        Store intrinsics, extrinsics, and confidence for one view.

    Limitations:
        Convention is OpenCV/NeRF-style 4x4 c2w when ``c2w`` is set;
        providers must document any alternate convention in metrics/metadata.
    """

    image_id: str
    image_name: str
    width: int = 0
    height: int = 0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    # Row-major 4x4 camera-to-world if available
    c2w: list[list[float]] | None = None
    confidence: float = 1.0
    registered: bool = True
    reprojection_error: float | None = None
    num_observations: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PoseGraph(BaseModel):
    """Set of cameras and optional connectivity metadata."""

    cameras: list[CameraPose] = Field(default_factory=list)
    registered_count: int = 0
    total_count: int = 0
    registered_fraction: float = 0.0
    coordinate_frame: str = "c2w_opencv"
    notes: list[str] = Field(default_factory=list)

    def recompute(self) -> None:
        self.total_count = len(self.cameras)
        self.registered_count = sum(1 for c in self.cameras if c.registered)
        self.registered_fraction = (
            self.registered_count / self.total_count if self.total_count else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameras": [c.to_dict() for c in self.cameras],
            "registered_count": self.registered_count,
            "total_count": self.total_count,
            "registered_fraction": self.registered_fraction,
            "coordinate_frame": self.coordinate_frame,
            "notes": self.notes,
        }


class PointCloud(BaseModel):
    """Dense or sparse 3D points with optional attributes.

    Points are stored as nested lists for JSON-friendliness; providers may keep
    large clouds on disk and only attach a path + summary here.
    """

    xyz: list[list[float]] = Field(default_factory=list)
    rgb: list[list[int]] | None = None
    confidence: list[float] | None = None
    source: str = "unknown"
    path: Path | None = None
    point_count: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("path", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Path | None:
        return Path(v) if v is not None else None

    @classmethod
    def from_numpy(
        cls,
        xyz: "np.ndarray",
        *,
        rgb: "np.ndarray | None" = None,
        confidence: "np.ndarray | None" = None,
        source: str = "unknown",
        path: Path | None = None,
        store_inline: bool = False,
    ) -> PointCloud:
        """Build a PointCloud from numpy arrays.

        When ``store_inline`` is False (default), only counts/path are kept to
        avoid huge JSON payloads — preferred for production runs.

        Requires numpy at call time (optional dependency for inline conversion).
        """
        count = int(xyz.shape[0]) if xyz is not None else 0
        if not store_inline:
            return cls(
                xyz=[],
                rgb=None,
                confidence=None,
                source=source,
                path=path,
                point_count=count,
            )
        return cls(
            xyz=xyz.tolist(),
            rgb=rgb.tolist() if rgb is not None else None,
            confidence=confidence.tolist() if confidence is not None else None,
            source=source,
            path=path,
            point_count=count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "path": str(self.path) if self.path else None,
            "point_count": self.point_count,
            "inline_points": len(self.xyz),
        }


class ConfidenceMap(BaseModel):
    """Multi-level geometric confidence.

    Purpose:
        Replace the legacy habit of assuming every camera/point is equally good.
    """

    global_score: float = 0.0
    per_camera: dict[str, float] = Field(default_factory=dict)
    per_point_summary: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class GeometryMetrics(BaseModel):
    """Honest metrics produced by a geometry provider (no fake 100% registration)."""

    registered_cameras: int = 0
    total_cameras: int = 0
    registered_fraction: float = 0.0
    point_count: int = 0
    mean_camera_confidence: float | None = None
    global_align_loss: float | None = None
    execution_time_s: float = 0.0
    peak_vram_gb: float | None = None
    model_id: str | None = None
    pair_graph: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class GeometryResult(BaseModel):
    """Canonical output of any GeometryProvider.

    Purpose:
        Interchange object between foundation reconstruction, fusion, validation,
        and dataset building.

    Non-responsibilities:
        Does not train Gaussians or export PLY product artifacts.
    """

    provider_name: str
    success: bool
    pose_graph: PoseGraph | None = None
    point_cloud: PointCloud | None = None
    confidence: ConfidenceMap | None = None
    metrics: GeometryMetrics = Field(default_factory=GeometryMetrics)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "success": self.success,
            "pose_graph": self.pose_graph.to_dict() if self.pose_graph else None,
            "point_cloud": self.point_cloud.to_dict() if self.point_cloud else None,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "metrics": self.metrics.to_dict(),
            "artifacts": self.artifacts,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class SceneDescriptor(BaseModel):
    """Lightweight scene classification for geometry routing.

    Purpose:
        Help GeometryRouter choose MASt3R vs DUSt3R (and future providers)
        without hardcoding scene type into the pipeline.
    """

    scene_type: str = "residential_indoor"
    texture_score: float = 50.0
    blur_ratio: float = 0.0
    is_reflective: bool = False
    is_low_light: bool = False
    frame_count: int = 0
    mean_confidence: float = 50.0
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
