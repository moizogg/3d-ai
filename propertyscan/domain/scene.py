"""PropertyScene — canonical product of the reconstruction engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from propertyscan.domain.depth import DepthResult
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.gaussian import GaussianScene
from propertyscan.domain.geometry import GeometryResult, PoseGraph, SceneDescriptor
from propertyscan.domain.quality import HealthReport, InspectionReport, QualityReport


class SceneMetadata(BaseModel):
    """High-level metadata for a reconstructed property scene."""

    scene_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    engine_version: str = "0.1.0"
    profile: str = "default"
    source_path: str | None = None
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PropertyScene(BaseModel):
    """Canonical internal representation of a reconstructed property.

    Purpose:
        Own the scene. Exporters only translate this object.

    Philosophy:
        PLY is one export — not the architecture.
    """

    metadata: SceneMetadata
    frame_set: FrameSet | None = None
    scene_descriptor: SceneDescriptor | None = None
    geometry: GeometryResult | None = None
    pose_graph: PoseGraph | None = None
    depth: DepthResult | None = None
    gaussian_scene: GaussianScene | None = None
    health: HealthReport | None = None
    inspection: InspectionReport | None = None
    quality: QualityReport | None = None
    processing_history: list[dict[str, Any]] = Field(default_factory=list)
    exports: dict[str, str] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "frame_set": self.frame_set.to_dict() if self.frame_set else None,
            "scene_descriptor": (
                self.scene_descriptor.to_dict() if self.scene_descriptor else None
            ),
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "pose_graph": self.pose_graph.to_dict() if self.pose_graph else None,
            "depth": self.depth.to_dict() if self.depth else None,
            "gaussian_scene": (
                self.gaussian_scene.to_dict() if self.gaussian_scene else None
            ),
            "health": self.health.to_dict() if self.health else None,
            "inspection": self.inspection.to_dict() if self.inspection else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "processing_history": self.processing_history,
            "exports": self.exports,
            "statistics": self.statistics,
        }
