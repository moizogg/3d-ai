"""Geometric validation of GeometryResult / fused output (learned-geometry aware)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from propertyscan.core.config import EngineConfig
from propertyscan.domain.geometry import GeometryResult
from propertyscan.geometry.fusion.fuse import FusedGeometry


@dataclass
class GeometryValidationReport:
    """Structured geometry checks before health gate / training."""

    passed: bool
    registered_fraction: float
    registered_cameras: int
    total_cameras: int
    point_count: int
    mean_confidence: float | None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "registered_fraction": self.registered_fraction,
            "registered_cameras": self.registered_cameras,
            "total_cameras": self.total_cameras,
            "point_count": self.point_count,
            "mean_confidence": self.mean_confidence,
            "issues": self.issues,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_geometry(
    geometry: GeometryResult | FusedGeometry,
    config: EngineConfig,
) -> GeometryValidationReport:
    """Validate geometry honesty and minimum registration for training readiness.

    Unlike COLMAP pipelines, we do not require reprojection error from SIFT.
    We require: success flag, registered cameras fraction, non-empty poses.
    """
    if isinstance(geometry, FusedGeometry):
        geom = geometry.geometry
        extra = {"fusion_notes": geometry.notes}
    else:
        geom = geometry
        extra = {}

    issues: list[str] = []
    warnings: list[str] = []
    min_frac = config.geometry.min_registered_fraction

    if not geom.success:
        issues.append(f"geometry.success=False: {geom.error_message}")

    reg_frac = geom.metrics.registered_fraction
    reg_n = geom.metrics.registered_cameras
    total = geom.metrics.total_cameras
    points = geom.metrics.point_count
    mean_conf = geom.metrics.mean_camera_confidence

    if geom.pose_graph is None:
        issues.append("missing pose_graph")
    else:
        # Honest check: registered cameras must have c2w
        missing_pose = [
            c.image_name
            for c in geom.pose_graph.cameras
            if c.registered and c.c2w is None
        ]
        if missing_pose:
            issues.append(
                f"{len(missing_pose)} cameras marked registered without c2w "
                f"(e.g. {missing_pose[:3]})"
            )

    if reg_frac < min_frac:
        issues.append(
            f"registered_fraction {reg_frac:.2%} < min {min_frac:.0%}"
        )

    if reg_n < 2:
        issues.append(f"need >= 2 registered cameras; got {reg_n}")

    if points == 0:
        warnings.append("point_count=0 (may still train if poses are strong)")

    if mean_conf is not None and mean_conf < 0.3:
        warnings.append(f"low mean camera confidence {mean_conf:.2f}")

    passed = len(issues) == 0
    return GeometryValidationReport(
        passed=passed,
        registered_fraction=reg_frac,
        registered_cameras=reg_n,
        total_cameras=total,
        point_count=points,
        mean_confidence=mean_conf,
        issues=issues,
        warnings=warnings,
        details={
            "provider": geom.provider_name,
            "model_id": geom.metrics.model_id,
            **extra,
        },
    )
