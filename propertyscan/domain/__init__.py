"""Canonical domain objects exchanged between pipeline stages."""

from __future__ import annotations

from propertyscan.domain.capture import CaptureKind, CaptureManifest
from propertyscan.domain.dataset import TrainingDataset
from propertyscan.domain.depth import DepthMap, DepthResult
from propertyscan.domain.frames import FrameMetadata, FrameSet, FrameStatus
from propertyscan.domain.gaussian import GaussianScene, GaussianStats
from propertyscan.domain.geometry import (
    CameraPose,
    ConfidenceMap,
    GeometryMetrics,
    GeometryResult,
    PointCloud,
    PoseGraph,
    SceneDescriptor,
)
from propertyscan.domain.provenance import ProvenanceRecord
from propertyscan.domain.quality import (
    HealthReport,
    InspectionReport,
    QualityReport,
)
from propertyscan.domain.scene import PropertyScene, SceneMetadata

__all__ = [
    "CaptureKind",
    "CaptureManifest",
    "TrainingDataset",
    "DepthMap",
    "DepthResult",
    "FrameMetadata",
    "FrameSet",
    "FrameStatus",
    "GaussianScene",
    "GaussianStats",
    "CameraPose",
    "ConfidenceMap",
    "GeometryMetrics",
    "GeometryResult",
    "PointCloud",
    "PoseGraph",
    "SceneDescriptor",
    "ProvenanceRecord",
    "HealthReport",
    "InspectionReport",
    "QualityReport",
    "PropertyScene",
    "SceneMetadata",
]
