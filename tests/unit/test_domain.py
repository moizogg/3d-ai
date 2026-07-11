"""Unit tests for domain models."""

from __future__ import annotations

from pathlib import Path

import pytest

from propertyscan.domain.capture import CaptureKind, CaptureManifest
from propertyscan.domain.frames import FrameMetadata, FrameSet, FrameStatus
from propertyscan.domain.geometry import (
    CameraPose,
    GeometryResult,
    PointCloud,
    PoseGraph,
    SceneDescriptor,
)
from propertyscan.domain.quality import HealthReport
from propertyscan.domain.scene import PropertyScene, SceneMetadata


def test_frame_set_recompute_counts() -> None:
    frames = [
        FrameMetadata(
            filename="a.jpg",
            filepath=Path("a.jpg"),
            status=FrameStatus.ACCEPTED,
            confidence_score=90,
        ),
        FrameMetadata(
            filename="b.jpg",
            filepath=Path("b.jpg"),
            status=FrameStatus.MOTION_SMEAR,
            reject_reason="smear",
        ),
        FrameMetadata(
            filename="c.jpg",
            filepath=Path("c.jpg"),
            status=FrameStatus.LOW_RANK,
            confidence_score=30,
        ),
    ]
    fs = FrameSet(source_type="images", source_path=Path("."), frames=frames)
    fs.recompute_counts()
    assert fs.accepted_count == 1
    assert fs.low_confidence_count == 1
    # non-accepted = motion_smear + low_rank
    assert fs.rejected_count == 2
    assert fs.rejection_stats.get("motion_smear") == 1
    assert fs.rejection_stats.get("low_rank") == 1
    assert len(fs.get_accepted_paths()) == 1


def test_pose_graph_recompute() -> None:
    cams = [
        CameraPose(image_id="0", image_name="a.jpg", registered=True, confidence=0.9),
        CameraPose(image_id="1", image_name="b.jpg", registered=False, confidence=0.1),
    ]
    graph = PoseGraph(cameras=cams)
    graph.recompute()
    assert graph.total_count == 2
    assert graph.registered_count == 1
    assert graph.registered_fraction == 0.5


def test_point_cloud_from_numpy_summary_only() -> None:
    np = pytest.importorskip("numpy")
    xyz = np.zeros((100, 3), dtype=np.float32)
    pc = PointCloud.from_numpy(xyz, source="test", store_inline=False)
    assert pc.point_count == 100
    assert pc.xyz == []
    d = pc.to_dict()
    assert d["point_count"] == 100


def test_geometry_result_serialization() -> None:
    from propertyscan.domain.geometry import GeometryMetrics

    result = GeometryResult(
        provider_name="mock",
        success=False,
        error_message="weights missing",
        metrics=GeometryMetrics(registered_fraction=0.0),
    )
    d = result.to_dict()
    assert d["success"] is False
    assert d["error_message"] == "weights missing"


def test_capture_manifest() -> None:
    m = CaptureManifest(
        kind=CaptureKind.VIDEO,
        source_path=Path("walk.mp4"),
        duration_s=90.0,
        fps=30.0,
    )
    assert m.to_dict()["kind"] == "video"


def test_property_scene_minimal() -> None:
    scene = PropertyScene(
        metadata=SceneMetadata(scene_id="test-001", profile="colab_t4"),
        scene_descriptor=SceneDescriptor(frame_count=10, texture_score=40),
        health=HealthReport(score=80, expected_quality="good", passed=True),
    )
    d = scene.to_dict()
    assert d["metadata"]["scene_id"] == "test-001"
    assert d["health"]["passed"] is True
