"""Helpers to build honest GeometryResult / PoseGraph objects."""

from __future__ import annotations

from pathlib import Path

from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import (
    CameraPose,
    ConfidenceMap,
    GeometryMetrics,
    GeometryResult,
    PointCloud,
    PoseGraph,
)


def identity_c2w() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pose_graph_from_frames(
    frame_set: FrameSet,
    *,
    poses: list[list[list[float]]] | None = None,
    focals: list[float] | None = None,
    confidences: list[float] | None = None,
    registered_mask: list[bool] | None = None,
) -> PoseGraph:
    """Build PoseGraph from accepted frames with optional pose data.

    If poses is None, cameras are created as unregistered (honest empty).
    """
    accepted = frame_set.get_accepted()
    cameras: list[CameraPose] = []
    for i, fr in enumerate(accepted):
        registered = True
        if registered_mask is not None and i < len(registered_mask):
            registered = registered_mask[i]
        c2w = None
        if poses is not None and i < len(poses):
            c2w = poses[i]
        elif registered and poses is None:
            # Caller must not claim registered without poses
            registered = False

        fx = fy = None
        if focals is not None and i < len(focals):
            fx = fy = float(focals[i])
        conf = 1.0
        if confidences is not None and i < len(confidences):
            conf = float(confidences[i])

        cameras.append(
            CameraPose(
                image_id=str(i),
                image_name=fr.filename,
                width=fr.width,
                height=fr.height,
                fx=fx,
                fy=fy,
                cx=fr.width / 2.0 if fr.width else None,
                cy=fr.height / 2.0 if fr.height else None,
                c2w=c2w,
                confidence=conf,
                registered=registered and c2w is not None,
            )
        )

    graph = PoseGraph(cameras=cameras)
    graph.recompute()
    return graph


def failure_result(
    provider_name: str,
    message: str,
    *,
    execution_time_s: float = 0.0,
    model_id: str | None = None,
) -> GeometryResult:
    """Honest failure — never success with empty poses."""
    return GeometryResult(
        provider_name=provider_name,
        success=False,
        error_message=message,
        metrics=GeometryMetrics(
            execution_time_s=execution_time_s,
            model_id=model_id,
            registered_fraction=0.0,
        ),
    )


def success_result(
    provider_name: str,
    *,
    pose_graph: PoseGraph,
    point_cloud: PointCloud | None = None,
    confidence: ConfidenceMap | None = None,
    model_id: str | None = None,
    pair_graph: str | None = None,
    execution_time_s: float = 0.0,
    peak_vram_gb: float | None = None,
    artifacts: dict[str, str] | None = None,
    metadata: dict | None = None,
    global_align_loss: float | None = None,
) -> GeometryResult:
    pose_graph.recompute()
    mean_conf = None
    if pose_graph.cameras:
        mean_conf = sum(c.confidence for c in pose_graph.cameras) / len(pose_graph.cameras)

    if confidence is None:
        confidence = ConfidenceMap(
            global_score=mean_conf or 0.0,
            per_camera={c.image_id: c.confidence for c in pose_graph.cameras},
        )

    metrics = GeometryMetrics(
        registered_cameras=pose_graph.registered_count,
        total_cameras=pose_graph.total_count,
        registered_fraction=pose_graph.registered_fraction,
        point_count=point_cloud.point_count if point_cloud else 0,
        mean_camera_confidence=mean_conf,
        global_align_loss=global_align_loss,
        execution_time_s=execution_time_s,
        peak_vram_gb=peak_vram_gb,
        model_id=model_id,
        pair_graph=pair_graph,
    )
    return GeometryResult(
        provider_name=provider_name,
        success=pose_graph.registered_count > 0,
        pose_graph=pose_graph,
        point_cloud=point_cloud,
        confidence=confidence,
        metrics=metrics,
        artifacts=artifacts or {},
        metadata=metadata or {},
        error_message=None
        if pose_graph.registered_count > 0
        else "No cameras registered",
    )


def write_transforms_json(path: Path, pose_graph: PoseGraph, frame_set: FrameSet) -> Path:
    """Write a Nerfstudio-style transforms.json for downstream training."""
    import json
    import math

    accepted = {f.filename: f for f in frame_set.get_accepted()}
    frames_out = []
    for cam in pose_graph.cameras:
        if not cam.registered or cam.c2w is None:
            continue
        fr = accepted.get(cam.image_name)
        w = cam.width or (fr.width if fr else 0)
        h = cam.height or (fr.height if fr else 0)
        fl_x = cam.fx or (0.9 * max(w, 1))
        fl_y = cam.fy or fl_x
        cx = cam.cx if cam.cx is not None else w / 2.0
        cy = cam.cy if cam.cy is not None else h / 2.0
        frames_out.append(
            {
                "file_path": f"images/{cam.image_name}",
                "transform_matrix": cam.c2w,
                "w": w,
                "h": h,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "camera_angle_x": 2 * math.atan(w / (2 * fl_x)) if fl_x else 0.0,
            }
        )

    payload = {
        "camera_model": "OPENCV",
        "frames": frames_out,
        "ply_file_path": "sparse_pc.ply",
        "provider": "propertyscan",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
