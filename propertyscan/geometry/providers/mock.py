"""Mock geometry provider for CI and pipeline wiring tests (no neural weights)."""

from __future__ import annotations

import math
import time
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, PointCloud, SceneDescriptor
from propertyscan.geometry.base import GeometryProvider
from propertyscan.geometry.result_builder import (
    pose_graph_from_frames,
    success_result,
    write_transforms_json,
)


class MockGeometryProvider(GeometryProvider):
    """Deterministic synthetic poses for testing pipeline contracts.

    Purpose:
        Exercise router / fusion / health without CUDA or model downloads.

    Limitations:
        Poses are synthetic (circular layout) — not suitable for real scans.
    """

    @property
    def name(self) -> str:
        return "mock"

    @property
    def provider_type(self) -> str:
        return "mock"

    @property
    def requires_cuda(self) -> bool:
        return False

    def can_handle(self, descriptor: SceneDescriptor) -> float:
        return 0.1  # low priority; only for tests / explicit mode

    def reconstruct(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
        descriptor: SceneDescriptor | None = None,
    ) -> GeometryResult:
        t0 = time.perf_counter()
        accepted = frame_set.get_accepted()
        n = len(accepted)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        poses: list[list[list[float]]] = []
        confs: list[float] = []
        for i in range(n):
            angle = 2 * math.pi * i / max(n, 1)
            # Simple c2w: camera on a circle looking at origin
            c2w = [
                [math.cos(angle), 0.0, math.sin(angle), 2.0 * math.sin(angle)],
                [0.0, 1.0, 0.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle), 2.0 * math.cos(angle)],
                [0.0, 0.0, 0.0, 1.0],
            ]
            poses.append(c2w)
            confs.append(0.85)

        focals = [0.9 * max(fr.width, 640) for fr in accepted]
        graph = pose_graph_from_frames(
            frame_set,
            poses=poses,
            focals=focals,
            confidences=confs,
            registered_mask=[True] * n,
        )
        # Tiny synthetic point cloud summary
        pc = PointCloud(
            source="mock",
            point_count=max(n * 100, 100),
            path=output_dir / "mock_points.ply",
        )
        transforms = write_transforms_json(output_dir / "transforms.json", graph, frame_set)
        dt = time.perf_counter() - t0
        return success_result(
            self.name,
            pose_graph=graph,
            point_cloud=pc,
            model_id="mock://synthetic",
            pair_graph="mock",
            execution_time_s=round(dt, 4),
            artifacts={"transforms": str(transforms)},
            metadata={"synthetic": True},
            global_align_loss=0.01,
        )
