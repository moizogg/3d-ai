"""DUSt3R geometry provider — Phase 4 real neural inference when deps+CUDA available."""

from __future__ import annotations

import time
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, PointCloud, SceneDescriptor
from propertyscan.geometry.base import GeometryProvider
from propertyscan.geometry.deps import probe_dust3r, probe_torch
from propertyscan.geometry.foundation_infer import run_foundation_reconstruction
from propertyscan.geometry.result_builder import (
    failure_result,
    pose_graph_from_frames,
    success_result,
    write_transforms_json,
)
from propertyscan.geometry.runtime import empty_cache


class DUSt3RProvider(GeometryProvider):
    """NAVER DUSt3R ViT-Large dense stereo foundation model.

    Same contract as MASt3RProvider; alternate engine for Auto routing.
    No COLMAP fallback.
    """

    @property
    def name(self) -> str:
        return "dust3r"

    @property
    def provider_type(self) -> str:
        return "foundation"

    @property
    def requires_cuda(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        t = probe_torch()
        d = probe_dust3r()
        if not (t.available and d.available):
            return False
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def can_handle(self, descriptor: SceneDescriptor) -> float:
        score = 0.80
        if descriptor.scene_type == "residential_indoor":
            score = 0.82
        if descriptor.frame_count > 80:
            score = 0.78
        return min(1.0, score)

    def reconstruct(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
        descriptor: SceneDescriptor | None = None,
    ) -> GeometryResult:
        t0 = time.perf_counter()
        model_id = config.geometry.dust3r_model
        accepted = frame_set.get_accepted()
        n = len(accepted)
        if n < 2:
            return failure_result(
                self.name,
                f"Need at least 2 keyframes for DUSt3R; got {n}.",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        paths = [f.filepath for f in accepted]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        infer = run_foundation_reconstruction(
            engine="dust3r",
            image_paths=paths,
            output_dir=out,
            model_id=model_id,
            pair_graph=config.geometry.pair_graph,
            batch_size=config.geometry.batch_size,
            global_align_iters=config.geometry.global_align_iters,
            prefer_cuda=config.device.prefer_cuda,
            allow_cpu=config.device.allow_cpu_geometry,
            image_size=512,
        )

        if not infer.success:
            empty_cache()
            return failure_result(
                self.name,
                infer.error_message or "DUSt3R failed",
                execution_time_s=infer.execution_time_s,
                model_id=model_id,
            )

        registered_mask = [c > 0.05 for c in infer.confidences]
        if not any(registered_mask) and infer.poses:
            registered_mask = [True] * len(infer.poses)

        graph = pose_graph_from_frames(
            frame_set,
            poses=infer.poses,
            focals=infer.focals,
            confidences=infer.confidences,
            registered_mask=registered_mask,
        )
        for cam, conf in zip(graph.cameras, infer.confidences):
            if conf <= 0.05:
                cam.registered = False
                cam.c2w = None
                cam.confidence = conf
        graph.recompute()

        pc = None
        if infer.ply_path is not None:
            pc = PointCloud(
                source="dust3r",
                path=infer.ply_path,
                point_count=infer.point_count,
            )
        elif infer.point_count:
            pc = PointCloud(source="dust3r", point_count=infer.point_count)

        artifacts: dict[str, str] = {}
        if infer.ply_path is not None:
            artifacts["dense_ply"] = str(infer.ply_path)
        tpath = write_transforms_json(out / "transforms.json", graph, frame_set)
        artifacts["transforms"] = str(tpath)

        empty_cache()
        return success_result(
            self.name,
            pose_graph=graph,
            point_cloud=pc,
            model_id=model_id,
            pair_graph=infer.pair_graph,
            execution_time_s=infer.execution_time_s,
            peak_vram_gb=infer.peak_vram_gb,
            artifacts=artifacts,
            metadata={
                **infer.metadata,
                "device": infer.device,
                "engine": "dust3r",
            },
            global_align_loss=infer.global_align_loss,
        )
