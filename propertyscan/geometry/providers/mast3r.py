"""MASt3R geometry provider — Phase 4 real neural inference when deps+CUDA available."""

from __future__ import annotations

import time
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, PointCloud, SceneDescriptor
from propertyscan.geometry.base import GeometryProvider
from propertyscan.geometry.deps import probe_mast3r, probe_torch
from propertyscan.geometry.foundation_infer import run_foundation_reconstruction
from propertyscan.geometry.result_builder import (
    failure_result,
    pose_graph_from_frames,
    success_result,
    write_transforms_json,
)
from propertyscan.geometry.runtime import empty_cache


class MASt3RProvider(GeometryProvider):
    """NAVER MASt3R ViT-Large foundation geometry (primary engine).

    Purpose:
        Dense point maps + camera poses for real-estate keyframes.

    Inputs:
        Accepted FrameSet keyframes.

    Outputs:
        GeometryResult with pose_graph, optional dense PLY, transforms.json.

    Limitations:
        Requires dust3r + mast3r packages and preferably CUDA.
        Does not fall back to COLMAP. Fails honestly if deps/weights missing.
    """

    @property
    def name(self) -> str:
        return "mast3r"

    @property
    def provider_type(self) -> str:
        return "foundation"

    @property
    def requires_cuda(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        t = probe_torch()
        m = probe_mast3r()
        if not (t.available and m.available):
            return False
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def can_handle(self, descriptor: SceneDescriptor) -> float:
        score = 0.85
        if descriptor.scene_type in ("low_texture_interior", "reflective_interior"):
            score = 0.92
        if descriptor.blur_ratio > 0.3:
            score = 0.88
        if descriptor.frame_count < 4:
            score *= 0.5
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
        model_id = config.geometry.mast3r_model
        accepted = frame_set.get_accepted()
        n = len(accepted)
        if n < 2:
            return failure_result(
                self.name,
                f"Need at least 2 keyframes for MASt3R; got {n}.",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        paths = [f.filepath for f in accepted]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        infer = run_foundation_reconstruction(
            engine="mast3r",
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
                infer.error_message or "MASt3R failed",
                execution_time_s=infer.execution_time_s,
                model_id=model_id,
            )

        registered_mask = [c > 0.05 for c in infer.confidences]
        # If all low conf but poses exist, still register with raw conf
        if not any(registered_mask) and infer.poses:
            registered_mask = [True] * len(infer.poses)

        graph = pose_graph_from_frames(
            frame_set,
            poses=infer.poses,
            focals=infer.focals,
            confidences=infer.confidences,
            registered_mask=registered_mask,
        )
        # Drop registration when confidence is ~0
        for cam, conf in zip(graph.cameras, infer.confidences):
            if conf <= 0.05:
                cam.registered = False
                cam.c2w = None
                cam.confidence = conf
        graph.recompute()

        pc = None
        if infer.ply_path is not None:
            pc = PointCloud(
                source="mast3r",
                path=infer.ply_path,
                point_count=infer.point_count,
            )
        elif infer.point_count:
            pc = PointCloud(source="mast3r", point_count=infer.point_count)

        artifacts: dict[str, str] = {}
        if infer.ply_path is not None:
            artifacts["dense_ply"] = str(infer.ply_path)
        try:
            tpath = write_transforms_json(out / "transforms.json", graph, frame_set)
            artifacts["transforms"] = str(tpath)
        except Exception as exc:
            return failure_result(
                self.name,
                f"MASt3R poses ok but transforms.json write failed: {exc}",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

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
                "engine": "mast3r",
            },
            global_align_loss=infer.global_align_loss,
        )
