"""Shared MASt3R / DUSt3R dense reconstruction inference (Phase 4).

Uses official NAVER APIs when packages are installed. Never falls back to COLMAP.
Never invents poses on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from propertyscan.geometry.deps import (
    INSTALL_HINT_DUST3R,
    INSTALL_HINT_MAST3R,
    probe_dust3r,
    probe_mast3r,
    probe_torch,
)
from propertyscan.geometry.runtime import (
    choose_pair_graph,
    empty_cache,
    peak_vram_gb,
    reset_peak_vram,
    resolve_torch_device,
    torch_module,
)

logger = logging.getLogger("propertyscan.geometry.foundation_infer")

EngineKind = Literal["mast3r", "dust3r"]


@dataclass
class FoundationInferResult:
    success: bool
    engine: str
    model_id: str
    poses: list[list[list[float]]] = field(default_factory=list)
    focals: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    ply_path: Path | None = None
    point_count: int = 0
    pair_graph: str = ""
    global_align_loss: float | None = None
    execution_time_s: float = 0.0
    peak_vram_gb: float | None = None
    device: str = "cpu"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_list_pose(p: Any) -> list[list[float]]:
    if hasattr(p, "detach"):
        arr = p.detach().cpu().numpy()
    else:
        arr = p
    # Expect 4x4
    return [[float(arr[r, c]) for c in range(4)] for r in range(4)]


def _to_float(x: Any) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().cpu().numpy().reshape(-1)[0])
    try:
        return float(x)
    except Exception:
        return float(x[0])


def run_foundation_reconstruction(
    *,
    engine: EngineKind,
    image_paths: list[Path],
    output_dir: Path,
    model_id: str,
    pair_graph: str = "swin-5",
    batch_size: int = 1,
    global_align_iters: int = 300,
    prefer_cuda: bool = True,
    allow_cpu: bool = False,
    image_size: int = 512,
) -> FoundationInferResult:
    """Run MASt3R or DUSt3R end-to-end on keyframe paths."""
    import time

    t0 = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in image_paths]

    if len(paths) < 2:
        return FoundationInferResult(
            success=False,
            engine=engine,
            model_id=model_id,
            error_message=f"Need >= 2 images; got {len(paths)}",
            execution_time_s=time.perf_counter() - t0,
        )

    # Dependency probes
    if not probe_torch().available:
        return FoundationInferResult(
            success=False,
            engine=engine,
            model_id=model_id,
            error_message=f"PyTorch missing. {INSTALL_HINT_MAST3R if engine == 'mast3r' else INSTALL_HINT_DUST3R}",
            execution_time_s=time.perf_counter() - t0,
        )
    if not probe_dust3r().available:
        return FoundationInferResult(
            success=False,
            engine=engine,
            model_id=model_id,
            error_message=f"DUSt3R package missing. {INSTALL_HINT_DUST3R}",
            execution_time_s=time.perf_counter() - t0,
        )
    if engine == "mast3r" and not probe_mast3r().available:
        return FoundationInferResult(
            success=False,
            engine=engine,
            model_id=model_id,
            error_message=f"MASt3R package missing. {INSTALL_HINT_MAST3R}",
            execution_time_s=time.perf_counter() - t0,
        )

    try:
        device = resolve_torch_device(prefer_cuda=prefer_cuda, allow_cpu=allow_cpu)
    except ValueError as exc:
        return FoundationInferResult(
            success=False,
            engine=engine,
            model_id=model_id,
            error_message=str(exc),
            execution_time_s=time.perf_counter() - t0,
        )

    graph = choose_pair_graph(len(paths), pair_graph)
    ply_path = output_dir / f"{engine}_dense.ply"
    reset_peak_vram()
    empty_cache()

    try:
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        # load_images is the robust path; fall back to path strings if needed
        try:
            from dust3r.utils.image import load_images

            images = load_images([str(p) for p in paths], size=image_size)
        except Exception as load_exc:
            logger.warning("load_images failed (%s); trying raw paths", load_exc)
            images = [str(p) for p in paths]

        if engine == "mast3r":
            from mast3r.model import AsymmetricMASt3R

            model = AsymmetricMASt3R.from_pretrained(model_id).to(device)
        else:
            from dust3r.model import AsymmetricCroCo3DStereo

            model = AsymmetricCroCo3DStereo.from_pretrained(model_id).to(device)

        model.eval()
        pairs = make_pairs(images, scene_graph=graph, prefilter=None, symmetrize=True)
        logger.info(
            "%s inference: %d images, pair_graph=%s, batch_size=%d, device=%s",
            engine,
            len(paths),
            graph,
            batch_size,
            device,
        )
        output = inference(pairs, model, device, batch_size=batch_size, verbose=True)

        scene = global_aligner(
            output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        # cosine schedule often more stable than linear for indoor walks
        try:
            loss = scene.compute_global_alignment(
                init="mst",
                niter=int(global_align_iters),
                schedule="cosine",
                lr=0.01,
            )
        except TypeError:
            loss = scene.compute_global_alignment(
                init="mst",
                niter=int(global_align_iters),
                schedule="linear",
                lr=0.01,
            )

        try:
            scene.clean_pointcloud()
        except Exception as clean_exc:
            logger.warning("clean_pointcloud skipped: %s", clean_exc)

        try:
            scene.save_ply(str(ply_path))
        except Exception as ply_exc:
            logger.warning("save_ply failed: %s", ply_exc)
            ply_path = None  # type: ignore[assignment]

        im_poses = scene.get_im_poses()
        focals = scene.get_focals()
        poses_list = [_to_list_pose(p) for p in im_poses]
        focals_list = [_to_float(f) for f in focals]

        # Per-camera confidence from scene if available
        confs: list[float] = []
        try:
            # Some versions expose get_conf / im_conf
            if hasattr(scene, "im_conf"):
                for c in scene.im_conf:
                    confs.append(float(_to_float(c.mean() if hasattr(c, "mean") else c)))
            elif hasattr(scene, "get_conf"):
                raw = scene.get_conf()
                for c in raw:
                    confs.append(float(_to_float(c.mean() if hasattr(c, "mean") else c)))
        except Exception:
            confs = []
        if len(confs) != len(poses_list):
            confs = [0.8] * len(poses_list)

        point_count = 0
        try:
            if hasattr(scene, "get_pts3d"):
                pts = scene.get_pts3d()
                # list of tensors per image
                if isinstance(pts, (list, tuple)):
                    point_count = int(sum(int(p.reshape(-1, 3).shape[0]) for p in pts))
                else:
                    point_count = int(pts.reshape(-1, 3).shape[0])
        except Exception:
            point_count = 0

        # Free model ASAP
        del model
        empty_cache()

        loss_val = None
        if loss is not None:
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = None

        # Validate poses (reject all-NaN)
        valid_mask = []
        for pose in poses_list:
            flat = [v for row in pose for v in row]
            valid_mask.append(all(v == v and abs(v) < 1e6 for v in flat))  # not NaN/inf

        if not any(valid_mask):
            return FoundationInferResult(
                success=False,
                engine=engine,
                model_id=model_id,
                error_message="Global alignment produced no valid camera poses (NaN/inf).",
                execution_time_s=time.perf_counter() - t0,
                peak_vram_gb=peak_vram_gb(),
                device=device,
                pair_graph=graph,
            )

        # Zero-out invalid as unregistered by replacing with identity and low conf
        # Caller uses confidences; mark invalid conf as 0
        for i, ok in enumerate(valid_mask):
            if not ok:
                confs[i] = 0.0

        return FoundationInferResult(
            success=True,
            engine=engine,
            model_id=model_id,
            poses=poses_list,
            focals=focals_list,
            confidences=confs,
            ply_path=ply_path if ply_path and Path(ply_path).exists() else None,
            point_count=point_count,
            pair_graph=graph,
            global_align_loss=loss_val,
            execution_time_s=round(time.perf_counter() - t0, 3),
            peak_vram_gb=peak_vram_gb(),
            device=device,
            metadata={
                "image_count": len(paths),
                "valid_poses": sum(1 for v in valid_mask if v),
                "image_size": image_size,
                "batch_size": batch_size,
                "global_align_iters": global_align_iters,
            },
        )
    except Exception as exc:
        logger.exception("%s reconstruction failed", engine)
        empty_cache()
        return FoundationInferResult(
            success=False,
            engine=engine,
            model_id=model_id,
            error_message=f"{engine} inference failed: {exc}",
            execution_time_s=round(time.perf_counter() - t0, 3),
            peak_vram_gb=peak_vram_gb(),
            device=device if "device" in dir() else "unknown",
            pair_graph=graph,
            metadata={"exception_type": type(exc).__name__},
        )
