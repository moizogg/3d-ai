"""Depth Anything V2 — Phase 4 real monocular depth when transformers+torch available."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from propertyscan.core.config import EngineConfig
from propertyscan.domain.depth import DepthMap, DepthResult
from propertyscan.domain.frames import FrameSet
from propertyscan.geometry.depth.base import DepthProvider
from propertyscan.geometry.deps import INSTALL_HINT_DEPTH, probe_depth_anything, probe_torch
from propertyscan.core.progress import ProgressHeartbeat
from propertyscan.geometry.runtime import empty_cache, peak_vram_gb, reset_peak_vram

logger = logging.getLogger("propertyscan.geometry.depth.anything_v2")

_MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2Provider(DepthProvider):
    """Monocular depth (Depth Anything V2) — first-class component.

    Purpose:
        Dense relative depth maps for fusion and future 3DGS depth supervision.

    Outputs:
        16-bit PNG depth maps (one per keyframe), relative scale.

    Limitations:
        Monocular depth is relative, not metric (unless later ARKit fusion).
        Requires transformers + torch; prefers CUDA.
    """

    @property
    def name(self) -> str:
        return "depth_anything_v2"

    @property
    def is_available(self) -> bool:
        return probe_torch().available and probe_depth_anything().available

    def estimate(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
    ) -> DepthResult:
        t0 = time.perf_counter()
        size = config.depth.size
        model_id = _MODEL_IDS.get(size, _MODEL_IDS["small"])
        accepted = frame_set.get_accepted()
        n = len(accepted)

        if not config.depth.enabled:
            return DepthResult(
                provider_name=self.name,
                success=False,
                error_message="Depth disabled in config (depth.enabled=false).",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        if n == 0:
            return DepthResult(
                provider_name=self.name,
                success=False,
                error_message="No accepted keyframes for depth estimation.",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        if not probe_torch().available or not probe_depth_anything().available:
            return DepthResult(
                provider_name=self.name,
                success=False,
                error_message=f"Depth dependencies missing. {INSTALL_HINT_DEPTH}",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except Exception as exc:
            return DepthResult(
                provider_name=self.name,
                success=False,
                error_message=f"Cannot import transformers/torch: {exc}. {INSTALL_HINT_DEPTH}",
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        use_cuda = config.device.prefer_cuda and torch.cuda.is_available()
        if not use_cuda and not config.device.allow_cpu_geometry:
            return DepthResult(
                provider_name=self.name,
                success=False,
                error_message=(
                    "Depth Anything V2 prefers CUDA (allow_cpu_geometry=false). "
                    f"Model {model_id}."
                ),
                execution_time_s=time.perf_counter() - t0,
                model_id=model_id,
            )

        device_arg = 0 if use_cuda else -1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reset_peak_vram()
        empty_cache()

        try:
            pipe = pipeline("depth-estimation", model=model_id, device=device_arg)
        except Exception as exc:
            # Fallback chain: requested → small → v1 small
            for fallback in (
                _MODEL_IDS["small"],
                "depth-anything/Depth-Anything-V1-Small-hf",
            ):
                if fallback == model_id:
                    continue
                try:
                    logger.warning("Failed to load %s (%s); trying %s", model_id, exc, fallback)
                    pipe = pipeline("depth-estimation", model=fallback, device=device_arg)
                    model_id = fallback
                    break
                except Exception:
                    pipe = None
            if pipe is None:
                return DepthResult(
                    provider_name=self.name,
                    success=False,
                    error_message=f"Failed to load depth model: {exc}",
                    execution_time_s=time.perf_counter() - t0,
                    model_id=model_id,
                )

        maps: list[DepthMap] = []
        max_res = config.depth.max_resolution
        errors = 0

        with ProgressHeartbeat("depth_anything_v2", interval_s=10.0) as hb:
            for i, fr in enumerate(accepted):
                if i == 0 or (i + 1) % 5 == 0 or i + 1 == n:
                    hb.set_status(f"depth {i + 1}/{n}: {fr.filename}")
                try:
                    img = Image.open(fr.filepath).convert("RGB")
                    w0, h0 = img.size
                    if max(w0, h0) > max_res:
                        scale = max_res / float(max(w0, h0))
                        img = img.resize(
                            (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                            Image.Resampling.BILINEAR,
                        )
                    res = pipe(img)
                    depth_img = res["depth"]
                    depth_np = np.array(depth_img, dtype=np.float32)
                    dmin, dmax = float(depth_np.min()), float(depth_np.max())
                    if dmax > dmin:
                        depth_u16 = (
                            (depth_np - dmin) / (dmax - dmin) * 65535.0
                        ).astype(np.uint16)
                    else:
                        depth_u16 = np.zeros_like(depth_np, dtype=np.uint16)

                    depth_pil = Image.fromarray(depth_u16, mode="I;16")
                    if depth_pil.size != (w0, h0):
                        depth_pil = depth_pil.resize(
                            (w0, h0), Image.Resampling.NEAREST
                        )

                    out_file = output_dir / f"{Path(fr.filename).stem}.png"
                    depth_pil.save(out_file)
                    maps.append(
                        DepthMap(
                            image_id=str(i),
                            image_name=fr.filename,
                            path=out_file,
                            width=w0,
                            height=h0,
                            scale="relative",
                            min_depth=dmin,
                            max_depth=dmax,
                        )
                    )
                except Exception as err:
                    errors += 1
                    logger.warning("Depth failed for %s: %s", fr.filename, err)

            empty_cache()
            del pipe
            empty_cache()

        if not maps:
            return DepthResult(
                provider_name=self.name,
                success=False,
                error_message=f"All depth estimates failed ({errors} errors).",
                execution_time_s=round(time.perf_counter() - t0, 3),
                model_id=model_id,
                peak_vram_gb=peak_vram_gb(),
            )

        return DepthResult(
            provider_name=self.name,
            success=True,
            depth_maps=maps,
            scale_hint="relative",
            execution_time_s=round(time.perf_counter() - t0, 3),
            model_id=model_id,
            peak_vram_gb=peak_vram_gb(),
            artifacts={"depth_dir": str(output_dir)},
            metadata={
                "maps": len(maps),
                "errors": errors,
                "size_setting": size,
                "device": "cuda" if use_cuda else "cpu",
            },
        )
