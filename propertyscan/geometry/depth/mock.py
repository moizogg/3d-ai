"""Mock depth provider for tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from propertyscan.core.config import EngineConfig
from propertyscan.domain.depth import DepthMap, DepthResult
from propertyscan.domain.frames import FrameSet
from propertyscan.geometry.depth.base import DepthProvider


class MockDepthProvider(DepthProvider):
    """Writes synthetic 16-bit depth PNGs for accepted keyframes."""

    @property
    def name(self) -> str:
        return "mock_depth"

    @property
    def requires_cuda(self) -> bool:
        return False

    def estimate(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
    ) -> DepthResult:
        t0 = time.perf_counter()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        maps: list[DepthMap] = []

        for i, fr in enumerate(frame_set.get_accepted()):
            h, w = max(fr.height, 64), max(fr.width, 64)
            # Gradient depth for tests
            yy = np.linspace(0.2, 1.0, h, dtype=np.float32)
            depth = np.tile(yy[:, None], (1, w))
            depth_u16 = (depth * 65535.0).astype(np.uint16)
            out = output_dir / f"{Path(fr.filename).stem}.png"
            Image.fromarray(depth_u16, mode="I;16").save(out)
            maps.append(
                DepthMap(
                    image_id=str(i),
                    image_name=fr.filename,
                    path=out,
                    width=w,
                    height=h,
                    scale="relative",
                    min_depth=0.2,
                    max_depth=1.0,
                )
            )

        return DepthResult(
            provider_name=self.name,
            success=True,
            depth_maps=maps,
            scale_hint="relative",
            execution_time_s=round(time.perf_counter() - t0, 4),
            model_id="mock://depth",
            artifacts={"depth_dir": str(output_dir)},
            metadata={"synthetic": True},
        )
