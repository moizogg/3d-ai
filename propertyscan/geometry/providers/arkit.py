"""Apple ARKit geometry provider stub (future metric poses + depth)."""

from __future__ import annotations

import time
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, SceneDescriptor
from propertyscan.geometry.base import GeometryProvider
from propertyscan.geometry.result_builder import failure_result


class AppleARKitProvider(GeometryProvider):
    """Future provider for transforms.json + metric scale + depth maps.

    Phase 3: interface reserved. Does not fabricate ARKit poses.
    """

    @property
    def name(self) -> str:
        return "arkit"

    @property
    def provider_type(self) -> str:
        return "arkit"

    @property
    def requires_cuda(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        return False  # not implemented yet

    def can_handle(self, descriptor: SceneDescriptor) -> float:
        # Only preferred when tags indicate ARKit capture (set by future adapter)
        if "arkit" in descriptor.tags or "metric_scale" in descriptor.tags:
            return 0.99
        return 0.05

    def reconstruct(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        config: EngineConfig,
        descriptor: SceneDescriptor | None = None,
    ) -> GeometryResult:
        return failure_result(
            self.name,
            (
                "AppleARKitProvider is a Phase 3+ stub. "
                "Provide video/image capture for MASt3R/DUSt3R for now."
            ),
            execution_time_s=0.0,
            model_id="arkit://future",
        )
