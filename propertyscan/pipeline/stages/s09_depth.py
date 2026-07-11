"""Stage 9 — Monocular depth (Depth Anything V2 / mock)."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.geometry.depth.anything_v2 import DepthAnythingV2Provider
from propertyscan.geometry.depth.mock import MockDepthProvider


class EstimateDepthStage(Stage):
    name = "estimate_depth"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("frame_set"):
            raise StageError("Missing frame_set", stage_name=self.name)

    def execute(self, ctx: RunContext) -> StageResult:
        if not ctx.config.depth.enabled:
            ctx.set("depth_result", None)
            return StageResult(
                stage_name=self.name,
                success=True,
                metrics={"skipped": True},
                message="Depth disabled in config",
            )

        # Use mock when geometry engine is mock or provider name is mock path
        use_mock = ctx.config.geometry.engine == "mock"
        provider = MockDepthProvider(ctx.config) if use_mock else DepthAnythingV2Provider(
            ctx.config
        )
        out = ctx.artifact_dir("depth")
        result = provider.estimate(
            ctx.require("frame_set"),
            output_dir=out,
            config=ctx.config,
        )
        ctx.set("depth_result", result)
        if ctx.provenance and result.model_id:
            ctx.provenance.set_model("depth", result.model_id)

        # Depth failure is soft in Phase 3 (fusion continues without it)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "provider": result.provider_name,
                "depth_success": result.success,
                "maps": len(result.depth_maps),
            },
            artifacts=result.artifacts,
            message=(
                f"Depth OK ({len(result.depth_maps)} maps)"
                if result.success
                else f"Depth unavailable: {result.error_message}"
            ),
        )
