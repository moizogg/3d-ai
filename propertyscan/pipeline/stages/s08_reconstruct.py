"""Stage 8 — Foundation geometry reconstruction (MASt3R / DUSt3R / mock)."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.domain.geometry import SceneDescriptor
from propertyscan.geometry.router import GeometryRouter


class ReconstructGeometryStage(Stage):
    name = "reconstruct_geometry"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("frame_set"):
            raise StageError("Missing frame_set", stage_name=self.name)

    def execute(self, ctx: RunContext) -> StageResult:
        frame_set = ctx.require("frame_set")
        descriptor = ctx.get("scene_descriptor") or SceneDescriptor(
            frame_count=frame_set.accepted_count
        )
        router = ctx.get("geometry_router")
        if router is None:
            router = GeometryRouter(
                ctx.config,
                include_mock=ctx.config.geometry.engine == "mock",
            )
            ctx.set("geometry_router", router)

        out_dir = ctx.artifact_dir("geometry")
        result = router.reconstruct(
            frame_set,
            output_dir=out_dir,
            descriptor=descriptor,
            config=ctx.config,
        )
        ctx.set("geometry_result", result)
        if ctx.provenance and result.metrics.model_id:
            ctx.provenance.set_model("geometry", result.metrics.model_id)

        return StageResult(
            stage_name=self.name,
            success=result.success,
            metrics={
                "provider": result.provider_name,
                "registered": result.metrics.registered_cameras,
                "total": result.metrics.total_cameras,
                "registered_fraction": result.metrics.registered_fraction,
                "points": result.metrics.point_count,
            },
            artifacts=result.artifacts,
            message=(
                f"Geometry OK via {result.provider_name}"
                if result.success
                else f"Geometry failed: {result.error_message}"
            ),
            error=None if result.success else result.error_message,
        )
