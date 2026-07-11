"""Stage 7 — Geometry provider selection (router rank)."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.domain.geometry import SceneDescriptor
from propertyscan.geometry.router import GeometryRouter


class RouteGeometryStage(Stage):
    name = "route_geometry"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("frame_set"):
            raise StageError(
                "Missing frame_set — run frame intelligence first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        descriptor = ctx.get("scene_descriptor") or SceneDescriptor(
            frame_count=ctx.require("frame_set").accepted_count
        )
        include_mock = ctx.config.geometry.engine == "mock"
        router = GeometryRouter(ctx.config, include_mock=include_mock)
        ranked = router.rank(descriptor)
        if not ranked:
            raise StageError(
                f"No geometry provider for engine={ctx.config.geometry.engine}",
                stage_name=self.name,
                suggestion="Use mast3r | dust3r | auto | mock. COLMAP is not supported.",
            )
        provider = ranked[0][1]
        ctx.set("geometry_router", router)
        ctx.set("geometry_provider_name", provider.name)
        ctx.set("geometry_rankings", [(s, p.name) for s, p in ranked])
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "selected": provider.name,
                "score": ranked[0][0],
                "mode": ctx.config.geometry.engine,
                "candidates": [p.name for _, p in ranked],
            },
            message=f"Selected geometry provider: {provider.name}",
        )
