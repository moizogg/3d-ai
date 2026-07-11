"""Stage 10 — Geometry + depth fusion."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.geometry.fusion.fuse import fuse_geometry_and_depth


class FuseGeometryStage(Stage):
    name = "fuse_geometry"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("geometry_result"):
            raise StageError("Missing geometry_result", stage_name=self.name)

    def execute(self, ctx: RunContext) -> StageResult:
        fused = fuse_geometry_and_depth(
            ctx.require("geometry_result"),
            ctx.get("depth_result"),
            output_dir=ctx.artifact_dir("fusion"),
        )
        ctx.set("fused_geometry", fused)
        return StageResult(
            stage_name=self.name,
            success=fused.geometry.success,
            metrics={
                "confidence": fused.confidence.global_score,
                "depth_attached": fused.depth is not None
                and bool(fused.depth.success),
                "notes": len(fused.notes),
            },
            artifacts=fused.artifacts,
            message=f"Fusion complete: {', '.join(fused.notes) or 'ok'}",
        )
