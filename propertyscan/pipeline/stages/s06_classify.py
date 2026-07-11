"""Stage 6 — Scene classification."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.intelligence.classify import classify_scene


class ClassifySceneStage(Stage):
    name = "classify_scene"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("frame_set"):
            raise StageError(
                "Missing frame_set — run keyframes first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        descriptor = classify_scene(ctx.require("frame_set"))
        ctx.set("scene_descriptor", descriptor)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "scene_type": descriptor.scene_type,
                "texture_score": descriptor.texture_score,
                "blur_ratio": descriptor.blur_ratio,
            },
            message=f"Scene classified as {descriptor.scene_type}",
        )
