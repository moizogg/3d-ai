"""Stage 5 — Keyframe selection."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.intelligence.keyframes import select_keyframes


class KeyframeStage(Stage):
    name = "keyframes"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("analyzed_frames"):
            raise StageError(
                "Missing analyzed_frames — run quality/dedup first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        selected_dir = ctx.artifact_dir("frames", "selected")
        frame_set = select_keyframes(
            ctx.require("analyzed_frames"),
            ctx.config,
            selected_dir=selected_dir,
        )
        ctx.set("frame_set", frame_set)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "accepted": frame_set.accepted_count,
                "rejected": frame_set.rejected_count,
            },
            artifacts={"selected_dir": str(selected_dir)},
            message=f"Selected {frame_set.accepted_count} keyframes",
        )
