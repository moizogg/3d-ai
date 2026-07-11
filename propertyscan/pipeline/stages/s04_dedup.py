"""Stage 4 — Frame deduplication."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.intelligence.dedup import mark_duplicates


class DedupStage(Stage):
    name = "dedup"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("analyzed_frames"):
            raise StageError(
                "Missing analyzed_frames — run frame_quality first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        frames = mark_duplicates(ctx.require("analyzed_frames"), ctx.config)
        ctx.set("analyzed_frames", frames)
        red = sum(1 for f in frames if f.status.value == "redundant")
        selectable = sum(
            1 for f in frames if f.status.value in ("candidate", "low_rank")
        )
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "redundant_stationary": red,
                "selectable": selectable,
                "mode": "motion_based",
            },
            message=f"Motion redundancy: {red} near-stationary, {selectable} selectable",
        )
