"""Stage 3 — Frame quality analysis."""

from __future__ import annotations

from pathlib import Path

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.intelligence.quality import analyze_frames


class FrameQualityStage(Stage):
    name = "frame_quality"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("candidate_paths"):
            raise StageError(
                "Missing candidate_paths — run decode_frames first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        paths: list[Path] = ctx.require("candidate_paths")
        frames = analyze_frames(paths, ctx.config)
        ctx.set("analyzed_frames", frames)
        hard = sum(1 for f in frames if f.is_hard_rejected())
        candidates = sum(
            1 for f in frames if f.status.value in ("candidate", "low_rank")
        )
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "analyzed": len(frames),
                "hard_rejected": hard,
                "selectable": candidates,
                "mean_quality": round(
                    sum(f.quality_score for f in frames) / max(len(frames), 1), 2
                ),
                "mode": "reliable_v2",
            },
            message=f"Reliable validation: {candidates} selectable, {hard} hard-rejected",
        )
