"""Stage 16 — Geometry-dominated final quality score."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.stage import Stage, StageResult
from propertyscan.quality.scorer import score_scene


class QualityScoreStage(Stage):
    name = "quality_score"

    def execute(self, ctx: RunContext) -> StageResult:
        report = score_scene(
            geometry=ctx.get("geometry_result"),
            health=ctx.get("health_report"),
            inspection=ctx.get("inspection_report"),
            frame_set=ctx.get("frame_set"),
            descriptor=ctx.get("scene_descriptor"),
        )
        ctx.set("quality_report", report)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "overall": report.overall,
                "status": report.status,
                "camera": report.camera,
                "artifacts": report.artifacts,
                "failure_class": report.failure_class or "",
            },
            message=f"Quality {report.overall}/100 ({report.status})",
        )
