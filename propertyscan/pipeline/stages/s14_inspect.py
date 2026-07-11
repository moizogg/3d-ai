"""Stage 14/15 — Post-train scene inspection and artifact removal."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.optimize.inspector import SceneInspector


class InspectSceneStage(Stage):
    name = "inspect_scene"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("gaussian_scene"):
            raise StageError(
                "Missing gaussian_scene — run training first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        scene = ctx.require("gaussian_scene")
        result = SceneInspector().inspect(scene, output_dir=ctx.artifact_dir("inspect"))
        ctx.set("inspection_report", result.report)
        ctx.set("gaussian_scene", result.cleaned_scene)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "before": result.report.total_gaussians_before,
                "after": result.report.total_gaussians_after,
                "needles": result.report.needles_removed,
                "floaters": result.report.floaters_removed,
                "huge": result.report.huge_removed,
                "tiny": result.report.tiny_removed,
                "reduction_pct": result.report.size_reduction_pct,
            },
            artifacts={
                "cleaned_ply": str(result.cleaned_path) if result.cleaned_path else "",
            },
            message=(
                f"Inspector: {result.report.total_gaussians_before}→"
                f"{result.report.total_gaussians_after} "
                f"(-{result.report.size_reduction_pct:.1f}%)"
            ),
        )
