"""Stage 12 — Build 3DGS training dataset from fused geometry."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import HealthGateError, StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.dataset.builder import DatasetBuildError, DatasetBuilder


class BuildDatasetStage(Stage):
    name = "build_dataset"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("frame_set"):
            raise StageError("Missing frame_set", stage_name=self.name)
        if not ctx.has("geometry_result") and not ctx.has("fused_geometry"):
            raise StageError("Missing geometry_result/fused_geometry", stage_name=self.name)

    def execute(self, ctx: RunContext) -> StageResult:
        if ctx.config.training.require_health_pass:
            health = ctx.get("health_report")
            if health is not None and not health.passed:
                raise HealthGateError(
                    f"Refusing dataset build: health gate failed (score={health.score}).",
                    suggestion="; ".join(health.recommendations)
                    or "Fix geometry before training.",
                    details=health.to_dict(),
                )

        geom = ctx.get("fused_geometry") or ctx.require("geometry_result")
        depth = ctx.get("depth_result")
        out = ctx.artifact_dir("dataset")
        try:
            dataset = DatasetBuilder().build(
                frame_set=ctx.require("frame_set"),
                geometry=geom,
                output_dir=out,
                config=ctx.config,
                depth=depth,
            )
        except DatasetBuildError as exc:
            raise StageError(
                str(exc),
                stage_name=self.name,
                suggestion=exc.suggestion,
                details=exc.details,
            ) from exc

        ctx.set("training_dataset", dataset)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "frames": dataset.frame_count,
                "has_depth": dataset.has_depth,
                "has_init_cloud": dataset.has_init_point_cloud,
                "downscale": dataset.downscale_factor,
            },
            artifacts={
                "dataset_root": str(dataset.root),
                "transforms": str(dataset.transforms_path),
            },
            message=(
                f"Dataset: {dataset.frame_count} frames "
                f"(depth={dataset.has_depth}, init_cloud={dataset.has_init_point_cloud})"
            ),
        )
