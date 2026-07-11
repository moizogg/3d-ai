"""Stage 13 — Gaussian training backend."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.training.factory import get_trainer


class TrainGaussiansStage(Stage):
    name = "train_gaussians"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("training_dataset"):
            raise StageError(
                "Missing training_dataset — run build_dataset first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        dataset = ctx.require("training_dataset")
        trainer = get_trainer(ctx.config)
        out = ctx.artifact_dir("training")

        if not trainer.is_available() and trainer.name != "mock":
            raise StageError(
                f"Trainer '{trainer.name}' is not available in this environment.",
                stage_name=self.name,
                suggestion="Install Nerfstudio (ns-train) or set training.backend=mock.",
            )

        result = trainer.train(dataset, output_dir=out, config=ctx.config)
        ctx.set("train_result", result)
        if result.scene is not None:
            ctx.set("gaussian_scene", result.scene)
        if ctx.provenance:
            ctx.provenance.set_model("trainer", result.backend)

        if not result.success:
            raise StageError(
                result.error_message or "Training failed",
                stage_name=self.name,
                suggestion="Check training logs under artifacts/training/.",
                details=result.to_dict(),
            )

        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "backend": result.backend,
                "iterations": result.iterations,
                "duration_s": result.execution_time_s,
                **{k: v for k, v in result.metrics.items() if isinstance(v, (int, float, str, bool))},
            },
            artifacts={
                "train_dir": str(result.train_dir) if result.train_dir else "",
                "ply": str(result.scene.path)
                if result.scene and result.scene.path
                else "",
            },
            message=(
                f"Training OK via {result.backend} "
                f"({result.iterations} iters, {result.execution_time_s:.1f}s)"
            ),
        )
