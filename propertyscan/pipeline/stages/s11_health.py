"""Stage 11 — Geometry validation + pre-training health gate."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import HealthGateError, StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.geometry.validation.geometric import validate_geometry
from propertyscan.geometry.validation.health import evaluate_pretrain_health


class HealthGateStage(Stage):
    name = "health_gate"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("fused_geometry") and not ctx.has("geometry_result"):
            raise StageError("Missing geometry for health gate", stage_name=self.name)

    def execute(self, ctx: RunContext) -> StageResult:
        fused = ctx.get("fused_geometry")
        target = fused if fused is not None else ctx.require("geometry_result")
        validation = validate_geometry(target, ctx.config)
        health = evaluate_pretrain_health(
            fused=fused if fused is not None else None,
            validation=validation,
            frame_set=ctx.get("frame_set"),
            descriptor=ctx.get("scene_descriptor"),
            config=ctx.config,
        )
        ctx.set("geometry_validation", validation)
        ctx.set("health_report", health)

        metrics = {
            "health_score": health.score,
            "passed": health.passed,
            "expected_quality": health.expected_quality,
            "needle_probability": health.needle_probability,
            "validation_passed": validation.passed,
        }

        abort = (
            ctx.config.health.abort_below_min_score
            and ctx.config.run.abort_on_health_fail
            and not health.passed
        )
        if abort:
            raise HealthGateError(
                (
                    f"Pre-training health gate FAILED (score={health.score}, "
                    f"needle_p={health.needle_probability}). "
                    f"Reasons: {'; '.join(health.reasons) or 'n/a'}"
                ),
                suggestion="; ".join(health.recommendations)
                or "Improve capture / geometry before training.",
                details=health.to_dict(),
            )

        return StageResult(
            stage_name=self.name,
            success=True,
            metrics=metrics,
            message=(
                f"Health {health.score}/100 ({health.expected_quality}) "
                f"passed={health.passed}"
            ),
        )
