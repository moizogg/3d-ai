"""Stage 1 — Input validation / capture manifest."""

from __future__ import annotations

from propertyscan.capture.detect import validate_and_load
from propertyscan.core.context import RunContext
from propertyscan.core.stage import Stage, StageResult


class ValidateCaptureStage(Stage):
    name = "validate_capture"

    def execute(self, ctx: RunContext) -> StageResult:
        adapter, manifest = validate_and_load(ctx.input_path, ctx.config)
        ctx.set("capture_adapter", adapter)
        ctx.set("capture_manifest", manifest)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "kind": manifest.kind.value,
                "file_count": manifest.file_count,
                "warnings": len(manifest.warnings),
            },
            message=f"Capture validated as {manifest.kind.value}",
        )
