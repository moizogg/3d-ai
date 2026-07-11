"""Stage 2 — Decode / materialize candidate frames."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult


class DecodeFramesStage(Stage):
    name = "decode_frames"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("capture_adapter") or not ctx.has("capture_manifest"):
            raise StageError(
                "Missing capture_adapter/manifest — run validate_capture first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        adapter = ctx.require("capture_adapter")
        manifest = ctx.require("capture_manifest")
        root = ctx.artifact_dir("frames")
        paths = adapter.materialize_frames(manifest, root, ctx.config)
        if not paths:
            raise StageError(
                "No candidate frames produced.",
                stage_name=self.name,
                suggestion="Check media readability and FFmpeg/OpenCV installation.",
            )
        ctx.set("candidate_paths", paths)
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={"candidate_count": len(paths)},
            artifacts={"candidates_dir": str(root / "candidates")},
            message=f"Materialized {len(paths)} candidate frames",
        )
