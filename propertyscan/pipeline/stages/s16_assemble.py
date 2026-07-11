"""Stage 16/19 — Assemble canonical PropertyScene."""

from __future__ import annotations

import json

from propertyscan.core.context import RunContext
from propertyscan.core.stage import Stage, StageResult
from propertyscan.scene.builder import build_property_scene


class AssembleSceneStage(Stage):
    name = "assemble_scene"

    def execute(self, ctx: RunContext) -> StageResult:
        scene = build_property_scene(ctx)
        ctx.set("property_scene", scene)
        # Persist archive early
        out = ctx.output_dir / "property_scene.json"
        out.write_text(
            json.dumps(scene.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "scene_id": scene.metadata.scene_id,
                "has_quality": scene.quality is not None,
                "has_inspection": scene.inspection is not None,
            },
            artifacts={"property_scene_json": str(out)},
            message=f"PropertyScene assembled: {scene.metadata.scene_id}",
        )
