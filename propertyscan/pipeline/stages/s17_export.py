"""Stage 17 — Export PropertyScene via registered exporters."""

from __future__ import annotations

from propertyscan.core.context import RunContext
from propertyscan.core.exceptions import StageError
from propertyscan.core.stage import Stage, StageResult
from propertyscan.export.ply import PlyExporter
from propertyscan.export.scene_json import PropertySceneJsonExporter


class ExportStage(Stage):
    name = "export"

    def validate(self, ctx: RunContext) -> None:
        if not ctx.has("property_scene"):
            raise StageError(
                "Missing property_scene — run assemble_scene first.",
                stage_name=self.name,
            )

    def execute(self, ctx: RunContext) -> StageResult:
        scene = ctx.require("property_scene")
        export_dir = ctx.output_dir / "export"
        exporters = [PlyExporter(), PropertySceneJsonExporter()]
        # Honor config formats if present
        wanted = set(ctx.config.export.formats or ["ply"])
        results = []
        exports: dict[str, str] = {}

        for exp in exporters:
            if exp.format_name == "ply" and "ply" not in wanted:
                continue
            if exp.format_name == "property_scene_json":
                # always write archive json
                pass
            res = exp.export(scene, export_dir)
            results.append(res)
            if res.success and res.path is not None:
                exports[exp.format_name] = str(res.path)

        # Also top-level scene.ply convenience
        if "ply" in exports:
            from pathlib import Path
            import shutil

            top = ctx.output_dir / "scene.ply"
            shutil.copy2(exports["ply"], top)
            exports["scene_ply"] = str(top)

        scene.exports = {**scene.exports, **exports}
        ctx.set("property_scene", scene)
        ctx.set("exports", exports)

        ok = any(r.success for r in results)
        if not ok:
            raise StageError(
                "All exporters failed: "
                + "; ".join(r.error_message or r.format for r in results),
                stage_name=self.name,
            )

        return StageResult(
            stage_name=self.name,
            success=True,
            metrics={
                "exports": len(exports),
                "formats": list(exports.keys()),
            },
            artifacts=exports,
            message=f"Exported: {', '.join(exports.keys())}",
        )
