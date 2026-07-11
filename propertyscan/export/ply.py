"""PLY exporter — prefers cleaned Gaussian PLY, falls back to raw."""

from __future__ import annotations

import shutil
from pathlib import Path

from propertyscan.domain.scene import PropertyScene
from propertyscan.export.base import Exporter, ExportResult


class PlyExporter(Exporter):
    """Copy the best available Gaussian PLY into the export directory."""

    @property
    def format_name(self) -> str:
        return "ply"

    def export(self, scene: PropertyScene, output_dir: Path) -> ExportResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gs = scene.gaussian_scene
        if gs is None:
            return ExportResult(
                success=False,
                format=self.format_name,
                error_message="PropertyScene has no gaussian_scene",
            )

        src: Path | None = None
        kind = "none"
        if gs.cleaned_path and Path(gs.cleaned_path).is_file():
            src = Path(gs.cleaned_path)
            kind = "cleaned"
        elif gs.path and Path(gs.path).is_file():
            src = Path(gs.path)
            kind = "raw"

        if src is None:
            return ExportResult(
                success=False,
                format=self.format_name,
                error_message="No Gaussian PLY path available on scene",
            )

        dest = output_dir / "scene.ply"
        shutil.copy2(src, dest)
        # Convenience alias
        if kind == "cleaned":
            shutil.copy2(src, output_dir / "cleaned_scene.ply")

        return ExportResult(
            success=True,
            format=self.format_name,
            path=dest,
            metrics={"source": kind, "bytes": dest.stat().st_size},
        )
