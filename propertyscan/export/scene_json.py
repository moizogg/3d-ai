"""Archive PropertyScene as JSON (native intermediate, not PLY)."""

from __future__ import annotations

import json
from pathlib import Path

from propertyscan.domain.scene import PropertyScene
from propertyscan.export.base import Exporter, ExportResult


class PropertySceneJsonExporter(Exporter):
    """Write property_scene.json for provenance / future .pscene."""

    @property
    def format_name(self) -> str:
        return "property_scene_json"

    def export(self, scene: PropertyScene, output_dir: Path) -> ExportResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "property_scene.json"
        # Avoid dumping entire huge frame lists if needed — keep full for now (research)
        path.write_text(
            json.dumps(scene.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return ExportResult(
            success=True,
            format=self.format_name,
            path=path,
            metrics={"bytes": path.stat().st_size},
        )
