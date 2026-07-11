"""Geometry + depth fusion (Phase 3 contract; richer math in later phases)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from propertyscan.domain.depth import DepthResult
from propertyscan.domain.geometry import ConfidenceMap, GeometryResult


@dataclass
class FusedGeometry:
    """Product of fusing foundation geometry with monocular depth.

    Phase 3:
        Attaches depth artifacts and adjusts confidence — does not re-solve BA.

    Future:
        Scale alignment, depth-guided densify, wall regularization.
    """

    geometry: GeometryResult
    depth: DepthResult | None
    confidence: ConfidenceMap
    scale_aligned: bool = False
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry.to_dict(),
            "depth": self.depth.to_dict() if self.depth else None,
            "confidence": self.confidence.to_dict(),
            "scale_aligned": self.scale_aligned,
            "notes": self.notes,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


def fuse_geometry_and_depth(
    geometry: GeometryResult,
    depth: DepthResult | None,
    *,
    output_dir: Path | None = None,
) -> FusedGeometry:
    """Fuse stereo/foundation geometry with monocular depth.

    Rules (Phase 3):
      - Geometry poses are primary (never overwritten by monocular depth).
      - Successful depth attaches paths and boosts confidence slightly.
      - Failed/missing depth is allowed; notes record it (honest).
      - Does not invent metric scale for monocular depth.
    """
    notes: list[str] = []
    conf = geometry.confidence or ConfidenceMap(global_score=0.0)
    artifacts: dict[str, str] = dict(geometry.artifacts)
    meta: dict[str, Any] = {"fusion_version": "phase3_attach"}

    if not geometry.success:
        notes.append("geometry_failed_skip_deep_fusion")
        return FusedGeometry(
            geometry=geometry,
            depth=depth,
            confidence=conf,
            notes=notes,
            artifacts=artifacts,
            metadata=meta,
        )

    global_score = conf.global_score
    if depth is not None and depth.success and depth.depth_maps:
        notes.append("depth_attached_relative_scale")
        artifacts["depth_dir"] = depth.artifacts.get(
            "depth_dir",
            str(depth.depth_maps[0].path.parent) if depth.depth_maps else "",
        )
        # Mild confidence boost when dense depth available (walls / low texture)
        global_score = min(1.0, max(global_score, 0.0) + 0.05)
        meta["depth_maps"] = len(depth.depth_maps)
        meta["depth_provider"] = depth.provider_name
        meta["depth_scale"] = depth.scale_hint
    elif depth is not None and not depth.success:
        notes.append(f"depth_unavailable: {depth.error_message}")
        meta["depth_error"] = depth.error_message
    else:
        notes.append("depth_skipped")

    fused_conf = ConfidenceMap(
        global_score=round(global_score, 4),
        per_camera=dict(conf.per_camera),
        per_point_summary=dict(conf.per_point_summary),
        notes=list(conf.notes) + notes,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        import json

        report = out / "fusion_report.json"
        fg = FusedGeometry(
            geometry=geometry,
            depth=depth,
            confidence=fused_conf,
            scale_aligned=False,
            notes=notes,
            artifacts=artifacts,
            metadata=meta,
        )
        report.write_text(json.dumps(fg.to_dict(), indent=2, default=str), encoding="utf-8")
        artifacts["fusion_report"] = str(report)

    return FusedGeometry(
        geometry=geometry,
        depth=depth,
        confidence=fused_conf,
        scale_aligned=False,
        notes=notes,
        artifacts=artifacts,
        metadata=meta,
    )
