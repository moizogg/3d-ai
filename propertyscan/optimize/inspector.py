"""Post-training Scene Inspector — surgical prune of needles / floaters / huge / tiny.

Threshold spirit from Reconstruction Bible Stage 15, applied to PLY attributes
when present. If the PLY is xyz-only (e.g. mock), we copy through with honest notes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from propertyscan.domain.gaussian import GaussianScene, GaussianStats
from propertyscan.domain.quality import InspectionReport
from propertyscan.optimize.ply_io import PlyData, col_or, read_ply, write_ply

logger = logging.getLogger("propertyscan.optimize.inspector")


@dataclass(frozen=True)
class InspectorThresholds:
    """Configurable prune thresholds."""

    min_opacity: float = 0.05
    needle_aspect: float = 20.0
    needle_max_scale: float = 1.5
    huge_scale: float = 5.0
    tiny_scale: float = 1e-4


@dataclass
class InspectResult:
    report: InspectionReport
    cleaned_scene: GaussianScene
    cleaned_path: Path | None


class SceneInspector:
    """Inspect and clean a trained Gaussian PLY.

    Purpose:
        Remove obvious artifacts before export without re-training.

    Non-responsibilities:
        Pose repair, re-training, format conversion beyond PLY clean copy.
    """

    def __init__(self, thresholds: InspectorThresholds | None = None) -> None:
        self.thresholds = thresholds or InspectorThresholds()

    def inspect(
        self,
        scene: GaussianScene,
        *,
        output_dir: Path,
    ) -> InspectResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        src = scene.path
        if src is None or not Path(src).is_file():
            report = InspectionReport(
                notes=["no_ply_to_inspect"],
                details={"reason": "GaussianScene.path missing"},
            )
            return InspectResult(report=report, cleaned_scene=scene, cleaned_path=None)

        try:
            ply = read_ply(Path(src))
        except Exception as exc:
            report = InspectionReport(
                notes=[f"ply_read_failed: {exc}"],
                details={"path": str(src)},
            )
            return InspectResult(report=report, cleaned_scene=scene, cleaned_path=None)

        if ply.is_binary:
            # Honest: do not silently invent scales/opacity for binary without parser
            dest = output_dir / "cleaned_scene.ply"
            write_ply(dest, ply)
            n = _guess_binary_vertex_count(ply)
            report = InspectionReport(
                total_gaussians_before=n,
                total_gaussians_after=n,
                notes=["binary_ply_passthrough", "no_surgical_prune_without_ascii_attrs"],
                details={"path": str(src)},
            )
            cleaned = scene.model_copy(
                update={
                    "cleaned_path": dest,
                    "stats": scene.stats.model_copy(update={"count": n or scene.stats.count}),
                }
            )
            return InspectResult(report=report, cleaned_scene=cleaned, cleaned_path=dest)

        before = ply.count
        keep_mask, counts = self._classify(ply)
        kept = [v for v, k in zip(ply.vertices, keep_mask) if k]
        after = len(kept)

        cleaned_ply = PlyData(
            comments=list(ply.comments) + ["propertyscan_inspector_v1"],
            properties=list(ply.properties),
            property_types=list(ply.property_types),
            vertices=kept,
        )
        dest = output_dir / "cleaned_scene.ply"
        write_ply(dest, cleaned_ply)

        reduction = 0.0 if before == 0 else (1.0 - after / before) * 100.0
        report = InspectionReport(
            total_gaussians_before=before,
            total_gaussians_after=after,
            needles_removed=counts["needles"],
            floaters_removed=counts["floaters"],
            huge_removed=counts["huge"],
            tiny_removed=counts["tiny"],
            size_reduction_pct=round(reduction, 2),
            notes=counts["notes"],
            details={"thresholds": self.thresholds.__dict__},
        )

        stats = self._stats_from_ply(cleaned_ply)
        cleaned = scene.model_copy(
            update={
                "cleaned_path": dest,
                "stats": stats,
                "metadata": {
                    **scene.metadata,
                    "inspection": report.to_dict(),
                },
            }
        )
        logger.info(
            "Inspector: %d→%d (needles=%d floaters=%d huge=%d tiny=%d)",
            before,
            after,
            report.needles_removed,
            report.floaters_removed,
            report.huge_removed,
            report.tiny_removed,
        )
        return InspectResult(report=report, cleaned_scene=cleaned, cleaned_path=dest)

    def _classify(self, ply: PlyData) -> tuple[list[bool], dict]:
        thr = self.thresholds
        n = ply.count
        notes: list[str] = []

        opacity = col_or(ply, ["opacity", "alpha", "o"], default=1.0)
        # scales: scale_0..2 or sx,sy,sz
        s0 = col_or(ply, ["scale_0", "sx", "scale_x"], default=-1.0)
        s1 = col_or(ply, ["scale_1", "sy", "scale_y"], default=-1.0)
        s2 = col_or(ply, ["scale_2", "sz", "scale_z"], default=-1.0)
        has_scale = any(v >= 0 for v in s0) and any(
            name in ply.properties for name in ("scale_0", "sx", "scale_x")
        )
        has_opacity = any(name in ply.properties for name in ("opacity", "alpha", "o"))

        if not has_scale and not has_opacity:
            notes.append("xyz_only_no_prune")
            return [True] * n, {
                "needles": 0,
                "floaters": 0,
                "huge": 0,
                "tiny": 0,
                "notes": notes,
            }

        # Splatfacto often stores log-scales
        def exp_scale(v: float) -> float:
            if v < 0:
                return 0.0
            # if already looks like metric scale small, use as-is; else exp
            if v > 2.0:  # likely log-space
                return math.exp(min(v, 20.0))
            return max(v, 0.0)

        keep = [True] * n
        needles = floaters = huge = tiny = 0

        for i in range(n):
            drop = False
            if has_opacity:
                # opacity may be logit; map loosely
                op = opacity[i]
                if op < 0:
                    op = 1.0 / (1.0 + math.exp(-op))  # sigmoid
                if op < thr.min_opacity:
                    floaters += 1
                    drop = True

            if has_scale and not drop:
                a = exp_scale(s0[i])
                b = exp_scale(s1[i])
                c = exp_scale(s2[i])
                scales = sorted([a, b, c])
                mx, mid, mn = scales[2], scales[1], scales[0]
                aspect = mx / max(mn, 1e-8)
                if mx > thr.huge_scale:
                    huge += 1
                    drop = True
                elif aspect > thr.needle_aspect and mx > thr.needle_max_scale:
                    needles += 1
                    drop = True
                elif mx < thr.tiny_scale:
                    tiny += 1
                    drop = True

            keep[i] = not drop

        if not has_opacity:
            notes.append("no_opacity_attr")
        if not has_scale:
            notes.append("no_scale_attr")
        notes.append("surgical_prune_applied")

        return keep, {
            "needles": needles,
            "floaters": floaters,
            "huge": huge,
            "tiny": tiny,
            "notes": notes,
        }

    def _stats_from_ply(self, ply: PlyData) -> GaussianStats:
        if ply.count == 0:
            return GaussianStats(count=0)
        xs = col_or(ply, ["x"], 0.0)
        ys = col_or(ply, ["y"], 0.0)
        zs = col_or(ply, ["z"], 0.0)
        return GaussianStats(
            count=ply.count,
            bounding_box_min=[min(xs), min(ys), min(zs)],
            bounding_box_max=[max(xs), max(ys), max(zs)],
        )


def _guess_binary_vertex_count(ply: PlyData) -> int:
    if not ply.raw_bytes:
        return 0
    header = ply.raw_bytes.split(b"end_header", 1)[0].decode("ascii", errors="replace")
    for line in header.splitlines():
        if line.startswith("element vertex"):
            try:
                return int(line.split()[-1])
            except ValueError:
                return 0
    return 0
