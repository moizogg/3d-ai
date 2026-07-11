"""Geometry-dominated quality engine (Reconstruction Bible Stage 17 spirit)."""

from __future__ import annotations

from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, SceneDescriptor
from propertyscan.domain.quality import HealthReport, InspectionReport, QualityReport


def score_scene(
    *,
    geometry: GeometryResult | None,
    health: HealthReport | None,
    inspection: InspectionReport | None,
    frame_set: FrameSet | None = None,
    descriptor: SceneDescriptor | None = None,
) -> QualityReport:
    """Compute overall quality 0–100 with camera/geometry dominant.

    Weights (approx):
      camera 35%, geometry 25%, artifacts 20%, completeness 15%, health 5%
    Photometric metrics are intentionally omitted (secondary; not available yet).
    """
    camera = 0.0
    geometry_s = 0.0
    artifacts = 100.0
    completeness = 50.0
    reasons: list[str] = []
    recs: list[str] = []
    failure_class: str | None = None

    if geometry is not None and geometry.success and geometry.pose_graph:
        reg = geometry.metrics.registered_fraction
        camera = min(100.0, reg * 100.0)
        if geometry.metrics.mean_camera_confidence is not None:
            camera = 0.7 * camera + 0.3 * (
                geometry.metrics.mean_camera_confidence * 100.0
            )
        geometry_s = camera
        if geometry.metrics.point_count > 0:
            geometry_s = min(100.0, geometry_s + 5.0)
        if reg < 0.7:
            reasons.append(f"low registration {reg:.0%}")
            failure_class = "camera_failure"
            recs.append("Re-capture with slower motion and more overlap.")
    else:
        camera = 0.0
        geometry_s = 0.0
        reasons.append("geometry missing or failed")
        failure_class = "geometry_failure"
        recs.append("Re-run foundation geometry before trusting this export.")

    if health is not None:
        if not health.passed:
            reasons.append(f"pre-train health failed ({health.score})")
            failure_class = failure_class or "geometry_failure"
        # blend health into camera/geometry slightly
        camera = 0.85 * camera + 0.15 * health.score
        geometry_s = 0.85 * geometry_s + 0.15 * health.score

    if inspection is not None:
        before = max(inspection.total_gaussians_before, 1)
        bad = (
            inspection.needles_removed
            + inspection.floaters_removed
            + inspection.huge_removed
        )
        # Penalty for high artifact density, but cleaning already removed them
        density = bad / before
        artifacts = max(0.0, 100.0 - density * 200.0)
        if density > 0.25:
            reasons.append(f"high artifact density before clean ({density:.0%})")
            failure_class = failure_class or "optimization_failure"
            recs.append("Prefer better poses/depth; reduce densification next run.")
        if inspection.total_gaussians_after == 0 and inspection.total_gaussians_before > 0:
            artifacts = 0.0
            reasons.append("all gaussians pruned")
            failure_class = "optimization_failure"

    if frame_set is not None:
        # Completeness proxy: accepted keyframes volume
        n = frame_set.accepted_count
        completeness = min(100.0, 40.0 + n * 2.0)
        if n < 8:
            reasons.append(f"few keyframes ({n})")
            recs.append("Capture longer walkthrough for room coverage.")

    if descriptor is not None and descriptor.texture_score < 8:
        reasons.append("low-texture interior")
        recs.append("MASt3R + depth priors recommended for white walls.")

    overall = (
        0.35 * camera
        + 0.25 * geometry_s
        + 0.20 * artifacts
        + 0.15 * completeness
        + 0.05 * (health.score if health else 50.0)
    )
    overall = max(0.0, min(100.0, overall))

    if overall >= 85:
        status = "excellent"
    elif overall >= 70:
        status = "accepted"
    elif overall >= 50:
        status = "marginal"
    else:
        status = "reject"

    if status == "reject" and failure_class is None:
        failure_class = "capture_failure"
        recs.append("Consider a new walkthrough capture.")

    diagnosis = (
        f"Overall {overall:.0f}/100 ({status}). "
        + ("; ".join(reasons) if reasons else "No major issues flagged.")
    )

    return QualityReport(
        overall=round(overall, 2),
        camera=round(camera, 2),
        geometry=round(geometry_s, 2),
        artifacts=round(artifacts, 2),
        completeness=round(completeness, 2),
        photometric=None,
        status=status,
        diagnosis=diagnosis,
        failure_class=failure_class,
        recommendations=recs or ["Scene suitable for review/export."],
        details={
            "reasons": reasons,
            "weights": {
                "camera": 0.35,
                "geometry": 0.25,
                "artifacts": 0.20,
                "completeness": 0.15,
                "health": 0.05,
            },
        },
    )
