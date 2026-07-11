"""Pre-training health gate for learned geometry (no COLMAP-only metrics)."""

from __future__ import annotations

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import SceneDescriptor
from propertyscan.domain.quality import HealthReport
from propertyscan.geometry.fusion.fuse import FusedGeometry
from propertyscan.geometry.validation.geometric import (
    GeometryValidationReport,
    validate_geometry,
)


def evaluate_pretrain_health(
    *,
    fused: FusedGeometry | None = None,
    validation: GeometryValidationReport | None = None,
    frame_set: FrameSet | None = None,
    descriptor: SceneDescriptor | None = None,
    config: EngineConfig,
) -> HealthReport:
    """Compute pre-training health and pass/fail gate.

    Factors (learned-geometry adapted):
      - registration fraction / validation pass
      - mean camera confidence
      - frame intelligence texture / blur signals
      - depth availability (soft)
      - needle/floater risk heuristics (pre-train estimate)
    """
    if validation is None and fused is not None:
        validation = validate_geometry(fused, config)
    if validation is None:
        return HealthReport(
            score=0.0,
            expected_quality="fail",
            passed=False,
            reasons=["no geometry validation available"],
            recommendations=["Run geometry reconstruction first."],
        )

    score = 100.0
    reasons: list[str] = []
    recs: list[str] = []

    # Registration dominates
    reg = validation.registered_fraction
    if reg < 0.5:
        score -= 50
        reasons.append(f"poor registration {reg:.0%}")
        recs.append("Recapture with more overlap; ensure keyframes have motion.")
    elif reg < config.geometry.min_registered_fraction:
        score -= 30
        reasons.append(f"registration below threshold {reg:.0%}")
    else:
        score -= max(0.0, (1.0 - reg) * 25.0)

    if not validation.passed:
        score -= 15
        reasons.extend(validation.issues[:3])

    mean_conf = validation.mean_confidence
    if mean_conf is not None:
        if mean_conf < 0.4:
            score -= 15
            reasons.append(f"low pose confidence {mean_conf:.2f}")
        elif mean_conf < 0.7:
            score -= 5

    # Frame intelligence soft signals
    if descriptor is not None:
        if descriptor.blur_ratio > 0.35:
            score -= 10
            reasons.append("high motion-smear ratio in frames")
            recs.append("Walk slower; reduce motion blur before geometry.")
        if descriptor.texture_score < 5:
            score -= 5
            reasons.append("very low texture (white walls)")
            recs.append("Prefer MASt3R; include door frames / furniture in pass.")

    depth_ok = False
    if fused is not None and fused.depth is not None and fused.depth.success:
        depth_ok = True
        score = min(100.0, score + 5)
    else:
        reasons.append("no monocular depth attached")

    # Pre-train artifact risk heuristics
    needle_p = 0.15
    floater_p = 0.15
    if reg < 0.7:
        needle_p += 0.35
        floater_p += 0.25
    if mean_conf is not None and mean_conf < 0.5:
        needle_p += 0.2
    if descriptor and descriptor.texture_score < 8:
        needle_p += 0.1
    if not depth_ok:
        floater_p += 0.1
    needle_p = min(0.95, needle_p)
    floater_p = min(0.95, floater_p)

    score = max(0.0, min(100.0, score))

    if score >= 80:
        expected = "excellent"
    elif score >= 60:
        expected = "good"
    elif score >= config.health.min_score:
        expected = "poor"
    else:
        expected = "fail"

    passed = score >= config.health.min_score and needle_p <= config.health.max_needle_probability
    if validation and not validation.passed and config.health.abort_below_min_score:
        passed = False

    if not passed:
        recs.append("Do not start Gaussian training until geometry health improves.")
    else:
        recs.append("Geometry health acceptable for training.")

    return HealthReport(
        score=round(score, 2),
        expected_quality=expected,  # type: ignore[arg-type]
        needle_probability=round(needle_p, 3),
        floater_probability=round(floater_p, 3),
        registered_fraction=reg,
        coverage_score=round(reg * 100.0, 2),
        texture_score=descriptor.texture_score if descriptor else None,
        passed=passed,
        reasons=reasons,
        recommendations=recs,
        details={
            "validation": validation.to_dict() if validation else None,
            "depth_ok": depth_ok,
            "accepted_keyframes": frame_set.accepted_count if frame_set else None,
        },
    )
