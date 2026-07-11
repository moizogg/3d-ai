"""Health, inspection, and quality report domain types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ExpectedQuality = Literal["excellent", "good", "poor", "fail"]


class HealthReport(BaseModel):
    """Pre-training scene health and artifact risk prediction.

    Purpose:
        Gate expensive Gaussian training when alignment / coverage is unsafe.

    Philosophy:
        Bad alignment must never silently become a bad tour.
    """

    score: float = 0.0
    expected_quality: ExpectedQuality = "fail"
    needle_probability: float = 0.0
    floater_probability: float = 0.0
    registered_fraction: float = 0.0
    coverage_score: float | None = None
    texture_score: float | None = None
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class InspectionReport(BaseModel):
    """Post-training Scene Inspector results (needles / floaters / huge / tiny)."""

    total_gaussians_before: int = 0
    total_gaussians_after: int = 0
    needles_removed: int = 0
    floaters_removed: int = 0
    huge_removed: int = 0
    tiny_removed: int = 0
    size_reduction_pct: float = 0.0
    notes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class QualityReport(BaseModel):
    """Geometry-dominated final quality score and diagnosis.

    Weighting spirit (from Reconstruction Bible):
      camera / geometry dominant; photometric metrics secondary.
    """

    overall: float = 0.0
    camera: float = 0.0
    geometry: float = 0.0
    artifacts: float = 0.0
    completeness: float = 0.0
    photometric: float | None = None
    status: str = "unknown"
    diagnosis: str | None = None
    failure_class: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
