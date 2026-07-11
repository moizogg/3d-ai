"""Run provenance for research reproducibility and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ProvenanceRecord:
    """Immutable-ish log of how a reconstruction run was produced.

    Purpose:
        Record config, device, model ids, stage digests for experiment tracking.

    Future:
        Will attach git commit, dataset fingerprint, and metric summaries.
    """

    job_id: str
    created_at: str
    config_profile: str
    config_snapshot: dict[str, Any]
    device: dict[str, Any]
    stages: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        config_profile: str,
        config_snapshot: dict[str, Any],
        device: dict[str, Any],
    ) -> ProvenanceRecord:
        return cls(
            job_id=job_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            config_profile=config_profile,
            config_snapshot=config_snapshot,
            device=device,
        )

    def add_stage(self, stage_dict: dict[str, Any]) -> None:
        self.stages.append(stage_dict)

    def set_model(self, role: str, model_id: str) -> None:
        self.models[role] = model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "config_profile": self.config_profile,
            "config_snapshot": self.config_snapshot,
            "device": self.device,
            "stages": self.stages,
            "models": self.models,
            "metrics": self.metrics,
            "notes": self.notes,
        }
