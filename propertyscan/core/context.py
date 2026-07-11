"""RunContext: shared state bag for a single reconstruction job.

Prefer typed getters/setters for domain objects over unstructured dict sprawl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from propertyscan.core.config import EngineConfig
from propertyscan.core.device import DeviceInfo, resolve_device
from propertyscan.core.stage import StageResult
from propertyscan.domain.provenance import ProvenanceRecord

T = TypeVar("T")


@dataclass
class RunContext:
    """Execution context for one pipeline run.

    Purpose:
        Carry configuration, filesystem roots, device info, typed domain state,
        stage history, and provenance across isolated stages.

    Inputs:
        Created by the CLI / orchestrator at job start.

    Outputs:
        Mutated in-place as stages complete; can flush provenance JSON.

    Non-responsibilities:
        Does not execute stages or load neural models.
    """

    config: EngineConfig
    input_path: Path
    output_dir: Path
    work_dir: Path
    job_id: str
    device: DeviceInfo = field(default_factory=lambda: resolve_device())
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _state: dict[str, Any] = field(default_factory=dict)
    stage_history: list[StageResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceRecord | None = None

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        self.work_dir = Path(self.work_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "logs").mkdir(parents=True, exist_ok=True)
        if self.provenance is None:
            self.provenance = ProvenanceRecord.create(
                job_id=self.job_id,
                config_profile=self.config.engine.profile,
                config_snapshot=self.config.to_dict(),
                device=self.device.to_dict(),
            )
        self.metadata.setdefault("job", {})
        self.metadata["job"].update(
            {
                "id": self.job_id,
                "started_at": self.started_at,
                "profile": self.config.engine.profile,
            }
        )

    # --- typed state helpers -------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store a domain object under a stable key."""
        self._state[key] = value

    def get(self, key: str, default: T | None = None) -> Any | T | None:
        """Retrieve a domain object by key."""
        return self._state.get(key, default)

    def require(self, key: str) -> Any:
        """Retrieve a domain object or raise KeyError with a clear message."""
        if key not in self._state:
            raise KeyError(
                f"RunContext is missing required state '{key}'. "
                f"Available keys: {sorted(self._state)}"
            )
        return self._state[key]

    def has(self, key: str) -> bool:
        return key in self._state

    # --- artifacts & history -------------------------------------------------

    def artifact_dir(self, *parts: str) -> Path:
        """Return (and create) a subdirectory under work_dir/artifacts."""
        path = self.work_dir / "artifacts"
        for part in parts:
            path = path / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_stage(self, result: StageResult) -> None:
        """Append a stage result and mirror into provenance."""
        self.stage_history.append(result)
        self.metadata.setdefault("stages", {})[result.stage_name] = result.to_dict()
        if self.provenance is not None:
            self.provenance.add_stage(result.to_dict())

    def write_metadata(self) -> Path:
        """Write job metadata JSON into the output directory."""
        path = self.output_dir / "metadata.json"
        finished = datetime.now(timezone.utc).isoformat()
        self.metadata["job"]["finished_at"] = finished
        path.write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
        return path

    def write_provenance(self) -> Path | None:
        """Write provenance JSON when enabled in config."""
        if not self.config.export.write_provenance or self.provenance is None:
            return None
        path = self.output_dir / "provenance.json"
        path.write_text(
            json.dumps(self.provenance.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return path
