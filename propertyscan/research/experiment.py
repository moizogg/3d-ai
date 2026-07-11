"""Experiment registry — append-only JSONL for reproducibility."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """One immutable experiment entry."""

    experiment_id: str
    created_at: str
    scene_id: str
    job_id: str
    profile: str
    geometry_engine: str
    train_backend: str
    quality_preset: str
    metrics: dict[str, Any] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    output_dir: str = ""
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        scene_id: str,
        job_id: str,
        profile: str,
        geometry_engine: str,
        train_backend: str,
        quality_preset: str,
        metrics: dict[str, Any] | None = None,
        models: dict[str, str] | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        output_dir: str = "",
    ) -> ExperimentRecord:
        return cls(
            experiment_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            scene_id=scene_id,
            job_id=job_id,
            profile=profile,
            geometry_engine=geometry_engine,
            train_backend=train_backend,
            quality_preset=quality_preset,
            metrics=metrics or {},
            models=models or {},
            tags=tags or [],
            notes=notes,
            output_dir=output_dir,
            git_commit=_try_git_commit(),
        )


class ExperimentRegistry:
    """Append-only experiment log (history.jsonl).

    Purpose:
        Long-term regression tracking without a database.

    Layout:
        <registry_dir>/history.jsonl
        <registry_dir>/experiments/<experiment_id>.json  (optional full dump)
    """

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.registry_dir / "history.jsonl"
        self.experiments_dir = self.registry_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExperimentRecord, *, write_full: bool = True) -> Path:
        line = json.dumps(record.to_dict(), default=str)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if write_full:
            path = self.experiments_dir / f"{record.experiment_id}.json"
            path.write_text(
                json.dumps(record.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            return path
        return self.history_path

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.history_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.history_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:]

    def filter_by_scene(self, scene_id: str) -> list[dict[str, Any]]:
        return [r for r in self.list_recent(limit=10_000) if r.get("scene_id") == scene_id]


def _try_git_commit() -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None
