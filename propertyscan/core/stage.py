"""Pipeline stage contract.

Every stage:
  validate input → process → metrics → logs → artifacts → return StageResult
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from propertyscan.core.exceptions import EngineError, StageError
from propertyscan.core.logging import get_logger
from propertyscan.core.progress import ProgressHeartbeat

if TYPE_CHECKING:
    from propertyscan.core.context import RunContext


@dataclass
class StageResult:
    """Outcome of a single stage execution.

    Purpose:
        Uniform return type so the orchestrator can record provenance and metrics.

    Fields:
        stage_name: Stage identifier.
        success: Whether the stage completed without error.
        duration_s: Wall-clock seconds.
        metrics: Numeric / categorical metrics for research and debugging.
        artifacts: Named paths written by this stage.
        message: Optional human summary.
        error: Optional error message when success is False.
    """

    stage_name: str
    success: bool
    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "message": self.message,
            "error": self.error,
        }


class Stage(ABC):
    """Abstract base class for pipeline stages.

    Purpose:
        Enforce a single-responsibility processing unit with consistent
        logging, timing, and error wrapping.

    Inputs:
        RunContext carrying config, paths, and typed domain state.

    Outputs:
        StageResult plus any domain objects stored on RunContext.

    Non-responsibilities:
        Must not call other stages directly; must not own global process state.
    """

    name: str = "unnamed_stage"

    def __init__(self) -> None:
        self.logger = get_logger(f"stage.{self.name}")

    def run(self, ctx: RunContext) -> StageResult:
        """Execute the stage with timing, 10s progress heartbeats, and error wrapping."""
        self.logger.info("stage started: %s", self.name)
        started = time.perf_counter()
        # Heartbeat every 10s so Colab/logs show the stage is alive (not stuck).
        with ProgressHeartbeat(f"stage:{self.name}", interval_s=10.0) as hb:
            try:
                self.validate(ctx)
                hb.set_status("executing")
                # Allow execute() to update detail via ctx if desired
                ctx.set("_progress_heartbeat", hb)
                result = self.execute(ctx)
                if result.duration_s <= 0:
                    result.duration_s = round(time.perf_counter() - started, 3)
                result.stage_name = self.name
                ctx.record_stage(result)
                self.logger.info(
                    "stage finished: %s (%.2fs, success=%s)",
                    self.name,
                    result.duration_s,
                    result.success,
                )
                return result
            except StageError:
                raise
            except EngineError:
                duration = round(time.perf_counter() - started, 3)
                failure = StageResult(
                    stage_name=self.name,
                    success=False,
                    duration_s=duration,
                    error=str(sys.exc_info()[1]),
                )
                ctx.record_stage(failure)
                raise
            except Exception as exc:
                duration = round(time.perf_counter() - started, 3)
                failure = StageResult(
                    stage_name=self.name,
                    success=False,
                    duration_s=duration,
                    error=str(exc),
                )
                ctx.record_stage(failure)
                raise StageError(
                    f"Stage '{self.name}' failed: {exc}",
                    stage_name=self.name,
                    suggestion="Inspect logs and stage inputs; see exception details.",
                    details={"error": str(exc)},
                ) from exc
            finally:
                if ctx.has("_progress_heartbeat"):
                    try:
                        del ctx._state["_progress_heartbeat"]  # type: ignore[attr-defined]
                    except Exception:
                        pass

    def validate(self, ctx: RunContext) -> None:
        """Optional pre-condition checks. Override in subclasses."""

    @abstractmethod
    def execute(self, ctx: RunContext) -> StageResult:
        """Perform stage work and return a StageResult."""
