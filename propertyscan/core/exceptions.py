"""Typed exceptions for PropertyScan engine failures.

Every recoverable failure should answer: what failed, why, and what to try next.
"""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """Base error for all PropertyScan engine failures.

    Attributes:
        message: Human-readable description of what failed.
        suggestion: Actionable next step for the operator.
        details: Optional structured context for logs / provenance.
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message} | suggestion: {self.suggestion}"
        return self.message


class ConfigurationError(EngineError):
    """Invalid or incomplete engine configuration."""


class ValidationError(EngineError):
    """Input or intermediate data failed validation."""


class StageError(EngineError):
    """A pipeline stage failed.

    Attributes:
        stage_name: Name of the failing stage.
    """

    def __init__(
        self,
        message: str,
        *,
        stage_name: str,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, suggestion=suggestion, details=details)
        self.stage_name = stage_name


class GeometryError(EngineError):
    """Foundation geometry reconstruction failed (poses / pointmaps / alignment)."""


class HealthGateError(EngineError):
    """Pre-training health gate rejected the scene (abort training on purpose)."""


class ExportError(EngineError):
    """Export of PropertyScene artifacts failed."""
