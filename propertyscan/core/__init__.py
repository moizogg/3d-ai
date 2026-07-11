"""Core runtime primitives: config, context, stages, device, logging, errors."""

from __future__ import annotations

from propertyscan.core.config import EngineConfig, load_config
from propertyscan.core.context import RunContext
from propertyscan.core.device import DeviceInfo, resolve_device
from propertyscan.core.exceptions import (
    ConfigurationError,
    EngineError,
    GeometryError,
    HealthGateError,
    StageError,
    ValidationError,
)
from propertyscan.core.stage import Stage, StageResult

__all__ = [
    "EngineConfig",
    "load_config",
    "RunContext",
    "DeviceInfo",
    "resolve_device",
    "EngineError",
    "ConfigurationError",
    "ValidationError",
    "StageError",
    "GeometryError",
    "HealthGateError",
    "Stage",
    "StageResult",
]
