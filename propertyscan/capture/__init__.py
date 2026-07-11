"""Capture adapters: turn user media into validated manifests and frame paths."""

from __future__ import annotations

from propertyscan.capture.base import CaptureAdapter
from propertyscan.capture.detect import detect_capture_kind, get_adapter, validate_and_load

__all__ = [
    "CaptureAdapter",
    "detect_capture_kind",
    "get_adapter",
    "validate_and_load",
]
