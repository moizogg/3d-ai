"""Unit tests for device helpers and exception formatting."""

from __future__ import annotations

from propertyscan.core.device import DeviceInfo, resolve_device
from propertyscan.core.exceptions import EngineError, GeometryError, StageError


def test_resolve_device_returns_device_info() -> None:
    info = resolve_device(prefer_cuda=True)
    assert isinstance(info, DeviceInfo)
    assert info.device in ("cpu", "cuda")
    d = info.to_dict()
    assert "cuda_available" in d


def test_engine_error_suggestion_in_str() -> None:
    err = GeometryError(
        "MASt3R weights failed to load",
        suggestion="Check network / HF cache and CUDA VRAM.",
    )
    text = str(err)
    assert "MASt3R weights failed to load" in text
    assert "suggestion:" in text


def test_stage_error_has_stage_name() -> None:
    err = StageError("boom", stage_name="alignment")
    assert err.stage_name == "alignment"
    assert isinstance(err, EngineError)
