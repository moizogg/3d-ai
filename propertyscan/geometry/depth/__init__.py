"""Depth providers (first-class)."""

from __future__ import annotations

from propertyscan.geometry.depth.base import DepthProvider
from propertyscan.geometry.depth.anything_v2 import DepthAnythingV2Provider
from propertyscan.geometry.depth.mock import MockDepthProvider

__all__ = ["DepthProvider", "DepthAnythingV2Provider", "MockDepthProvider"]
