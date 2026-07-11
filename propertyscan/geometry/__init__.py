"""Geometry subsystem — foundation providers, router, fusion, validation.

Primary engines: MASt3R / DUSt3R (real weights in Phase 4).
COLMAP is intentionally not included.
"""

from __future__ import annotations

from propertyscan.geometry.base import GeometryProvider
from propertyscan.geometry.router import GeometryRouter

__all__ = ["GeometryProvider", "GeometryRouter"]
