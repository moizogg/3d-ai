"""Frame intelligence: reliable validation (not legacy Laplacian/dHash)."""

from __future__ import annotations

from propertyscan.intelligence.classify import classify_scene
from propertyscan.intelligence.dedup import mark_duplicates
from propertyscan.intelligence.keyframes import select_keyframes
from propertyscan.intelligence.quality import analyze_frame, analyze_frames

__all__ = [
    "analyze_frame",
    "analyze_frames",
    "mark_duplicates",
    "select_keyframes",
    "classify_scene",
]
