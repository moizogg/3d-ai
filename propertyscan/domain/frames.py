"""Frame-level domain types for capture and frame intelligence stages."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FrameStatus(str, Enum):
    """Lifecycle status of a candidate frame.

    Hard rejects (truly unusable):
      UNREADABLE, CLIPPED_BLACK, CLIPPED_WHITE, MOTION_SMEAR

    Soft states (still usable for ranking / fill):
      CANDIDATE, LOW_RANK

    Redundancy (only when camera barely moved):
      REDUNDANT

    Selection:
      ACCEPTED (selected keyframe), REJECTED (not selected — not a quality failure)
    """

    ACCEPTED = "accepted"
    CANDIDATE = "candidate"
    LOW_RANK = "low_rank"
    REDUNDANT = "redundant"
    UNREADABLE = "unreadable"
    CLIPPED_BLACK = "clipped_black"
    CLIPPED_WHITE = "clipped_white"
    MOTION_SMEAR = "motion_smear"
    REJECTED = "rejected"

    # Legacy aliases kept so old tests/reports don't explode if referenced
    LOW_CONFIDENCE = "low_rank"
    BLURRY = "motion_smear"  # only used if something still maps old name
    DARK = "clipped_black"
    OVEREXPOSED = "clipped_white"
    LOW_CONTRAST = "low_rank"
    DUPLICATE = "redundant"


class FrameMetadata(BaseModel):
    """Quality metrics and status for a single frame.

    Purpose:
        Carry reliable intelligence for ranking and keyframe selection.
        Does NOT use legacy absolute Laplacian/dHash hard rejects.

    Philosophy:
        Hard-reject only truly broken frames. Soft-score everything else.
        Redundancy is about camera motion, not thumbnail hash similarity.
    """

    filename: str
    filepath: Path
    width: int = 0
    height: int = 0
    index: int = 0

    # Raw measurements
    sharpness_raw: float = 0.0  # Tenengrad
    texture_score: float = 0.0  # edge density 0–100
    brightness: float = 0.0
    contrast: float = 0.0
    clip_low_pct: float = 0.0
    clip_high_pct: float = 0.0

    # Clip-relative scores (0–100)
    sharpness_percentile: float = 50.0
    quality_score: float = 50.0  # soft overall quality
    rank_score: float = 50.0  # used for keyframe greed

    # Motion / redundancy
    motion_from_prev: float | None = None  # mean optical flow or feature motion
    is_stationary: bool = False

    # Compat fields (legacy names still serialized for reports)
    blur_score: float = 0.0  # mirrors sharpness_raw for older report consumers
    confidence_score: float = 50.0  # mirrors quality_score
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    dhash_hex: str | None = None
    noise_score: float | None = None

    status: FrameStatus = FrameStatus.CANDIDATE
    reject_reason: str | None = None
    notes: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("filepath", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            legacy = {
                "low_confidence": "low_rank",
                "blurry": "motion_smear",
                "dark": "clipped_black",
                "overexposed": "clipped_white",
                "low_contrast": "low_rank",
                "duplicate": "redundant",
            }
            return legacy.get(v, v)
        return v

    def is_hard_rejected(self) -> bool:
        return self.status in {
            FrameStatus.UNREADABLE,
            FrameStatus.CLIPPED_BLACK,
            FrameStatus.CLIPPED_WHITE,
            FrameStatus.MOTION_SMEAR,
        }

    def is_selectable(self) -> bool:
        """Frames eligible for keyframe selection."""
        return self.status in {
            FrameStatus.CANDIDATE,
            FrameStatus.LOW_RANK,
            FrameStatus.ACCEPTED,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["filepath"] = str(self.filepath)
        data["status"] = self.status.value
        return data


class FrameSet(BaseModel):
    """Collection of frames after intelligence filtering / keyframe selection."""

    source_type: str
    source_path: Path
    frames: list[FrameMetadata] = Field(default_factory=list)
    accepted_count: int = 0
    rejected_count: int = 0
    low_confidence_count: int = 0
    rejection_stats: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    validation_mode: str = "reliable_v2"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("source_path", mode="before")
    @classmethod
    def _pathify(cls, v: Any) -> Path:
        return Path(v)

    def get_accepted(self) -> list[FrameMetadata]:
        return [f for f in self.frames if f.status == FrameStatus.ACCEPTED]

    def get_accepted_paths(self) -> list[Path]:
        return [f.filepath for f in self.get_accepted()]

    def get_low_confidence(self) -> list[FrameMetadata]:
        return [f for f in self.frames if f.status == FrameStatus.LOW_RANK]

    def recompute_counts(self) -> None:
        self.accepted_count = len(self.get_accepted())
        self.low_confidence_count = len(self.get_low_confidence())
        rejected = [
            f
            for f in self.frames
            if f.status not in (FrameStatus.ACCEPTED, FrameStatus.CANDIDATE)
        ]
        # For reporting: rejected = everything not accepted as keyframe
        not_accepted = [f for f in self.frames if f.status != FrameStatus.ACCEPTED]
        self.rejected_count = len(not_accepted)
        stats: dict[str, int] = {}
        for frame in not_accepted:
            key = frame.status.value
            stats[key] = stats.get(key, 0) + 1
        self.rejection_stats = stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_path": str(self.source_path),
            "validation_mode": self.validation_mode,
            "frames": [f.to_dict() for f in self.frames],
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "low_confidence_count": self.low_confidence_count,
            "rejection_stats": self.rejection_stats,
            "notes": self.notes,
        }
