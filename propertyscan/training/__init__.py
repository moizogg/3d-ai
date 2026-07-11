"""Gaussian training backends."""

from __future__ import annotations

from propertyscan.training.base import TrainerBackend, TrainResult
from propertyscan.training.factory import get_trainer

__all__ = ["TrainerBackend", "TrainResult", "get_trainer"]
