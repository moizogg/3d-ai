"""Resolve trainer backend from config."""

from __future__ import annotations

from propertyscan.core.config import EngineConfig
from propertyscan.training.base import TrainerBackend
from propertyscan.training.mock import MockTrainer
from propertyscan.training.splatfacto import SplatfactoTrainer


def get_trainer(config: EngineConfig) -> TrainerBackend:
    name = config.training.backend
    if name == "mock":
        return MockTrainer()
    if name == "splatfacto":
        return SplatfactoTrainer()
    raise ValueError(f"Unknown training.backend={name!r} (use mock|splatfacto)")
