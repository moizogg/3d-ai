"""Resolve training hyperparameters from config + dataset characteristics."""

from __future__ import annotations

from dataclasses import dataclass

from propertyscan.core.config import EngineConfig
from propertyscan.domain.dataset import TrainingDataset


@dataclass(frozen=True)
class TrainPreset:
    iterations: int
    downscale_factor: int
    cull_alpha_thresh: float
    reset_alpha_every: int
    stop_split_at: int
    notes: tuple[str, ...] = ()


def resolve_train_preset(config: EngineConfig, dataset: TrainingDataset) -> TrainPreset:
    """Compute conservative training knobs for quality + T4 friendliness."""
    t = config.training
    iters = t.resolved_iterations()
    notes: list[str] = [f"quality={t.quality}"]

    if t.reduce_iters_for_dense_geometry and dataset.has_init_point_cloud:
        capped = min(iters, t.dense_geometry_max_iters)
        if capped < iters:
            notes.append(
                f"dense_init: iterations {iters}→{capped} (faster convergence)"
            )
            iters = capped

    stop_split = min(iters, max(iters // 2, min(iters, 12000)))
    return TrainPreset(
        iterations=iters,
        downscale_factor=max(1, t.downscale_factor),
        cull_alpha_thresh=t.cull_alpha_thresh,
        reset_alpha_every=t.reset_alpha_every,
        stop_split_at=stop_split,
        notes=tuple(notes),
    )
