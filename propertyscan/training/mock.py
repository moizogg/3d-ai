"""Mock trainer for CI / pipeline wiring without Nerfstudio."""

from __future__ import annotations

import time
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.dataset import TrainingDataset
from propertyscan.domain.gaussian import GaussianScene, GaussianStats
from propertyscan.training.base import TrainerBackend, TrainResult
from propertyscan.training.presets import resolve_train_preset


class MockTrainer(TrainerBackend):
    """Writes a minimal placeholder PLY so export/inspector stages can be tested.

    Not a visual quality backend — use splatfacto on Colab for real tours.
    """

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def train(
        self,
        dataset: TrainingDataset,
        *,
        output_dir: Path,
        config: EngineConfig,
    ) -> TrainResult:
        t0 = time.perf_counter()
        preset = resolve_train_preset(config, dataset)
        train_dir = Path(output_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        ply_path = train_dir / "scene.ply"

        # Minimal ASCII PLY (3 points) — valid enough for file-existence checks
        ply_path.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        meta = {
            "mock": True,
            "dataset_frames": dataset.frame_count,
            "preset_notes": list(preset.notes),
            "has_depth": dataset.has_depth,
            "has_init_point_cloud": dataset.has_init_point_cloud,
        }
        (train_dir / "train_meta.json").write_text(
            __import__("json").dumps(meta, indent=2), encoding="utf-8"
        )

        scene = GaussianScene(
            path=ply_path,
            stats=GaussianStats(count=3, mean_opacity=0.5, mean_scale=0.1),
            training_iterations=preset.iterations,
            training_time_s=round(time.perf_counter() - t0, 4),
            trainer_name=self.name,
            metadata=meta,
        )
        return TrainResult(
            success=True,
            backend=self.name,
            scene=scene,
            train_dir=train_dir,
            iterations=preset.iterations,
            execution_time_s=scene.training_time_s or 0.0,
            metrics={
                "gaussian_count": 3,
                "notes": list(preset.notes),
            },
        )
