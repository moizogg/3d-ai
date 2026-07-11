"""Splatfacto trainer via Nerfstudio ``ns-train`` (optional dependency)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.domain.dataset import TrainingDataset
from propertyscan.domain.gaussian import GaussianScene, GaussianStats
from propertyscan.training.base import TrainerBackend, TrainResult
from propertyscan.training.presets import resolve_train_preset

logger = logging.getLogger("propertyscan.training.splatfacto")

INSTALL_HINT = (
    "Install Nerfstudio with splatfacto support, e.g. "
    "`pip install nerfstudio` (see Nerfstudio docs for CUDA). "
    "Or set training.backend=mock for CI."
)


class SplatfactoTrainer(TrainerBackend):
    """Train 3D Gaussians with Nerfstudio splatfacto.

    Purpose:
        Consume TrainingDataset (poses already fixed); do not re-estimate cameras.

    Limitations:
        Requires ``ns-train`` on PATH. Export of final PLY may need ``ns-export``;
        if export is unavailable, success still reports the checkpoint config path.
    """

    @property
    def name(self) -> str:
        return "splatfacto"

    def is_available(self) -> bool:
        return shutil.which("ns-train") is not None

    def train(
        self,
        dataset: TrainingDataset,
        *,
        output_dir: Path,
        config: EngineConfig,
    ) -> TrainResult:
        t0 = time.perf_counter()
        if not self.is_available():
            return TrainResult(
                success=False,
                backend=self.name,
                error_message=f"ns-train not found on PATH. {INSTALL_HINT}",
                execution_time_s=time.perf_counter() - t0,
            )

        preset = resolve_train_preset(config, dataset)
        train_dir = Path(output_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        log_path = train_dir / "ns_train.log"

        cmd = [
            "ns-train",
            "splatfacto",
            "--output-dir",
            str(train_dir),
            "--max-num-iterations",
            str(preset.iterations),
            "--pipeline.model.cull-alpha-thresh",
            str(preset.cull_alpha_thresh),
            "--pipeline.model.reset-alpha-every",
            str(preset.reset_alpha_every),
            "--pipeline.model.stop-split-at",
            str(preset.stop_split_at),
            "--viewer.quit-on-train-completion",
            "True",
            "nerfstudio-data",
            "--data",
            str(dataset.root),
            "--downscale-factor",
            str(preset.downscale_factor),
        ]

        logger.info("Running: %s", " ".join(cmd))
        for note in preset.notes:
            logger.info("preset: %s", note)

        try:
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=config.training.timeout_s or None,
                )
        except subprocess.TimeoutExpired:
            return TrainResult(
                success=False,
                backend=self.name,
                train_dir=train_dir,
                iterations=preset.iterations,
                execution_time_s=time.perf_counter() - t0,
                error_message=f"ns-train timed out after {config.training.timeout_s}s",
            )
        except Exception as exc:
            return TrainResult(
                success=False,
                backend=self.name,
                train_dir=train_dir,
                iterations=preset.iterations,
                execution_time_s=time.perf_counter() - t0,
                error_message=f"ns-train failed to launch: {exc}",
            )

        if proc.returncode != 0:
            return TrainResult(
                success=False,
                backend=self.name,
                train_dir=train_dir,
                iterations=preset.iterations,
                execution_time_s=time.perf_counter() - t0,
                error_message=(
                    f"ns-train exited {proc.returncode}. See {log_path}. {INSTALL_HINT}"
                ),
                metrics={"log": str(log_path)},
            )

        config_path = _find_latest_train_config(train_dir)
        if config_path is None:
            return TrainResult(
                success=False,
                backend=self.name,
                train_dir=train_dir,
                iterations=preset.iterations,
                execution_time_s=time.perf_counter() - t0,
                error_message=(
                    f"Training finished but no checkpoint config found under {train_dir}."
                ),
                metrics={"log": str(log_path)},
            )

        ply_path = _try_export_ply(config_path, train_dir)
        elapsed = round(time.perf_counter() - t0, 3)
        scene = GaussianScene(
            path=ply_path,
            stats=GaussianStats(count=0),
            training_iterations=preset.iterations,
            training_time_s=elapsed,
            trainer_name=self.name,
            metadata={
                "train_config": str(config_path),
                "log": str(log_path),
                "preset_notes": list(preset.notes),
                "exported_ply": ply_path is not None,
            },
        )
        return TrainResult(
            success=True,
            backend=self.name,
            scene=scene,
            train_dir=train_dir,
            config_path=config_path,
            iterations=preset.iterations,
            execution_time_s=elapsed,
            metrics={
                "log": str(log_path),
                "train_config": str(config_path),
                "ply": str(ply_path) if ply_path else None,
                "notes": list(preset.notes),
            },
        )


def _find_latest_train_config(train_dir: Path) -> Path | None:
    valid: list[Path] = []
    for cfg in train_dir.rglob("config.yml"):
        models = cfg.parent / "nerfstudio_models"
        if models.is_dir() and any(models.glob("*.ckpt")):
            valid.append(cfg)
    if not valid:
        # Some versions use config.yml without ckpt yet — still accept newest config
        configs = list(train_dir.rglob("config.yml"))
        return max(configs, key=lambda p: p.stat().st_mtime) if configs else None
    return max(valid, key=lambda p: p.stat().st_mtime)


def _try_export_ply(config_path: Path, train_dir: Path) -> Path | None:
    """Best-effort PLY export; None if ns-export missing or fails."""
    if shutil.which("ns-export") is None:
        logger.info("ns-export not found; leaving checkpoint only (export in Phase 6)")
        return None
    out_ply = train_dir / "exports" / "point_cloud.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(out_ply.parent),
    ]
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)
    except Exception as exc:
        logger.warning("ns-export failed: %s", exc)
        return None
    # Nerfstudio may write nested paths
    candidates = list(out_ply.parent.rglob("*.ply"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
