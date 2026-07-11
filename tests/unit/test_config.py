"""Unit tests for configuration loading and merge order."""

from __future__ import annotations

from pathlib import Path

import pytest

from propertyscan.core.config import EngineConfig, default_configs_dir, load_config
from propertyscan.core.exceptions import ConfigurationError


def test_default_configs_dir_exists() -> None:
    root = default_configs_dir()
    assert root.is_dir()
    assert (root / "default.yaml").is_file()


def test_load_default_config() -> None:
    cfg = load_config(apply_env=False)
    assert isinstance(cfg, EngineConfig)
    assert cfg.engine.name == "propertyscan"
    assert cfg.geometry.engine == "mast3r"
    assert "MASt3R_ViTLarge" in cfg.geometry.mast3r_model
    assert "DUSt3R_ViTLarge" in cfg.geometry.dust3r_model
    assert cfg.device.allow_cpu_geometry is False


def test_colab_t4_profile_merge() -> None:
    cfg = load_config(profile="colab_t4", apply_env=False)
    assert cfg.engine.profile == "colab_t4"
    assert cfg.frame_intelligence.max_keyframes == 80
    assert cfg.depth.size == "small"
    # Still full large foundation model IDs
    assert "ViTLarge" in cfg.geometry.mast3r_model


def test_quality_gpu_profile_merge() -> None:
    cfg = load_config(profile="quality_gpu", apply_env=False)
    assert cfg.engine.profile == "quality_gpu"
    assert cfg.depth.size == "base"
    assert cfg.training.quality == "high"
    assert cfg.frame_intelligence.max_keyframes == 160


def test_quality_overlay_draft() -> None:
    cfg = load_config(quality="draft", apply_env=False)
    assert cfg.training.quality == "draft"
    assert cfg.training.resolved_iterations() == 7000


def test_config_to_dict_roundtrip_keys() -> None:
    cfg = load_config(apply_env=False)
    data = cfg.to_dict()
    assert "geometry" in data
    assert "depth" in data
    restored = EngineConfig.model_validate(data)
    assert restored.geometry.mast3r_model == cfg.geometry.mast3r_model


def test_missing_profile_raises() -> None:
    with pytest.raises(ConfigurationError):
        load_config(profile="does_not_exist_xyz", apply_env=False)
