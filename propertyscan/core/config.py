"""Configuration system: YAML profiles + environment overrides + Pydantic models.

Design:
  - ``configs/default.yaml`` is the base.
  - Optional profile files (``colab_t4.yaml``, ``quality_gpu.yaml``) deep-merge on top.
  - Optional quality presets under ``configs/quality/``.
  - Environment variables with prefix ``PSCAN_`` can override a subset of fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from propertyscan.core.exceptions import ConfigurationError

GeometryEngine = Literal["mast3r", "dust3r", "auto", "mock", "arkit"]
DepthSize = Literal["small", "base", "large"]
TrainingQuality = Literal["draft", "standard", "high"]
LogFormat = Literal["console", "json"]
PairGraph = Literal["complete", "swin-5", "swin-3"]


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base (overlay wins)."""
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(
            f"Config file not found: {path}",
            suggestion="Pass an existing YAML path or use a known profile name.",
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Config root must be a mapping: {path}",
            suggestion="Ensure the YAML file starts with key/value pairs, not a list.",
        )
    return data


def default_configs_dir() -> Path:
    """Return the packaged configs directory (revamped_code/configs)."""
    # propertyscan/core/config.py → parents[2] = revamped_code/
    return Path(__file__).resolve().parents[2] / "configs"


class EngineSection(BaseModel):
    name: str = "propertyscan"
    version: str = "0.1.0"
    profile: str = "default"


class RunSection(BaseModel):
    experiment_id: str | None = None
    seed: int = 42
    abort_on_health_fail: bool = True


class DeviceSection(BaseModel):
    prefer_cuda: bool = True
    allow_cpu_geometry: bool = False
    empty_cache_between_stages: bool = True


class CaptureSection(BaseModel):
    video_fps: float = 2.0
    max_candidate_frames: int = 300
    min_frames: int = 8
    min_resolution: int = 480
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )
    video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    )


class FrameIntelligenceSection(BaseModel):
    """Reliable frame validation settings (NOT legacy Laplacian/dHash hard-reject).

    Philosophy:
      - Hard-reject only truly broken frames (clipping, unreadable, motion smear).
      - Soft-rank everything else with clip-relative sharpness + texture awareness.
      - Redundancy uses camera motion, not aggressive perceptual hashes.
    """

    # Selection budget
    max_keyframes: int = 120
    # Soft rank: below this percentile-quality → LOW_RANK (still selectable)
    low_rank_threshold: float = 25.0

    # Hard clip rejects (fraction of pixels near black/white)
    clip_black_pct: float = 0.92
    clip_white_pct: float = 0.92
    clip_pixel_low: float = 8.0
    clip_pixel_high: float = 247.0

    # Texture-aware sharpness
    # Low-texture frames (white walls) must NOT be labeled blurry via absolute laplacian.
    low_texture_edge_density: float = 0.04  # fraction of strong edge pixels
    # Motion smear: only when relative sharpness is poor AND texture is present
    motion_smear_sharpness_percentile: float = 8.0
    motion_smear_min_texture: float = 0.06
    # Also require elevated motion if flow available
    motion_smear_min_flow: float = 2.5

    # Redundancy / motion (optical flow or feature motion)
    # Frame is REDUNDANT only if motion from last kept is below this.
    min_motion_to_keep: float = 0.8
    # Analysis size for flow (speed)
    motion_max_side: int = 320
    # Optional: minimum index gap when motion unavailable (fallback)
    min_index_gap_fallback: int = 1

    # Legacy fields retained for YAML compatibility (IGNORED by reliable validator)
    blur_threshold: float = 6.0
    phash_threshold: int = 2
    min_brightness: float = 15.0
    max_brightness: float = 245.0
    min_contrast: float = 8.0
    low_confidence_threshold: float = 40.0


class GeometrySection(BaseModel):
    """Foundation geometry settings.

    Defaults pin official NAVER ViT-Large checkpoints — not stubs.
    """

    engine: GeometryEngine = "mast3r"
    prefer_quality: bool = True
    mast3r_model: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    dust3r_model: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    batch_size: int = 1
    pair_graph: PairGraph = "swin-5"
    global_align_iters: int = 300
    min_registered_fraction: float = 0.85

    @field_validator("batch_size")
    @classmethod
    def _batch_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("geometry.batch_size must be >= 1")
        return v


class DepthSection(BaseModel):
    enabled: bool = True
    provider: str = "depth_anything_v2"
    size: DepthSize = "small"
    max_resolution: int = 1080


class HealthSection(BaseModel):
    min_score: float = 40.0
    max_needle_probability: float = 0.75
    abort_below_min_score: bool = True


TrainerBackendName = Literal["mock", "splatfacto"]


class TrainingSection(BaseModel):
    """Gaussian training settings.

    ``backend=mock`` writes a synthetic scene for CI.
    ``backend=splatfacto`` shells out to Nerfstudio ``ns-train`` when installed.
    """

    quality: TrainingQuality = "standard"
    backend: TrainerBackendName = "splatfacto"
    iterations: dict[str, int] = Field(
        default_factory=lambda: {"draft": 7000, "standard": 15000, "high": 30000}
    )
    downscale_factor: int = 4
    cull_alpha_thresh: float = 0.01
    reset_alpha_every: int = 30
    # Dense foundation init often needs fewer splat iterations
    reduce_iters_for_dense_geometry: bool = True
    dense_geometry_max_iters: int = 8000
    require_health_pass: bool = True
    # ns-train timeout (seconds); 0 = no timeout
    timeout_s: int = 0

    def resolved_iterations(self) -> int:
        return int(self.iterations.get(self.quality, 15000))


class ExportSection(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["ply"])
    write_provenance: bool = True


class LoggingSection(BaseModel):
    level: str = "INFO"
    format: LogFormat = "console"


class EngineConfig(BaseModel):
    """Canonical engine configuration for a single reconstruction run.

    Purpose:
        Single typed object consumed by all stages.

    Inputs:
        Built via ``load_config`` from YAML profiles + optional env overrides.

    Outputs:
        Validated nested settings for capture, geometry, depth, health, training, export.

    Non-responsibilities:
        Does not load neural weights or touch the filesystem beyond config paths.
    """

    engine: EngineSection = Field(default_factory=EngineSection)
    run: RunSection = Field(default_factory=RunSection)
    device: DeviceSection = Field(default_factory=DeviceSection)
    capture: CaptureSection = Field(default_factory=CaptureSection)
    frame_intelligence: FrameIntelligenceSection = Field(
        default_factory=FrameIntelligenceSection
    )
    geometry: GeometrySection = Field(default_factory=GeometrySection)
    depth: DepthSection = Field(default_factory=DepthSection)
    health: HealthSection = Field(default_factory=HealthSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    export: ExportSection = Field(default_factory=ExportSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EnvOverrides(BaseSettings):
    """Optional environment overrides (prefix PSCAN_)."""

    model_config = SettingsConfigDict(env_prefix="PSCAN_", extra="ignore")

    geometry_engine: GeometryEngine | None = None
    profile: str | None = None
    log_level: str | None = None
    prefer_cuda: bool | None = None
    allow_cpu_geometry: bool | None = None


def load_config(
    *,
    profile: str | None = None,
    quality: TrainingQuality | None = None,
    configs_dir: Path | None = None,
    extra_yaml: Path | None = None,
    apply_env: bool = True,
) -> EngineConfig:
    """Load and merge configuration into a validated ``EngineConfig``.

    Merge order (later wins):
      1. default.yaml
      2. profile YAML (``colab_t4``, ``quality_gpu``, or custom path stem)
      3. quality preset (``draft`` / ``standard`` / ``high``)
      4. optional extra_yaml
      5. environment overrides (when apply_env=True)

    Raises:
        ConfigurationError: missing files or invalid structure.
    """
    root = configs_dir or default_configs_dir()
    if not root.is_dir():
        raise ConfigurationError(
            f"Configs directory not found: {root}",
            suggestion="Run from the revamped_code package or pass configs_dir= explicitly.",
        )

    data = _load_yaml(root / "default.yaml")

    env = EnvOverrides() if apply_env else None
    resolved_profile = profile or (env.profile if env else None)
    if resolved_profile and resolved_profile != "default":
        profile_path = Path(resolved_profile)
        if not profile_path.is_file():
            profile_path = root / f"{resolved_profile}.yaml"
        data = _deep_merge(data, _load_yaml(profile_path))
        data.setdefault("engine", {})["profile"] = resolved_profile

    if quality:
        quality_path = root / "quality" / f"{quality}.yaml"
        if quality_path.is_file():
            data = _deep_merge(data, _load_yaml(quality_path))
        data.setdefault("training", {})["quality"] = quality

    if extra_yaml is not None:
        data = _deep_merge(data, _load_yaml(extra_yaml))

    if env is not None:
        if env.geometry_engine is not None:
            data.setdefault("geometry", {})["engine"] = env.geometry_engine
        if env.log_level is not None:
            data.setdefault("logging", {})["level"] = env.log_level
        if env.prefer_cuda is not None:
            data.setdefault("device", {})["prefer_cuda"] = env.prefer_cuda
        if env.allow_cpu_geometry is not None:
            data.setdefault("device", {})["allow_cpu_geometry"] = env.allow_cpu_geometry

    try:
        return EngineConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(
            f"Invalid engine configuration: {exc}",
            suggestion="Check YAML keys against EngineConfig schema in core/config.py.",
            details={"error": str(exc)},
        ) from exc
