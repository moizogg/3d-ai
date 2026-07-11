"""Geometry router — MASt3R / DUSt3R / Auto (no COLMAP)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from propertyscan.core.config import EngineConfig
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult, SceneDescriptor
from propertyscan.geometry.base import GeometryProvider
from propertyscan.geometry.providers.arkit import AppleARKitProvider
from propertyscan.geometry.providers.dust3r import DUSt3RProvider
from propertyscan.geometry.providers.mast3r import MASt3RProvider
from propertyscan.geometry.providers.mock import MockGeometryProvider
from propertyscan.geometry.result_builder import failure_result

logger = logging.getLogger("propertyscan.geometry.router")

EngineMode = Literal["mast3r", "dust3r", "auto", "mock", "arkit"]


class GeometryRouter:
    """Select and run foundation geometry providers.

    Modes:
      - mast3r / dust3r / mock / arkit: force one provider
      - auto: rank by can_handle(SceneDescriptor), try in order until success

    Never falls back to COLMAP.
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        providers: list[GeometryProvider] | None = None,
        include_mock: bool = False,
    ) -> None:
        self.config = config
        if providers is not None:
            self.providers = providers
        else:
            self.providers = [
                MASt3RProvider(config),
                DUSt3RProvider(config),
                AppleARKitProvider(config),
            ]
            if include_mock:
                self.providers.append(MockGeometryProvider(config))

    def rank(
        self,
        descriptor: SceneDescriptor,
        *,
        mode: str | None = None,
    ) -> list[tuple[float, GeometryProvider]]:
        mode = (mode or self.config.geometry.engine).lower()
        ranked: list[tuple[float, GeometryProvider]] = []

        for p in self.providers:
            if mode == "auto":
                score = p.can_handle(descriptor)
            elif mode == p.name or mode == p.provider_type:
                score = 1000.0
            elif mode == "mock" and p.name == "mock":
                score = 1000.0
            else:
                score = -1.0
            if score >= 0:
                ranked.append((score, p))

        ranked.sort(key=lambda x: x[0], reverse=True)
        # Drop non-matching forced modes
        if mode not in ("auto",) and ranked:
            ranked = [(s, p) for s, p in ranked if s >= 1000.0]
        return ranked

    def select(
        self,
        descriptor: SceneDescriptor,
        *,
        mode: str | None = None,
    ) -> GeometryProvider:
        ranked = self.rank(descriptor, mode=mode)
        if not ranked:
            raise ValueError(
                f"No geometry provider for mode={mode or self.config.geometry.engine}. "
                "Valid: mast3r | dust3r | auto | mock | arkit. COLMAP is not supported."
            )
        score, provider = ranked[0]
        logger.info(
            "GeometryRouter selected [%s] score=%.2f mode=%s scene=%s",
            provider.name,
            score,
            mode or self.config.geometry.engine,
            descriptor.scene_type,
        )
        return provider

    def reconstruct(
        self,
        frame_set: FrameSet,
        *,
        output_dir: Path,
        descriptor: SceneDescriptor,
        config: EngineConfig | None = None,
        mode: str | None = None,
    ) -> GeometryResult:
        """Route reconstruction with ordered fallback among foundation providers only."""
        cfg = config or self.config
        mode = (mode or cfg.geometry.engine).lower()
        ranked = self.rank(descriptor, mode=mode)

        if not ranked:
            return failure_result(
                "router",
                f"No provider registered for geometry.engine={mode}. COLMAP is not available.",
            )

        last: GeometryResult | None = None
        for score, provider in ranked:
            logger.info(
                "Attempting geometry via [%s] (score=%.2f)", provider.name, score
            )
            out = Path(output_dir) / provider.name
            res = provider.reconstruct(
                frame_set,
                output_dir=out,
                config=cfg,
                descriptor=descriptor,
            )
            last = res
            if res.success:
                logger.info(
                    "Provider [%s] succeeded: %d/%d registered",
                    provider.name,
                    res.metrics.registered_cameras,
                    res.metrics.total_cameras,
                )
                return res
            logger.warning(
                "Provider [%s] failed: %s",
                provider.name,
                res.error_message,
            )

        assert last is not None
        # Re-wrap to show router exhausted options
        return GeometryResult(
            provider_name=last.provider_name,
            success=False,
            pose_graph=last.pose_graph,
            point_cloud=last.point_cloud,
            confidence=last.confidence,
            metrics=last.metrics,
            artifacts=last.artifacts,
            error_message=(
                f"All geometry providers failed. Last [{last.provider_name}]: "
                f"{last.error_message}"
            ),
            metadata={**last.metadata, "router_exhausted": True, "mode": mode},
        )
