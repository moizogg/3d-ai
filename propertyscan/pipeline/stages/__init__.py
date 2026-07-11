"""Thin pipeline stage wrappers for Phase 2+."""

from __future__ import annotations

from propertyscan.pipeline.stages.s01_validate import ValidateCaptureStage
from propertyscan.pipeline.stages.s02_decode import DecodeFramesStage
from propertyscan.pipeline.stages.s03_quality import FrameQualityStage
from propertyscan.pipeline.stages.s04_dedup import DedupStage
from propertyscan.pipeline.stages.s05_keyframes import KeyframeStage
from propertyscan.pipeline.stages.s06_classify import ClassifySceneStage
from propertyscan.pipeline.stages.s07_route import RouteGeometryStage
from propertyscan.pipeline.stages.s08_reconstruct import ReconstructGeometryStage
from propertyscan.pipeline.stages.s09_depth import EstimateDepthStage
from propertyscan.pipeline.stages.s10_fusion import FuseGeometryStage
from propertyscan.pipeline.stages.s11_health import HealthGateStage
from propertyscan.pipeline.stages.s12_dataset import BuildDatasetStage
from propertyscan.pipeline.stages.s13_train import TrainGaussiansStage
from propertyscan.pipeline.stages.s14_inspect import InspectSceneStage
from propertyscan.pipeline.stages.s15_quality import QualityScoreStage
from propertyscan.pipeline.stages.s16_assemble import AssembleSceneStage
from propertyscan.pipeline.stages.s17_export import ExportStage

__all__ = [
    "ValidateCaptureStage",
    "DecodeFramesStage",
    "FrameQualityStage",
    "DedupStage",
    "KeyframeStage",
    "ClassifySceneStage",
    "RouteGeometryStage",
    "ReconstructGeometryStage",
    "EstimateDepthStage",
    "FuseGeometryStage",
    "HealthGateStage",
    "BuildDatasetStage",
    "TrainGaussiansStage",
    "InspectSceneStage",
    "QualityScoreStage",
    "AssembleSceneStage",
    "ExportStage",
]
