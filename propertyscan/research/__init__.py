"""Research platform: experiments, benchmarks, standard artifact trees."""

from __future__ import annotations

from propertyscan.research.artifacts import write_research_layout
from propertyscan.research.benchmark import BenchmarkRunner, BenchmarkSceneResult
from propertyscan.research.experiment import ExperimentRecord, ExperimentRegistry
from propertyscan.research.metrics import collect_run_metrics

__all__ = [
    "ExperimentRecord",
    "ExperimentRegistry",
    "BenchmarkRunner",
    "BenchmarkSceneResult",
    "write_research_layout",
    "collect_run_metrics",
]
