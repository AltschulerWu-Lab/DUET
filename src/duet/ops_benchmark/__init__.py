# src/duet/ops_benchmark/__init__.py
"""OPS benchmark pipeline (multi-trial orchestration over DUET + baselines)."""

from duet.ops_benchmark.config import (
    CandidatePoolConfig,
    FeldmanConfig,
    OpsBenchmarkConfig,
    SivanandanConfig,
)
from duet.ops_benchmark.runner import run_ops_benchmark

__all__ = [
    "run_ops_benchmark",
    "OpsBenchmarkConfig",
    "CandidatePoolConfig",
    "SivanandanConfig",
    "FeldmanConfig",
]
