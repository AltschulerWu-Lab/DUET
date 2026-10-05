"""The DUET runners: the advanced, provisional layer of the API.

``run_duet_ops``, ``run_duet_merfish``, ``DuetOptimizerConfig`` and
``ObjectiveSpec`` are documented and importable, but may change in minor
versions until 1.0 (docs/adr/0003-public-api.md). Most users want the public
facade instead: ``duet.design_ops``, ``duet.design_merfish``.
``run_duet_from_primitives`` is internal (kept importable for an internal
benchmark script).
"""

from duet.runner.core import (
    DuetOptimizerConfig,
    ObjectiveSpec,
    run_duet_from_primitives,
)
from duet.runner.ops import run_duet_ops
from duet.runner.merfish import run_duet_merfish

__all__ = [
    "run_duet_ops",
    "run_duet_merfish",
    "DuetOptimizerConfig",
    "ObjectiveSpec",
]
