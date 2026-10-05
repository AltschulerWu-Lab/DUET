# src/duet/runner/ops.py
"""OPS entry point for DUET: decode vs. activity score."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from duet.candidate_pool import CandidatePool
from duet.evaluator_config import EvaluatorConfig
from duet.pareto_optimization import ScoreSwapCache
from duet.runner.core import (
    DuetOptimizerConfig,
    ObjectiveSpec,
    _run_duet_core,
)


# Module-level helpers keep ObjectiveSpec picklable.
def _build_score_cache(
    candidates: CandidatePool,
    *,
    codeword_to_group,
    candidate_to_sequence_idx: np.ndarray | None = None,
) -> ScoreSwapCache:
    # candidate_to_sequence_idx is accepted for forward-compat with the
    # unified factory signature but is not used by ScoreSwapCache.
    del candidate_to_sequence_idx
    return ScoreSwapCache(
        candidates.scores,
        candidates.group_to_candidates,
        codeword_to_group=codeword_to_group,
    )


def _ops_weights(lambda_: float) -> Dict[str, float]:
    """Decode + score weights for the OPS objective.

    J = lambda * decode + (1 - lambda) * score, matching the manuscript's
    Methods: lambda = 1 is decode-only, lambda = 0 is score-only.
    """
    return {"decode": lambda_, "score": 1.0 - lambda_}


def run_duet_ops(
    candidates: CandidatePool,
    pep_config: EvaluatorConfig,
    optimizer_config: DuetOptimizerConfig,
    init: np.ndarray,
    alphabet_size: int,
    seed: int | None = None,
    cache_dir: Path | None = None,
    use_mmap: bool = True,
    device: str = "cpu",
    force_rebuild: bool = False,
    sym_mem_budget_gb: float = 8.0,
) -> Dict:
    """Run DUET on OPS data. Balances decoding accuracy vs. activity score.

    Returns the dict shape documented on _run_duet_core.
    """
    spec = ObjectiveSpec(
        cache_factories={"score": _build_score_cache},
        weight_schedule=_ops_weights,
    )
    return _run_duet_core(
        candidates=candidates,
        pep_config=pep_config,
        optimizer_config=optimizer_config,
        init=init,
        alphabet_size=alphabet_size,
        objective_spec=spec,
        seed=seed,
        cache_dir=cache_dir,
        use_mmap=use_mmap,
        device=device,
        force_rebuild=force_rebuild,
        sym_mem_budget_gb=sym_mem_budget_gb,
    )
