# src/duet/runner/merfish.py
"""MERFISH entry point for DUET: decode vs. optical crowding."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np

from duet.candidate_pool import CandidatePool
from duet.evaluator_config import EvaluatorConfig
from duet.pareto_optimization import CrowdingSwapCache
from duet.runner.core import (
    DuetOptimizerConfig,
    ObjectiveSpec,
    _run_duet_core,
)


# Module-level helper + functools.partial to bind the arrays without a closure.
def _build_crowding_cache(
    codewords: np.ndarray,
    expression: np.ndarray,
    candidates: CandidatePool,
    *,
    codeword_to_group,
    candidate_to_sequence_idx: np.ndarray | None = None,
) -> CrowdingSwapCache:
    # candidate_to_sequence_idx is accepted for forward-compat with the
    # unified factory signature but is not used by CrowdingSwapCache.
    del candidate_to_sequence_idx
    return CrowdingSwapCache(
        codewords,
        expression,
        candidates.group_to_candidates,
        codeword_to_group=codeword_to_group,
    )


def _merfish_weights(lambda_: float) -> Dict[str, float]:
    """Decode + crowding weights for the MERFISH multi-objective.

    Both DecodingSwapCache.compute_objective and the C(S_0)-anchored
    CrowdingSwapCache.compute_objective (Candidate C) return scores in
    approximately [0, 1] (decode accuracy, and 1 - C(S)/C(S_0) with
    O_C(S_0) = 0 exactly), so a plain linear mix in lambda is already
    commensurate in score space -- no per-experiment retuning.

    J = lambda * decode + (1 - lambda) * crowding, matching the manuscript's
    Methods: lambda = 1 is decode-only, lambda = 0 is crowding-only.
    """
    return {"decode": lambda_, "crowding": 1.0 - lambda_}


def run_duet_merfish(
    candidates: CandidatePool,
    pep_config: EvaluatorConfig,
    optimizer_config: DuetOptimizerConfig,
    init: np.ndarray,
    codewords: np.ndarray,
    expression: np.ndarray,
    alphabet_size: int = 2,
    seed: int | None = None,
    cache_dir: Path | None = None,
    use_mmap: bool = True,
    device: str = "cpu",
    force_rebuild: bool = False,
    sym_mem_budget_gb: float = 8.0,
) -> Dict:
    """Run DUET on MERFISH data. Balances decoding accuracy vs. optical crowding.

    Returns the dict shape documented on _run_duet_core.
    """
    spec = ObjectiveSpec(
        cache_factories={
            "crowding": partial(_build_crowding_cache, codewords, expression),
        },
        weight_schedule=_merfish_weights,
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
