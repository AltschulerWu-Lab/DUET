# src/duet/ops_benchmark/baselines.py
"""Orchestration helpers that wrap the generic Sivanandan/Feldman runners."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from duet.benchmark.feldman_runner import FeldmanParams, run_feldman
from duet.benchmark.sivanandan_runner import SivanandanParams, run_sivanandan
from duet.candidate_pool import CandidatePool
from duet.codebook_evaluator import get_encoder
from duet.dual_guide_factory import get_dual_guide_encoder
from duet.ops_benchmark.config import FeldmanConfig, SivanandanConfig
from duet.utils import Stopwatch


logger = logging.getLogger(__name__)


def encode_for_sivanandan(
    candidates: CandidatePool,
    chemistry: str,
) -> np.ndarray:
    """Encode sequences into the observed space used by Sivanandan's optimizer."""
    if chemistry == "dna":
        return get_encoder(4).encode(candidates.sequences)
    dual_encoder = get_dual_guide_encoder(chemistry)
    hex_encoded = dual_encoder.encode(candidates.sequences)
    return dual_encoder.encode_to_observed(hex_encoded)


def run_sivanandan_for_eds(
    candidates: CandidatePool,
    config: SivanandanConfig,
    encoded_sequences: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Run Sivanandan for each edit distance; return dict keyed by ED."""
    results = {}
    control_groups = candidates.metadata.get("control_groups", set())
    for ed in config.edit_distances:
        with Stopwatch(f"Sivanandan (ED={ed})"):
            idx = run_sivanandan(
                encoded_sequences,
                candidates.group_to_candidates,
                candidates.quotas,
                candidates.scores,
                SivanandanParams(
                    target_min_hamming=ed,
                    control_groups=control_groups,
                ),
                num_cpus=config.num_cpus,
            )
        results[ed] = idx
    return results


def run_feldman_for_eds(
    candidates: CandidatePool,
    config: FeldmanConfig,
    seq_rounds: int,
    quota: int,
    num_controls: int,
) -> Dict[int, np.ndarray]:
    """Run Feldman for each edit distance; return dict keyed by ED.

    Each group's quota comes from ``candidates.quotas``, the quotas the
    validity check uses, so pools with several control groups (CRISPick) get
    each group's own quota. ``quota`` and ``num_controls`` are the fallback
    quotas of ``run_feldman`` and are not used when the pool's quotas are
    passed. Pools above 80,000 sequences run too: the OPS wrapper bypasses the
    OPS library's unfinished >80k branch (see ``external_ops/ops_wrapper.py``).
    """
    with Stopwatch("Feldman et al."):
        feldman_df = candidates.metadata["source_dataframe"]
        return run_feldman(
            feldman_df,
            seq_rounds,
            quota,
            num_controls,
            FeldmanParams(
                conda_env=config.conda_env,
                edit_distances=config.edit_distances,
            ),
            group_quotas=dict(candidates.quotas),
        )
