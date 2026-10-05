"""PEP-dedup-with-distinct-scores tests for the per-row candidate fix."""
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from duet.candidate_pool import CandidatePool
from duet.candidate_pool_factory import create_pool_from_source
from duet.runner import DuetOptimizerConfig, run_duet_ops
from duet.evaluator_config import EvaluatorConfig, create_evaluator
from duet.pareto_optimization import DecodingSwapCache


def _dup_pool():
    """Pool of 4 candidates; candidates 0 and 2 share sequence 'ATCG' with
    distinct scores. U = 3 unique sequences."""
    return CandidatePool(
        sequences=["ATCG", "GGCA", "ATCG", "TTAC"],
        group_to_candidates={"G1": [0, 1], "G2": [2, 3]},
        quotas={"G1": 1, "G2": 1},
        scores=np.array([0.9, 0.8, 0.6, 0.7]),
    )


def test_producer_builds_pep_over_unique_sequences():
    """run_duet_ops on a duplicate pool must build a U x U PEP matrix (U=3),
    not pool_size x pool_size (4)."""
    pool = _dup_pool()
    assert len(pool.unique_sequences) == 3 and pool.pool_size == 4

    pep_cfg = EvaluatorConfig.from_dict({
        "noise_channel": {"type": "symmetric", "epsilon": 0.1},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 100,
        "num_cpus": 1,
    })
    opt_cfg = DuetOptimizerConfig(
        lambda_=[0.5], max_iter=50, max_patience=20, num_cpus=1,
    )
    init = pool.sample_initial_selection(strategy="random", seed=0)
    result = run_duet_ops(
        candidates=pool, pep_config=pep_cfg, optimizer_config=opt_cfg,
        init=init, alphabet_size=4, seed=0,
    )
    pep = result["pep_count_matrix"]
    n_rows = pep.shape[0] if hasattr(pep, "shape") else pep.N
    assert n_rows == 3, f"expected U=3 PEP rows, got {n_rows}"


def test_decode_objective_requires_matrix_over_unique_sequences():
    """[s0, s0, s1]-shaped pool forces index renumbering (compacted u != candidate c).
    A PEP matrix over UNIQUE sequences gives the correct union-bound objective; a
    matrix over per-candidate sequences silently reads the wrong rows."""
    # Two unique sequences s0, s1; symmetric off-diagonal confusability 0.30.
    M_unique = np.array([[0.0, 0.30],
                         [0.30, 0.0]], dtype=np.float64)
    # Candidates [s0, s0, s1]  ->  candidate_to_sequence_idx = [0, 0, 1].
    c_to_u = np.array([0, 0, 1], dtype=np.int32)
    # Per-candidate ("over sequences") matrix replicates rows/cols per c_to_u.
    M_full = M_unique[np.ix_(c_to_u, c_to_u)]          # shape (3, 3)

    group_to_candidates = {"g0": [0, 1], "g1": [2]}
    codeword_to_group = ["g0", "g1"]
    S = np.array([1, 2], dtype=np.int64)               # candidate 1 (=s0), candidate 2 (=s1)

    # Reference union-bound objective over the ACTUAL selected sequences (s0, s1).
    sel = np.array([0, 1])                             # unique-seq indices of the selection
    sub = M_unique[np.ix_(sel, sel)]
    ref = 1.0 - (sub.sum() - np.trace(sub)) / len(S)   # = 0.70

    fixed = DecodingSwapCache.from_pep_matrix(
        M_unique, group_to_candidates, codeword_to_group,
        candidate_to_sequence_idx=c_to_u, duplicate_offset=0.0,
    )
    fixed.build_cache(S)

    broken = DecodingSwapCache.from_pep_matrix(
        M_full, group_to_candidates, codeword_to_group,
        candidate_to_sequence_idx=c_to_u, duplicate_offset=0.0,
    )
    broken.build_cache(S)

    np.testing.assert_allclose(fixed.compute_objective(S), ref)
    assert not np.isclose(broken.compute_objective(S), ref), (
        "matrix over per-candidate sequences should mis-decode (this is the bug "
        "Change 3 fixes)"
    )


def test_merfish_ndarray_library_cache_key_unchanged():
    """For an ndarray pool of unique codewords, the producer library
    (unique_sequences) hashes identically to sequences -> MERFISH cache no-op."""
    from duet.providers import hash_library

    seqs = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=np.int8)
    pool = CandidatePool(
        sequences=seqs, group_to_candidates={"g": [0, 1, 2]},
        quotas={"g": 1}, scores=np.ones(3),
    )
    assert hash_library(pool.unique_sequences) == hash_library(pool.sequences)


def test_baseline_evaluator_is_candidate_indexed_on_duplicate_pool():
    """Baselines build their own evaluator over pool.sequences (candidate space),
    so num_codewords == pool_size even when sequences contain duplicates — they
    never consume the deduped DUET matrix."""
    pool = _dup_pool()  # pool_size 4, U 3
    cfg = EvaluatorConfig.from_dict({
        "noise_channel": {"type": "symmetric", "epsilon": 0.1},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 50,
        "num_cpus": 1,
    })
    ev = create_evaluator(pool.sequences, cfg, alphabet_size=4)
    assert ev.num_codewords == pool.pool_size == 4


_CRISPRI_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "Horlbeck_2016" / "CRISPRi_v2_1.csv"
)


@pytest.mark.skipif(not _CRISPRI_CSV.exists(), reason="Weissman CRISPRi CSV not present")
def test_weissman_crispri_pool_loads_and_retains_duplicates(monkeypatch):
    """The exact load that crashed on main now succeeds, retaining prefix-collision
    guides as separate candidates that share PEP rows."""
    # Point the factory's default at the repository data (it only resolves by
    # itself from a source checkout, not from an installed wheel).
    monkeypatch.setenv("DUET_DATA_DIR", str(_CRISPRI_CSV.parents[2]))
    pool = create_pool_from_source(
        source="WeissmanCRISPRi",
        seq_rounds=10, quota=2, num_controls=200, min_rank=10,
        num_groups=1000, seed=191664963,   # trial-1 seed from the reproducer
    )
    assert pool.pool_size > len(pool.unique_sequences), (
        "expected duplicate truncated sequences retained as separate candidates"
    )
    # candidate_to_sequence_idx is no longer the identity (real dedup map).
    assert not np.array_equal(
        pool.candidate_to_sequence_idx, np.arange(pool.pool_size)
    )
