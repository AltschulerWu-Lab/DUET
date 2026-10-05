"""Unit tests for the OPS benchmark codebook validity check.

A codebook is valid iff it has the right cardinality, selects no candidate index
twice, and fills each group to exactly its quota. Duplicate *sequences* within a
group are allowed (two distinct candidates that share a codeword both decode to
the same gene); the rejected degeneracy is reusing the same candidate index.
"""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import numpy as np

from duet.candidate_pool import CandidatePool
from duet.ops_benchmark.runner import _is_valid_codebook


def _make_pool(sequences, group_to_candidates, quotas) -> CandidatePool:
    return CandidatePool(
        sequences=list(sequences),
        group_to_candidates=group_to_candidates,
        quotas=quotas,
        scores=np.ones(len(sequences), dtype=float),
    )


def _pool_distinct() -> CandidatePool:
    # G1: 3 distinct candidates (quota 2); G2: 2 distinct candidates (quota 2).
    return _make_pool(
        ["AAAA", "CCCC", "GGGG", "ATAT", "CGCG"],
        {"G1": [0, 1, 2], "G2": [3, 4]},
        {"G1": 2, "G2": 2},
    )


def test_valid_codebook_passes():
    pool = _pool_distinct()
    assert _is_valid_codebook([0, 1, 3, 4], pool) is True


def test_wrong_cardinality_fails():
    pool = _pool_distinct()
    assert _is_valid_codebook([0, 1, 3], pool) is False


def test_per_group_quota_violation_fails():
    # Right total (4) but G1 gets 3 and G2 gets 1.
    pool = _pool_distinct()
    assert _is_valid_codebook([0, 1, 2, 3], pool) is False


def test_foreign_index_fails_membership():
    pool = _pool_distinct()
    assert _is_valid_codebook([0, 1, 3, 99], pool) is False


def test_repeated_index_fails():
    # The duplicate-stacking degeneracy: the same candidate index twice. Per-group
    # counts would look right (G1 gets 2, G2 gets 2), but index 0 is reused.
    pool = _pool_distinct()
    assert _is_valid_codebook([0, 0, 3, 4], pool) is False


def test_within_group_duplicate_sequence_allowed():
    # idx 0 and 1 are DISTINCT candidates that share sequence "AAAA" within G1.
    # Both decode to the same gene, so selecting both is valid.
    pool = _make_pool(
        ["AAAA", "AAAA", "GGGG", "ATAT", "CGCG"],
        {"G1": [0, 1, 2], "G2": [3, 4]},
        {"G1": 2, "G2": 2},
    )
    assert _is_valid_codebook([0, 1, 3, 4], pool) is True
    assert _is_valid_codebook([0, 2, 3, 4], pool) is True


def test_across_group_duplicate_sequence_allowed():
    # A multi-targeting reagent listed once per gene: sequence "AAAA" appears as
    # candidate 0 (G1) and candidate 2 (G2) — distinct indices, shared codeword.
    pool = _make_pool(
        ["AAAA", "CCCC", "AAAA", "CGCG"],
        {"G1": [0, 1], "G2": [2, 3]},
        {"G1": 1, "G2": 1},
    )
    assert _is_valid_codebook([0, 2], pool) is True
