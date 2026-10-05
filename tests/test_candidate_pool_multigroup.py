"""Tests for the multi-group CandidatePool data model.

The container still supports multi-group membership when constructed directly
(candidate index in several groups). ``from_dataframe`` no longer produces it: it
builds one candidate per row, and sequence-level dedup is left to the PEP layer.
"""
import numpy as np
import pandas as pd
import pytest

from duet.candidate_pool import CandidatePool


def test_from_dataframe_keeps_duplicate_sequences_as_distinct_candidates():
    """A sequence shared across rows yields one candidate per row, not a collapse.

    Two rows whose (truncated) sequences coincide become DISTINCT candidates that
    share a codeword; sequence-level PEP dedup is handled downstream by
    unique_sequences / candidate_to_sequence_idx. Each row keeps its own group
    and its own score, so differing scores are not a conflict.
    """
    df = pd.DataFrame({
        "Group":    ["G1", "G1", "G2", "G2"],
        "Sequence": ["ATCG", "GGCA", "ATCG", "TTAC"],  # ATCG shared across genes
        "Score":    [0.9, 0.8, 0.5, 0.7],              # differing scores: allowed
        "Quota":    [1, 1, 1, 1],
    })
    pool = CandidatePool.from_dataframe(df)

    assert pool.pool_size == 4, "no collapse: one candidate per row"
    # The two ATCG rows are distinct candidates, each in exactly one group.
    assert pool.group_to_candidates == {"G1": [0, 1], "G2": [2, 3]}
    assert pool.candidate_to_groups[0] == frozenset({"G1"})
    assert pool.candidate_to_groups[2] == frozenset({"G2"})
    # They share one PEP row via sequence dedup.
    assert pool.candidate_to_sequence_idx[0] == pool.candidate_to_sequence_idx[2]
    assert len(pool.unique_sequences) == 3
    # Each row kept its own score.
    np.testing.assert_array_equal(pool.scores, [0.9, 0.8, 0.5, 0.7])


def test_from_dataframe_allows_inconsistent_scores_for_shared_sequence():
    """Distinct guides sharing a truncated prefix keep their distinct scores;
    this no longer raises."""
    df = pd.DataFrame({
        "Group":    ["G1", "G2"],
        "Sequence": ["ATCG", "ATCG"],
        "Score":    [0.9, 0.5],   # previously raised ValueError; now kept per-row
        "Quota":    [1, 1],
    })
    pool = CandidatePool.from_dataframe(df)  # must NOT raise

    assert pool.pool_size == 2
    assert list(pool.scores) == [0.9, 0.5]
    assert pool.candidate_to_sequence_idx[0] == pool.candidate_to_sequence_idx[1]
    assert len(pool.unique_sequences) == 1


def test_codeword_to_group_order_matches_quotas():
    """codeword_to_group is built from quotas.items() in insertion order."""
    pool = CandidatePool(
        sequences=["A", "B", "C"],
        group_to_candidates={"g1": [0, 1], "g2": [2]},
        quotas={"g1": 2, "g2": 1},
    )
    assert pool.codeword_to_group == ["g1", "g1", "g2"]
    assert len(pool.codeword_to_group) == sum(pool.quotas.values())


def test_unique_sequences_and_sequence_idx():
    pool = CandidatePool(
        sequences=["A", "B", "A", "C"],
        group_to_candidates={"g": [0, 1, 2, 3]},
        quotas={"g": 1},
    )
    assert pool.unique_sequences == ["A", "B", "C"]
    assert pool.candidate_to_sequence_idx.tolist() == [0, 1, 0, 2]
    assert pool.candidate_to_sequence_idx.dtype == np.int32


def test_multi_group_membership_and_lazy_reverse_map():
    pool = CandidatePool(
        sequences=["A", "B", "C"],
        group_to_candidates={"g1": [0, 1], "g2": [0, 2]},  # candidate 0 in both
        quotas={"g1": 1, "g2": 1},
    )
    assert pool.candidate_to_groups[0] == frozenset({"g1", "g2"})
    assert pool.candidate_to_groups[1] == frozenset({"g1"})
    assert pool.candidate_to_groups[2] == frozenset({"g2"})


def test_validation_rejects_unreachable_candidate():
    """A candidate not present in any group is dead weight; reject."""
    with pytest.raises(ValueError, match="are in no group"):
        CandidatePool(
            sequences=["A", "B", "C"],
            group_to_candidates={"g": [0, 1]},  # 2 is unreachable
            quotas={"g": 1},
        )


def test_unique_sequences_type_preserving_and_merfish_cache_noop():
    """ndarray pool with all-unique codewords: unique_sequences stays an ndarray
    byte-identical to sequences, so the PEP cache key is unchanged (MERFISH no-op)."""
    from duet.providers import hash_library

    seqs = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=np.int8)
    pool = CandidatePool(
        sequences=seqs,
        group_to_candidates={"g": [0, 1, 2]},
        quotas={"g": 1},
        scores=np.ones(3),
    )
    us = pool.unique_sequences
    assert isinstance(us, np.ndarray)
    np.testing.assert_array_equal(us, seqs)
    assert hash_library(pool.unique_sequences) == hash_library(pool.sequences)
    np.testing.assert_array_equal(
        pool.candidate_to_sequence_idx, np.arange(3, dtype=np.int32)
    )


def test_unique_sequences_dedup_ndarray_rows_first_occurrence():
    """ndarray pool with a duplicate row: unique_sequences drops the dup row,
    in first-occurrence order, and candidate_to_sequence_idx points both copies
    at the same unique row."""
    seqs = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.int8)
    pool = CandidatePool(
        sequences=seqs,
        group_to_candidates={"g0": [0, 2], "g1": [1]},
        quotas={"g0": 1, "g1": 1},
        scores=np.ones(3),
    )
    us = pool.unique_sequences
    assert isinstance(us, np.ndarray)
    assert us.shape == (2, 4)
    np.testing.assert_array_equal(
        us, np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.int8)
    )
    np.testing.assert_array_equal(
        pool.candidate_to_sequence_idx, np.array([0, 0, 1], dtype=np.int32)
    )
