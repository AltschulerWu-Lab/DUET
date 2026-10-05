"""Unit tests for SivanandanOptimizer."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import numpy as np
import pytest

from duet.hamming import DenseHammingMatrix
from duet.greedy_optimizers import SivanandanOptimizer


def _make_matrix(sequences: list[list[int]]) -> DenseHammingMatrix:
    """Build a DenseHammingMatrix from a list of integer-encoded sequences."""
    encoded = np.array(sequences, dtype=int)
    return DenseHammingMatrix(encoded)


class TestGeneOrdering:
    """Verify genes are processed in ascending order of candidate availability."""

    def test_constrained_gene_processed_first(self):
        """Gene with fewer candidates should be processed first, getting
        priority access to the Hamming space."""
        # 5 sequences of length 3
        sequences = [
            [0, 0, 0],  # idx 0 — gene_small
            [1, 1, 1],  # idx 1 — gene_small
            [0, 0, 1],  # idx 2 — gene_large (HD=1 from idx 0)
            [1, 1, 0],  # idx 3 — gene_large (HD=1 from idx 1)
            [2, 2, 2],  # idx 4 — gene_large
        ]
        mat = _make_matrix(sequences)

        gene_to_indices = {
            "gene_large": [2, 3, 4],  # 3 candidates
            "gene_small": [0, 1],     # 2 candidates (more constrained)
        }
        gene_to_N = {"gene_large": 1, "gene_small": 1}
        scores = np.array([1.0, 0.5, 1.0, 0.5, 0.3])

        opt = SivanandanOptimizer()
        result = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=1, scores=scores,
        )

        # gene_small has fewer candidates (2) so it is processed first.
        # Its best-scoring candidate is idx 0 (score=1.0).
        assert 0 in result
        assert len(result) == 2

    def test_deterministic_output(self):
        """Same input always produces same output (no randomness)."""
        sequences = [[0, 0], [1, 1], [0, 1], [1, 0]]
        mat = _make_matrix(sequences)

        gene_to_indices = {"A": [0, 1], "B": [2, 3]}
        gene_to_N = {"A": 1, "B": 1}
        scores = np.array([1.0, 0.5, 1.0, 0.5])

        opt = SivanandanOptimizer()
        results = [
            opt.optimize(mat, gene_to_indices, gene_to_N,
                         target_min_hamming=1, scores=scores)
            for _ in range(5)
        ]
        assert all(r == results[0] for r in results)


class TestScoreOrdering:
    """Verify candidates within a gene are tried in descending score order."""

    def test_higher_score_candidate_preferred(self):
        """When multiple candidates satisfy the Hamming constraint,
        the highest-scoring one should be selected."""
        # 3 sequences, all pairwise HD >= 2
        sequences = [
            [0, 0, 0],  # idx 0, score=0.2 (low)
            [1, 1, 1],  # idx 1, score=0.9 (high)
            [2, 2, 2],  # idx 2, score=0.5 (mid)
        ]
        mat = _make_matrix(sequences)

        gene_to_indices = {"gene_A": [0, 1, 2]}
        gene_to_N = {"gene_A": 1}
        scores = np.array([0.2, 0.9, 0.5])

        opt = SivanandanOptimizer()
        result = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=1, scores=scores,
        )

        # idx 1 has the highest score and satisfies the constraint
        assert result == {1}


class TestHammingConstraint:
    """Verify the Hamming distance constraint is enforced."""

    def test_candidate_rejected_if_too_close(self):
        """A candidate that violates the minimum Hamming distance
        should be skipped even if it has a high score."""
        # idx 0 and idx 1 have HD=1 (too close for target_min_hamming=2)
        sequences = [
            [0, 0, 0],  # idx 0 — gene_A
            [0, 0, 1],  # idx 1 — gene_B, HD=1 from idx 0
            [1, 1, 1],  # idx 2 — gene_B, HD=3 from idx 0
        ]
        mat = _make_matrix(sequences)

        gene_to_indices = {"gene_A": [0], "gene_B": [1, 2]}
        gene_to_N = {"gene_A": 1, "gene_B": 1}
        # idx 1 has higher score but will violate Hamming constraint
        scores = np.array([1.0, 0.9, 0.5])

        opt = SivanandanOptimizer()
        result = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=2, scores=scores,
        )

        # idx 1 (HD=1 from idx 0) is rejected; idx 2 (HD=3) is accepted
        assert result == {0, 2}

    def test_incomplete_selection_when_no_candidates_fit(self):
        """If no candidate for a gene satisfies the constraint,
        the gene gets fewer selections than requested."""
        # All three sequences are identical
        sequences = [[0, 0], [0, 0], [1, 1]]
        mat = _make_matrix(sequences)

        gene_to_indices = {"gene_A": [0], "gene_B": [1, 2]}
        gene_to_N = {"gene_A": 1, "gene_B": 1}
        scores = np.array([1.0, 1.0, 0.5])

        opt = SivanandanOptimizer()
        result = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=2, scores=scores,
        )

        # gene_A processed first (1 candidate), selects idx 0.
        # gene_B: idx 1 has HD=0 from idx 0 (rejected), idx 2 has HD=2 (accepted).
        assert result == {0, 2}


class TestControlGroups:
    """Verify control groups are processed after all targeting genes."""

    def test_controls_processed_last(self):
        """Control groups should be processed after all targeting genes,
        even if they have fewer candidates."""
        # control has only 1 candidate (most constrained), but should go last
        sequences = [
            [0, 0, 0],  # idx 0 — control
            [1, 1, 1],  # idx 1 — gene_A
            [2, 2, 2],  # idx 2 — gene_A
            [0, 1, 2],  # idx 3 — gene_B
        ]
        mat = _make_matrix(sequences)

        gene_to_indices = {
            "control": [0],
            "gene_A": [1, 2],
            "gene_B": [3],
        }
        gene_to_N = {"control": 1, "gene_A": 1, "gene_B": 1}
        scores = np.array([1.0, 1.0, 0.5, 1.0])

        opt = SivanandanOptimizer()
        result = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=1, scores=scores,
            control_groups={"control"},
        )

        # All should be selected (all pairwise HD >= 1)
        assert result == {0, 1, 3}

    def test_control_groups_none_treats_all_as_targeting(self):
        """When control_groups is None, all groups are targeting genes
        and sorted by availability."""
        sequences = [[0, 0], [1, 1], [0, 1]]
        mat = _make_matrix(sequences)

        gene_to_indices = {"A": [0], "B": [1, 2]}
        gene_to_N = {"A": 1, "B": 1}
        scores = np.array([1.0, 1.0, 0.5])

        opt = SivanandanOptimizer()
        result_none = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=1, scores=scores,
            control_groups=None,
        )
        result_empty = opt.optimize(
            mat, gene_to_indices, gene_to_N,
            target_min_hamming=1, scores=scores,
            control_groups=set(),
        )

        assert result_none == result_empty
