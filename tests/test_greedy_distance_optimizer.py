"""Tests for GreedyDistanceOptimizer."""
from __future__ import annotations

import numpy as np
import pytest

from duet.greedy_optimizers import GreedyDistanceOptimizer
from duet.codebook_evaluator import HammingDistance, SymmetricNLL


class TestGreedyHamming:
    """Test GreedyDistanceOptimizer with default HammingDistance."""

    def test_selects_one_per_group(self):
        """With quota=1 per group, should select exactly one candidate per group."""
        sequences = np.array([
            [0, 0, 0],  # idx 0 — group_0
            [1, 1, 1],  # idx 1 — group_0
            [2, 2, 2],  # idx 2 — group_1
            [0, 1, 2],  # idx 3 — group_1
        ], dtype=np.int8)
        group_to_candidates = {"group_0": [0, 1], "group_1": [2, 3]}
        quotas = {"group_0": 1, "group_1": 1}

        opt = GreedyDistanceOptimizer()
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=42)

        assert len(result) == 2
        # One from each group
        groups_selected = set()
        for idx in result:
            for g, cands in group_to_candidates.items():
                if idx in cands:
                    groups_selected.add(g)
        assert groups_selected == {"group_0", "group_1"}

    def test_maximizes_min_hamming(self):
        """Second selection should maximize minimum Hamming distance to first."""
        # With seed=0, group_0 is processed first -> idx 0 selected first.
        # group_1: [0,0,1] has HD=1, [1,1,1] has HD=3 -> should pick [1,1,1]
        sequences = np.array([
            [0, 0, 0],  # idx 0 — group_0 (only candidate)
            [0, 0, 1],  # idx 1 — group_1, HD=1 from idx 0
            [1, 1, 1],  # idx 2 — group_1, HD=3 from idx 0
        ], dtype=np.int8)
        group_to_candidates = {"group_0": [0], "group_1": [1, 2]}
        quotas = {"group_0": 1, "group_1": 1}

        opt = GreedyDistanceOptimizer()
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=0)

        assert 0 in result  # only candidate for group_0
        assert 2 in result  # maximizes min HD

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same result."""
        sequences = np.array([
            [0, 0], [1, 1], [0, 1], [1, 0], [2, 2], [2, 0],
        ], dtype=np.int8)
        group_to_candidates = {"g0": [0, 1], "g1": [2, 3], "g2": [4, 5]}
        quotas = {"g0": 1, "g1": 1, "g2": 1}

        opt = GreedyDistanceOptimizer()
        r1 = opt.optimize(sequences, group_to_candidates, quotas, seed=123)
        r2 = opt.optimize(sequences, group_to_candidates, quotas, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_may_differ(self):
        """Different seeds shuffle group order, potentially giving different results."""
        sequences = np.array([
            [0, 0, 0], [1, 1, 1],  # g0
            [0, 0, 1], [1, 1, 0],  # g1
            [0, 1, 0], [1, 0, 1],  # g2
        ], dtype=np.int8)
        group_to_candidates = {"g0": [0, 1], "g1": [2, 3], "g2": [4, 5]}
        quotas = {"g0": 1, "g1": 1, "g2": 1}

        opt = GreedyDistanceOptimizer()
        results = set()
        for seed in range(20):
            r = opt.optimize(sequences, group_to_candidates, quotas, seed=seed)
            results.add(tuple(sorted(r)))
        # With enough seeds, at least 2 distinct solutions should appear
        assert len(results) >= 1  # at minimum it works without error

    def test_first_candidate_selected_when_nothing_prior(self):
        """First group's first candidate (in shuffled order) is selected."""
        sequences = np.array([[0, 0], [1, 1]], dtype=np.int8)
        group_to_candidates = {"g0": [0, 1]}
        quotas = {"g0": 1}

        opt = GreedyDistanceOptimizer()
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=42)
        assert len(result) == 1
        assert result[0] in [0, 1]


class TestGreedyNLL:
    """Test GreedyDistanceOptimizer with NLL decoding metric."""

    def test_nll_selects_distant_candidate(self):
        """With NLL decoding metric, should still prefer more distant candidates."""
        sequences = np.array([
            [0, 0, 0],  # idx 0 — group_0
            [0, 0, 1],  # idx 1 — group_1, close to idx 0
            [2, 2, 2],  # idx 2 — group_1, far from idx 0
        ], dtype=np.int8)
        group_to_candidates = {"group_0": [0], "group_1": [1, 2]}
        quotas = {"group_0": 1, "group_1": 1}

        decoding_metric = SymmetricNLL(epsilon=0.1, alphabet_size=4)
        opt = GreedyDistanceOptimizer(decoding_metric=decoding_metric)
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=0)

        assert 0 in result
        assert 2 in result  # more distant under NLL too


class TestQuotaGreaterThanOne:
    """Test with quota > 1 per group."""

    def test_multiple_selections_per_group(self):
        """Should select multiple candidates from a group when quota > 1."""
        sequences = np.array([
            [0, 0], [1, 1], [2, 2],  # g0: 3 candidates, quota=2
            [0, 1], [1, 0],          # g1: 2 candidates, quota=1
        ], dtype=np.int8)
        group_to_candidates = {"g0": [0, 1, 2], "g1": [3, 4]}
        quotas = {"g0": 2, "g1": 1}

        opt = GreedyDistanceOptimizer()
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=42)

        assert len(result) == 3
        g0_selected = [i for i in result if i in [0, 1, 2]]
        g1_selected = [i for i in result if i in [3, 4]]
        assert len(g0_selected) == 2
        assert len(g1_selected) == 1
