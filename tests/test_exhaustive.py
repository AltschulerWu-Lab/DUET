"""Tests for exhaustive enumeration module."""
from __future__ import annotations

import numpy as np
import pytest

from duet.benchmark.exhaustive import (
    enumerate_all_codebooks,
    compute_total_codebooks,
    find_optimal_codebook,
)


class TestEnumerateAllCodebooks:
    def test_simple_enumeration(self):
        group_to_candidates = {"g0": [0, 1], "g1": [2, 3]}
        quotas = {"g0": 1, "g1": 1}
        codebooks = enumerate_all_codebooks(group_to_candidates, quotas)
        assert len(codebooks) == 4
        for cb in codebooks:
            assert len(cb) == 2
            assert cb[0] in [0, 1]
            assert cb[1] in [2, 3]

    def test_three_groups(self):
        group_to_candidates = {"g0": [0, 1, 2], "g1": [3, 4], "g2": [5, 6]}
        quotas = {"g0": 1, "g1": 1, "g2": 1}
        codebooks = enumerate_all_codebooks(group_to_candidates, quotas)
        assert len(codebooks) == 3 * 2 * 2

    def test_quota_two(self):
        group_to_candidates = {"g0": [0, 1, 2]}
        quotas = {"g0": 2}
        codebooks = enumerate_all_codebooks(group_to_candidates, quotas)
        assert len(codebooks) == 3  # C(3,2) = 3
        for cb in codebooks:
            assert len(cb) == 2
            assert len(set(cb)) == 2


class TestComputeTotalCodebooks:
    def test_matches_enumeration(self):
        group_to_candidates = {"g0": [0, 1, 2], "g1": [3, 4], "g2": [5, 6]}
        quotas = {"g0": 1, "g1": 1, "g2": 1}
        total = compute_total_codebooks(group_to_candidates, quotas)
        codebooks = enumerate_all_codebooks(group_to_candidates, quotas)
        assert total == len(codebooks)

    def test_benchmark_scale(self):
        group_to_candidates = {f"g{i}": list(range(i*5, (i+1)*5)) for i in range(7)}
        quotas = {f"g{i}": 1 for i in range(7)}
        total = compute_total_codebooks(group_to_candidates, quotas)
        assert total == 5**7


class _MockEvaluator:
    """Mock evaluator that returns predefined accuracies."""
    def __init__(self, accuracy_map):
        self.accuracy_map = accuracy_map
    def get_accuracy(self, indices):
        return self.accuracy_map[tuple(sorted(indices))]


_CODEBOOKS = [
    np.array([0, 2]),
    np.array([0, 3]),
    np.array([1, 2]),
    np.array([1, 3]),
]

_ACCURACY_MAP = {
    (0, 2): 0.80,
    (0, 3): 0.95,
    (1, 2): 0.70,
    (1, 3): 0.85,
}


class TestFindOptimalCodebook:
    def test_finds_best(self):
        evaluator = _MockEvaluator(_ACCURACY_MAP)
        best_indices, best_accuracy = find_optimal_codebook(_CODEBOOKS, evaluator)
        assert best_accuracy == 0.95
        np.testing.assert_array_equal(sorted(best_indices), [0, 3])


