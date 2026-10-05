"""Tests for synthetic candidate pool generation."""
import numpy as np
import pytest

from duet.benchmark.synthetic_pool import create_synthetic_pool


class TestCreateSyntheticPool:
    """Test create_synthetic_pool function."""

    def test_basic_creation(self):
        pool = create_synthetic_pool(
            num_groups=3, candidates_per_group=4,
            seq_length=5, alphabet_size=4, quota=1, seed=42,
        )
        assert pool.pool_size == 12
        assert pool.seq_length == 5
        assert pool.num_groups == 3
        assert pool.total_selections == 3

    def test_sequences_are_integer_array(self):
        pool = create_synthetic_pool(
            num_groups=2, candidates_per_group=3,
            seq_length=4, alphabet_size=4, quota=1, seed=42,
        )
        assert isinstance(pool.sequences, np.ndarray)
        assert np.issubdtype(pool.sequences.dtype, np.integer)
        assert pool.sequences.shape == (6, 4)

    def test_alphabet_range(self):
        pool = create_synthetic_pool(
            num_groups=5, candidates_per_group=10,
            seq_length=8, alphabet_size=4, quota=1, seed=42,
        )
        assert pool.sequences.min() >= 0
        assert pool.sequences.max() < 4

    def test_scores_are_none_internally_ones(self):
        pool = create_synthetic_pool(
            num_groups=2, candidates_per_group=3,
            seq_length=4, alphabet_size=4, quota=1, seed=42,
        )
        np.testing.assert_array_equal(pool.scores, np.ones(6))

    def test_group_names(self):
        pool = create_synthetic_pool(
            num_groups=3, candidates_per_group=2,
            seq_length=4, alphabet_size=4, quota=1, seed=42,
        )
        unique_groups = sorted(pool.group_to_candidates.keys())
        assert unique_groups == ["group_0", "group_1", "group_2"]

    def test_candidates_per_group(self):
        pool = create_synthetic_pool(
            num_groups=3, candidates_per_group=5,
            seq_length=4, alphabet_size=4, quota=1, seed=42,
        )
        for group, indices in pool.group_to_candidates.items():
            assert len(indices) == 5

    def test_quotas(self):
        pool = create_synthetic_pool(
            num_groups=3, candidates_per_group=5,
            seq_length=4, alphabet_size=4, quota=2, seed=42,
        )
        for group, q in pool.quotas.items():
            assert q == 2

    def test_reproducible_with_same_seed(self):
        pool1 = create_synthetic_pool(
            num_groups=3, candidates_per_group=5,
            seq_length=8, alphabet_size=4, quota=1, seed=99,
        )
        pool2 = create_synthetic_pool(
            num_groups=3, candidates_per_group=5,
            seq_length=8, alphabet_size=4, quota=1, seed=99,
        )
        np.testing.assert_array_equal(pool1.sequences, pool2.sequences)

    def test_different_seeds_differ(self):
        pool1 = create_synthetic_pool(
            num_groups=3, candidates_per_group=5,
            seq_length=8, alphabet_size=4, quota=1, seed=1,
        )
        pool2 = create_synthetic_pool(
            num_groups=3, candidates_per_group=5,
            seq_length=8, alphabet_size=4, quota=1, seed=2,
        )
        assert not np.array_equal(pool1.sequences, pool2.sequences)
