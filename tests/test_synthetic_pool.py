"""Tests for synthetic pool generation (1-D and 2-D)."""
from __future__ import annotations

import numpy as np

from duet.benchmark.synthetic_pool import (
    create_synthetic_pool,
    create_2d_synthetic_pool,
)


class TestCreateSyntheticPoolShapeAndDeterminism:
    """Round-trip tests for the existing 1-D create_synthetic_pool."""

    def test_sequence_shape(self):
        pool = create_synthetic_pool(
            num_groups=4,
            candidates_per_group=3,
            seq_length=6,
            alphabet_size=2,
            quota=1,
            seed=0,
        )
        assert pool.sequences.shape == (12, 6)

    def test_quotas_cover_all_groups(self):
        pool = create_synthetic_pool(
            num_groups=4,
            candidates_per_group=3,
            seq_length=6,
            alphabet_size=2,
            quota=1,
            seed=0,
        )
        assert set(pool.quotas.keys()) == {f"group_{i}" for i in range(4)}

    def test_same_seed_reproduces_sequences(self):
        a = create_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=42,
        )
        b = create_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=42,
        )
        np.testing.assert_array_equal(a.sequences, b.sequences)

    def test_different_seeds_give_different_sequences(self):
        a = create_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=0,
        )
        b = create_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=1,
        )
        assert not np.array_equal(a.sequences, b.sequences)


class TestCreate2dSyntheticPoolScores:
    """Tests for the new 2-D variant that attaches U(0,1) scores."""

    def test_scores_shape_finite_in_unit_interval(self):
        pool = create_2d_synthetic_pool(
            num_groups=4,
            candidates_per_group=3,
            seq_length=6,
            alphabet_size=2,
            quota=1,
            seed=0,
        )
        pool_size = 4 * 3
        assert pool.scores.shape == (pool_size,)
        assert np.all(np.isfinite(pool.scores))
        assert np.all(pool.scores >= 0.0)
        assert np.all(pool.scores <= 1.0)

    def test_same_seed_reproduces_scores(self):
        a = create_2d_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=42,
        )
        b = create_2d_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=42,
        )
        np.testing.assert_array_equal(a.scores, b.scores)
        np.testing.assert_array_equal(a.sequences, b.sequences)

    def test_different_seeds_give_different_scores(self):
        a = create_2d_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=0,
        )
        b = create_2d_synthetic_pool(
            num_groups=4, candidates_per_group=3, seq_length=6,
            alphabet_size=2, quota=1, seed=1,
        )
        assert not np.array_equal(a.scores, b.scores)

    def test_iterating_seeds_does_not_couple_adjacent_trials(self):
        """SeedSequence.spawn ensures trial t's score stream is independent
        of trial (t+1)'s sequence stream, even though their integer seeds
        differ by 1.
        """
        T = 4
        seq_arrays = []
        score_arrays = []
        for t in range(T):
            pool = create_2d_synthetic_pool(
                num_groups=4, candidates_per_group=3, seq_length=6,
                alphabet_size=2, quota=1, seed=100 + t,
            )
            seq_arrays.append(pool.sequences.copy())
            score_arrays.append(pool.scores.copy())

        # All score arrays pairwise distinct.
        for i in range(T):
            for j in range(i + 1, T):
                assert not np.array_equal(score_arrays[i], score_arrays[j])
        # All sequence arrays pairwise distinct.
        for i in range(T):
            for j in range(i + 1, T):
                assert not np.array_equal(seq_arrays[i], seq_arrays[j])
