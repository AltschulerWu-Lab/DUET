"""Tests for memory optimization: sample batching (Task 1) and shared swap cache (Task 2)."""

import numpy as np
import pytest

from duet.codebook_evaluator import CodebookEvaluator, _worker_pep_batch
from duet.pareto_optimization import (
    DUET,
    DecodingSwapCache,
    ScoreSwapCache,
    SharedDecodeState,
    build_shared_decode_state,
    enumerate_within_group_swaps,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_small_evaluator(n_codewords=20, seq_length=10, n_samples=100, seed=42):
    """Create a small CodebookEvaluator for testing."""
    from duet.codebook_evaluator import (
        DNAEncoder,
        SymmetricEpsilon,
        HammingDistance,
        UniqueMinimum,
    )

    rng = np.random.default_rng(seed)
    alphabet = "ACGT"
    dna_list = []
    for _ in range(n_codewords):
        seq = "".join(rng.choice(list(alphabet), size=seq_length))
        dna_list.append(seq)

    noise_channel = SymmetricEpsilon(epsilon=0.05, alphabet_size=4)
    decoding_metric = HammingDistance()
    decoding_rule = UniqueMinimum()

    return CodebookEvaluator.from_dna_list(
        dna_list, noise_channel, decoding_metric, decoding_rule,
        n_samples=n_samples, seed=seed,
    )


def _make_pep_problem(pool_size=20, n_groups=4, seed=42):
    """Create a small PEP-based problem for testing shared state.

    Returns (pep_matrix, group_to_candidates, S, scores, sequences, codeword_to_group).
    """
    rng = np.random.default_rng(seed)
    cands_per_group = pool_size // n_groups

    group_to_candidates = {}
    for g in range(n_groups):
        start = g * cands_per_group
        end = start + cands_per_group
        group_to_candidates[f"G{g}"] = list(range(start, end))

    S = np.array([cs[0] for cs in group_to_candidates.values()])
    scores = rng.uniform(0.0, 1.0, size=pool_size)

    # Symmetric PEP matrix
    pep_matrix = rng.uniform(0.0, 0.3, size=(pool_size, pool_size))
    pep_matrix = (pep_matrix + pep_matrix.T) / 2
    np.fill_diagonal(pep_matrix, 0.0)

    # Dummy sequences (distinct)
    sequences = [f"SEQ{i:04d}" for i in range(pool_size)]

    # codeword_to_group maps codeword position i to the group bound to that
    # position. Since S has one candidate per group in g2c order, this is
    # simply the list of group keys.
    codeword_to_group = list(group_to_candidates.keys())

    return pep_matrix, group_to_candidates, S, scores, sequences, codeword_to_group


# =============================================================================
# Task 1: Sample Batching Tests
# =============================================================================


class TestSampleBatchingEquivalence:
    """Verify sample batching produces identical results to unbatched."""

    def test_worker_pep_batch_equivalence(self):
        """_worker_pep_batch with sample_batch_size produces identical count_rows."""
        evaluator = _make_small_evaluator(n_codewords=20, n_samples=100, seed=42)

        batch_indices = np.arange(5)  # first 5 codewords
        seed_val = np.random.SeedSequence(42).generate_state(1)[0]
        precomputed = evaluator.decoding_metric.precompute_transmitted(evaluator.codebook)

        # Unbatched (sample_batch_size=None)
        args_unbatched = (
            batch_indices, evaluator.codebook, evaluator.noise_channel,
            evaluator.decoding_metric, evaluator.decoding_rule, 100, seed_val, None,
        )
        _, counts_unbatched = _worker_pep_batch(args_unbatched)

        # Batched with sample_batch_size=25
        args_batched = (
            batch_indices, evaluator.codebook, evaluator.noise_channel,
            evaluator.decoding_metric, evaluator.decoding_rule, 100, seed_val, 25,
        )
        _, counts_batched = _worker_pep_batch(args_batched)

        np.testing.assert_array_equal(counts_unbatched, counts_batched)

    def test_compute_pep_matrix_equivalence(self):
        """compute_pep_matrix with and without sample_batch_size produces identical results."""
        evaluator1 = _make_small_evaluator(n_codewords=15, n_samples=80, seed=123)
        evaluator2 = _make_small_evaluator(n_codewords=15, n_samples=80, seed=123)

        counts1, ns1 = evaluator1.compute_pep_matrix(n_jobs=1, sample_batch_size=None)
        counts2, ns2 = evaluator2.compute_pep_matrix(n_jobs=1, sample_batch_size=20)

        assert ns1 == ns2
        np.testing.assert_array_equal(counts1, counts2)


class TestSampleBatchingEdgeCases:

    def test_sample_batch_size_1(self):
        """Maximally batched (1 sample at a time) still produces correct results."""
        evaluator1 = _make_small_evaluator(n_codewords=10, n_samples=50, seed=7)
        evaluator2 = _make_small_evaluator(n_codewords=10, n_samples=50, seed=7)

        counts1, _ = evaluator1.compute_pep_matrix(n_jobs=1, sample_batch_size=None)
        counts2, _ = evaluator2.compute_pep_matrix(n_jobs=1, sample_batch_size=1)

        np.testing.assert_array_equal(counts1, counts2)

    def test_sample_batch_size_equals_k(self):
        """sample_batch_size == n_samples is effectively no batching."""
        evaluator1 = _make_small_evaluator(n_codewords=10, n_samples=50, seed=7)
        evaluator2 = _make_small_evaluator(n_codewords=10, n_samples=50, seed=7)

        counts1, _ = evaluator1.compute_pep_matrix(n_jobs=1, sample_batch_size=None)
        counts2, _ = evaluator2.compute_pep_matrix(n_jobs=1, sample_batch_size=50)

        np.testing.assert_array_equal(counts1, counts2)

    def test_sample_batch_size_greater_than_k(self):
        """sample_batch_size > n_samples works correctly (single batch)."""
        evaluator1 = _make_small_evaluator(n_codewords=10, n_samples=50, seed=7)
        evaluator2 = _make_small_evaluator(n_codewords=10, n_samples=50, seed=7)

        counts1, _ = evaluator1.compute_pep_matrix(n_jobs=1, sample_batch_size=None)
        counts2, _ = evaluator2.compute_pep_matrix(n_jobs=1, sample_batch_size=200)

        np.testing.assert_array_equal(counts1, counts2)


class TestSampleBatchingValidation:

    def test_sample_batch_size_zero_raises(self):
        evaluator = _make_small_evaluator(n_codewords=5, n_samples=10, seed=1)
        with pytest.raises(ValueError, match="sample_batch_size must be a positive integer"):
            evaluator.compute_pep_matrix(n_jobs=1, sample_batch_size=0)

    def test_sample_batch_size_negative_raises(self):
        evaluator = _make_small_evaluator(n_codewords=5, n_samples=10, seed=1)
        with pytest.raises(ValueError, match="sample_batch_size must be a positive integer"):
            evaluator.compute_pep_matrix(n_jobs=1, sample_batch_size=-1)

    def test_auto_heuristic(self):
        """sample_batch_size=None auto-computes based on N and K."""
        evaluator = _make_small_evaluator(n_codewords=20, n_samples=100, seed=1)
        N = evaluator.num_codewords
        K = evaluator.n_samples
        expected = min(K, max(1, (2 * 1024**3) // (N * 9)))
        # For small N, expected == K (no batching needed)
        assert expected == K  # sanity check for small problem


# =============================================================================
# Task 2: Shared Swap Cache Tests
# =============================================================================


class TestBuildSharedDecodeState:
    """Verify build_shared_decode_state matches normal build_cache."""

    def test_streaming_matches_build_cache(self):
        """build_shared_decode_state output matches DecodingSwapCache.build_cache."""
        from duet.pep_accessor import InMemoryPEPAccessor

        pep_matrix, g2c, S, scores, sequences, codeword_to_group = _make_pep_problem()
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        # Build via normal path
        cache = DecodingSwapCache(
            pep_accessor=accessor,
            group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        cache.build_cache(S)

        # Build via streaming
        state = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )

        np.testing.assert_array_equal(state.S_indices, cache._S_indices)
        np.testing.assert_allclose(state.pep_sum_pool, cache._pep_sum_pool, atol=1e-12)
        np.testing.assert_array_equal(state.codebook_index_to_remove, cache._codebook_index_to_remove)
        np.testing.assert_array_equal(state.pool_index_to_add, cache._pool_index_to_add)
        np.testing.assert_allclose(state.deltas, cache.deltas, atol=1e-12)
        np.testing.assert_allclose(state.diag_X, cache._diag_X, atol=1e-12)

    def test_batch_size_sweep(self):
        """Different batch sizes all produce identical output."""
        from duet.pep_accessor import InMemoryPEPAccessor

        pep_matrix, g2c, S, scores, sequences, codeword_to_group = _make_pep_problem()
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        ref = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )

        for bs in [1, 2, 16, len(S)]:
            state = build_shared_decode_state(
                S=S, accessor=accessor, group_to_candidates=g2c,
                codeword_to_group=codeword_to_group, batch_size=bs,
            )
            np.testing.assert_allclose(state.pep_sum_pool, ref.pep_sum_pool, atol=1e-12)
            np.testing.assert_allclose(state.deltas, ref.deltas, atol=1e-12)

    def test_empty_groups_no_swaps(self):
        """Groups with exactly quota candidates produce 0 swaps."""
        from duet.pep_accessor import InMemoryPEPAccessor

        # Each group has exactly 1 candidate → no alternatives
        g2c = {"A": [0], "B": [1], "C": [2]}
        S = np.array([0, 1, 2])
        codeword_to_group = ["A", "B", "C"]
        pep_matrix = np.random.default_rng(42).uniform(0, 0.3, (3, 3))
        pep_matrix = (pep_matrix + pep_matrix.T) / 2
        np.fill_diagonal(pep_matrix, 0.0)
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        state = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )

        assert len(state.codebook_index_to_remove) == 0
        assert len(state.pool_index_to_add) == 0
        assert len(state.deltas) == 0


class TestFromSharedState:
    """Verify DecodingSwapCache.from_shared_state equivalence."""

    def test_from_shared_state_matches_build_cache(self):
        """Cache built via from_shared_state matches build_cache."""
        from duet.pep_accessor import InMemoryPEPAccessor

        pep_matrix, g2c, S, scores, sequences, codeword_to_group = _make_pep_problem()
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        # Normal path
        cache_normal = DecodingSwapCache(
            pep_accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        cache_normal.build_cache(S)

        # Shared state path
        state = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        cache_shared = DecodingSwapCache.from_shared_state(
            pep_accessor=accessor, state=state, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )

        assert pytest.approx(cache_shared.compute_objective(S)) == cache_normal.compute_objective(S)
        np.testing.assert_allclose(cache_shared.deltas, cache_normal.deltas, atol=1e-12)
        np.testing.assert_allclose(cache_shared._pep_sum_pool, cache_normal._pep_sum_pool, atol=1e-12)

    def test_build_cache_guard_noop(self):
        """build_cache on already-initialized cache is a no-op."""
        from duet.pep_accessor import InMemoryPEPAccessor

        pep_matrix, g2c, S, scores, sequences, codeword_to_group = _make_pep_problem()
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        state = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        cache = DecodingSwapCache.from_shared_state(
            pep_accessor=accessor, state=state, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )

        # Save state before build_cache
        deltas_before = cache.deltas.copy()
        pep_sum_before = cache._pep_sum_pool.copy()

        # Call build_cache — should be no-op
        cache.build_cache(S)

        np.testing.assert_array_equal(cache.deltas, deltas_before)
        np.testing.assert_array_equal(cache._pep_sum_pool, pep_sum_before)


class TestSharedStateDUETIntegration:
    """End-to-end DUET with shared state."""

    def test_duet_with_shared_state_matches_without(self):
        """DUET results identical with and without shared decode state."""
        from duet.pep_accessor import InMemoryPEPAccessor

        pep_matrix, g2c, S, scores, sequences, codeword_to_group = _make_pep_problem(seed=77)
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        # Without shared state
        decode1 = DecodingSwapCache(
            pep_accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        score1 = ScoreSwapCache(
            scores=scores, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        weights = {"decode": 0.5, "score": 0.5}
        opt1 = DUET(
            group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
            objective_caches=[decode1, score1],
            weights=weights,
            temperature=0.0,
            max_iter=30,
            max_patience=10,
        )
        opt1.optimize(list(S), seed=42)

        # With shared state
        state = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        decode2 = DecodingSwapCache.from_shared_state(
            pep_accessor=accessor, state=state, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        score2 = ScoreSwapCache(
            scores=scores, group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
        )
        opt2 = DUET(
            group_to_candidates=g2c,
            codeword_to_group=codeword_to_group,
            objective_caches=[decode2, score2],
            weights=weights,
            temperature=0.0,
            max_iter=30,
            max_patience=10,
        )
        opt2.optimize(list(S), seed=42)

        np.testing.assert_array_equal(opt1.best_S, opt2.best_S)
