"""
Performance benchmark tests for DUET algorithm and PEP matrix construction.

These tests verify that performance characteristics remain acceptable during
refactoring. They use simple timing assertions rather than statistical
benchmarking to avoid additional dependencies.

Run with: pytest test_performance.py -v -s
The -s flag shows timing output for debugging.

Note: These tests may be slower than unit tests. Consider marking them with
@pytest.mark.slow and skipping in CI with: pytest -m "not slow"
"""

import time
import pytest
import numpy as np
from typing import List, Dict, Callable
from functools import wraps

from duet.codebook_evaluator import (
    SymmetricEpsilon,
    HammingDistance,
    UniqueMinimum,
    CodebookEvaluator,
    create_evaluator_for_dna,
)
from duet.pareto_optimization import (
    DecodingSwapCache,
    ScoreSwapCache,
    DUET,
)
from duet.pep_accessor import InMemoryPEPAccessor


# =============================================================================
# Timing Utilities
# =============================================================================


def timed(func: Callable) -> Callable:
    """Decorator to time function execution and print result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n  {func.__name__}: {elapsed:.4f}s")
        return result, elapsed
    return wrapper


def generate_random_dna_library(n_sequences: int, seq_length: int, seed: int = 42) -> List[str]:
    """Generate random DNA sequences."""
    np.random.seed(seed)
    alphabet = "ATCG"
    return [
        ''.join(np.random.choice(list(alphabet), size=seq_length))
        for _ in range(n_sequences)
    ]


def generate_group_structure(n_groups: int, candidates_per_group: int) -> Dict[str, List[int]]:
    """Generate balanced group structure."""
    return {
        f"group_{i}": list(range(i * candidates_per_group, (i + 1) * candidates_per_group))
        for i in range(n_groups)
    }


def generate_scores(n_candidates: int, seed: int = 42) -> np.ndarray:
    """Generate random candidate scores."""
    np.random.seed(seed)
    return np.random.uniform(0.3, 1.0, size=n_candidates)


def generate_pep_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Generate a synthetic PEP matrix with realistic properties."""
    np.random.seed(seed)
    # Create symmetric matrix with high diagonal
    pep = np.random.uniform(0.05, 0.3, size=(n, n))
    pep = (pep + pep.T) / 2  # Make symmetric
    np.fill_diagonal(pep, 1.0)  # Diagonal is 1
    return pep


def generate_codeword_to_group(group_to_candidates: Dict[str, List[int]]) -> List[str]:
    """Slot-indexed codeword -> group map for a one-codeword-per-group codebook.

    The tests build S as ``[guides[0] for guides in group_to_candidates.values()]``,
    so codeword position i belongs to the i-th group in insertion order.
    """
    return list(group_to_candidates.keys())


def make_duet_perf(pep_matrix, group_to_candidates, scores, **kwargs):
    """Helper to construct DUET with new API for performance tests.

    Lambda convention (docs/adr/0001): J = lambda*decode + (1-lambda)*score.
    """
    codeword_to_group = generate_codeword_to_group(group_to_candidates)
    decode_cache = DecodingSwapCache.from_pep_matrix(
        pep_matrix=pep_matrix,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
    )
    score_cache = ScoreSwapCache(
        scores=scores,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
    )
    lambda_ = kwargs.pop("lambda_", 0.5)
    weights = {"decode": lambda_, "score": 1.0 - lambda_}
    return DUET(
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
        objective_caches=[decode_cache, score_cache],
        weights=weights,
        **kwargs,
    )


# =============================================================================
# PEP Matrix Performance Tests
# =============================================================================


class TestPEPMatrixPerformance:
    """Performance tests for PEP matrix computation."""

    @pytest.mark.slow
    def test_pep_matrix_small_library(self):
        """PEP matrix for 20 guides should complete in < 5 seconds."""
        library = generate_random_dna_library(20, 20, seed=42)

        evaluator = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=1000,
            seed=42,
        )

        start = time.perf_counter()
        pep, _ns = evaluator.compute_pep_matrix(n_jobs=1)
        elapsed = time.perf_counter() - start

        print(f"\n  PEP matrix (n=20, samples=1000): {elapsed:.4f}s")

        assert elapsed < 5.0, f"PEP computation too slow: {elapsed:.2f}s > 5s"
        assert pep.shape == (20, 20)

    @pytest.mark.slow
    def test_pep_matrix_medium_library(self):
        """PEP matrix for 50 guides should complete in < 20 seconds."""
        library = generate_random_dna_library(50, 20, seed=42)

        evaluator = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=1000,
            seed=42,
        )

        start = time.perf_counter()
        pep, _ns = evaluator.compute_pep_matrix(n_jobs=1)
        elapsed = time.perf_counter() - start

        print(f"\n  PEP matrix (n=50, samples=1000): {elapsed:.4f}s")

        assert elapsed < 20.0, f"PEP computation too slow: {elapsed:.2f}s > 20s"
        assert pep.shape == (50, 50)

    @pytest.mark.slow
    def test_pep_matrix_scaling(self):
        """Verify PEP computation scales roughly as O(n²)."""
        times = []
        sizes = [10, 20, 40]

        # Untimed warm-up: the first compute_pep_matrix call in a process pays
        # a one-off overhead of a few ms, comparable to the whole n=10 run, which
        # would otherwise deflate the 20/10 ratio when this test runs first.
        create_evaluator_for_dna(
            dna_list=generate_random_dna_library(sizes[0], 20, seed=0),
            epsilon=0.1,
            n_samples=500,
            seed=0,
        ).compute_pep_matrix(n_jobs=1)

        for n in sizes:
            library = generate_random_dna_library(n, 20, seed=42)
            evaluator = create_evaluator_for_dna(
                dna_list=library,
                epsilon=0.1,
                n_samples=500,
                seed=42,
            )

            start = time.perf_counter()
            _ = evaluator.compute_pep_matrix(n_jobs=1)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"\n  PEP matrix (n={n}): {elapsed:.4f}s")

        # Check that doubling n roughly quadruples time (within 10x tolerance)
        ratio_1_to_2 = times[1] / max(times[0], 1e-6)
        ratio_2_to_3 = times[2] / max(times[1], 1e-6)

        print(f"\n  Scaling ratio (20/10): {ratio_1_to_2:.2f} (expected ~4)")
        print(f"  Scaling ratio (40/20): {ratio_2_to_3:.2f} (expected ~4)")

        assert ratio_1_to_2 > 1.5, "Should scale faster than linear"
        assert ratio_2_to_3 > 1.5, "Should scale faster than linear"

    @pytest.mark.slow
    def test_pep_matrix_samples_scaling(self):
        """Verify PEP computation scales linearly with sample count."""
        library = generate_random_dna_library(20, 20, seed=42)
        times = []
        sample_counts = [500, 1000, 2000]

        for n_samples in sample_counts:
            evaluator = create_evaluator_for_dna(
                dna_list=library,
                epsilon=0.1,
                n_samples=n_samples,
                seed=42,
            )

            start = time.perf_counter()
            _ = evaluator.compute_pep_matrix(n_jobs=1)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"\n  PEP matrix (samples={n_samples}): {elapsed:.4f}s")

        ratio_1_to_2 = times[1] / max(times[0], 1e-6)
        ratio_2_to_3 = times[2] / max(times[1], 1e-6)

        print(f"\n  Scaling ratio (1000/500): {ratio_1_to_2:.2f} (expected ~2)")
        print(f"  Scaling ratio (2000/1000): {ratio_2_to_3:.2f} (expected ~2)")

        assert 0.5 < ratio_1_to_2 < 4.0, f"Unexpected scaling: {ratio_1_to_2}"
        assert 0.5 < ratio_2_to_3 < 4.0, f"Unexpected scaling: {ratio_2_to_3}"

    @pytest.mark.slow
    def test_parallel_speedup(self):
        """Parallel PEP computation should be faster than serial."""
        library = generate_random_dna_library(40, 20, seed=42)

        evaluator_serial = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=1000,
            seed=42,
        )

        start = time.perf_counter()
        pep_serial, _ = evaluator_serial.compute_pep_matrix(n_jobs=1)
        time_serial = time.perf_counter() - start

        evaluator_parallel = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=1000,
            seed=42,
        )

        start = time.perf_counter()
        pep_parallel, _ = evaluator_parallel.compute_pep_matrix(n_jobs=2)
        time_parallel = time.perf_counter() - start

        print(f"\n  Serial (n_jobs=1): {time_serial:.4f}s")
        print(f"  Parallel (n_jobs=2): {time_parallel:.4f}s")
        print(f"  Speedup: {time_serial / max(time_parallel, 1e-6):.2f}x")

        np.testing.assert_array_equal(pep_serial, pep_parallel)

        assert time_parallel < time_serial * 1.5, \
            f"Parallel slower than expected: {time_parallel:.2f}s vs {time_serial:.2f}s"


# =============================================================================
# Swap Cache Performance Tests
# =============================================================================


class TestSwapCachePerformance:
    """Performance tests for DecodingSwapCache."""

    def test_cache_build_small(self):
        """Cache build for 20 guides should complete in < 0.1 seconds."""
        n_candidates = 20
        n_groups = 5
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=generate_codeword_to_group(group_to_candidates),
        )

        S = np.array([guides[0] for guides in group_to_candidates.values()])

        start = time.perf_counter()
        cache.build_cache(S)
        elapsed = time.perf_counter() - start

        print(f"\n  Cache build (n=20, genes=5): {elapsed:.4f}s")

        assert elapsed < 0.1, f"Cache build too slow: {elapsed:.4f}s > 0.1s"

    def test_cache_build_medium(self):
        """Cache build for 100 guides should complete in < 0.5 seconds."""
        n_candidates = 100
        n_groups = 20
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=generate_codeword_to_group(group_to_candidates),
        )

        S = np.array([guides[0] for guides in group_to_candidates.values()])

        start = time.perf_counter()
        cache.build_cache(S)
        elapsed = time.perf_counter() - start

        print(f"\n  Cache build (n=100, genes=20): {elapsed:.4f}s")

        assert elapsed < 0.5, f"Cache build too slow: {elapsed:.4f}s > 0.5s"

    def test_incremental_update_faster_than_rebuild(self):
        """Incremental cache update should be significantly faster than rebuild."""
        n_candidates = 500
        n_groups = 50
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)
        codeword_to_group = generate_codeword_to_group(group_to_candidates)

        # One shared accessor so the rebuild timing below measures build_cache
        # only, not accessor construction (symmetrization of the PEP matrix).
        accessor = InMemoryPEPAccessor.from_pep_matrix(pep_matrix)

        def fresh_cache():
            return DecodingSwapCache(
                pep_accessor=accessor,
                group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )

        cache = fresh_cache()

        S = np.array([guides[0] for guides in group_to_candidates.values()])
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        remove_idx = int(remove_arr[0])
        add_guide = int(add_arr[0])
        old_guide = S[remove_idx]

        # Time incremental update (100 iterations)
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            cache.update_after_swap(remove_idx, old_guide, add_guide)
            cache.update_after_swap(remove_idx, add_guide, old_guide)
        time_update = time.perf_counter() - start

        # Time full rebuild (100 iterations). build_cache() is a no-op on an
        # already-valid cache (so caches restored via from_shared_state are not
        # rebuilt), so each rebuild runs on a fresh, not-yet-built cache.
        rebuild_caches = [fresh_cache() for _ in range(n_iterations)]
        start = time.perf_counter()
        for rebuild_cache in rebuild_caches:
            rebuild_cache.build_cache(S)
        time_rebuild = time.perf_counter() - start

        speedup = time_rebuild / max(time_update, 1e-9)

        print(f"\n  Incremental update ({n_iterations}x): {time_update:.4f}s")
        print(f"  Full rebuild ({n_iterations}x): {time_rebuild:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

        assert speedup > 2.0, f"Incremental update not fast enough: {speedup:.1f}x < 2x"

    def test_get_deltas_performance(self):
        """Getting deltas from cache should be very fast."""
        n_candidates = 100
        n_groups = 20
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=generate_codeword_to_group(group_to_candidates),
        )

        S = np.array([guides[0] for guides in group_to_candidates.values()])
        cache.build_cache(S)

        # Time 1000 delta retrievals
        n_iterations = 1000
        start = time.perf_counter()
        for _ in range(n_iterations):
            deltas = cache.get_deltas()
        elapsed = time.perf_counter() - start

        time_per_call = elapsed / n_iterations * 1000  # milliseconds

        print(f"\n  get_deltas ({n_iterations}x): {elapsed:.4f}s")
        print(f"  Time per call: {time_per_call:.3f}ms")

        assert time_per_call < 1.0, f"Delta retrieval too slow: {time_per_call:.3f}ms > 1ms"


# =============================================================================
# DUET Optimization Performance Tests
# =============================================================================


class TestDUETPerformance:
    """Performance tests for DUET optimization."""

    def test_duet_iteration_speed(self):
        """Single DUET iteration should be very fast."""
        n_candidates = 100
        n_groups = 20
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        scores = generate_scores(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        optimizer = make_duet_perf(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            scores=scores,
            lambda_=0.5,
            temperature=0.0,
        )

        S = np.array([guides[0] for guides in group_to_candidates.values()])
        optimizer.initialize(S)

        n_iterations = 1000
        start = time.perf_counter()
        for _ in range(n_iterations):
            S = optimizer.step(S)
        elapsed = time.perf_counter() - start

        time_per_iter = elapsed / n_iterations * 1000  # milliseconds

        print(f"\n  DUET step ({n_iterations}x, n=100): {elapsed:.4f}s")
        print(f"  Time per iteration: {time_per_iter:.3f}ms")

        assert time_per_iter < 5.0, f"Iteration too slow: {time_per_iter:.3f}ms > 5ms"

    def test_duet_full_optimization(self):
        """Full DUET optimization should complete in reasonable time."""
        n_candidates = 50
        n_groups = 10
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        scores = generate_scores(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        optimizer = make_duet_perf(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            scores=scores,
            lambda_=0.5,
            temperature=0.0,
            max_iter=1000,
            max_patience=100,
        )

        init = [guides[0] for guides in group_to_candidates.values()]

        start = time.perf_counter()
        optimizer.optimize(init, seed=42)
        elapsed = time.perf_counter() - start

        print(f"\n  Full DUET optimization (n=50, max_iter=1000): {elapsed:.4f}s")

        assert elapsed < 5.0, f"Optimization too slow: {elapsed:.2f}s > 5s"

    def test_duet_scaling_with_library_size(self):
        """DUET iteration should scale reasonably with library size."""
        times = []
        sizes = [25, 50, 100]
        n_iterations = 500

        for n_candidates in sizes:
            n_groups = n_candidates // 5
            candidates_per_group = 5

            pep_matrix = generate_pep_matrix(n_candidates)
            scores = generate_scores(n_candidates)
            group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

            optimizer = make_duet_perf(
                pep_matrix=pep_matrix,
                group_to_candidates=group_to_candidates,
                scores=scores,
                lambda_=0.5,
                temperature=0.0,
            )

            S = np.array([guides[0] for guides in group_to_candidates.values()])
            optimizer.initialize(S)

            start = time.perf_counter()
            for _ in range(n_iterations):
                S = optimizer.step(S)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"\n  DUET {n_iterations} iters (n={n_candidates}): {elapsed:.4f}s")

        ratio_1_to_2 = times[1] / max(times[0], 1e-6)
        ratio_2_to_3 = times[2] / max(times[1], 1e-6)

        print(f"\n  Scaling ratio (50/25): {ratio_1_to_2:.2f}")
        print(f"  Scaling ratio (100/50): {ratio_2_to_3:.2f}")

        assert ratio_1_to_2 < 6.0, f"Scaling too steep: {ratio_1_to_2:.2f}x"
        assert ratio_2_to_3 < 6.0, f"Scaling too steep: {ratio_2_to_3:.2f}x"


# =============================================================================
# Memory Performance Tests
# =============================================================================


class TestMemoryPerformance:
    """Tests for memory usage (basic checks)."""

    def test_pep_matrix_memory_size(self):
        """PEP matrix memory should be predictable (n² floats)."""
        n = 100
        library = generate_random_dna_library(n, 20)

        evaluator = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=500,
            seed=42,
        )

        pep, _ns = evaluator.compute_pep_matrix(n_jobs=1)

        expected_bytes = n * n * 2  # uint16 = 2 bytes
        actual_bytes = pep.nbytes

        print(f"\n  PEP matrix (n={n}): {actual_bytes / 1024:.1f} KB")
        print(f"  Expected: {expected_bytes / 1024:.1f} KB")

        assert actual_bytes == expected_bytes

    def test_swap_cache_reasonable_memory(self):
        """Swap cache should use reasonable memory."""
        n_candidates = 200
        n_groups = 40
        candidates_per_group = n_candidates // n_groups

        pep_matrix = generate_pep_matrix(n_candidates)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=generate_codeword_to_group(group_to_candidates),
        )

        S = np.array([guides[0] for guides in group_to_candidates.values()])
        cache.build_cache(S)

        n_swaps = len(cache.deltas)
        print(f"\n  Cache for n={n_candidates}: {n_swaps} potential swaps")

        expected_swaps = n_groups * (candidates_per_group - 1)
        assert n_swaps == expected_swaps, f"Expected {expected_swaps} swaps, got {n_swaps}"


# =============================================================================
# Regression Guards
# =============================================================================


class TestPerformanceRegression:
    """Tests that guard against performance regressions."""

    def test_pep_matrix_baseline(self):
        """Baseline PEP matrix computation (guards against major regressions)."""
        library = generate_random_dna_library(30, 20, seed=12345)

        evaluator = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=500,
            seed=12345,
        )

        start = time.perf_counter()
        pep, _ns = evaluator.compute_pep_matrix(n_jobs=1)
        elapsed = time.perf_counter() - start

        print(f"\n  Baseline PEP (n=30, samples=500): {elapsed:.4f}s")

        assert elapsed < 10.0, f"Possible performance regression: {elapsed:.2f}s"

        assert pep.shape == (30, 30)
        assert pep.dtype == np.uint16
        np.testing.assert_array_almost_equal(np.diag(pep), np.full(30, _ns))
        assert np.all(pep >= 0) and np.all(pep <= _ns)

    def test_duet_baseline(self):
        """Baseline DUET optimization (guards against major regressions)."""
        n_candidates = 40
        n_groups = 8
        candidates_per_group = 5

        pep_matrix = generate_pep_matrix(n_candidates, seed=12345)
        scores = generate_scores(n_candidates, seed=12345)
        group_to_candidates = generate_group_structure(n_groups, candidates_per_group)

        optimizer = make_duet_perf(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            scores=scores,
            lambda_=0.5,
            temperature=0.0,
            max_iter=500,
            max_patience=100,
        )

        init = [guides[0] for guides in group_to_candidates.values()]

        start = time.perf_counter()
        optimizer.optimize(init, seed=12345)
        elapsed = time.perf_counter() - start

        print(f"\n  Baseline DUET (n=40, max_iter=500): {elapsed:.4f}s")

        assert elapsed < 5.0, f"Possible performance regression: {elapsed:.2f}s"

        assert optimizer.best_S is not None
        assert len(optimizer.best_S) == n_groups
