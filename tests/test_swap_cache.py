"""
Tests for DecodingSwapCache, ScoreSwapCache, and DUET algorithm.

This module tests the core correctness invariants of the DUET optimization
algorithm, focusing on:
1. DecodingSwapCache - decode delta computation and incremental updates
2. ScoreSwapCache - score delta computation and incremental updates
3. DUET class - initialization, stepping, and optimization
4. Key invariants like gene constraints and numerical precision
"""

import pytest
import numpy as np
from collections import Counter
from typing import Dict, List

from duet.pareto_optimization import (
    DecodingSwapCache,
    ScoreSwapCache,
    DUET,
    enumerate_within_group_swaps,
    ParetoOptimizer,
)


# =============================================================================
# Helper Functions
# =============================================================================


def slot_groups(S, group_to_candidates):
    """Slot-indexed codeword_to_group for selection S.

    Returns a list of length len(S) whose i-th entry is the group of codeword
    position i, derived from the group containing S[i]. Assumes each candidate
    belongs to exactly one group (OPS-style pools). Swaps stay within a group,
    so the result is the same for any selection reachable from S.
    """
    cand_to_group = {
        c: g for g, cs in group_to_candidates.items() for c in cs
    }
    return [cand_to_group[int(c)] for c in S]


def make_duet(
    pep_matrix,
    group_to_candidates,
    scores,
    init,
    lambda_=0.5,
    duplicate_offset=200.0,
    **kwargs,
):
    """Helper to construct DUET with a decode cache and a score cache.

    Translates lambda_ to weights (docs/adr/0001):
    J = lambda_ * decode + (1 - lambda_) * score, so lambda_=1 is decode-only
    and lambda_=0 is score-only.

    codeword_to_group is slot-indexed and derived from `init` via slot_groups.
    duplicate_offset defaults to 2 * 100.0, the diagonal offset the runner
    uses for the default DuetOptimizerConfig(avoid_duplicates=True,
    duplicate_penalty=100.0). The fixture pools have distinct sequences, so
    no candidate_to_sequence_idx dedup map is needed.
    """
    codeword_to_group = slot_groups(init, group_to_candidates)
    decode_cache = DecodingSwapCache.from_pep_matrix(
        pep_matrix=pep_matrix,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
        duplicate_offset=duplicate_offset,
    )
    score_cache = ScoreSwapCache(
        scores=scores,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
    )
    weights = {"decode": lambda_, "score": 1.0 - lambda_}
    return DUET(
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
        objective_caches=[decode_cache, score_cache],
        weights=weights,
        **kwargs,
    )


def compute_pep_sum_explicit(pep_matrix: np.ndarray, S: np.ndarray, target_idx: int) -> float:
    """
    Compute PEP sum explicitly for a target guide.

    pep_sum(target) = Σ_{s ∈ S} [PEP[s, target] + PEP[target, s]]
    """
    total = 0.0
    for s in S:
        total += pep_matrix[s, target_idx] + pep_matrix[target_idx, s]
    return total


def compute_decode_delta_explicit(
    pep_matrix: np.ndarray,
    S: np.ndarray,
    remove_idx: int,
    old_guide: int,
    new_guide: int,
) -> float:
    """
    Compute decode delta explicitly using the formula:
    delta = pep_sum(old) - pep_sum(new) + E(old,new) + E(new,old) - 2
    """
    # Compute current S (before swap)
    pep_sum_old = compute_pep_sum_explicit(pep_matrix, S, old_guide)
    pep_sum_new = compute_pep_sum_explicit(pep_matrix, S, new_guide)

    e_old_new = pep_matrix[old_guide, new_guide]
    e_new_old = pep_matrix[new_guide, old_guide]

    return pep_sum_old - pep_sum_new + e_old_new + e_new_old - 2


def compute_union_bound_accuracy(pep_matrix: np.ndarray, S: np.ndarray) -> float:
    """
    Compute decode accuracy using union bound approximation.

    P(correct) ≈ 1 - (1/|S|) Σ_{i ∈ S} Σ_{j ∈ S, j ≠ i} PEP[i, j]
    """
    pep_submatrix = pep_matrix[np.ix_(S, S)]
    total_pep = pep_submatrix.sum() - np.trace(pep_submatrix)
    return 1.0 - total_pep / len(S)


def get_gene_counts(S: np.ndarray, candidate_to_group: Dict[int, str]) -> Dict[str, int]:
    """Get count of guides per gene for a selection."""
    genes = [candidate_to_group[int(g)] for g in S]
    return dict(Counter(genes))


# =============================================================================
# Tests for enumerate_within_group_swaps
# =============================================================================


class TestEnumerateWithinGroupSwaps:
    """Tests for the swap enumeration helper."""

    def test_basic_enumeration(self, simple_swap_cache_setup):
        """Enumerates correct number of swaps."""
        setup = simple_swap_cache_setup
        group_to_candidates = setup["group_to_candidates"]
        S = setup["init"].copy()

        remove_arr, add_arr = enumerate_within_group_swaps(
            S, slot_groups(S, group_to_candidates), group_to_candidates
        )
        # S = [0, 2], groups = {A: [0,1], B: [2,3]}
        # Position 0 (guide 0): swap to 1 → 1 swap
        # Position 1 (guide 2): swap to 3 → 1 swap
        assert len(remove_arr) == 2
        assert len(add_arr) == 2

    def test_swaps_within_group(self, simple_swap_cache_setup):
        """All swaps stay within the same group."""
        setup = simple_swap_cache_setup
        S = setup["init"].copy()

        remove_arr, add_arr = enumerate_within_group_swaps(
            S, slot_groups(S, setup["group_to_candidates"]), setup["group_to_candidates"]
        )
        for r, a in zip(remove_arr, add_arr):
            old = S[r]
            assert setup["candidate_to_group"][old] == setup["candidate_to_group"][a]

    def test_no_self_swaps(self, simple_swap_cache_setup):
        """No swap replaces a guide with itself."""
        setup = simple_swap_cache_setup
        S = setup["init"].copy()

        remove_arr, add_arr = enumerate_within_group_swaps(
            S, slot_groups(S, setup["group_to_candidates"]), setup["group_to_candidates"]
        )
        for r, a in zip(remove_arr, add_arr):
            assert S[r] != a

    def test_deterministic_ordering(self, simple_swap_cache_setup):
        """Same inputs produce same ordering."""
        setup = simple_swap_cache_setup
        S = setup["init"].copy()

        r1, a1 = enumerate_within_group_swaps(
            S, slot_groups(S, setup["group_to_candidates"]), setup["group_to_candidates"]
        )
        r2, a2 = enumerate_within_group_swaps(
            S, slot_groups(S, setup["group_to_candidates"]), setup["group_to_candidates"]
        )
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(a1, a2)


# =============================================================================
# Tests for DecodingSwapCache
# =============================================================================


class TestDecodingSwapCacheBuild:
    """Tests for DecodingSwapCache building."""

    def test_build_cache_creates_valid_structure(self, simple_swap_cache_setup):
        """Cache has correct structure after build."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        assert cache._is_valid
        assert len(cache.deltas) > 0
        assert len(cache._codebook_index_to_remove) > 0
        assert len(cache._pool_index_to_add) > 0

        n_swaps = len(cache.deltas)
        assert len(cache._codebook_index_to_remove) == n_swaps
        assert len(cache._pool_index_to_add) == n_swaps

    def test_build_cache_correct_swap_count(self, simple_swap_cache_setup):
        """Cache contains correct number of possible swaps."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        # S = [0, 2]: 1 swap from each group = 2 total
        assert len(cache.deltas) == 2

    def test_build_cache_swaps_within_gene(self, simple_swap_cache_setup):
        """All swaps are within the same gene."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        for r, a in zip(remove_arr, add_arr):
            old = S[r]
            assert setup["candidate_to_group"][old] == setup["candidate_to_group"][a]

    def test_build_cache_no_self_swaps(self, simple_swap_cache_setup):
        """No swap replaces a guide with itself."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        for r, a in zip(remove_arr, add_arr):
            assert S[r] != a


class TestDecodingSwapCacheDecodeDeltas:
    """Tests for decode delta computation."""

    def test_decode_delta_formula(self, simple_swap_cache_setup):
        """Decode deltas match the explicit formula."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        for i, (remove_idx, add_guide) in enumerate(zip(remove_arr, add_arr)):
            old_guide = S[remove_idx]
            expected_delta = compute_decode_delta_explicit(
                setup["pep_matrix"], S, remove_idx, old_guide, add_guide,
            )
            actual_delta = cache.deltas[i]
            assert np.isclose(actual_delta, expected_delta, rtol=1e-6)

    def test_pep_sum_computation(self, simple_swap_cache_setup):
        """Internal PEP sum computation matches explicit calculation."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        for target_idx in range(len(setup["pep_matrix"])):
            expected = compute_pep_sum_explicit(setup["pep_matrix"], S, target_idx)
            actual = cache._pep_sum_pool[target_idx]
            assert np.isclose(actual, expected, rtol=1e-6)


class TestDecodingSwapCacheUpdate:
    """Tests for incremental cache updates after swaps."""

    def test_update_after_swap_decode_deltas(self, simple_swap_cache_setup):
        """Decode deltas update correctly after swap."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_idx = 0
        old_guide = S[remove_idx]
        new_guide = 1

        S[remove_idx] = new_guide
        cache.update_after_swap(remove_idx, old_guide, new_guide)

        fresh_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )
        fresh_cache.build_cache(S)

        assert np.allclose(cache.deltas, fresh_cache.deltas, rtol=1e-6)

    def test_incremental_equals_rebuild_multiple_swaps(self, simple_swap_cache_setup):
        """After multiple swaps, incremental update equals fresh rebuild."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        swaps_to_perform = [
            (0, 0, 1),
            (1, 2, 3),
            (0, 1, 0),
        ]

        for remove_idx, old_guide, new_guide in swaps_to_perform:
            S[remove_idx] = new_guide
            cache.update_after_swap(remove_idx, old_guide, new_guide)

        fresh_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )
        fresh_cache.build_cache(S)

        assert np.allclose(cache.deltas, fresh_cache.deltas, rtol=1e-6)

    def test_pep_sum_updates_correctly(self, simple_swap_cache_setup):
        """PEP sum updates correctly after swap."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_idx = 0
        old_guide = S[remove_idx]
        new_guide = 1

        S[remove_idx] = new_guide
        cache.update_after_swap(remove_idx, old_guide, new_guide)

        for target_idx in range(len(setup["pep_matrix"])):
            expected = compute_pep_sum_explicit(setup["pep_matrix"], S, target_idx)
            actual = cache._pep_sum_pool[target_idx]
            assert np.isclose(actual, expected, rtol=1e-6)


class TestDecodingSwapCacheObjective:
    """Tests for compute_objective."""

    def test_compute_objective_matches_formula(self, simple_swap_cache_setup):
        """compute_objective matches explicit union bound formula."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        expected = compute_union_bound_accuracy(setup["pep_matrix"], S)
        actual = cache.compute_objective(S)
        assert np.isclose(actual, expected)

    def test_cache_invalid_before_build(self, simple_swap_cache_setup):
        """Cache raises error if accessed before build."""
        setup = simple_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        with pytest.raises(ValueError, match="Cache is not valid"):
            cache.get_deltas()


# =============================================================================
# Tests for ScoreSwapCache
# =============================================================================


class TestScoreSwapCacheBuild:
    """Tests for ScoreSwapCache building."""

    def test_build_cache_creates_valid_structure(self, simple_swap_cache_setup):
        """Cache has correct structure after build."""
        setup = simple_swap_cache_setup
        cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        assert cache._is_valid
        assert len(cache.deltas) == 2  # 1 swap per group


class TestScoreSwapCacheActivityDeltas:
    """Tests for activity delta computation."""

    def test_activity_delta_computation(self, simple_swap_cache_setup):
        """Activity deltas are computed correctly."""
        setup = simple_swap_cache_setup
        cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        for i, (remove_idx, add_guide) in enumerate(zip(remove_arr, add_arr)):
            old_guide = S[remove_idx]
            expected_delta = setup["scores"][add_guide] - setup["scores"][old_guide]
            actual_delta = cache.deltas[i]
            assert np.isclose(actual_delta, expected_delta)

    def test_activity_delta_sign(self, simple_swap_cache_setup):
        """Activity delta is negative when swapping to lower-score guide."""
        setup = simple_swap_cache_setup
        # Scores are [0.9, 0.7, 0.8, 0.6]
        # S = [0, 2] means we have guides with scores [0.9, 0.8]
        # Swapping 0->1: delta = 0.7 - 0.9 = -0.2
        # Swapping 2->3: delta = 0.6 - 0.8 = -0.2
        cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        for delta in cache.deltas:
            assert delta < 0, "Expected negative activity delta"


class TestScoreSwapCacheUpdate:
    """Tests for incremental ScoreSwapCache updates."""

    def test_update_after_swap_score_deltas(self, simple_swap_cache_setup):
        """Score deltas update correctly after swap."""
        setup = simple_swap_cache_setup
        cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        remove_idx = 0
        old_guide = S[remove_idx]
        new_guide = 1

        S[remove_idx] = new_guide
        cache.update_after_swap(remove_idx, old_guide, new_guide)

        fresh_cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )
        fresh_cache.build_cache(S)

        assert np.allclose(cache.deltas, fresh_cache.deltas)

    def test_incremental_equals_rebuild_multiple_swaps(self, simple_swap_cache_setup):
        """After multiple swaps, incremental update equals fresh rebuild."""
        setup = simple_swap_cache_setup
        cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        swaps_to_perform = [
            (0, 0, 1),
            (1, 2, 3),
            (0, 1, 0),
        ]

        for remove_idx, old_guide, new_guide in swaps_to_perform:
            S[remove_idx] = new_guide
            cache.update_after_swap(remove_idx, old_guide, new_guide)

        fresh_cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )
        fresh_cache.build_cache(S)

        assert np.allclose(cache.deltas, fresh_cache.deltas)


class TestScoreSwapCacheObjective:
    """Tests for ScoreSwapCache.compute_objective."""

    def test_compute_objective(self, simple_swap_cache_setup):
        """compute_objective returns mean score."""
        setup = simple_swap_cache_setup
        cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        expected = float(np.mean(setup["scores"][S]))
        actual = cache.compute_objective(S)
        assert np.isclose(actual, expected)


# =============================================================================
# Tests for swap ordering consistency across caches
# =============================================================================


class TestSwapOrderingConsistency:
    """Verify that DecodingSwapCache and ScoreSwapCache produce identical swap orderings."""

    def test_same_swap_ordering(self, simple_swap_cache_setup):
        """Both caches enumerate identical swaps."""
        setup = simple_swap_cache_setup

        decode_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )
        score_cache = ScoreSwapCache(
            scores=setup["scores"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        decode_cache.build_cache(S)
        score_cache.build_cache(S)

        np.testing.assert_array_equal(
            decode_cache._codebook_index_to_remove,
            score_cache._codebook_index_to_remove,
        )
        np.testing.assert_array_equal(
            decode_cache._pool_index_to_add,
            score_cache._pool_index_to_add,
        )


# =============================================================================
# Tests for DUET Class
# =============================================================================


class TestDUETInitialization:
    """Tests for DUET initialization."""

    def test_initialize_sets_running_objectives(self, simple_duet_setup):
        """Initialize sets running objectives dict."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        objectives = optimizer.evaluate(S)
        assert "decode" in objectives
        assert "score" in objectives

        expected_accuracy = compute_union_bound_accuracy(setup["pep_matrix"], S)
        expected_activity = np.mean(setup["scores"][S])

        assert np.isclose(objectives["decode"], expected_accuracy)
        assert np.isclose(objectives["score"], expected_activity)

    def test_initialize_sets_best_solution(self, simple_duet_setup):
        """Initialize sets best_S and best_objectives."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        assert optimizer.best_S is not None
        assert optimizer.best_objectives is not None
        np.testing.assert_array_equal(optimizer.best_S, S)


class TestDUETStep:
    """Tests for DUET step function."""

    def test_step_returns_modified_S(self, simple_duet_setup):
        """Step returns S with exactly one element changed."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=0.0,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        S_before = S.copy()
        S_after = optimizer.step(S)

        differences = np.sum(S_before != S_after)
        assert differences == 1

    def test_step_respects_gene_constraints(self, simple_duet_setup):
        """Step maintains gene constraints."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=0.0,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        candidate_to_group = {g: gene for gene, guides in setup["group_to_candidates"].items() for g in guides}
        counts_before = get_gene_counts(S, candidate_to_group)

        S_after = optimizer.step(S)

        counts_after = get_gene_counts(S_after, candidate_to_group)
        assert counts_before == counts_after

    def test_step_updates_running_objectives(self, simple_duet_setup):
        """Step updates running objectives incrementally."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=0.0,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        optimizer.step(S)

        objectives = optimizer.evaluate(S)
        assert objectives["decode"] is not None
        assert objectives["score"] is not None

    def test_greedy_selection_chooses_max(self, simple_duet_setup):
        """With temperature=0, step selects maximum score swap."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=0.0,
        )

        np.random.seed(42)
        S = setup["init"].copy()
        optimizer.initialize(S)

        # Get combined deltas before step
        combined = optimizer._compute_combined_deltas()
        max_score = np.max(combined)
        max_indices = np.where(combined == max_score)[0]

        remove_arr = optimizer._caches[0]._codebook_index_to_remove
        add_arr = optimizer._caches[0]._pool_index_to_add

        S_before = S.copy()
        S_after = optimizer.step(S)

        # Find which swap was performed
        for i in range(len(remove_arr)):
            if S_before[remove_arr[i]] != S_after[remove_arr[i]]:
                assert i in max_indices
                break


class TestDUETEvaluate:
    """Tests for DUET evaluation functions."""

    def test_evaluate_returns_running_values(self, simple_duet_setup):
        """Evaluate returns the running objective dict."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        objectives = optimizer.evaluate(S)
        assert isinstance(objectives, dict)
        assert "decode" in objectives
        assert "score" in objectives

    def test_compute_approximate_matches_formula(self, simple_duet_setup):
        """compute_approximate_objectives matches direct computation."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        S = setup["init"].copy()
        optimizer.initialize(S)

        approx = optimizer.compute_approximate_objectives(S)

        expected_decode = compute_union_bound_accuracy(setup["pep_matrix"], S)
        expected_activity = np.mean(setup["scores"][S])

        assert np.isclose(approx["decode"], expected_decode)
        assert np.isclose(approx["score"], expected_activity)

    def test_running_vs_recomputed_after_steps(self, simple_duet_setup):
        """Running values stay close to recomputed after multiple steps."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
        )

        np.random.seed(42)
        S = setup["init"].copy()
        optimizer.initialize(S)

        for _ in range(10):
            S = optimizer.step(S)

        running = optimizer.evaluate(S)
        recomputed = optimizer.compute_approximate_objectives(S)

        assert np.isclose(running["decode"], recomputed["decode"], rtol=1e-6)
        assert np.isclose(running["score"], recomputed["score"], rtol=1e-6)


class TestDUETOptimize:
    """Tests for DUET optimization loop."""

    def test_optimize_improves_objective(self, simple_duet_setup):
        """Optimization improves or maintains the scalarized objective."""
        setup = simple_duet_setup
        lambda_ = 0.5

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=lambda_,
            temperature=0.0,
            max_iter=100,
            max_patience=50,
            track_history=True,
        )

        init = list(setup["init"])
        optimizer.optimize(init, seed=42)

        initial_decode = compute_union_bound_accuracy(setup["pep_matrix"], np.array(init))
        initial_activity = np.mean(setup["scores"][np.array(init)])
        initial_scalarized = lambda_ * initial_decode + (1 - lambda_) * initial_activity

        assert optimizer.best_scalarized_objective >= initial_scalarized - 1e-6

    def test_optimize_respects_max_iter(self, simple_duet_setup):
        """Optimization stops at max_iter."""
        setup = simple_duet_setup
        max_iter = 50

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
            max_iter=max_iter,
            max_patience=1000,
            track_history=True,
        )

        init = list(setup["init"])
        optimizer.optimize(init, seed=42)

        assert len(optimizer.history['iteration']) <= max_iter + 1

    def test_optimize_respects_patience(self, simple_duet_setup):
        """Optimization stops after max_patience non-improving steps."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=0.0,
            max_iter=10000,
            max_patience=10,
            track_history=True,
        )

        init = list(setup["init"])
        optimizer.optimize(init, seed=42)

        assert len(optimizer.history['iteration']) < 10000

    def test_optimize_tracks_best_solution(self, simple_duet_setup):
        """best_S is the actual best solution seen."""
        setup = simple_duet_setup
        lambda_ = 0.5

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=lambda_,
            temperature=1.0,
            max_iter=100,
            max_patience=50,
            track_history=True,
            store_all_solutions=True,
        )

        init = list(setup["init"])
        optimizer.optimize(init, seed=42)

        best_decode = compute_union_bound_accuracy(setup["pep_matrix"], optimizer.best_S)
        best_activity = np.mean(setup["scores"][optimizer.best_S])
        best_scalarized = lambda_ * best_decode + (1 - lambda_) * best_activity

        assert np.isclose(best_scalarized, optimizer.best_scalarized_objective, rtol=1e-5)

    def test_optimize_reproducible_with_seed(self, simple_duet_setup):
        """Same seed produces identical results."""
        setup = simple_duet_setup

        optimizer1 = make_duet(
            pep_matrix=setup["pep_matrix"].copy(),
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"].copy(),
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
            max_iter=50,
            max_patience=25,
        )
        optimizer1.optimize(list(setup["init"]), seed=12345)

        optimizer2 = make_duet(
            pep_matrix=setup["pep_matrix"].copy(),
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"].copy(),
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
            max_iter=50,
            max_patience=25,
        )
        optimizer2.optimize(list(setup["init"]), seed=12345)

        np.testing.assert_array_equal(optimizer1.best_S, optimizer2.best_S)
        assert optimizer1.best_scalarized_objective == optimizer2.best_scalarized_objective

    def test_solution_valid_throughout(self, simple_duet_setup):
        """All solutions throughout optimization are valid."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
            max_iter=50,
            max_patience=25,
            track_history=True,
            store_all_solutions=True,
        )

        init = list(setup["init"])
        optimizer.optimize(init, seed=42)

        for solution in optimizer.all_solutions:
            assert optimizer.validate_solution(solution, np.array(init))


class TestDUETNumericalPrecision:
    """Tests for numerical precision and stability."""

    def test_no_numerical_drift_many_steps(self, simple_duet_setup):
        """Running values don't drift significantly from recomputed after many steps."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
            max_iter=500,
            max_patience=500,
        )

        np.random.seed(42)
        S = np.array(setup["init"])
        optimizer.initialize(S)

        for _ in range(200):
            S = optimizer.step(S)

        running = optimizer.evaluate(S)
        recomputed = optimizer.compute_approximate_objectives(S)

        assert abs(running["decode"] - recomputed["decode"]) < 1e-9
        assert abs(running["score"] - recomputed["score"]) < 1e-9

    def test_no_nan_or_inf(self, simple_duet_setup):
        """No NaN or Inf values during optimization."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
            temperature=1.0,
            max_iter=100,
            max_patience=50,
            track_history=True,
        )

        optimizer.optimize(list(setup["init"]), seed=42)

        for key, values in optimizer.history.items():
            for v in values:
                assert not np.isnan(v), f"NaN found in history[{key}]"
                assert not np.isinf(v), f"Inf found in history[{key}]"

        assert not np.isnan(optimizer.best_scalarized_objective)
        assert not np.isinf(optimizer.best_scalarized_objective)


class TestDUETEdgeCases:
    """Tests for edge cases."""

    def test_single_guide_per_gene_no_swaps(self):
        """With one guide per gene, no swaps are possible."""
        pep_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
        group_to_candidates = {"geneA": [0], "geneB": [1]}
        scores = np.array([0.9, 0.8])
        init = [0, 1]

        optimizer = make_duet(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            scores=scores,
            init=init,
            lambda_=0.5,
            max_iter=100,
            max_patience=10,
        )

        optimizer.optimize(init, seed=42)

        np.testing.assert_array_equal(optimizer.best_S, np.array(init))

    def test_lambda_one_pure_decode(self, simple_duet_setup):
        """With lambda=1 (weight on decode only), optimization focuses on decode accuracy."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=1.0,
            temperature=0.0,
            max_iter=100,
            max_patience=50,
        )

        optimizer.optimize(list(setup["init"]), seed=42)

        objectives = optimizer.evaluate(optimizer.best_S)
        assert objectives["decode"] is not None

    def test_lambda_zero_pure_activity(self, simple_duet_setup):
        """With lambda=0, optimization focuses purely on activity."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.0,
            temperature=0.0,
            max_iter=100,
            max_patience=50,
        )

        optimizer.optimize(list(setup["init"]), seed=42)

        expected = np.array([0, 2])
        np.testing.assert_array_equal(sorted(optimizer.best_S), sorted(expected))


def _deltas_by_swap(cache):
    """Build a dict from (position, candidate) → delta for order-invariant comparison.

    After incremental updates, alternatives within a position may be in a
    different array order than a fresh build_cache. This helper allows comparing
    delta values by their logical identity (which position, which alternative)
    rather than array index.
    """
    remove_arr, add_arr = cache.get_swaps()
    return {
        (int(r), int(a)): float(d)
        for r, a, d in zip(remove_arr, add_arr, cache.deltas)
    }


class TestIncrementalDeltaCorrectness:
    """Tests for the incremental delta update formula.

    These verify that the O(N) incremental update produces identical results
    to a full from-scratch build_cache after each swap.
    """

    def test_incremental_deltas_match_full_recompute(self):
        """After each swap, incremental deltas match from-scratch build_cache.

        Uses an asymmetric PEP matrix to exercise X = M + M^T correctness.
        Comparison is by (position, candidate) pairs since the array ordering
        within each position may differ after incremental updates.
        """
        np.random.seed(42)
        N = 20
        n_groups = 5
        cands_per_group = N // n_groups

        # Asymmetric PEP matrix
        M = np.random.uniform(0.05, 0.3, size=(N, N))
        np.fill_diagonal(M, 1.0)

        group_to_candidates = {
            f"g{i}": list(range(i * cands_per_group, (i + 1) * cands_per_group))
            for i in range(n_groups)
        }

        S = np.array([g[0] for g in group_to_candidates.values()])
        codeword_to_group = slot_groups(S, group_to_candidates)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=M, group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        cache.build_cache(S)

        for iteration in range(50):
            remove_arr, add_arr = cache.get_swaps()
            if len(remove_arr) == 0:
                break
            idx = np.random.randint(len(remove_arr))
            remove_idx = int(remove_arr[idx])
            add_candidate = int(add_arr[idx])
            old_candidate = S[remove_idx]

            S[remove_idx] = add_candidate
            cache.update_after_swap(remove_idx, old_candidate, add_candidate)

            # Fresh rebuild for comparison
            fresh = DecodingSwapCache.from_pep_matrix(
                pep_matrix=M, group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )
            fresh.build_cache(S)

            incr_map = _deltas_by_swap(cache)
            fresh_map = _deltas_by_swap(fresh)
            assert incr_map.keys() == fresh_map.keys(), \
                f"Swap set mismatch at iteration {iteration}"
            for key in incr_map:
                np.testing.assert_allclose(
                    incr_map[key], fresh_map[key], rtol=1e-6,
                    err_msg=f"Delta mismatch for swap {key} at iteration {iteration}",
                )

    def test_incremental_pep_sum_matches_full_recompute(self):
        """After each swap, pep_sum_pool matches from-scratch build_cache."""
        np.random.seed(123)
        N = 15
        n_groups = 3
        cands_per_group = N // n_groups

        M = np.random.uniform(0.05, 0.3, size=(N, N))
        np.fill_diagonal(M, 1.0)

        group_to_candidates = {
            f"g{i}": list(range(i * cands_per_group, (i + 1) * cands_per_group))
            for i in range(n_groups)
        }

        S = np.array([g[0] for g in group_to_candidates.values()])
        codeword_to_group = slot_groups(S, group_to_candidates)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=M, group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        cache.build_cache(S)

        for iteration in range(50):
            remove_arr, add_arr = cache.get_swaps()
            if len(remove_arr) == 0:
                break
            idx = np.random.randint(len(remove_arr))
            remove_idx = int(remove_arr[idx])
            add_candidate = int(add_arr[idx])
            old_candidate = S[remove_idx]

            S[remove_idx] = add_candidate
            cache.update_after_swap(remove_idx, old_candidate, add_candidate)

            fresh = DecodingSwapCache.from_pep_matrix(
                pep_matrix=M, group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )
            fresh.build_cache(S)

            np.testing.assert_allclose(
                cache._pep_sum_pool, fresh._pep_sum_pool, rtol=1e-6,
                err_msg=f"pep_sum_pool mismatch at iteration {iteration}",
            )

    def test_category_c_formula_exact(self):
        """For a single swap, verify Category C correction matches delta difference."""
        np.random.seed(789)
        N = 12
        n_groups = 3
        cands_per_group = N // n_groups

        M = np.random.uniform(0.05, 0.3, size=(N, N))
        np.fill_diagonal(M, 1.0)
        X = M + M.T

        group_to_candidates = {
            f"g{i}": list(range(i * cands_per_group, (i + 1) * cands_per_group))
            for i in range(n_groups)
        }

        S = np.array([g[0] for g in group_to_candidates.values()])
        codeword_to_group = slot_groups(S, group_to_candidates)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=M, group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        cache.build_cache(S)

        deltas_before = cache.deltas.copy()
        remove_arr = cache._codebook_index_to_remove.copy()
        add_arr = cache._pool_index_to_add.copy()
        S_before = S.copy()

        # Pick the first available swap
        swap_idx = 0
        remove_idx = int(remove_arr[swap_idx])
        add_candidate = int(add_arr[swap_idx])
        old_candidate = S[remove_idx]

        # Compute delta_alpha manually
        x_row_old = X[old_candidate, :]
        x_row_new = X[add_candidate, :]
        delta_alpha = x_row_new - x_row_old

        S[remove_idx] = add_candidate
        cache.update_after_swap(remove_idx, old_candidate, add_candidate)

        # For Category C swaps (positions != remove_idx), verify the formula:
        # delta_new = delta_old + delta_alpha[S_old[i]] - delta_alpha[c_j]
        mask_other = (remove_arr != remove_idx)
        if np.any(mask_other):
            old_cands_c = S_before[remove_arr[mask_other]]
            new_cands_c = add_arr[mask_other]
            expected_deltas_c = (
                deltas_before[mask_other]
                + delta_alpha[old_cands_c]
                - delta_alpha[new_cands_c]
            )
            np.testing.assert_allclose(
                cache.deltas[mask_other], expected_deltas_c, rtol=1e-6,
            )

    def test_step_ordering_bookkeeping(self):
        """Verify that position p's alternatives after a swap correctly include
        old_candidate and exclude new_candidate."""
        N = 12
        n_groups = 3
        cands_per_group = N // n_groups

        np.random.seed(111)
        M = np.random.uniform(0.05, 0.3, size=(N, N))
        np.fill_diagonal(M, 1.0)

        group_to_candidates = {
            f"g{i}": list(range(i * cands_per_group, (i + 1) * cands_per_group))
            for i in range(n_groups)
        }

        S = np.array([g[0] for g in group_to_candidates.values()])
        codeword_to_group = slot_groups(S, group_to_candidates)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=M, group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        swap_idx = 0
        remove_idx = int(remove_arr[swap_idx])
        add_candidate = int(add_arr[swap_idx])
        old_candidate = S[remove_idx]

        S[remove_idx] = add_candidate
        cache.update_after_swap(remove_idx, old_candidate, add_candidate)

        # After swap, position p's alternatives should include old_candidate
        # and NOT include new_candidate
        mask_p = (cache._codebook_index_to_remove == remove_idx)
        alternatives_p = set(cache._pool_index_to_add[mask_p].tolist())
        assert old_candidate in alternatives_p, \
            "old_candidate should be available as alternative after swap"
        assert add_candidate not in alternatives_p, \
            "new_candidate (now in codebook) should not be an alternative"


class TestBackendEquivalence:
    """Tests verifying in-memory and mmap backends produce identical results."""

    def test_inmemory_vs_mmap_build_cache(self, asymmetric_pep_matrix):
        """Build cache with both backends, verify identical pep_sum_pool and deltas."""
        import tempfile
        from duet.pep_accessor import InMemoryPEPAccessor, create_symmetric_mmap

        M = asymmetric_pep_matrix
        group_to_candidates = {"gA": [0, 1], "gB": [2, 3]}
        codeword_to_group = ["gA", "gB"]  # slot-indexed, for S = [0, 2]

        # In-memory
        inmem_acc = InMemoryPEPAccessor.from_pep_matrix(M)
        inmem_cache = DecodingSwapCache(
            pep_accessor=inmem_acc,
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )

        S = np.array([0, 2])
        inmem_cache.build_cache(S)

        # Mmap
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)
            mmap_cache = DecodingSwapCache(
                pep_accessor=mmap_acc,
                group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )
            mmap_cache.build_cache(S)

            np.testing.assert_allclose(
                inmem_cache._pep_sum_pool, mmap_cache._pep_sum_pool, rtol=1e-6,
            )
            np.testing.assert_allclose(
                inmem_cache.deltas, mmap_cache.deltas, rtol=1e-6,
            )

    def test_inmemory_vs_mmap_after_swaps(self, asymmetric_pep_matrix):
        """Run identical swaps on both backends, verify identical state."""
        import tempfile
        from duet.pep_accessor import InMemoryPEPAccessor, create_symmetric_mmap

        M = asymmetric_pep_matrix
        group_to_candidates = {"gA": [0, 1], "gB": [2, 3]}
        codeword_to_group = ["gA", "gB"]  # slot-indexed, for S = [0, 2]

        inmem_acc = InMemoryPEPAccessor.from_pep_matrix(M)
        inmem_cache = DecodingSwapCache(
            pep_accessor=inmem_acc,
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )

        S_inmem = np.array([0, 2])
        inmem_cache.build_cache(S_inmem)

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)
            mmap_cache = DecodingSwapCache(
                pep_accessor=mmap_acc,
                group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )

            S_mmap = np.array([0, 2])
            mmap_cache.build_cache(S_mmap)

            # Perform identical swaps
            swaps = [(0, 0, 1), (1, 2, 3), (0, 1, 0)]
            for remove_idx, old_c, new_c in swaps:
                S_inmem[remove_idx] = new_c
                S_mmap[remove_idx] = new_c
                inmem_cache.update_after_swap(remove_idx, old_c, new_c)
                mmap_cache.update_after_swap(remove_idx, old_c, new_c)

                np.testing.assert_allclose(
                    inmem_cache._pep_sum_pool, mmap_cache._pep_sum_pool, rtol=1e-6,
                )
                np.testing.assert_allclose(
                    inmem_cache.deltas, mmap_cache.deltas, rtol=1e-6,
                )

    def test_inmemory_vs_mmap_optimization(self, asymmetric_pep_matrix):
        """Run full DUET.optimize with both backends, verify identical results."""
        import tempfile
        from duet.pep_accessor import InMemoryPEPAccessor, create_symmetric_mmap

        M = asymmetric_pep_matrix
        group_to_candidates = {"gA": [0, 1], "gB": [2, 3]}
        codeword_to_group = ["gA", "gB"]  # slot-indexed, for S = [0, 2]
        scores = np.array([0.9, 0.7, 0.8, 0.6])
        init = [0, 2]

        # In-memory run
        inmem_acc = InMemoryPEPAccessor.from_pep_matrix(M)
        inmem_dc = DecodingSwapCache(
            pep_accessor=inmem_acc,
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        inmem_sc = ScoreSwapCache(
            scores=scores, group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        inmem_opt = DUET(
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
            objective_caches=[inmem_dc, inmem_sc],
            weights={"decode": 0.5, "score": 0.5},
            temperature=0.0, max_iter=50, max_patience=20,
        )
        inmem_opt.optimize(list(init), seed=42)

        # Mmap run
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)
            mmap_dc = DecodingSwapCache(
                pep_accessor=mmap_acc,
                group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )
            mmap_sc = ScoreSwapCache(
                scores=scores, group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )
            mmap_opt = DUET(
                group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
                objective_caches=[mmap_dc, mmap_sc],
                weights={"decode": 0.5, "score": 0.5},
                temperature=0.0, max_iter=50, max_patience=20,
            )
            mmap_opt.optimize(list(init), seed=42)

            np.testing.assert_array_equal(inmem_opt.best_S, mmap_opt.best_S)
            for key in inmem_opt.best_objectives:
                np.testing.assert_allclose(
                    inmem_opt.best_objectives[key],
                    mmap_opt.best_objectives[key],
                    rtol=1e-5,
                )


class TestDUETValidateSolution:
    """Tests for solution validation."""

    def test_validate_solution_correct(self, simple_duet_setup):
        """Valid solution passes validation."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        init = np.array(setup["init"])
        valid_solution = np.array([1, 3])

        assert optimizer.validate_solution(valid_solution, init)

    def test_validate_solution_wrong_genes(self, simple_duet_setup):
        """Invalid solution fails validation."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        init = np.array(setup["init"])
        invalid_solution = np.array([0, 1])

        assert not optimizer.validate_solution(invalid_solution, init)

    def test_validate_solution_invalid_index(self, simple_duet_setup):
        """Solution with invalid index fails validation."""
        setup = simple_duet_setup

        optimizer = make_duet(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            scores=setup["scores"],
            init=setup["init"],
            lambda_=0.5,
        )

        init = np.array(setup["init"])
        invalid_solution = np.array([0, 99])

        assert not optimizer.validate_solution(invalid_solution, init)


# =============================================================================
# Tests for Uint16 Count Path Through DecodingSwapCache
# =============================================================================


class TestUint16SwapCache:
    """Tests that uint16 count path produces identical results to float path."""

    def test_uint16_cache_matches_float_cache(self, asymmetric_pep_matrix, asymmetric_count_matrix, simple_group_to_candidates):
        """DecodingSwapCache from uint16 counts matches float path."""
        M_counts, n_samples = asymmetric_count_matrix
        S = np.array([0, 2])
        codeword_to_group = slot_groups(S, simple_group_to_candidates)

        # Float path (legacy)
        cache_float = DecodingSwapCache.from_pep_matrix(
            pep_matrix=asymmetric_pep_matrix,
            group_to_candidates=simple_group_to_candidates,
            codeword_to_group=codeword_to_group,
            duplicate_offset=0.0,
        )

        # Uint16 path
        cache_uint16 = DecodingSwapCache.from_pep_matrix(
            pep_matrix=M_counts,
            group_to_candidates=simple_group_to_candidates,
            codeword_to_group=codeword_to_group,
            duplicate_offset=0.0,
            n_samples=n_samples,
        )

        cache_float.build_cache(S)
        cache_uint16.build_cache(S)

        obj_float = cache_float.compute_objective(S)
        obj_uint16 = cache_uint16.compute_objective(S)
        np.testing.assert_allclose(obj_uint16, obj_float, rtol=1e-5)


class TestMultiRepIncrementalUpdate:
    """Regression tests for multi-representative groups.

    When a group has multiple representatives in the codebook (guides_per_gene > 1),
    a swap at one position changes the deltas at same-group sibling positions
    (the uniform Category C correction), while the swap relabel is scoped to the
    swapped position only. These tests verify that the incremental update
    handles this correctly.
    """

    def test_incremental_equals_rebuild_single_swap(self, multi_rep_swap_cache_setup):
        """After one swap, incremental deltas match fresh rebuild (multi-rep)."""
        setup = multi_rep_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        # Swap position 0 (candidate 0) for candidate 2 (sibling at position 1
        # is also in groupA, so its deltas take the Category C correction)
        remove_idx = 0
        old_guide = S[remove_idx]
        new_guide = 2
        S[remove_idx] = new_guide
        cache.update_after_swap(remove_idx, old_guide, new_guide)

        fresh = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )
        fresh.build_cache(S)

        incr_map = _deltas_by_swap(cache)
        fresh_map = _deltas_by_swap(fresh)
        assert incr_map.keys() == fresh_map.keys(), "Swap set mismatch"
        for key in incr_map:
            np.testing.assert_allclose(
                incr_map[key], fresh_map[key], atol=1e-10,
                err_msg=f"Delta mismatch for swap {key}",
            )

    def test_incremental_equals_rebuild_multi_swap(self, multi_rep_swap_cache_setup):
        """After multiple swaps (including sibling swaps), incremental matches rebuild."""
        setup = multi_rep_swap_cache_setup
        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=setup["pep_matrix"],
            group_to_candidates=setup["group_to_candidates"],
            codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
        )

        S = setup["init"].copy()
        cache.build_cache(S)

        # Sequence of swaps that exercises sibling interactions:
        # S starts as [0, 1, 4, 5]
        swaps = [
            (0, 0, 2),   # groupA: pos 0: 0->2, sibling at pos 1 affected
            (1, 1, 3),   # groupA: pos 1: 1->3, sibling at pos 0 affected
            (2, 4, 6),   # groupB: pos 2: 4->6, sibling at pos 3 affected
            (0, 2, 0),   # groupA: pos 0: 2->0, swap back
            (3, 5, 7),   # groupB: pos 3: 5->7, sibling at pos 2 affected
        ]

        for remove_idx, old_guide, new_guide in swaps:
            S[remove_idx] = new_guide
            cache.update_after_swap(remove_idx, old_guide, new_guide)

            fresh = DecodingSwapCache.from_pep_matrix(
                pep_matrix=setup["pep_matrix"],
                group_to_candidates=setup["group_to_candidates"],
                codeword_to_group=slot_groups(setup["init"], setup["group_to_candidates"]),
            )
            fresh.build_cache(S)

            incr_map = _deltas_by_swap(cache)
            fresh_map = _deltas_by_swap(fresh)
            assert incr_map.keys() == fresh_map.keys(), \
                f"Swap set mismatch after swap ({remove_idx}, {old_guide}->{new_guide})"
            for key in incr_map:
                np.testing.assert_allclose(
                    incr_map[key], fresh_map[key], atol=1e-10,
                    err_msg=(
                        f"Delta mismatch for swap {key} after "
                        f"swap ({remove_idx}, {old_guide}->{new_guide})"
                    ),
                )

    def test_incremental_equals_rebuild_random_swaps(self):
        """Stress test: random swaps with multi-rep groups match rebuild."""
        np.random.seed(2026)
        N = 24
        n_groups = 4
        cands_per_group = N // n_groups  # 6 candidates per group
        reps_per_group = 3

        M = np.random.uniform(0.05, 0.3, size=(N, N))
        np.fill_diagonal(M, 1.0)

        group_to_candidates = {
            f"g{i}": list(range(i * cands_per_group, (i + 1) * cands_per_group))
            for i in range(n_groups)
        }

        # Initial selection: first `reps_per_group` from each group
        S = np.array([
            g[j]
            for g in group_to_candidates.values()
            for j in range(reps_per_group)
        ])
        codeword_to_group = slot_groups(S, group_to_candidates)

        cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=M, group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
        )
        cache.build_cache(S)

        for iteration in range(100):
            remove_arr, add_arr = cache.get_swaps()
            if len(remove_arr) == 0:
                break
            idx = np.random.randint(len(remove_arr))
            remove_idx = int(remove_arr[idx])
            add_candidate = int(add_arr[idx])
            old_candidate = S[remove_idx]

            S[remove_idx] = add_candidate
            cache.update_after_swap(remove_idx, old_candidate, add_candidate)

            fresh = DecodingSwapCache.from_pep_matrix(
                pep_matrix=M, group_to_candidates=group_to_candidates,
                codeword_to_group=codeword_to_group,
            )
            fresh.build_cache(S)

            incr_map = _deltas_by_swap(cache)
            fresh_map = _deltas_by_swap(fresh)
            assert incr_map.keys() == fresh_map.keys(), \
                f"Swap set mismatch at iteration {iteration}"
            for key in incr_map:
                np.testing.assert_allclose(
                    incr_map[key], fresh_map[key], atol=1e-10,
                    err_msg=f"Delta mismatch for swap {key} at iteration {iteration}",
                )
