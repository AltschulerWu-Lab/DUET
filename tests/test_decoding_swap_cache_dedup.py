"""Tests for the multi-group + dedup'd DecodingSwapCache."""
import numpy as np
import pytest

from duet.pareto_optimization import enumerate_within_group_swaps


def test_enumerate_excludes_only_self_not_S():
    """With duplicates-allowed, swaps may add a candidate already in S
    (at a different codeword position)."""
    S = np.array([0, 1, 2], dtype=np.int32)
    codeword_to_group = ["g1", "g2", "g1"]
    group_to_candidates = {"g1": [0, 1, 2], "g2": [0, 1, 2]}
    remove, add = enumerate_within_group_swaps(
        S, codeword_to_group, group_to_candidates,
    )
    # At codeword 0 (group g1, S[0]=0): swaps add 1 and 2 (exclude self 0).
    # At codeword 1 (group g2, S[1]=1): swaps add 0 and 2 (exclude self 1).
    #   Note: 0 is in S at codeword 0, but still enumerated — duplicates allowed.
    # At codeword 2 (group g1, S[2]=2): swaps add 0 and 1 (exclude self 2).
    assert sorted(zip(remove.tolist(), add.tolist())) == [
        (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
    ]
    assert remove.dtype == np.int32
    assert add.dtype == np.int32


def test_enumerate_uses_codeword_group_not_candidate_group():
    """A multi-group candidate's codeword position determines the enumeration set."""
    S = np.array([0], dtype=np.int32)
    codeword_to_group = ["g1"]
    # Candidate 0 is in both g1 and g2; only g1's pool is enumerated at codeword 0.
    group_to_candidates = {"g1": [0, 1, 2], "g2": [0, 3, 4]}
    remove, add = enumerate_within_group_swaps(
        S, codeword_to_group, group_to_candidates,
    )
    assert sorted(zip(remove.tolist(), add.tolist())) == [(0, 1), (0, 2)]


# --- C1 regression tests (DecodingSwapCache _diag_const) ---


def _make_small_decoding_cache(offset: float):
    """Build a small DecodingSwapCache with controllable duplicate_offset.

    4 distinct-sequence candidates; one group; quota 2 -> |S|=2.
    Symmetrized X = M + M^T, with diagonal X(s,s) = 2.0 raw.
    """
    from duet.pep_accessor import InMemoryPEPAccessor
    from duet.pareto_optimization import DecodingSwapCache

    M = np.array([
        [1.0, 0.2, 0.3, 0.1],
        [0.2, 1.0, 0.2, 0.4],
        [0.3, 0.2, 1.0, 0.3],
        [0.1, 0.4, 0.3, 1.0],
    ], dtype=np.float32)
    X = M + M.T
    acc = InMemoryPEPAccessor(X, n_samples=None, duplicate_offset=offset)
    cache = DecodingSwapCache(
        pep_accessor=acc,
        group_to_candidates={"g": [0, 1, 2, 3]},
        codeword_to_group=["g", "g"],
    )
    S = np.array([0, 1], dtype=np.int32)
    cache.build_cache(S)
    return cache, acc, S


@pytest.mark.parametrize("offset", [0.0, 50.0, 200.0, 1000.0])
def test_compute_objective_invariant_under_offset_for_nonduplicate_S(offset):
    """C1 regression: with no duplicates in S, varying duplicate_offset must not
    change compute_objective. The -X(s,s) constant in delta formulas must use
    the offset-aware diagonal (= 2.0 + offset), not hard-coded -2.0."""
    cache, _, S = _make_small_decoding_cache(offset)
    objective = cache.compute_objective(S)

    ref_cache, _, _ = _make_small_decoding_cache(0.0)
    ref_objective = ref_cache.compute_objective(S)

    np.testing.assert_allclose(objective, ref_objective, atol=1e-10)


def test_running_delta_matches_compute_objective_with_offset():
    """After applying a swap and updating the cache, compute_objective must
    equal the value implied by running deltas. Tests _diag_const at the
    Step 5 recompute site of update_after_swap."""
    cache, _, S = _make_small_decoding_cache(200.0)
    initial_obj = cache.compute_objective(S)

    remove_idx = int(cache._codebook_index_to_remove[0])
    new_candidate = int(cache._pool_index_to_add[0])
    old_candidate = int(S[remove_idx])
    delta_before = cache.deltas[0]

    cache.update_after_swap(remove_idx, old_candidate, new_candidate)
    new_S = S.copy()
    new_S[remove_idx] = new_candidate
    new_obj = cache.compute_objective(new_S)

    np.testing.assert_allclose(new_obj - initial_obj, delta_before / len(S), atol=1e-8)


def test_duplicate_in_S_adds_2_plus_offset_to_total_pep():
    """With a duplicate at two codeword positions, total_pep increases by
    exactly (2.0 + offset) compared to a baseline where the off-diagonal
    between the two clean candidates is zero (so no other terms contribute)."""
    from duet.pep_accessor import InMemoryPEPAccessor
    from duet.pareto_optimization import DecodingSwapCache

    offset = 200.0
    # Diagonal X(s, s) = 2.0 raw; all off-diagonals 0 → clean baseline has
    # no cross-term, so the delta to total_pep from a duplicate is exactly
    # (2.0 + offset) per duplicate pair.
    X = np.eye(4, dtype=np.float32) * 2.0
    acc = InMemoryPEPAccessor(X, n_samples=None, duplicate_offset=offset)
    group_to_candidates = {"g": [0, 1, 2, 3]}

    cache_clean = DecodingSwapCache(
        pep_accessor=acc,
        group_to_candidates=group_to_candidates,
        codeword_to_group=["g", "g"],
    )
    cache_clean.build_cache(np.array([0, 1], dtype=np.int32))
    obj_clean = cache_clean.compute_objective(np.array([0, 1], dtype=np.int32))

    cache_dup = DecodingSwapCache(
        pep_accessor=acc,
        group_to_candidates=group_to_candidates,
        codeword_to_group=["g", "g"],
    )
    cache_dup.build_cache(np.array([0, 0], dtype=np.int32))
    obj_dup = cache_dup.compute_objective(np.array([0, 0], dtype=np.int32))

    expected_delta_total_pep = 2.0 + offset
    np.testing.assert_allclose(
        (obj_clean - obj_dup) * 2,  # |S| = 2
        expected_delta_total_pep,
        atol=1e-8,
    )


def test_score_swap_cache_scoped_relabel_does_not_bump_siblings():
    """C4 regression: when a multi-group candidate c is at codeword i and a
    different candidate c' is at codeword j, a swap at codeword i must NOT
    bump deltas at codeword j just because j's swap list contains c."""
    from duet.pareto_optimization import ScoreSwapCache

    scores = np.array([0.0, 1.0, 2.0, 3.0])
    # Two groups; candidate 0 is in both groups (multi-group).
    group_to_candidates = {"g_i": [0, 1], "g_j": [0, 2, 3]}
    codeword_to_group = ["g_i", "g_j"]

    cache = ScoreSwapCache(
        scores=scores,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
    )
    S = np.array([1, 2], dtype=np.int32)        # candidate 0 not at either codeword
    cache.build_cache(S)

    # Find the swap (j=1, c_new=0): it sits at codeword 1, adds candidate 0.
    # Also find any swap (i=0, c_new=1)? No — c != S[0]=1, so (0,1) is excluded.
    # Try the swap (0, c_new=0): at codeword 0, add candidate 0.
    # Apply it; the (1, 0) sibling swap must NOT have its delta bumped.
    pairs = list(zip(
        cache._codebook_index_to_remove.tolist(),
        cache._pool_index_to_add.tolist(),
    ))
    idx_10 = pairs.index((1, 0))
    delta_10_before = cache.deltas[idx_10]

    cache.update_after_swap(remove_idx=0, old_candidate=1, new_candidate=0)

    # Find the (1, 0) entry post-swap: relabel is scoped, so (1, 0) is unchanged.
    pairs_after = list(zip(
        cache._codebook_index_to_remove.tolist(),
        cache._pool_index_to_add.tolist(),
    ))
    idx_10_after = pairs_after.index((1, 0))
    assert cache.deltas[idx_10_after] == delta_10_before, (
        "Sibling delta at codeword 1 was wrongly bumped — second += score_diff "
        "is not scoped to mask_at_i"
    )
