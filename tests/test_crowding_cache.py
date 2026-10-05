"""Unit tests for CrowdingSwapCache and binary alphabet support."""

import tracemalloc

import numpy as np
import pytest

from duet.candidate_pool import CandidatePool
from duet.pareto_optimization import (
    DUET,
    CrowdingSwapCache,
    DecodingSwapCache,
    ScoreSwapCache,
    enumerate_within_group_swaps,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_problem():
    """Small problem: 5 codewords (pool), 3 positions in S, 4 rounds.

    Groups: {A: [0,1], B: [2,3], C: [4]}
    S = [0, 2, 4]  (one from each group)
    """
    codewords = np.array(
        [
            [1, 0, 1, 0],  # candidate 0 (group A)
            [0, 1, 1, 0],  # candidate 1 (group A)
            [1, 1, 0, 0],  # candidate 2 (group B)
            [0, 0, 1, 1],  # candidate 3 (group B)
            [1, 0, 0, 1],  # candidate 4 (group C)
        ],
        dtype=float,
    )
    expression = np.array([3.0, 2.0, 1.0])  # per position in S
    group_to_candidates = {"A": [0, 1], "B": [2, 3], "C": [4]}
    S = np.array([0, 2, 4])
    return codewords, expression, group_to_candidates, S


def _compute_cost_brute(codewords, expression, S):
    """Brute-force C(S) = sum_r TEV[r]^2."""
    TEV = np.zeros(codewords.shape[1])
    for pos, cand in enumerate(S):
        TEV += expression[pos] * codewords[cand]
    return np.sum(TEV**2)


def _compute_O_crowd_brute(codewords, expression, S, C_anchor):
    """Brute-force O_C(S) = 1 - C(S) / C(S_0).

    Uses an externally provided anchor (the realised C frozen at
    build_cache time for the initial selection).
    """
    C = _compute_cost_brute(codewords, expression, S)
    return 1.0 - C / C_anchor


def _c2g_from_S(S: np.ndarray, group_to_candidates: dict[str, list[int]]) -> list[str]:
    """Derive codeword_to_group[i] = group containing S[i].

    Inverse of group_to_candidates restricted to S. Used to wire the new
    CrowdingSwapCache(codeword_to_group=...) required arg from test fixtures
    that only carry (S, group_to_candidates).
    """
    cand_to_group = {c: g for g, cs in group_to_candidates.items() for c in cs}
    return [cand_to_group[int(c)] for c in np.asarray(S)]


def _c2g_for_pool_size(
    n_pool: int, group_to_candidates: dict[str, list[int]]
) -> list[str]:
    """codeword_to_group when |S| == n_pool and S = arange(n_pool) -- only used by
    a couple of memory/structural tests that build a cache but never call build_cache.
    For those, codeword_to_group is wired but never consumed; any consistent labeling
    works. We pick the natural candidate-indexed labels for clarity.
    """
    cand_to_group = {c: g for g, cs in group_to_candidates.items() for c in cs}
    return [cand_to_group[i] for i in range(n_pool)]


def _make_constant_hw_codewords(
    n: int, R: int, hw: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate n binary codewords of length R, each with exactly `hw` ones."""
    cw = np.zeros((n, R), dtype=float)
    for i in range(n):
        positions = rng.choice(R, size=hw, replace=False)
        cw[i, positions] = 1.0
    return cw


def _make_valid_random_codewords(
    n: int, R: int, hw_choices: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Generate n binary codewords of length R with HW drawn from hw_choices.

    Guarantees HW(c_i) > 0 for every codeword, matching the
    CrowdingSwapCache precondition. (HW < R is a side-effect of choosing
    hw_choices < R but is not required by the precondition.)
    """
    cw = np.zeros((n, R), dtype=float)
    for i in range(n):
        hw = int(rng.choice(hw_choices))
        positions = rng.choice(R, size=hw, replace=False)
        cw[i, positions] = 1.0
    return cw


# =============================================================================
# Tests
# =============================================================================


class TestTEVComputation:
    def test_tev_matches_manual(self, small_problem):
        codewords, expression, group_to_candidates, S = small_problem
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        # Manual: TEV[r] = 3*[1,0,1,0] + 2*[1,1,0,0] + 1*[1,0,0,1]
        #       = [3+2+1, 0+2+0, 3+0+0, 0+0+1] = [6, 2, 3, 1]
        expected_TEV = np.array([6.0, 2.0, 3.0, 1.0])
        np.testing.assert_array_almost_equal(cache._TEV, expected_TEV)


class TestComputeObjective:
    def test_objective_at_init_is_zero(self, small_problem):
        """O_C(S_0) = 0 exactly by construction (anchor is C(S_0))."""
        codewords, expression, group_to_candidates, S = small_problem
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)
        assert cache.compute_objective(S) == pytest.approx(0.0, abs=1e-12)

    def test_objective_matches_brute_force_after_swap(self, small_problem):
        """compute_objective(S_swap) = 1 - C(S_swap)/C(S_0)."""
        codewords, expression, group_to_candidates, S = small_problem
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        S_swap = S.copy()
        S_swap[0] = 1  # swap group-A candidate 0 -> 1

        expected = _compute_O_crowd_brute(
            codewords, expression, S_swap, cache._C_anchor
        )
        assert cache.compute_objective(S_swap) == pytest.approx(expected, abs=1e-12)


class TestBruteForceDeltaVerification:
    def test_deltas_match_brute_force(self, small_problem):
        """Cache deltas equal |S| * (O_C(S_swap) - O_C(S)) to 1e-10.

        Sum-form delta convention: cache.deltas[k] = |S| * delta_objective.
        See ObjectiveSwapCache docstring.
        """
        codewords, expression, group_to_candidates, S = small_problem
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)
        C_anchor = cache._C_anchor

        remove_arr, add_arr = cache.get_swaps()
        deltas = cache.get_deltas()

        # O_C(S) = 0 at init, so expected_delta == |S| * O_C(S_swap).
        for k in range(len(deltas)):
            pos = remove_arr[k]
            new_cand = add_arr[k]
            S_swap = S.copy()
            S_swap[pos] = new_cand

            expected_delta = len(S) * _compute_O_crowd_brute(
                codewords, expression, S_swap, C_anchor
            )
            assert pytest.approx(deltas[k], abs=1e-10) == expected_delta, (
                f"Swap {k}: pos={pos}, old={S[pos]}, new={new_cand}; "
                f"expected delta={expected_delta}, got {deltas[k]}"
            )


class TestIncrementalUpdate:
    def test_tev_and_beta_match_after_swap(self, small_problem):
        codewords, expression, group_to_candidates, S = small_problem
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        # Perform a swap: position 0, candidate 0 -> candidate 1
        remove_idx = 0
        old_candidate = S[remove_idx]
        new_candidate = 1
        S[remove_idx] = new_candidate
        cache.update_after_swap(remove_idx, old_candidate, new_candidate)

        # Rebuild from scratch on the new S to compare
        cache2 = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache2.build_cache(S)

        np.testing.assert_array_almost_equal(cache._TEV, cache2._TEV)
        np.testing.assert_array_almost_equal(cache._beta, cache2._beta)

    def test_deltas_match_after_swap(self, small_problem):
        """After one swap, cache deltas equal brute-force O_C swap deltas."""
        codewords, expression, group_to_candidates, S = small_problem
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)
        C_anchor = cache._C_anchor  # frozen at build_cache(S_0)

        # Perform a swap: position 1, candidate 2 -> candidate 3
        remove_idx = 1
        old_candidate = S[remove_idx]
        new_candidate = 3
        S[remove_idx] = new_candidate
        cache.update_after_swap(remove_idx, old_candidate, new_candidate)

        # The anchor stays frozen at C(S_0); deltas are O_C(S_swap) - O_C(S_current).
        O_current = _compute_O_crowd_brute(
            codewords, expression, S, C_anchor
        )
        remove_arr, add_arr = cache.get_swaps()
        deltas = cache.get_deltas()

        for k in range(len(deltas)):
            pos = remove_arr[k]
            swap_cand = add_arr[k]
            S_swap = S.copy()
            S_swap[pos] = swap_cand
            O_swap = _compute_O_crowd_brute(
                codewords, expression, S_swap, C_anchor
            )
            # Sum-form delta: cache.deltas[k] = |S| * (O_swap - O_current).
            expected_delta = len(S) * (O_swap - O_current)
            assert pytest.approx(deltas[k], abs=1e-10) == expected_delta


class TestPreconditions:
    """build_cache enforces the math preconditions of the C(S_0)-anchored score.

    C(S_0) > 0 is guaranteed by at least one position with e_i > 0 AND
    HW(c_{S[i]}) > 0; zero-expression positions are crowding-neutral but
    legal. Negative expression remains rejected. R >= 2 is retained as a
    sanity guard (R = 1 collapses the score to a function of M alone).
    """

    def test_negative_expression_rejected(self):
        codewords = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        group_to_candidates = {"A": [0, 1], "B": [2, 3]}
        S = np.array([0, 2])

        cache = CrowdingSwapCache(
            codewords, np.array([-1.0, 5.0]), group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        with pytest.raises(ValueError, match="e_i >= 0"):
            cache.build_cache(S)

    def test_all_zero_expression_rejected(self):
        codewords = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        group_to_candidates = {"A": [0, 1], "B": [2, 3]}
        S = np.array([0, 2])

        cache = CrowdingSwapCache(
            codewords, np.array([0.0, 0.0]), group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        with pytest.raises(ValueError, match="e_i > 0 AND HW"):
            cache.build_cache(S)

    def test_mixed_zero_expression_accepted(self):
        # At least one position has e_i > 0 AND HW(c_{S[i]}) > 0 -> C(S_0) > 0.
        # Zero-expression positions are crowding-neutral; build_cache must succeed.
        codewords = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        group_to_candidates = {"A": [0, 1], "B": [2, 3]}
        S = np.array([0, 2])

        cache = CrowdingSwapCache(
            codewords, np.array([0.0, 5.0]), group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)
        # C_anchor > 0 because position 1 (e=5) selects codeword [1,1,0] with HW>0.
        assert cache._C_anchor > 0.0
        # Per-swap delta at the zero-expression position must be 0 (e_i = 0
        # zeroes both the linear and quadratic terms in the closed form).
        e_per_swap = np.asarray(cache._expression)[cache._codebook_index_to_remove]
        assert np.all(cache.deltas[e_per_swap == 0.0] == 0.0)

    def test_all_zero_codeword_rejected(self):
        codewords = np.array(
            [[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float
        )
        expression = np.array([3.0, 2.0])
        group_to_candidates = {"A": [0, 1], "B": [2, 3]}
        S = np.array([0, 2])  # selects the all-zero codeword

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        with pytest.raises(ValueError, match="HW.*> 0"):
            cache.build_cache(S)

    def test_R_less_than_2_rejected(self):
        codewords = np.array([[1], [0], [1], [0]], dtype=float)  # R = 1
        expression = np.array([1.0, 1.0])
        group_to_candidates = {"A": [0, 1], "B": [2, 3]}
        S = np.array([0, 2])
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        with pytest.raises(ValueError, match="R >= 2"):
            cache.build_cache(S)


class TestDUETIntegration:
    def test_three_objective_duet_runs(self):
        """Smoke test: DUET with decode + score + crowding runs without error."""
        rng = np.random.default_rng(42)
        pool_size = 10
        n_rounds = 4
        n_groups = 3
        cands_per_group = pool_size // n_groups

        # Random binary codewords with valid HW for the CrowdingSwapCache
        # precondition (HW > 0).
        codewords = _make_valid_random_codewords(
            pool_size, n_rounds, hw_choices=(1, 2, 3), rng=rng,
        )

        # Groups
        group_to_candidates = {}
        for g in range(n_groups):
            start = g * cands_per_group
            end = start + cands_per_group
            if g == n_groups - 1:
                end = pool_size
            group_to_candidates[f"G{g}"] = list(range(start, end))

        # Initial selection: one per group
        S = np.array([cs[0] for cs in group_to_candidates.values()])
        n_positions = len(S)

        expression = rng.uniform(1.0, 10.0, size=n_positions)
        scores = rng.uniform(0.0, 1.0, size=pool_size)

        # Symmetric PEP matrix with zeros on diagonal
        pep_matrix = rng.uniform(0.0, 0.3, size=(pool_size, pool_size))
        pep_matrix = (pep_matrix + pep_matrix.T) / 2
        np.fill_diagonal(pep_matrix, 0.0)

        c2g = _c2g_from_S(S, group_to_candidates)
        decode_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        score_cache = ScoreSwapCache(
            scores=scores,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        crowding_cache = CrowdingSwapCache(
            codewords=codewords,
            expression=expression,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )

        weights = {"decode": 0.5, "score": 0.3, "crowding": 0.2}

        optimizer = DUET(
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
            objective_caches=[decode_cache, score_cache, crowding_cache],
            weights=weights,
            temperature=1.0,
            max_iter=50,
            max_patience=50,
            track_history=True,
        )
        optimizer.optimize(list(S), seed=42)

        # History should have all 3 objectives
        assert "decode" in optimizer.history
        assert "score" in optimizer.history
        assert "crowding" in optimizer.history
        assert "scalarized_objective" in optimizer.history
        assert len(optimizer.history["iteration"]) > 1

    def test_crowding_objective_improves(self):
        """Crowding-heavy DUET should never make O_crowd worse than O_crowd(S_0) = 0."""
        rng = np.random.default_rng(123)
        pool_size = 8
        n_rounds = 4

        codewords = _make_valid_random_codewords(
            pool_size, n_rounds, hw_choices=(1, 2, 3), rng=rng,
        )
        group_to_candidates = {
            "A": [0, 1, 2, 3],
            "B": [4, 5, 6, 7],
        }
        S = np.array([0, 4])
        expression = np.array([5.0, 3.0])

        # Dummy PEP matrix (not used for crowding-only, but needed for decode cache)
        pep_matrix = np.zeros((pool_size, pool_size))

        c2g = _c2g_from_S(S, group_to_candidates)
        decode_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        crowding_cache = CrowdingSwapCache(
            codewords=codewords,
            expression=expression,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        score_cache = ScoreSwapCache(
            scores=np.zeros(pool_size),
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )

        # Heavily weight crowding
        weights = {"decode": 0.01, "score": 0.01, "crowding": 0.98}

        optimizer = DUET(
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
            objective_caches=[decode_cache, score_cache, crowding_cache],
            weights=weights,
            temperature=0.0,
            max_iter=100,
            max_patience=100,
            track_history=True,
        )
        optimizer.optimize(list(S), seed=123)

        # Crowding should not have gotten worse
        crowding_values = optimizer.history["crowding"]
        assert crowding_values[-1] >= crowding_values[0]


class TestPairwiseDecomposition:
    def test_cost_matches_pairwise_form(self, small_problem):
        """Verify C(S) = sum_{g,h} e_g * e_h * <c_g, c_h>."""
        codewords, expression, group_to_candidates, S = small_problem

        # Direct TEV computation
        C_direct = _compute_cost_brute(codewords, expression, S)

        # Pairwise decomposition
        C_pairwise = 0.0
        for i, ci in enumerate(S):
            for j, cj in enumerate(S):
                ip = np.dot(codewords[ci], codewords[cj])
                C_pairwise += expression[i] * expression[j] * ip

        assert pytest.approx(C_direct) == C_pairwise


class TestBinaryAlphabet:
    """Test CandidatePool.get_sequences_as_array(alphabet_size=2)."""

    def test_binary_encoding_shape_and_dtype(self):
        """Verify shape (N, L) and dtype int8."""
        sequences = ["1100", "1010", "0110"]
        pool = CandidatePool.from_dataframe(
            _make_single_group_df(sequences, n_select=2),
        )
        arr = pool.get_sequences_as_array(alphabet_size=2)
        assert arr.shape == (3, 4)
        assert arr.dtype == np.int8

    def test_binary_encoding_values(self):
        """Verify correct 0/1 encoding."""
        sequences = ["1100", "0011", "1001"]
        pool = CandidatePool.from_dataframe(
            _make_single_group_df(sequences, n_select=2),
        )
        arr = pool.get_sequences_as_array(alphabet_size=2)
        expected = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=np.int8)
        np.testing.assert_array_equal(arr, expected)


class TestMERFISHCrowdingIntegration:
    """MERFISH-like single-group setup with CrowdingSwapCache."""

    def test_single_group_deltas_match_brute_force(self):
        """Binary codewords in a single group, verify O_C deltas vs. brute force."""
        sequences = ["1100", "1010", "0110", "0011", "1001"]
        pool = CandidatePool.from_dataframe(
            _make_single_group_df(sequences, n_select=3),
        )
        codewords = pool.get_sequences_as_array(alphabet_size=2).astype(np.float64)
        expression = np.array([10.0, 5.0, 2.0])
        S = np.array([0, 1, 2])

        cache = CrowdingSwapCache(
            codewords=codewords,
            expression=expression,
            group_to_candidates=pool.group_to_candidates,
            codeword_to_group=_c2g_from_S(S, pool.group_to_candidates),
        )
        cache.build_cache(S)
        C_anchor = cache._C_anchor

        remove_arr, add_arr = cache.get_swaps()
        deltas = cache.get_deltas()

        # O_C(S_0) = 0 at init, so expected_delta == |S| * O_C(S_swap)
        # (sum-form delta convention; see ObjectiveSwapCache docstring).
        for k in range(len(deltas)):
            pos = remove_arr[k]
            new_cand = add_arr[k]
            S_swap = S.copy()
            S_swap[pos] = new_cand
            expected_delta = len(S) * _compute_O_crowd_brute(
                codewords, expression, S_swap, C_anchor
            )
            assert pytest.approx(deltas[k], abs=1e-10) == expected_delta


def _make_single_group_df(sequences, n_select):
    """Helper: build a DataFrame for CandidatePool.from_dataframe."""
    import pandas as pd

    return pd.DataFrame({
        "Group": ["MERFISH"] * len(sequences),
        "Sequence": sequences,
        "Score": [1.0] * len(sequences),
        "Quota": [n_select] * len(sequences),
    })


class TestAnchoredCObjective:
    """Direct tests of the C(S_0)-anchored O_C score (Candidate C)."""

    def _achievable_uniform_fixture(self):
        """Fixture with a known uniform-TEV target reachable by swap.

        S_0 = [4, 5, 6, 7] (HW = 2,2,2,3) has non-uniform TEV [4,2,2,1];
        S_uniform = [0, 1, 2, 3] (one-hot codewords) has TEV [1,1,1,1].
        Expression is uniform so the uniform-TEV target lands exactly.
        """
        codewords = np.array(
            [
                [1, 0, 0, 0],  # 0
                [0, 1, 0, 0],  # 1
                [0, 0, 1, 0],  # 2
                [0, 0, 0, 1],  # 3
                [1, 1, 0, 0],  # 4
                [1, 0, 1, 0],  # 5
                [1, 0, 0, 1],  # 6
                [1, 1, 1, 0],  # 7
            ],
            dtype=float,
        )
        expression = np.array([1.0, 1.0, 1.0, 1.0])
        group_to_candidates = {"A": [0, 4], "B": [1, 5], "C": [2, 6], "D": [3, 7]}
        S0 = np.array([4, 5, 6, 7])
        S_uniform = np.array([0, 1, 2, 3])
        return codewords, expression, group_to_candidates, S0, S_uniform

    def test_O_crowd_at_lower_C_selection_matches_formula(self):
        codewords, expression, group_to_candidates, S0, S_uniform = (
            self._achievable_uniform_fixture()
        )
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S0, group_to_candidates),
        )
        cache.build_cache(S0)

        # S0 has TEV = [4, 2, 2, 1] -> C(S_0) = 16 + 4 + 4 + 1 = 25.
        # S_uniform has TEV = [1, 1, 1, 1] -> C(S_uniform) = 4.
        # O_C(S_uniform) = 1 - 4/25 = 0.84.
        TEV_uniform = (expression[:, None] * codewords[S_uniform]).sum(axis=0)
        assert np.allclose(TEV_uniform, TEV_uniform[0])
        assert cache.compute_objective(S_uniform) == pytest.approx(
            1.0 - 4.0 / 25.0, abs=1e-12
        )

    def test_O_crowd_at_init_is_zero_across_pool_scales(self):
        """O_crowd(S_0) = 0 holds exactly across a range of problem sizes."""
        rng = np.random.default_rng(20260520)
        for S_size in (20, 60, 140):
            n_pool = 4 * S_size
            R = 16
            codewords = _make_valid_random_codewords(
                n_pool, R, hw_choices=(3, 4, 5), rng=rng,
            )
            expression = rng.uniform(100.0, 1000.0, size=S_size)
            group_to_candidates = {
                str(i): list(range(i * 4, (i + 1) * 4)) for i in range(S_size)
            }
            S0 = np.array([4 * i for i in range(S_size)])

            cache = CrowdingSwapCache(
                codewords, expression, group_to_candidates,
                _c2g_from_S(S0, group_to_candidates),
            )
            cache.build_cache(S0)
            assert cache.compute_objective(S0) == pytest.approx(0.0, abs=1e-12), (
                f"O_crowd(S_0) != 0 at |S|={S_size}"
            )

    def test_C_anchor_frozen_across_swaps(self):
        """The anchor C(S_0) does not change as swaps are applied."""
        rng = np.random.default_rng(20260520)
        n_pool, R, S_size = 80, 16, 20
        codewords = _make_valid_random_codewords(
            n_pool, R, hw_choices=(3, 4, 5), rng=rng,
        )
        expression = rng.uniform(100.0, 1000.0, size=S_size)
        group_to_candidates = {"all": list(range(n_pool))}
        S = rng.choice(n_pool, size=S_size, replace=False).copy()

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)
        frozen_anchor = cache._C_anchor

        for _ in range(8):
            remove_arr, add_arr = cache.get_swaps()
            k = 0
            remove_idx = remove_arr[k]
            new_cand = add_arr[k]
            old_cand = S[remove_idx]
            S[remove_idx] = new_cand
            cache.update_after_swap(remove_idx, old_cand, new_cand)
            assert cache._C_anchor == frozen_anchor


class TestIPElimination:
    """Post-Fix-1 structural + Candidate-C brute-force parity at realistic scale.

    The IP-matrix elimination property (Fix 1) lives in
    test_cache_does_not_allocate_ip_matrix and test_cache_memory_bytes_subquadratic.
    The two parity tests verify that the cache's deltas (now C(S_0)-anchored
    O_C swap deltas) match a direct brute-force computation to 1e-12 -- both
    at build time and after several incremental swap updates.
    """

    @staticmethod
    def _brute_force_O_crowd_deltas(codewords, expression, S, remove_arr, add_arr, C_anchor):
        """Brute-force sum-form ΔO_C for each enumerated swap.

        Returns |S| * (O_C(S_swap) - O_C(S)) for each (pos, cand) pair in
        (remove_arr, add_arr); the |S| factor matches the sum-form delta
        convention documented on ObjectiveSwapCache. O(num_swaps * |S| * R)
        -- only used for test assertions.
        """
        C_current = _compute_cost_brute(codewords, expression, S)
        O_current = 1.0 - C_current / C_anchor
        deltas = np.empty(len(remove_arr))
        S_size = len(S)
        for k, (pos, new_cand) in enumerate(zip(remove_arr, add_arr)):
            S_swap = S.copy()
            S_swap[pos] = new_cand
            C_swap = _compute_cost_brute(codewords, expression, S_swap)
            O_swap = 1.0 - C_swap / C_anchor
            deltas[k] = S_size * (O_swap - O_current)
        return deltas

    def test_initial_deltas_match_C_brute_force(self):
        rng = np.random.default_rng(20260514)
        n_pool, R, S_size = 60, 16, 12
        codewords = _make_constant_hw_codewords(n_pool, R, hw=4, rng=rng)
        expression = rng.uniform(100.0, 1000.0, size=S_size)
        group_to_candidates = {"all": list(range(n_pool))}
        S = rng.choice(n_pool, size=S_size, replace=False)

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        expected = self._brute_force_O_crowd_deltas(
            codewords, expression, S, remove_arr, add_arr, cache._C_anchor,
        )
        np.testing.assert_allclose(cache.get_deltas(), expected, atol=1e-12)

    def test_deltas_match_C_brute_force_after_five_swaps(self):
        rng = np.random.default_rng(20260514)
        n_pool, R, S_size = 60, 16, 12
        codewords = _make_constant_hw_codewords(n_pool, R, hw=4, rng=rng)
        expression = rng.uniform(100.0, 1000.0, size=S_size)
        group_to_candidates = {"all": list(range(n_pool))}
        S = rng.choice(n_pool, size=S_size, replace=False).copy()

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)
        C_anchor = cache._C_anchor  # frozen at S_0

        for i in range(5):
            remove_arr, add_arr = cache.get_swaps()
            k = 0  # deterministic: first available swap
            remove_idx = remove_arr[k]
            new_cand = add_arr[k]
            old_cand = S[remove_idx]
            S[remove_idx] = new_cand
            cache.update_after_swap(remove_idx, old_cand, new_cand)

            remove_arr, add_arr = cache.get_swaps()
            expected = self._brute_force_O_crowd_deltas(
                codewords, expression, S, remove_arr, add_arr, C_anchor,
            )
            np.testing.assert_allclose(
                cache.get_deltas(), expected, atol=1e-12,
                err_msg=f"Mismatch at swap iteration {i}",
            )

    def test_cache_does_not_allocate_ip_matrix(self):
        """Fix 1: CrowdingSwapCache must not maintain an N×N IP attribute."""
        codewords = np.array(
            [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=float
        )
        expression = np.array([1.0])
        group_to_candidates = {"A": [0, 1, 2]}
        # build_cache is not called here; codeword_to_group is wired but unused.
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates, ["A"],
        )
        assert not hasattr(cache, "_IP"), (
            "Fix 1 requires removing the N×N IP matrix from CrowdingSwapCache."
        )

    def test_cache_memory_bytes_subquadratic(self):
        """Cache state size must scale as O(N*R + N), not O(N^2).

        Without Fix 1, _IP would add N*N*8 bytes; at N=5000 that is 200 MB
        and dominates every other allocation in the cache.
        """
        rng = np.random.default_rng(0)
        n_pool, R = 5000, 16
        codewords = _make_constant_hw_codewords(n_pool, R, hw=4, rng=rng)
        expression = np.array([1.0, 1.0])
        group_to_candidates = {"all": list(range(n_pool))}
        # build_cache is not called here; codeword_to_group is wired but unused.
        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates, ["all", "all"],
        )

        cache_bytes = sum(
            v.nbytes for v in vars(cache).values() if isinstance(v, np.ndarray)
        )

        # Allowance: codewords (N*R*8) + HW_per_cand (N*8) + small slack.
        # The IP matrix would add N*N*8 = 200 MB at N=5000, so a 2x bound on
        # the codeword footprint cleanly excludes it.
        max_allowed = 2 * (n_pool * R * 8 + n_pool * 8)
        assert cache_bytes < max_allowed, (
            f"Cache state at N={n_pool}, R={R} is {cache_bytes / 1e6:.1f} MB; "
            f"max allowed {max_allowed / 1e6:.1f} MB. The pre-Fix-1 IP matrix "
            f"would add {n_pool * n_pool * 8 / 1e6:.1f} MB."
        )


class TestComputeCrowdingDeltasMemory:
    """Bounds peak transient memory of `_compute_crowding_deltas`.

    Regression guard for the full-array (num_swaps,) gather + arithmetic
    pileup that blew the per-call transient peak to ~9.3 GiB at
    zhang_2023_scale (N=100k, |S|=1147, R=32). At the smaller N=50k, |S|=500
    used here, pre-BLAS-slab peak was ~14-19 GiB, BLAS-slab-only peak is
    ~2 GiB, and chunked peak is ~500 MiB; the 800 MiB bound cleanly
    separates the chunked variant from the BLAS-slab-only variant.

    tracemalloc tracks Python-managed allocations (numpy buffers via
    PyDataMem_NEW). It does NOT track BLAS scratch from libopenblas / MKL,
    which lives outside Python's heap. That's fine here -- the numpy
    intermediates are the regression we are guarding against.
    """

    def test_update_after_swap_peak_below_800mib_at_N50k(self):
        """Bounds peak transient memory of `_compute_crowding_deltas`.

        At N=50k, |S|=500, R=32, single-group (num_swaps = 500 × 49500 =
        24.75M), the chunked variant's expected components are:

            BLAS slab (|S|, N) float64       = 500 × 50_000 × 8   ≈ 200 MiB
            deltas_out (num_swaps,) float64  = 24.75M × 8         ≈ 198 MiB
            per-chunk arithmetic working set = 7 × 1M × 8         ≈  56 MiB

        Predicted peak ~500 MiB; the bound is set to 800 MiB to leave
        comfortable headroom for numpy internal scratch but to fail
        loudly if anyone reintroduces a full-array (num_swaps,) gather.

        Pre-chunking (BLAS-slab-only): peak ~2.0 GiB.
        Pre-BLAS-slab fix: peak ~14-19 GiB.

        tracemalloc tracks Python-managed numpy allocations
        (PyDataMem_NEW). BLAS scratch from libopenblas/MKL lives outside
        Python's heap and is not counted — that's fine here because we
        guard against the numpy-side gather regression, not BLAS scratch.
        """
        rng = np.random.default_rng(20260523)
        N = 50_000
        R = 32
        S_size = 500

        codewords = _make_constant_hw_codewords(N, R, hw=4, rng=rng)
        expression = rng.uniform(100.0, 1000.0, size=S_size)
        group_to_candidates = {"all": list(range(N))}
        S = rng.choice(N, size=S_size, replace=False)

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        remove_arr, add_arr = cache.get_swaps()
        remove_idx = int(remove_arr[0])
        new_candidate = int(add_arr[0])
        old_candidate = int(cache._S_indices[remove_idx])

        tracemalloc.start()
        try:
            cache.update_after_swap(remove_idx, old_candidate, new_candidate)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_mib = peak / 1024**2
        assert peak < 800 * 1024**2, (
            f"_compute_crowding_deltas peak tracemalloc allocation "
            f"{peak_mib:.1f} MiB exceeds 800 MiB bound at N={N}, "
            f"|S|={S_size}, R={R}. Likely regression to a full-array "
            f"(num_swaps,) gather pattern; expected chunked variant peak "
            f"~500 MiB."
        )
        print(f"\ntracemalloc peak: {peak_mib:.1f} MiB")


class TestCrowdingChunking:
    """Verify chunk_size kwarg and chunked output bit-equivalence."""

    def _build_problem(self, N: int, R: int, S_size: int):
        rng = np.random.default_rng(20260523)
        codewords = _make_constant_hw_codewords(N, R, hw=4, rng=rng)
        expression = rng.uniform(100.0, 1000.0, size=S_size)
        group_to_candidates = {"all": list(range(N))}
        S = rng.choice(N, size=S_size, replace=False)
        return codewords, expression, group_to_candidates, S

    def test_default_chunk_size(self):
        """chunk_size=None uses CrowdingSwapCache._DEFAULT_CHUNK_SIZE."""
        cw, exp, g2c, _ = self._build_problem(N=100, R=8, S_size=10)
        cache = CrowdingSwapCache(cw, exp, g2c, ["all"] * 10)
        assert cache._chunk_size == CrowdingSwapCache._DEFAULT_CHUNK_SIZE
        assert CrowdingSwapCache._DEFAULT_CHUNK_SIZE == 1_000_000

    def test_explicit_chunk_size_accepted(self):
        cw, exp, g2c, _ = self._build_problem(N=100, R=8, S_size=10)
        cache = CrowdingSwapCache(cw, exp, g2c, ["all"] * 10, chunk_size=42)
        assert cache._chunk_size == 42

    def test_zero_or_negative_chunk_size_rejected(self):
        cw, exp, g2c, _ = self._build_problem(N=100, R=8, S_size=10)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            CrowdingSwapCache(cw, exp, g2c, ["all"] * 10, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            CrowdingSwapCache(cw, exp, g2c, ["all"] * 10, chunk_size=-1)

    def test_chunked_matches_full_array_output(self):
        """At N=2000, |S|=50 single-group, several chunk_size values
        all produce bit-identical deltas (atol=0)."""
        cw, exp, g2c, S = self._build_problem(N=2000, R=16, S_size=50)
        c2g = _c2g_from_S(S, g2c)
        cache_ref = CrowdingSwapCache(cw, exp, g2c, c2g, chunk_size=10**9)  # one chunk
        cache_ref.build_cache(S)
        ref = cache_ref.deltas.copy()

        for cs in [13, 100, 1000, 10_000, 10**9]:
            cache = CrowdingSwapCache(cw, exp, g2c, c2g, chunk_size=cs)
            cache.build_cache(S)
            np.testing.assert_array_equal(
                cache.deltas, ref,
                err_msg=f"chunk_size={cs} produced different deltas vs ref",
            )

    def test_chunk_size_larger_than_num_swaps(self):
        """chunk_size > num_swaps falls through to a single chunk; no edge errors."""
        cw, exp, g2c, S = self._build_problem(N=100, R=8, S_size=10)
        cache = CrowdingSwapCache(
            cw, exp, g2c, _c2g_from_S(S, g2c), chunk_size=10**9,
        )
        cache.build_cache(S)
        assert len(cache.deltas) > 0


class TestSwapArrayDtypes:
    """Swap-index arrays are int32 (halves their persistent footprint vs int64).

    Indices fit comfortably: codebook_index_to_remove ∈ [0, |S|),
    pool_index_to_add ∈ [0, N), S_indices ∈ [0, N). At zhang_2023_scale
    N ≈ 10^5, |S| ≈ 10^3, both << 2^31.

    int32 is asserted (not just `np.issubdtype(..., np.integer)`) because
    we care about the byte width specifically — the whole point of this
    change is to halve the bytes per (num_swaps,) index array.
    """

    def test_enumerate_within_group_swaps_returns_int32(self):
        S = np.array([0, 2, 4], dtype=np.int32)
        g2c = {"A": [0, 1, 3], "B": [2, 5], "C": [4, 6]}
        c2g = {c: g for g, cs in g2c.items() for c in cs}
        rem, add = enumerate_within_group_swaps(S, c2g, g2c)
        assert rem.dtype == np.int32, f"got {rem.dtype}"
        assert add.dtype == np.int32, f"got {add.dtype}"

    def test_enumerate_empty_arrays_are_int32(self):
        """Empty result still uses int32 dtype (no inference fallback)."""
        S = np.array([0], dtype=np.int32)
        g2c = {"A": [0]}  # singleton group → no swaps
        c2g = {0: "A"}
        rem, add = enumerate_within_group_swaps(S, c2g, g2c)
        assert len(rem) == 0 and rem.dtype == np.int32, f"got {rem.dtype}"
        assert len(add) == 0 and add.dtype == np.int32, f"got {add.dtype}"

    def test_crowding_cache_swap_arrays_int32(self):
        rng = np.random.default_rng(20260523)
        N, R, S_size = 200, 8, 20
        cw = _make_constant_hw_codewords(N, R, hw=4, rng=rng)
        exp = rng.uniform(100.0, 1000.0, size=S_size)
        g2c = {"all": list(range(N))}
        S = rng.choice(N, size=S_size, replace=False).astype(np.int32)
        cache = CrowdingSwapCache(cw, exp, g2c, _c2g_from_S(S, g2c))
        cache.build_cache(S)
        assert cache._codebook_index_to_remove.dtype == np.int32
        assert cache._pool_index_to_add.dtype == np.int32
        assert cache._S_indices.dtype == np.int32

    def test_decoding_cache_swap_arrays_int32(self):
        from duet.pep_accessor import InMemoryPEPAccessor
        rng = np.random.default_rng(20260523)
        N, S_size = 50, 5
        X = rng.uniform(0, 0.1, size=(N, N)).astype(np.float32)
        X = (X + X.T) / 2.0
        np.fill_diagonal(X, 1.0)
        accessor = InMemoryPEPAccessor.from_pep_matrix(X * 2.0)  # X(s,s)=2
        g2c = {"A": list(range(N))}
        S = rng.choice(N, size=S_size, replace=False).astype(np.int32)
        cache = DecodingSwapCache(
            pep_accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=_c2g_from_S(S, g2c),
        )
        cache.build_cache(S)
        assert cache._codebook_index_to_remove.dtype == np.int32
        assert cache._pool_index_to_add.dtype == np.int32
        assert cache._S_indices.dtype == np.int32

    def test_shared_decode_state_swap_arrays_int32(self):
        from duet.pareto_optimization import build_shared_decode_state
        from duet.pep_accessor import InMemoryPEPAccessor
        rng = np.random.default_rng(20260523)
        N, S_size = 50, 5
        X = rng.uniform(0, 0.1, size=(N, N)).astype(np.float32)
        X = (X + X.T) / 2.0
        np.fill_diagonal(X, 1.0)
        accessor = InMemoryPEPAccessor.from_pep_matrix(X * 2.0)
        g2c = {"A": list(range(N))}
        S = rng.choice(N, size=S_size, replace=False).astype(np.int32)
        state = build_shared_decode_state(
            S=S, accessor=accessor, group_to_candidates=g2c,
            codeword_to_group=_c2g_from_S(S, g2c),
        )
        assert state.codebook_index_to_remove.dtype == np.int32
        assert state.pool_index_to_add.dtype == np.int32
        assert state.S_indices.dtype == np.int32


class TestCandidateC:
    """TDD harness for the Candidate C refactor.

    Runs against the cache's current state. Under the chi^2-anchored
    implementation these tests must fail; under the Candidate C
    implementation they must pass.
    """

    def test_deltas_match_C_brute_force(self):
        rng = np.random.default_rng(20260521)
        n_pool, R, S_size = 60, 16, 12
        codewords = _make_constant_hw_codewords(n_pool, R, hw=4, rng=rng)
        expression = rng.uniform(100.0, 1000.0, size=S_size)
        group_to_candidates = {"all": list(range(n_pool))}
        S = rng.choice(n_pool, size=S_size, replace=False)

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        # Brute-force C_anchor directly so this test is independent of
        # the cache attribute name (which changes during the refactor).
        C_anchor = _compute_cost_brute(codewords, expression, S)

        remove_arr, add_arr = cache.get_swaps()
        deltas = cache.get_deltas()

        # Sum-form delta: |S| * ΔO_C = -|S| * (C_swap - C_anchor) / C_anchor.
        for k in range(len(deltas)):
            pos = remove_arr[k]
            new_cand = add_arr[k]
            S_swap = S.copy()
            S_swap[pos] = new_cand
            C_swap = _compute_cost_brute(codewords, expression, S_swap)
            expected_delta = -len(S) * (C_swap - C_anchor) / C_anchor
            assert pytest.approx(deltas[k], abs=1e-10) == expected_delta, (
                f"Swap {k}: pos={pos}, new={new_cand}; "
                f"expected ΔO_C={expected_delta}, got {deltas[k]}"
            )


class TestCandidateCProperties:
    """Direction and invariance properties that distinguish Candidate C from
    the chi^2-anchored predecessor. Each test would fail against the prior
    chi^2-anchored implementation; passing here verifies the new objective
    captures the physics the May 2026 regression exposed.
    """

    def test_HW_reducing_swap_is_strictly_improving(self):
        """A swap that drops exactly one bit from a selected codeword (no
        other change) must have positive Delta O_C. This is the formal
        version of "lower HW -> less crowding" and the property the
        chi^2 objective famously failed: dropping a bit is detected as a
        strict improvement under Candidate C.
        """
        rng = np.random.default_rng(20260521)
        # Pool: each "drop-one-bit" candidate is the bit-flipped version of
        # its peer. Constructed deterministically so we know exactly which
        # swap to apply.
        R = 8
        c_old = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=float)  # HW = 5
        c_new = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=float)  # HW = 4, bit 0 dropped
        # Filler codewords so the cache has work to do; their HW is in
        # range and they're disjoint from c_old / c_new.
        filler = _make_valid_random_codewords(8, R, hw_choices=(3, 4), rng=rng)
        codewords = np.vstack([c_old[None, :], c_new[None, :], filler])
        # Position 0 is the swappable position; group A holds c_old (0) and c_new (1).
        group_to_candidates = {
            "A": [0, 1],
            **{f"B{i}": [2 + i] for i in range(8)},
        }
        S = np.array([0] + [2 + i for i in range(len(group_to_candidates) - 1)])
        expression = np.full(len(S), 100.0)

        cache = CrowdingSwapCache(
            codewords, expression, group_to_candidates,
            _c2g_from_S(S, group_to_candidates),
        )
        cache.build_cache(S)

        # Locate the (pos=0, new=1) swap in the enumeration and assert
        # its delta is strictly positive.
        remove_arr, add_arr = cache.get_swaps()
        deltas = cache.get_deltas()
        match = (remove_arr == 0) & (add_arr == 1)
        assert match.any(), "Expected drop-one-bit swap not enumerated."
        assert deltas[match][0] > 0.0, (
            f"Drop-one-bit swap not detected as improving: "
            f"delta = {deltas[match][0]}"
        )

    def test_crowding_heavy_run_makes_substantive_progress(self):
        """On a realistic-sized fixture (|S|=10, pool=40, R=12), running
        DUET with crowding-heavy weights must make sustained progress and
        reach a non-trivial O_C. The chi^2 predecessor terminated after
        ~7 iterations on the codebook_1 fixture with O_C ~ 0 (the May
        2026 regression); Candidate C should be qualitatively better here.

        Thresholds are intentionally conservative (any reasonable
        Candidate C implementation passes; chi^2 borderline-passes
        n_iter but never reaches the O_C floor).
        """
        rng = np.random.default_rng(20260521)
        pool_size = 40
        R = 12
        codewords = _make_valid_random_codewords(
            pool_size, R, hw_choices=(3, 4, 5), rng=rng,
        )
        group_to_candidates = {
            f"G{g}": list(range(g * 4, (g + 1) * 4)) for g in range(pool_size // 4)
        }
        S = np.array([g * 4 for g in range(pool_size // 4)])
        expression = rng.uniform(50.0, 500.0, size=len(S))

        pep_matrix = np.zeros((pool_size, pool_size))
        c2g = _c2g_from_S(S, group_to_candidates)
        decode_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=pep_matrix,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        score_cache = ScoreSwapCache(
            scores=np.zeros(pool_size),
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        crowding_cache = CrowdingSwapCache(
            codewords=codewords,
            expression=expression,
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
        )
        # Crowding-heavy with epsilon weight on decode/score (matches the
        # pattern used by the existing test_crowding_objective_improves).
        weights = {"decode": 0.01, "score": 0.01, "crowding": 0.98}
        optimizer = DUET(
            group_to_candidates=group_to_candidates,
            codeword_to_group=c2g,
            objective_caches=[decode_cache, score_cache, crowding_cache],
            weights=weights,
            temperature=0.0,
            max_iter=500,
            max_patience=500,
            track_history=True,
        )
        optimizer.optimize(list(S), seed=20260521)

        crowding_values = optimizer.history["crowding"]
        n_iter = len(optimizer.history["iteration"]) - 1

        assert n_iter >= 10, (
            f"Optimizer terminated after only {n_iter} iterations; "
            f"Candidate C should sustain >= 10 swaps on this fixture."
        )
        assert crowding_values[-1] >= 0.02, (
            f"Final O_C = {crowding_values[-1]:.4f}; expected >= 0.02 "
            f"under Candidate C. (chi^2 would land near 0.)"
        )
        assert crowding_values[-1] > crowding_values[0], (
            f"O_C did not strictly improve from 0: final {crowding_values[-1]}."
        )

    def test_e_scale_invariance(self):
        """O_C and ΔO_C are scale-invariant under uniform scaling of all
        e_i by a constant k (both C(S) and C(S_0) scale by k^2). Verifies
        the score and the delta vector are byte-identical between e and
        2*e on the same fixture.
        """
        rng = np.random.default_rng(20260521)
        pool_size = 20
        R = 8
        codewords = _make_valid_random_codewords(
            pool_size, R, hw_choices=(2, 3, 4), rng=rng,
        )
        group_to_candidates = {
            "A": list(range(10)),
            "B": list(range(10, 20)),
        }
        S = np.array([0, 10])
        expression = rng.uniform(50.0, 500.0, size=len(S))

        c2g = _c2g_from_S(S, group_to_candidates)
        cache1 = CrowdingSwapCache(codewords, expression, group_to_candidates, c2g)
        cache1.build_cache(S)

        cache2 = CrowdingSwapCache(
            codewords, 2.0 * expression, group_to_candidates, c2g,
        )
        cache2.build_cache(S)

        # Score at S_0 is 0 in both cases (trivially).
        assert cache1.compute_objective(S) == pytest.approx(0.0, abs=1e-12)
        assert cache2.compute_objective(S) == pytest.approx(0.0, abs=1e-12)

        # Score at a swapped S: pick the first available swap.
        remove_arr, add_arr = cache1.get_swaps()
        S_swap = S.copy()
        S_swap[remove_arr[0]] = add_arr[0]
        assert cache1.compute_objective(S_swap) == pytest.approx(
            cache2.compute_objective(S_swap), abs=1e-12
        )

        # Deltas vectors should also match.
        np.testing.assert_allclose(
            cache1.get_deltas(), cache2.get_deltas(), atol=1e-12,
        )
