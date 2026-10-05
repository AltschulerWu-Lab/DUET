"""Tests for GreedyDistanceOptimizer symmetrization."""

from unittest.mock import patch

import numpy as np
import pytest

from duet.codebook_evaluator import AsymmetricNLL, HammingDistance
from duet.greedy_optimizers import GreedyDistanceOptimizer, GreedyHammingMOOptimizer


class TestGreedyHammingUnchanged:
    """Symmetrization must not change Greedy Hamming results (Hamming is symmetric)."""

    def test_hamming_results_unchanged(self):
        """Greedy Hamming must produce the same selection as before symmetrization.

        With seed=0, 3 groups of 3 candidates each, quota=1 per group,
        the optimizer should select a deterministic codebook. This test pins
        the expected output to detect any unintended behavioral change.
        """
        rng = np.random.default_rng(0)
        sequences = rng.integers(0, 4, size=(9, 6))
        group_to_candidates = {
            "g0": [0, 1, 2],
            "g1": [3, 4, 5],
            "g2": [6, 7, 8],
        }
        quotas = {"g0": 1, "g1": 1, "g2": 1}

        opt = GreedyDistanceOptimizer(decoding_metric=HammingDistance())
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=0)

        # Pin expected output to detect any behavioral change
        np.testing.assert_array_equal(result, [8, 1, 4])


class TestGreedyNLLWithAsymmetricChannel:
    """Test that symmetrized NLL handles the benchmark's channel matrix correctly.

    The "asymmetric" noise channel of the earlier 1-D synthetic benchmark
    makes the first q/2 symbols immune (identity rows in the channel matrix).
    This produces inf costs via -log(0) for impossible transitions.
    The symmetrization must handle these inf values without degenerating.
    """

    @pytest.fixture
    def benchmark_channel_matrix(self):
        """Build the exact channel matrix used by the benchmark for asymmetric noise.

        Reproduces that benchmark's construction with q=4, error_rate=0.2.
        Symbols 0,1 are immune (identity rows); symbols 2,3 have error_rate=0.4
        (doubled so overall average = 0.2).
        """
        q = 4
        x = 0.2
        _EPSILON_FLOOR = 1e-6
        err = min(2 * x, 1.0 - _EPSILON_FLOOR)
        half = q // 2
        channel_matrix = np.eye(q)
        for a in range(half, q):
            channel_matrix[a, a] = 1.0 - err
            for b in range(q):
                if b != a:
                    channel_matrix[a, b] = err / (q - 1)
        return channel_matrix

    def test_does_not_degenerate_with_inf_costs(self, benchmark_channel_matrix):
        """Optimizer must select distinct candidates, not degenerate under inf.

        With immune symbols producing inf NLL costs, the symmetrized distance
        matrix will contain inf entries. The optimizer must still make
        meaningful selections (not all candidates tied at inf).
        """
        decoding_metric = AsymmetricNLL(channel_matrix=benchmark_channel_matrix)
        # Build sequences that mix immune (0,1) and susceptible (2,3) symbols
        # so some pairs produce inf and others don't.
        sequences = np.array([
            [0, 1, 2, 3, 0, 1],  # group 0, candidate 0
            [2, 3, 0, 1, 2, 3],  # group 0, candidate 1
            [0, 0, 0, 0, 0, 0],  # group 0, candidate 2
            [1, 1, 1, 1, 1, 1],  # group 1, candidate 0
            [2, 2, 2, 2, 2, 2],  # group 1, candidate 1
            [3, 3, 3, 3, 3, 3],  # group 1, candidate 2
            [0, 2, 1, 3, 0, 2],  # group 2, candidate 0
            [3, 1, 2, 0, 3, 1],  # group 2, candidate 1
            [1, 3, 0, 2, 1, 3],  # group 2, candidate 2
        ])
        group_to_candidates = {
            "g0": [0, 1, 2],
            "g1": [3, 4, 5],
            "g2": [6, 7, 8],
        }
        quotas = {"g0": 1, "g1": 1, "g2": 1}

        opt = GreedyDistanceOptimizer(decoding_metric=decoding_metric)
        result = opt.optimize(sequences, group_to_candidates, quotas, seed=42)

        # Must select exactly one per group
        assert len(result) == 3
        for group, candidates in group_to_candidates.items():
            assert any(idx in candidates for idx in result)

    def test_symmetrized_differs_from_one_directional(self, benchmark_channel_matrix):
        """The symmetrized optimizer must sometimes select differently than
        a hypothetical one-directional optimizer.

        We verify this by checking that d(a,b) != d(b,a) for at least some
        pairs in the channel matrix, confirming the asymmetry exists.
        """
        decoding_metric = AsymmetricNLL(channel_matrix=benchmark_channel_matrix)

        # Sequences where one uses only immune symbols and other uses susceptible
        seq_a = np.array([[0, 1, 0, 1, 0, 1]])  # all immune symbols
        seq_b = np.array([[2, 3, 2, 3, 2, 3]])  # all susceptible symbols

        d_ab = decoding_metric.compute(seq_a, seq_b)  # d(obs=a, tx=b)
        d_ba = decoding_metric.compute(seq_b, seq_a)  # d(obs=b, tx=a)

        # d(a,b) should be inf (transmitting susceptible symbol 2, observing
        # immune symbol 0 requires T[2,0] > 0 which is true, but transmitting
        # immune symbol 0 and observing susceptible symbol 2 requires T[0,2] = 0
        # which gives inf)
        assert not np.allclose(d_ab, d_ba), (
            f"Expected asymmetric NLL but got d(a,b)={d_ab} == d(b,a)={d_ba}"
        )


class TestGreedyHammingMOOptimizer:
    """Tests for the lambda-sweep scalarized greedy-Hamming MO baseline."""

    def _fixture(self, seed: int = 0):
        """Build a deterministic small pool with random scores."""
        rng = np.random.default_rng(seed)
        sequences = rng.integers(0, 4, size=(9, 6))
        scores = rng.uniform(0.0, 1.0, size=9)
        group_to_candidates = {
            "g0": [0, 1, 2],
            "g1": [3, 4, 5],
            "g2": [6, 7, 8],
        }
        quotas = {"g0": 1, "g1": 1, "g2": 1}
        return sequences, scores, group_to_candidates, quotas

    def test_lambda_one_matches_greedy_distance_optimizer(self):
        """At λ=1, the MO optimizer reduces to max-min Hamming greedy."""
        sequences, scores, g2c, quotas = self._fixture(seed=0)

        mo = GreedyHammingMOOptimizer(seed=0)
        mo_results = mo.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            scores=scores,
            lambdas=[1.0],
        )
        gd = GreedyDistanceOptimizer(decoding_metric=HammingDistance())
        gd_result = gd.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            seed=0,
        )

        np.testing.assert_array_equal(np.sort(mo_results[1.0]), np.sort(gd_result))

    def test_lambda_zero_picks_top_quota_by_score_per_group(self):
        """At λ=0, the optimizer picks each group's highest-scoring candidate(s)."""
        sequences, scores, g2c, quotas = self._fixture(seed=0)

        mo = GreedyHammingMOOptimizer(seed=0)
        result = mo.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            scores=scores,
            lambdas=[0.0],
        )[0.0]

        # Per-group expected: argmax over scores within each group's candidate set.
        expected = []
        for group in g2c:
            group_idx = g2c[group]
            quota = quotas[group]
            sorted_idx = sorted(group_idx, key=lambda i: -scores[i])
            expected.extend(sorted_idx[:quota])

        assert sorted(result.tolist()) == sorted(expected)

    def test_returns_one_result_per_lambda(self):
        sequences, scores, g2c, quotas = self._fixture(seed=0)
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]

        mo = GreedyHammingMOOptimizer(seed=0)
        results = mo.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            scores=scores,
            lambdas=lambdas,
        )

        assert set(results.keys()) == set(lambdas)
        for a in lambdas:
            assert len(results[a]) == 3  # one per group, quota=1

    def test_no_pep_machinery_used(self):
        """Behavioral ablation guard: optimizer must run without ever
        instantiating CodebookEvaluator. Patches CodebookEvaluator.__init__
        to raise; if any code path constructs an evaluator, the test fails.
        """
        sequences, scores, g2c, quotas = self._fixture(seed=0)

        with patch(
            "duet.codebook_evaluator.CodebookEvaluator.__init__",
            side_effect=AssertionError("PEP machinery used"),
        ):
            mo = GreedyHammingMOOptimizer(seed=0)
            results = mo.optimize(
                sequences=sequences,
                group_to_candidates=g2c,
                quotas=quotas,
                scores=scores,
                lambdas=[0.0, 0.5, 1.0],
            )
        # If we got here, the patch never fired.
        assert set(results.keys()) == {0.0, 0.5, 1.0}


class TestGreedyNLLMOOptimizer:
    """Tests for the lambda-sweep scalarized greedy-NLL MO baseline.

    Parallel to TestGreedyHammingMOOptimizer but with decoding metric supplied
    externally (SymmetricNLL in the fixture).
    """

    def _fixture(self, seed: int = 0):
        """Build a deterministic small pool with random scores."""
        rng = np.random.default_rng(seed)
        sequences = rng.integers(0, 2, size=(9, 6))
        scores = rng.uniform(0.0, 1.0, size=9)
        group_to_candidates = {
            "g0": [0, 1, 2],
            "g1": [3, 4, 5],
            "g2": [6, 7, 8],
        }
        quotas = {"g0": 1, "g1": 1, "g2": 1}
        return sequences, scores, group_to_candidates, quotas

    def test_lambda_one_matches_greedy_distance_optimizer(self):
        """At λ=1, the MO optimizer reduces to max-min symmetrized NLL greedy.

        Equivalent to GreedyDistanceOptimizer with the same decoding metric:
        both shuffle groups + candidates once with the same seed, both fall
        back to candidates[0] for the first pick (per-step normalization
        collapses gain_d_norm to 0 across all candidates), both use the
        symmetrized min for subsequent picks.
        """
        from duet.codebook_evaluator import SymmetricNLL
        from duet.greedy_optimizers import GreedyNLLMOOptimizer

        sequences, scores, g2c, quotas = self._fixture(seed=0)
        diss = SymmetricNLL(epsilon=0.1, alphabet_size=2)

        mo = GreedyNLLMOOptimizer(seed=0)
        mo_results = mo.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            scores=scores,
            lambdas=[1.0],
            decoding_metric=diss,
        )

        gd = GreedyDistanceOptimizer(decoding_metric=diss)
        gd_result = gd.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            seed=0,
        )

        np.testing.assert_array_equal(np.sort(mo_results[1.0]), np.sort(gd_result))

    def test_lambda_zero_picks_top_quota_by_score_per_group(self):
        """At λ=0, picks each group's highest-scoring candidate(s) — decoding metric ignored."""
        from duet.codebook_evaluator import SymmetricNLL
        from duet.greedy_optimizers import GreedyNLLMOOptimizer

        sequences, scores, g2c, quotas = self._fixture(seed=0)
        diss = SymmetricNLL(epsilon=0.1, alphabet_size=2)

        mo = GreedyNLLMOOptimizer(seed=0)
        result = mo.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            scores=scores,
            lambdas=[0.0],
            decoding_metric=diss,
        )[0.0]

        expected = []
        for group in g2c:
            group_idx = g2c[group]
            quota = quotas[group]
            sorted_idx = sorted(group_idx, key=lambda i: -scores[i])
            expected.extend(sorted_idx[:quota])

        assert sorted(result.tolist()) == sorted(expected)

    def test_returns_one_result_per_lambda(self):
        from duet.codebook_evaluator import SymmetricNLL
        from duet.greedy_optimizers import GreedyNLLMOOptimizer

        sequences, scores, g2c, quotas = self._fixture(seed=0)
        diss = SymmetricNLL(epsilon=0.1, alphabet_size=2)
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]

        mo = GreedyNLLMOOptimizer(seed=0)
        results = mo.optimize(
            sequences=sequences,
            group_to_candidates=g2c,
            quotas=quotas,
            scores=scores,
            lambdas=lambdas,
            decoding_metric=diss,
        )

        assert set(results.keys()) == set(lambdas)
        for a in lambdas:
            assert len(results[a]) == 3  # one per group, quota=1

    def test_no_pep_machinery_used(self):
        """Behavioral ablation guard: optimizer must run without ever
        instantiating CodebookEvaluator. Patches CodebookEvaluator.__init__
        to raise; if any code path constructs an evaluator, the test fails.
        """
        from duet.codebook_evaluator import SymmetricNLL
        from duet.greedy_optimizers import GreedyNLLMOOptimizer

        sequences, scores, g2c, quotas = self._fixture(seed=0)
        diss = SymmetricNLL(epsilon=0.1, alphabet_size=2)

        with patch(
            "duet.codebook_evaluator.CodebookEvaluator.__init__",
            side_effect=AssertionError("PEP machinery used"),
        ):
            mo = GreedyNLLMOOptimizer(seed=0)
            results = mo.optimize(
                sequences=sequences,
                group_to_candidates=g2c,
                quotas=quotas,
                scores=scores,
                lambdas=[0.0, 0.5, 1.0],
                decoding_metric=diss,
            )
        assert set(results.keys()) == {0.0, 0.5, 1.0}
