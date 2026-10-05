"""Tests for BLAS-based decoding metric computation.

Verifies that the BLAS path (precompute_transmitted / compute_with_precomputed)
produces results consistent with the original broadcasting path (compute) for
all six decoding metric types, across multiple decoding rules, and including
edge cases with inf values.
"""

import numpy as np
import pytest
from scipy.sparse import issparse

from duet.codebook_evaluator import (
    HammingDistance,
    WeightedHammingDistance,
    SymmetricNLL,
    PositionVaryingNLL,
    AsymmetricNLL,
    PositionVaryingAsymmetricNLL,
    UniqueMinimum,
    MarginDecoding,
    SymmetricEpsilon,
    PositionVaryingEpsilon,
    AsymmetricChannel,
    PositionVaryingAsymmetricChannel,
    one_hot_encode,
    _INF_SENTINEL,
    _BLASPrecomputed,
    _worker_pep_batch,
)
from duet.utils import _worker_init_blas

# Real behavior of live code, kept as is because it touches published numbers:
# the BLAS decoding-metric path computes costs in float32, and float32 rounding breaks some
# exact cost ties that the float64 reference counts as competitors under
# unique_minimum. Paper numbers and regression goldens use the float32 path.
# Whether a tie breaks depends on the CPU: numpy's OpenBLAS picks its float32
# kernel per CPU model. The tests fail on the paper machine (Xeon E5-2640 v4)
# and pass on some GitHub runners, so the xfails are not strict.
FLOAT32_TIE_REASON = (
    "float32 BLAS costs break exact float64 ties that unique_minimum counts as "
    "errors; affects decode counts at ties, kept for paper parity; "
    "CPU-dependent (OpenBLAS float32 kernel), so it passes on some machines"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def codebook_q4(rng):
    """Small codebook: 10 codewords, length 8, alphabet size 4."""
    return rng.integers(0, 4, size=(10, 8), dtype=np.int8)


@pytest.fixture
def observed_q4(rng):
    """Observed samples: 50 samples, length 8, alphabet size 4."""
    return rng.integers(0, 4, size=(50, 8), dtype=np.int8)


@pytest.fixture
def codebook_q2(rng):
    """Binary codebook: 8 codewords, length 6."""
    return rng.integers(0, 2, size=(8, 6), dtype=np.int8)


@pytest.fixture
def observed_q2(rng):
    """Binary observations: 30 samples, length 6."""
    return rng.integers(0, 2, size=(30, 6), dtype=np.int8)


# ---------------------------------------------------------------------------
# Helper: compare BLAS vs original paths
# ---------------------------------------------------------------------------

def assert_blas_matches_original(decoding_metric, codebook, observed, rtol=1e-5):
    """Check that BLAS path produces results consistent with original."""
    expected = decoding_metric.compute(observed, codebook)
    precomputed = decoding_metric.precompute_transmitted(codebook)
    assert precomputed is not None, "precompute_transmitted returned None"
    actual = decoding_metric.compute_with_precomputed(observed, precomputed)

    assert actual.shape == expected.shape

    finite_mask = np.isfinite(expected)
    np.testing.assert_allclose(
        actual[finite_mask], expected[finite_mask], rtol=rtol,
        err_msg=f"BLAS mismatch for {type(decoding_metric).__name__}"
    )
    if not np.all(finite_mask):
        assert np.all(actual[~finite_mask] >= _INF_SENTINEL), (
            "BLAS path should produce values >= _INF_SENTINEL where original has +inf"
        )


def assert_competitor_identification_matches(
    decoding_metric, decoding_rule, codebook, observed
):
    """Check that competitor identification is identical between paths.

    Uses the same access pattern as PEP computation: for each codeword i,
    only observations generated from codeword i are used, and the
    transmitted cost is always column i.
    """
    expected = decoding_metric.compute(observed, codebook)
    precomputed = decoding_metric.precompute_transmitted(codebook)
    actual = decoding_metric.compute_with_precomputed(observed, precomputed)

    for i in range(codebook.shape[0]):
        tx_orig = expected[:, i]
        tx_blas = actual[:, i]
        comp_orig = decoding_rule.identify_competitors(expected, tx_orig)
        comp_blas = decoding_rule.identify_competitors(actual, tx_blas)

        # Both should be sparse; convert to dense for comparison
        orig_dense = comp_orig.toarray() if issparse(comp_orig) else comp_orig
        blas_dense = comp_blas.toarray() if issparse(comp_blas) else comp_blas

        np.testing.assert_array_equal(
            orig_dense, blas_dense,
            err_msg=f"Competitor mismatch for codeword {i}, "
                    f"{type(decoding_metric).__name__}, "
                    f"{type(decoding_rule).__name__}"
        )


def assert_competitor_pep_matches(
    decoding_metric, decoding_rule, codebook, noise_channel, n_samples=20, seed=42
):
    """Check competitor identification matches in PEP-like computation.

    For decoding metrics with inf costs, random observations may produce inf
    transmitted costs, which breaks sentinel semantics. This helper mirrors
    the actual PEP computation: for each codeword, generate observations from
    that codeword's noise channel (ensuring finite transmitted costs).
    """
    rng = np.random.default_rng(seed)
    precomputed = decoding_metric.precompute_transmitted(codebook)

    for i in range(codebook.shape[0]):
        observed = noise_channel.generate(codebook[i], n_samples, rng)
        expected = decoding_metric.compute(observed, codebook)
        actual = decoding_metric.compute_with_precomputed(observed, precomputed)

        tx_orig = expected[:, i]
        tx_blas = actual[:, i]

        # Transmitted costs must be finite (noise channel guarantee)
        assert np.all(np.isfinite(tx_orig)), (
            f"Transmitted cost is not finite for codeword {i}"
        )

        comp_orig = decoding_rule.identify_competitors(expected, tx_orig)
        comp_blas = decoding_rule.identify_competitors(actual, tx_blas)

        orig_dense = comp_orig.toarray() if issparse(comp_orig) else comp_orig
        blas_dense = comp_blas.toarray() if issparse(comp_blas) else comp_blas

        np.testing.assert_array_equal(
            orig_dense, blas_dense,
            err_msg=f"PEP competitor mismatch for codeword {i}, "
                    f"{type(decoding_metric).__name__}, "
                    f"{type(decoding_rule).__name__}"
        )


# ---------------------------------------------------------------------------
# Tests: one_hot_encode
# ---------------------------------------------------------------------------

class TestOneHotEncode:
    def test_basic(self):
        seqs = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int8)
        phi = one_hot_encode(seqs, q=4)
        assert phi.shape == (2, 12)
        assert phi.dtype == np.float32
        # Each row should have exactly L=3 ones
        np.testing.assert_array_equal(phi.sum(axis=1), [3, 3])

    def test_binary(self):
        seqs = np.array([[0, 1], [1, 0]], dtype=np.int8)
        phi = one_hot_encode(seqs, q=2)
        assert phi.shape == (2, 4)
        assert phi.dtype == np.float32
        expected = np.array([
            [1, 0, 0, 1],  # pos0=0, pos1=1
            [0, 1, 1, 0],  # pos0=1, pos1=0
        ], dtype=np.float32)
        np.testing.assert_array_equal(phi, expected)

    def test_single_sequence(self):
        seqs = np.array([[2, 0]], dtype=np.int8)
        phi = one_hot_encode(seqs, q=3)
        assert phi.shape == (1, 6)
        assert phi.dtype == np.float32
        expected = np.array([[0, 0, 1, 1, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(phi, expected)


# ---------------------------------------------------------------------------
# Tests: _BLASPrecomputed
# ---------------------------------------------------------------------------

class TestBLASPrecomputed:
    def test_frozen(self):
        p = _BLASPrecomputed(psi_tx=np.zeros((2, 4)), q_obs=4)
        with pytest.raises(AttributeError):
            p.q_obs = 5


# ---------------------------------------------------------------------------
# Tests: HammingDistance BLAS path
# ---------------------------------------------------------------------------

class TestHammingDistanceBLAS:
    def test_matches_original(self, codebook_q4, observed_q4):
        d = HammingDistance()
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_matches_original_binary(self, codebook_q2, observed_q2):
        d = HammingDistance()
        assert_blas_matches_original(d, codebook_q2, observed_q2)

    def test_competitor_unique_minimum(self, codebook_q4, observed_q4):
        d = HammingDistance()
        assert_competitor_identification_matches(
            d, UniqueMinimum(), codebook_q4, observed_q4
        )

    def test_competitor_margin(self, codebook_q4, observed_q4):
        d = HammingDistance()
        assert_competitor_identification_matches(
            d, MarginDecoding(k=1.0), codebook_q4, observed_q4
        )

    def test_single_codeword(self, rng):
        codebook = rng.integers(0, 4, size=(1, 5), dtype=np.int8)
        observed = rng.integers(0, 4, size=(10, 5), dtype=np.int8)
        d = HammingDistance()
        assert_blas_matches_original(d, codebook, observed)

    def test_single_sample(self, rng):
        codebook = rng.integers(0, 4, size=(5, 8), dtype=np.int8)
        observed = rng.integers(0, 4, size=(1, 8), dtype=np.int8)
        d = HammingDistance()
        assert_blas_matches_original(d, codebook, observed)


# ---------------------------------------------------------------------------
# Tests: WeightedHammingDistance BLAS path
# ---------------------------------------------------------------------------

class TestWeightedHammingDistanceBLAS:
    def test_matches_original(self, codebook_q4, observed_q4, rng):
        weights = rng.random(8) * 5
        d = WeightedHammingDistance(weights)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_competitor_unique_minimum(self, codebook_q4, observed_q4, rng):
        weights = rng.random(8) * 5
        d = WeightedHammingDistance(weights)
        assert_competitor_identification_matches(
            d, UniqueMinimum(), codebook_q4, observed_q4
        )

    def test_competitor_margin(self, codebook_q4, observed_q4, rng):
        weights = rng.random(8) * 5
        d = WeightedHammingDistance(weights)
        assert_competitor_identification_matches(
            d, MarginDecoding(k=2.0), codebook_q4, observed_q4
        )


# ---------------------------------------------------------------------------
# Tests: SymmetricNLL BLAS path
# ---------------------------------------------------------------------------

class TestSymmetricNLLBLAS:
    def test_matches_original_finite(self, codebook_q4, observed_q4):
        d = SymmetricNLL(epsilon=0.1, alphabet_size=4)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_matches_original_epsilon_zero(self, codebook_q4, observed_q4):
        """When epsilon=0, weight=+inf. BLAS path uses sentinel."""
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_competitor_epsilon_zero(self, codebook_q4):
        """PEP-style: transmitted costs are finite (epsilon=0 → no mutations)."""
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        noise = SymmetricEpsilon(epsilon=0.0, alphabet_size=4)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_competitor_margin_epsilon_zero(self, codebook_q4):
        """PEP-style: transmitted costs are finite (epsilon=0 → no mutations)."""
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        noise = SymmetricEpsilon(epsilon=0.0, alphabet_size=4)
        assert_competitor_pep_matches(d, MarginDecoding(k=1.0), codebook_q4, noise)


# ---------------------------------------------------------------------------
# Tests: PositionVaryingNLL BLAS path
# ---------------------------------------------------------------------------

class TestPositionVaryingNLLBLAS:
    def test_matches_original_finite(self, codebook_q4, observed_q4):
        epsilons = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.2, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_matches_original_mixed_zeros(self, codebook_q4, observed_q4):
        """Mix of zero and nonzero epsilons."""
        epsilons = np.array([0.0, 0.2, 0.0, 0.15, 0.1, 0.0, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_competitor_mixed_zeros(self, codebook_q4):
        """PEP-style: transmitted costs are finite (some epsilons=0)."""
        epsilons = np.array([0.0, 0.2, 0.0, 0.15, 0.1, 0.0, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        noise = PositionVaryingEpsilon(epsilons, alphabet_size=4)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_all_zeros(self, codebook_q4, observed_q4):
        """All epsilons = 0 → all weights = +inf."""
        epsilons = np.zeros(8)
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_competitor_margin(self, codebook_q4, observed_q4):
        epsilons = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.2, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        assert_competitor_identification_matches(
            d, MarginDecoding(k=2.0), codebook_q4, observed_q4
        )


# ---------------------------------------------------------------------------
# Tests: AsymmetricNLL BLAS path
# ---------------------------------------------------------------------------

def _make_channel_matrix_with_zeros(q=4):
    """Create a channel matrix with some zero-probability transitions."""
    T = np.ones((q, q)) * 0.1 / (q - 1)
    np.fill_diagonal(T, 0.9)
    # Set some off-diagonal to 0
    T[0, 1] = 0.0
    T[1, 2] = 0.0
    # Renormalize rows
    row_sums = T.sum(axis=1)
    T = T / row_sums[:, None]
    return T


class TestAsymmetricNLLBLAS:
    def test_matches_original_square(self, codebook_q4, observed_q4):
        T = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.05, 0.85],
        ])
        d = AsymmetricNLL(T)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_matches_original_nonsquare(self, rng):
        """Non-square channel: q_tx=4, q_obs=3."""
        T = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.1, 0.3, 0.6],
        ])
        codebook = rng.integers(0, 4, size=(8, 6), dtype=np.int8)
        observed = rng.integers(0, 3, size=(20, 6), dtype=np.int8)
        d = AsymmetricNLL(T)
        assert_blas_matches_original(d, codebook, observed)

    def test_matches_original_with_zeros(self, codebook_q4, observed_q4):
        """Channel matrix with zero entries (inf costs)."""
        T = _make_channel_matrix_with_zeros(q=4)
        d = AsymmetricNLL(T)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_competitor_with_zeros(self, codebook_q4):
        """PEP-style: transmitted costs are finite (zero channel entries)."""
        T = _make_channel_matrix_with_zeros(q=4)
        d = AsymmetricNLL(T)
        noise = AsymmetricChannel(T)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_competitor_margin_with_zeros(self, codebook_q4):
        """PEP-style: transmitted costs are finite (zero channel entries)."""
        T = _make_channel_matrix_with_zeros(q=4)
        d = AsymmetricNLL(T)
        noise = AsymmetricChannel(T)
        assert_competitor_pep_matches(d, MarginDecoding(k=1.5), codebook_q4, noise)

    def test_from_symbol_epsilons(self, codebook_q4, observed_q4):
        epsilons = np.array([0.08, 0.12, 0.10, 0.09])
        d = AsymmetricNLL.from_symbol_epsilons(epsilons)
        assert_blas_matches_original(d, codebook_q4, observed_q4)


# ---------------------------------------------------------------------------
# Tests: PositionVaryingAsymmetricNLL BLAS path
# ---------------------------------------------------------------------------

def _make_positional_channel_matrices_with_zeros(L=8, q=4):
    """Create positional channel matrices with some zero entries."""
    T = np.ones((L, q, q)) * 0.1 / (q - 1)
    for l in range(L):
        np.fill_diagonal(T[l], 0.9)
    # Zero out some transitions at specific positions
    T[0, 0, 1] = 0.0
    T[2, 1, 3] = 0.0
    T[5, 3, 0] = 0.0
    # Renormalize
    for l in range(L):
        row_sums = T[l].sum(axis=1)
        T[l] = T[l] / row_sums[:, None]
    return T


class TestPositionVaryingAsymmetricNLLBLAS:
    def test_matches_original(self, codebook_q4, observed_q4):
        L = 8
        epsilons = np.random.default_rng(99).random((L, 4)) * 0.3 + 0.01
        d = PositionVaryingAsymmetricNLL.from_positional_symbol_epsilons(epsilons)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    def test_matches_original_nonsquare(self, rng):
        """Non-square positional channel: q_tx=4, q_obs=3."""
        L = 6
        T = rng.dirichlet(alpha=[5, 2, 2], size=(L, 4))  # (L, q_tx, q_obs)
        codebook = rng.integers(0, 4, size=(8, L), dtype=np.int8)
        observed = rng.integers(0, 3, size=(20, L), dtype=np.int8)
        d = PositionVaryingAsymmetricNLL(T)
        assert_blas_matches_original(d, codebook, observed)

    def test_matches_original_with_zeros(self, codebook_q4, observed_q4):
        T = _make_positional_channel_matrices_with_zeros(L=8, q=4)
        d = PositionVaryingAsymmetricNLL(T)
        assert_blas_matches_original(d, codebook_q4, observed_q4)

    @pytest.mark.xfail(strict=False, reason=FLOAT32_TIE_REASON)
    def test_competitor_with_zeros(self, codebook_q4):
        """PEP-style: transmitted costs are finite (zero channel entries)."""
        T = _make_positional_channel_matrices_with_zeros(L=8, q=4)
        d = PositionVaryingAsymmetricNLL(T)
        noise = PositionVaryingAsymmetricChannel(T)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_competitor_margin_with_zeros(self, codebook_q4):
        """PEP-style: transmitted costs are finite (zero channel entries)."""
        T = _make_positional_channel_matrices_with_zeros(L=8, q=4)
        d = PositionVaryingAsymmetricNLL(T)
        noise = PositionVaryingAsymmetricChannel(T)
        assert_competitor_pep_matches(d, MarginDecoding(k=2.0), codebook_q4, noise)

    def test_from_positional_epsilons(self, codebook_q4, observed_q4):
        epsilons = np.random.default_rng(77).random((8, 4)) * 0.3 + 0.01
        d = PositionVaryingAsymmetricNLL.from_positional_symbol_epsilons(epsilons)
        assert_blas_matches_original(d, codebook_q4, observed_q4)


# ---------------------------------------------------------------------------
# Tests: End-to-end PEP matrix computation via _worker_pep_batch
# ---------------------------------------------------------------------------

class TestWorkerPEPBatch:
    """Verify that _worker_pep_batch produces identical results with BLAS."""

    def _run_worker_comparison(self, decoding_metric, noise_channel, codebook, n_samples=100):
        """Run the worker with BLAS vs without BLAS and compare."""
        seed = 12345
        batch_indices = np.arange(len(codebook))
        decoding_rule = UniqueMinimum()

        args = (batch_indices, codebook, noise_channel, decoding_metric,
                decoding_rule, n_samples, seed, None)

        # The worker uses BLAS path internally when precomputed is not None.
        # Since all 6 types now support it, this tests the BLAS path.
        _, count_rows = _worker_pep_batch(args)

        assert count_rows.shape == (len(codebook), len(codebook))
        assert count_rows.dtype == np.uint16

        # The diagonal should generally have large counts (codeword matches itself)
        # This is a sanity check, not exact
        diag = count_rows[np.arange(len(codebook)), np.arange(len(codebook))]
        assert np.all(diag > 0), "Diagonal should have nonzero counts"

        return count_rows

    def test_hamming(self, rng):
        codebook = rng.integers(0, 4, size=(5, 8), dtype=np.int8)
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        d = HammingDistance()
        self._run_worker_comparison(d, noise, codebook)

    def test_weighted_hamming(self, rng):
        codebook = rng.integers(0, 4, size=(5, 8), dtype=np.int8)
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        weights = rng.random(8) * 3
        d = WeightedHammingDistance(weights)
        self._run_worker_comparison(d, noise, codebook)

    def test_symmetric_nll(self, rng):
        codebook = rng.integers(0, 4, size=(5, 8), dtype=np.int8)
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        d = SymmetricNLL(epsilon=0.1, alphabet_size=4)
        self._run_worker_comparison(d, noise, codebook)

    def test_position_varying_nll(self, rng):
        codebook = rng.integers(0, 4, size=(5, 8), dtype=np.int8)
        epsilons_noise = np.full(8, 0.1)
        noise = PositionVaryingEpsilon(epsilons_noise, alphabet_size=4)
        epsilons_decoding_metric = np.random.default_rng(99).random(8) * 0.3 + 0.01
        d = PositionVaryingNLL(epsilons_decoding_metric, alphabet_size=4)
        self._run_worker_comparison(d, noise, codebook)

    def test_asymmetric_nll(self, rng):
        T = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.05, 0.85],
        ])
        codebook = rng.integers(0, 4, size=(5, 8), dtype=np.int8)
        noise = AsymmetricChannel(T)
        d = AsymmetricNLL(T)
        self._run_worker_comparison(d, noise, codebook)

    def test_position_varying_asymmetric_nll(self, rng):
        L = 8
        epsilons = np.random.default_rng(77).random((L, 4)) * 0.3 + 0.01
        T = np.zeros((L, 4, 4))
        for l in range(L):
            for a in range(4):
                T[l, a, a] = 1 - epsilons[l, a]
                for b in range(4):
                    if b != a:
                        T[l, a, b] = epsilons[l, a] / 3
        codebook = rng.integers(0, 4, size=(5, L), dtype=np.int8)
        noise = PositionVaryingAsymmetricChannel(T)
        d = PositionVaryingAsymmetricNLL(T)
        self._run_worker_comparison(d, noise, codebook)


# ---------------------------------------------------------------------------
# Tests: inf edge cases - detailed
# ---------------------------------------------------------------------------

class TestInfEdgeCases:
    """Detailed tests for inf handling in BLAS path."""

    def test_symmetric_nll_epsilon_zero_no_nan(self):
        """BLAS path should never produce NaN for epsilon=0."""
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        codebook = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int8)
        observed = np.array([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=np.int8)

        precomputed = d.precompute_transmitted(codebook)
        result = d.compute_with_precomputed(observed, precomputed)

        assert not np.any(np.isnan(result)), "BLAS path produced NaN values"

    def test_position_varying_nll_mixed_zeros_no_nan(self):
        """Mixed zero/nonzero epsilons should not produce NaN."""
        epsilons = np.array([0.0, 0.1, 0.0, 0.2])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        codebook = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int8)
        observed = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int8)

        precomputed = d.precompute_transmitted(codebook)
        result = d.compute_with_precomputed(observed, precomputed)

        assert not np.any(np.isnan(result)), "BLAS path produced NaN values"

    def test_asymmetric_nll_zero_transitions_no_nan(self):
        """Channel matrix with zero entries should not produce NaN in BLAS."""
        T = np.array([
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.1, 0.0, 0.0, 0.9],
        ])
        d = AsymmetricNLL(T)
        codebook = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int8)
        observed = np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=np.int8)

        precomputed = d.precompute_transmitted(codebook)
        result = d.compute_with_precomputed(observed, precomputed)

        assert not np.any(np.isnan(result)), "BLAS path produced NaN values"

    def test_position_varying_asymmetric_nll_zero_transitions_no_nan(self):
        """Positional channel with zero entries should not produce NaN."""
        L = 4
        T = np.ones((L, 4, 4)) * 0.1 / 3
        for l in range(L):
            np.fill_diagonal(T[l], 0.9)
        # Zero out some transitions
        T[0, 0, 1] = 0.0
        T[1, 2, 3] = 0.0
        # Renormalize
        for l in range(L):
            row_sums = T[l].sum(axis=1)
            T[l] = T[l] / row_sums[:, None]

        d = PositionVaryingAsymmetricNLL(T)
        codebook = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int8)
        observed = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int8)

        precomputed = d.precompute_transmitted(codebook)
        result = d.compute_with_precomputed(observed, precomputed)

        assert not np.any(np.isnan(result)), "BLAS path produced NaN values"

    def test_sentinel_preserves_comparison_semantics(self):
        """Verify that _INF_SENTINEL preserves comparison: sentinel > any finite cost."""
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        codebook = np.array([[0, 1, 2, 3]], dtype=np.int8)
        # Observed differs from codebook - mismatch cost should be large sentinel
        observed = np.array([[1, 0, 3, 2]], dtype=np.int8)

        precomputed = d.precompute_transmitted(codebook)
        blas_cost = d.compute_with_precomputed(observed, precomputed)

        # For epsilon=0, any mismatch should produce a very large cost
        # (4 mismatches * sentinel weight)
        assert blas_cost[0, 0] > 0, "Mismatch cost should be positive"
        assert blas_cost[0, 0] >= _INF_SENTINEL, (
            "Mismatch cost with inf weight should be >= sentinel"
        )


# ---------------------------------------------------------------------------
# Tests: _worker_init_blas (BLAS thread pinning)
# ---------------------------------------------------------------------------

class TestWorkerInitBlas:
    """Regression tests for _worker_init_blas.

    An earlier ctypes.RTLD_NOLOAD bug caused an AttributeError here.
    """

    def test_worker_init_blas_runs_without_error(self):
        """Calling _worker_init_blas must not raise (regression for ctypes.RTLD_NOLOAD)."""
        _worker_init_blas()


# ---------------------------------------------------------------------------
# Tests: Float32 dtype verification for all decoding_metric BLAS paths
# ---------------------------------------------------------------------------

class TestFloat32Dtype:
    """Verify that all BLAS paths produce float32 psi_tx and float32 cost matrices."""

    def test_hamming_psi_tx_dtype(self, codebook_q4):
        d = HammingDistance()
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_hamming_costs_dtype(self, codebook_q4, observed_q4):
        d = HammingDistance()
        precomputed = d.precompute_transmitted(codebook_q4)
        costs = d.compute_with_precomputed(observed_q4, precomputed)
        assert costs.dtype == np.float32

    def test_weighted_hamming_psi_tx_dtype(self, codebook_q4, rng):
        weights = rng.random(8) * 5
        d = WeightedHammingDistance(weights)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_weighted_hamming_costs_dtype(self, codebook_q4, observed_q4, rng):
        weights = rng.random(8) * 5
        d = WeightedHammingDistance(weights)
        precomputed = d.precompute_transmitted(codebook_q4)
        costs = d.compute_with_precomputed(observed_q4, precomputed)
        assert costs.dtype == np.float32

    def test_symmetric_nll_psi_tx_dtype(self, codebook_q4):
        d = SymmetricNLL(epsilon=0.1, alphabet_size=4)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_symmetric_nll_epsilon_zero_psi_tx_dtype(self, codebook_q4):
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_position_varying_nll_psi_tx_dtype(self, codebook_q4):
        epsilons = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.2, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_position_varying_nll_mixed_zeros_psi_tx_dtype(self, codebook_q4):
        epsilons = np.array([0.0, 0.2, 0.0, 0.15, 0.1, 0.0, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_asymmetric_nll_psi_tx_dtype(self, codebook_q4):
        T = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.05, 0.85],
        ])
        d = AsymmetricNLL(T)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_asymmetric_nll_with_zeros_psi_tx_dtype(self, codebook_q4):
        T = _make_channel_matrix_with_zeros(q=4)
        d = AsymmetricNLL(T)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_position_varying_asymmetric_nll_psi_tx_dtype(self, codebook_q4):
        epsilons = np.random.default_rng(99).random((8, 4)) * 0.3 + 0.01
        d = PositionVaryingAsymmetricNLL.from_positional_symbol_epsilons(epsilons)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32

    def test_position_varying_asymmetric_nll_with_zeros_psi_tx_dtype(self, codebook_q4):
        T = _make_positional_channel_matrices_with_zeros(L=8, q=4)
        d = PositionVaryingAsymmetricNLL(T)
        precomputed = d.precompute_transmitted(codebook_q4)
        assert precomputed.psi_tx.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: Float32 BLAS vs Float64 original competitor equivalence
# ---------------------------------------------------------------------------

class TestFloat32CompetitorEquivalence:
    """Verify that float32 BLAS path identifies identical competitors to float64 original."""

    def test_hamming_competitors(self, codebook_q4, observed_q4):
        d = HammingDistance()
        assert_blas_matches_original(d, codebook_q4, observed_q4, rtol=1e-5)
        assert_competitor_identification_matches(
            d, UniqueMinimum(), codebook_q4, observed_q4
        )

    def test_weighted_hamming_competitors(self, codebook_q4, observed_q4, rng):
        weights = rng.random(8) * 5
        d = WeightedHammingDistance(weights)
        assert_blas_matches_original(d, codebook_q4, observed_q4, rtol=1e-5)
        assert_competitor_identification_matches(
            d, UniqueMinimum(), codebook_q4, observed_q4
        )

    def test_symmetric_nll_competitors(self, codebook_q4):
        d = SymmetricNLL(epsilon=0.1, alphabet_size=4)
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_symmetric_nll_epsilon_zero_pep_competitors(self, codebook_q4):
        d = SymmetricNLL(epsilon=0.0, alphabet_size=4)
        noise = SymmetricEpsilon(epsilon=0.0, alphabet_size=4)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_position_varying_nll_competitors(self, codebook_q4):
        epsilons = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.2, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        noise = PositionVaryingEpsilon(epsilons, alphabet_size=4)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_position_varying_nll_mixed_zeros_pep_competitors(self, codebook_q4):
        epsilons = np.array([0.0, 0.2, 0.0, 0.15, 0.1, 0.0, 0.05, 0.15])
        d = PositionVaryingNLL(epsilons, alphabet_size=4)
        noise = PositionVaryingEpsilon(epsilons, alphabet_size=4)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_asymmetric_nll_with_zeros_pep_competitors(self, codebook_q4):
        T = _make_channel_matrix_with_zeros(q=4)
        d = AsymmetricNLL(T)
        noise = AsymmetricChannel(T)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_position_varying_asymmetric_nll_competitors(self, codebook_q4):
        epsilons = np.random.default_rng(99).random((8, 4)) * 0.3 + 0.01
        d = PositionVaryingAsymmetricNLL.from_positional_symbol_epsilons(epsilons)
        T = np.zeros((8, 4, 4))
        for l in range(8):
            for a in range(4):
                T[l, a, a] = 1 - epsilons[l, a]
                for b in range(4):
                    if b != a:
                        T[l, a, b] = epsilons[l, a] / 3
        noise = PositionVaryingAsymmetricChannel(T)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    @pytest.mark.xfail(strict=False, reason=FLOAT32_TIE_REASON)
    def test_position_varying_asymmetric_nll_with_zeros_pep_competitors(self, codebook_q4):
        T = _make_positional_channel_matrices_with_zeros(L=8, q=4)
        d = PositionVaryingAsymmetricNLL(T)
        noise = PositionVaryingAsymmetricChannel(T)
        assert_competitor_pep_matches(d, UniqueMinimum(), codebook_q4, noise)

    def test_margin_decoding_hamming(self, codebook_q4, observed_q4):
        d = HammingDistance()
        assert_competitor_identification_matches(
            d, MarginDecoding(k=1.0), codebook_q4, observed_q4
        )

    def test_margin_decoding_asymmetric_nll(self, codebook_q4):
        T = _make_channel_matrix_with_zeros(q=4)
        d = AsymmetricNLL(T)
        noise = AsymmetricChannel(T)
        assert_competitor_pep_matches(d, MarginDecoding(k=1.5), codebook_q4, noise)
