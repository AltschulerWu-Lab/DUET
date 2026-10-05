"""Tests for GPU-accelerated PEP matrix computation.

These tests verify that the GPU path produces identical uint16 count
matrices to the CPU path (n_jobs=1) for the same seed. Tests are skipped
when no GPU or CuPy is available.
"""

import numpy as np
import pytest
from unittest.mock import patch

from duet.gpu_pep import has_gpu

_skip_without_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="No GPU available (CuPy not installed or no CUDA device)"
)


def requires_gpu(obj):
    """Mark a test class as needing a GPU (the `gpu` marker) and skip it without one."""
    return pytest.mark.gpu(_skip_without_gpu(obj))


# Known behavior, documented in docs/gpu_and_scaling.md (not a code change):
# CPU and GPU PEP/cache counts differ for NLL decoding metrics because the
# float32 costs are summed in different orders by OpenBLAS and cuBLAS, which
# moves a few near-tie reads. Hamming metrics match exactly.
GPU_NLL_REASON = (
    "CPU and GPU float32 NLL costs are summed in different orders (OpenBLAS vs "
    "cuBLAS), so a few near-tie reads differ; Hamming metrics match exactly"
)


from duet.codebook_evaluator import (
    CodebookEvaluator,
    SymmetricEpsilon,
    PositionVaryingEpsilon,
    AsymmetricChannel,
    PositionVaryingAsymmetricChannel,
    HammingDistance,
    WeightedHammingDistance,
    SymmetricNLL,
    PositionVaryingNLL,
    AsymmetricNLL,
    PositionVaryingAsymmetricNLL,
    UniqueMinimum,
    MarginDecoding,
    PairwisePosteriorThreshold,
    ApproximatePosteriorThreshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N, L, Q = 50, 8, 4
N_SAMPLES = 500
SEED = 123


def _make_codebook(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, Q, size=(N, L), dtype=np.int8)


def _build_symmetric_channel(q, epsilon):
    """Build a q x q symmetric channel matrix from uniform epsilon."""
    T = np.full((q, q), epsilon / (q - 1))
    np.fill_diagonal(T, 1.0 - epsilon)
    return T


def _build_nonsquare_channel(q_tx, q_obs):
    """Build a (q_tx, q_obs) row-stochastic channel matrix."""
    rng = np.random.default_rng(99)
    T = rng.dirichlet(np.ones(q_obs), size=q_tx)
    return T


def _assert_cpu_gpu_match(evaluator):
    """Run CPU (n_jobs=1) and GPU, assert identical uint16 results."""
    cpu_counts, cpu_n = evaluator.compute_pep_matrix(n_jobs=1, device="cpu")
    gpu_counts, gpu_n = evaluator.compute_pep_matrix(device="gpu")
    assert cpu_n == gpu_n
    np.testing.assert_array_equal(cpu_counts, gpu_counts)


# ---------------------------------------------------------------------------
# Core exact-match test
# ---------------------------------------------------------------------------

@requires_gpu
class TestGPUMatchesCPU:
    """GPU and CPU must produce identical uint16 count matrices."""

    def test_basic_hamming_uniform(self):
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_match(evaluator)


# ---------------------------------------------------------------------------
# All decoding metric types
# ---------------------------------------------------------------------------

@requires_gpu
class TestAllDecodingMetrics:

    def _make_evaluator(self, decoding_metric):
        return CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=decoding_metric,
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )

    def test_hamming_distance(self):
        _assert_cpu_gpu_match(self._make_evaluator(
            HammingDistance(alphabet_size=Q)
        ))

    def test_weighted_hamming_distance(self):
        weights = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 2.0, 1.5, 0.5])
        _assert_cpu_gpu_match(self._make_evaluator(
            WeightedHammingDistance(weights=weights, alphabet_size=Q)
        ))

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_symmetric_nll(self):
        _assert_cpu_gpu_match(self._make_evaluator(
            SymmetricNLL(epsilon=0.1, alphabet_size=Q)
        ))

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_position_varying_nll(self):
        epsilons = np.array([0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1])
        _assert_cpu_gpu_match(self._make_evaluator(
            PositionVaryingNLL(epsilons=epsilons, alphabet_size=Q)
        ))

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_asymmetric_nll(self):
        T = _build_symmetric_channel(Q, 0.1)
        _assert_cpu_gpu_match(self._make_evaluator(
            AsymmetricNLL(channel_matrix=T)
        ))

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_position_varying_asymmetric_nll(self):
        T = _build_symmetric_channel(Q, 0.1)
        Ts = np.stack([T] * L)  # (L, Q, Q)
        _assert_cpu_gpu_match(self._make_evaluator(
            PositionVaryingAsymmetricNLL(channel_matrices=Ts)
        ))


# ---------------------------------------------------------------------------
# All noise channels
# ---------------------------------------------------------------------------

@requires_gpu
class TestAllNoiseChannels:

    def _make_evaluator(self, noise_channel):
        return CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=noise_channel,
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )

    def test_uniform_epsilon(self):
        _assert_cpu_gpu_match(self._make_evaluator(
            SymmetricEpsilon(epsilon=0.1, alphabet_size=Q)
        ))

    def test_positional_epsilon(self):
        epsilons = np.array([0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1])
        _assert_cpu_gpu_match(self._make_evaluator(
            PositionVaryingEpsilon(epsilons=epsilons, alphabet_size=Q)
        ))

    def test_channel_noise(self):
        T = _build_symmetric_channel(Q, 0.1)
        _assert_cpu_gpu_match(self._make_evaluator(
            AsymmetricChannel(channel_matrix=T)
        ))

    def test_positional_channel_noise(self):
        T = _build_symmetric_channel(Q, 0.1)
        Ts = np.stack([T] * L)  # (L, Q, Q)
        _assert_cpu_gpu_match(self._make_evaluator(
            PositionVaryingAsymmetricChannel(channel_matrices=Ts)
        ))


# ---------------------------------------------------------------------------
# Non-square channels (q_tx != q_obs)
# ---------------------------------------------------------------------------

@requires_gpu
class TestNonSquareChannels:
    """Verify GPU handles q_tx != q_obs correctly."""

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_nonsquare_asymmetric_nll(self):
        q_tx, q_obs = 4, 3
        T = _build_nonsquare_channel(q_tx, q_obs)
        codebook = np.random.default_rng(42).integers(0, q_tx, size=(N, L), dtype=np.int8)
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=AsymmetricChannel(channel_matrix=T),
            decoding_metric=AsymmetricNLL(channel_matrix=T),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_match(evaluator)

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_nonsquare_position_varying_asymmetric_nll(self):
        q_tx, q_obs = 4, 3
        T = _build_nonsquare_channel(q_tx, q_obs)
        Ts = np.stack([T] * L)  # (L, q_tx, q_obs)
        codebook = np.random.default_rng(42).integers(0, q_tx, size=(N, L), dtype=np.int8)
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=PositionVaryingAsymmetricChannel(channel_matrices=Ts),
            decoding_metric=PositionVaryingAsymmetricNLL(channel_matrices=Ts),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_match(evaluator)


# ---------------------------------------------------------------------------
# Decoding rules
# ---------------------------------------------------------------------------

@requires_gpu
class TestDecodingRules:

    def _make_evaluator(self, decoding_rule):
        return CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=decoding_rule,
            n_samples=N_SAMPLES,
            seed=SEED,
        )

    def test_unique_minimum(self):
        _assert_cpu_gpu_match(self._make_evaluator(UniqueMinimum()))

    def test_margin_decoding(self):
        _assert_cpu_gpu_match(self._make_evaluator(MarginDecoding(k=1.0)))

    def test_pairwise_posterior_threshold(self):
        _assert_cpu_gpu_match(self._make_evaluator(
            PairwisePosteriorThreshold(threshold=0.9)
        ))

    def test_approximate_posterior_threshold(self):
        _assert_cpu_gpu_match(self._make_evaluator(
            ApproximatePosteriorThreshold(threshold=0.9, n_eff=3)
        ))


# ---------------------------------------------------------------------------
# Sample batching on GPU
# ---------------------------------------------------------------------------

@requires_gpu
class TestSampleBatching:

    def test_gpu_sample_batching_matches_unbatched(self):
        """sample_batch_size on GPU must match unbatched."""
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )

        gpu_full, _ = evaluator.compute_pep_matrix(
            device="gpu", sample_batch_size=N_SAMPLES,
        )
        gpu_batched, _ = evaluator.compute_pep_matrix(
            device="gpu", sample_batch_size=100,
        )
        np.testing.assert_array_equal(gpu_full, gpu_batched)


# ---------------------------------------------------------------------------
# competitor_margin() on all decoding rules
# ---------------------------------------------------------------------------

class TestCompetitorMargin:

    def test_unique_minimum(self):
        assert UniqueMinimum().competitor_margin() == 0.0

    def test_margin_decoding(self):
        assert MarginDecoding(k=2.5).competitor_margin() == 2.5

    def test_pairwise_posterior_threshold(self):
        rule = PairwisePosteriorThreshold(threshold=0.9)
        assert rule.competitor_margin() == pytest.approx(rule._margin)

    def test_approximate_posterior_threshold(self):
        rule = ApproximatePosteriorThreshold(threshold=0.9, n_eff=3)
        assert rule.competitor_margin() == pytest.approx(rule._margin)


# ---------------------------------------------------------------------------
# Graceful failure without CuPy
# ---------------------------------------------------------------------------

class TestNoCuPy:
    """device='gpu' without CuPy should raise ImportError with install hint."""

    def test_no_cupy_raises(self):
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        with patch("duet.gpu_utils.HAS_CUPY", False):
            with pytest.raises(ImportError, match="CuPy is required"):
                evaluator.compute_pep_matrix(device="gpu")


# ---------------------------------------------------------------------------
# MAX_SAMPLES_UINT16 guard applies to GPU path
# ---------------------------------------------------------------------------

class TestMaxSamplesGuard:
    """device='gpu' must enforce MAX_SAMPLES_UINT16."""

    def test_gpu_rejects_large_n_samples(self):
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=40000,  # exceeds MAX_SAMPLES_UINT16 = 32767
            seed=SEED,
        )
        with pytest.raises(NotImplementedError, match="exceeds maximum"):
            evaluator.compute_pep_matrix(device="gpu")


# ---------------------------------------------------------------------------
# Multi-GPU support
# ---------------------------------------------------------------------------

@requires_gpu
class TestMultiGPUPEP:
    """Multi-GPU device strings must produce bit-identical results."""

    def _make_evaluator(self):
        return CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )

    def test_single_gpu_explicit_matches_default(self):
        """device='gpu:0' must match device='gpu'."""
        evaluator = self._make_evaluator()
        default_counts, _ = evaluator.compute_pep_matrix(device="gpu")
        explicit_counts, _ = evaluator.compute_pep_matrix(device="gpu:0")
        np.testing.assert_array_equal(default_counts, explicit_counts)

    def test_two_threads_same_gpu_matches_single(self):
        """device='gpu:0,0' (two threads, one GPU) must match device='gpu:0'."""
        evaluator = self._make_evaluator()
        single_counts, _ = evaluator.compute_pep_matrix(device="gpu:0")
        dual_counts, _ = evaluator.compute_pep_matrix(device="gpu:0,0")
        np.testing.assert_array_equal(single_counts, dual_counts)
