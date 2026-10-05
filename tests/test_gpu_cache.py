"""Tests for GPU-accelerated cache initialization.

These tests verify that the GPU path produces identical sparse matrices
and observed sequences to the CPU path (n_jobs=1) for the same seed.
Tests are skipped when no GPU or CuPy is available.
"""

import numpy as np
import pytest
from unittest.mock import patch

from duet.gpu_utils import has_gpu

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

DUPLICATE_GPU_ID_REASON = (
    "device='gpu:0,0' (a repeated GPU id) makes the two workers share one "
    "indices file (gpu_cache keys its per-GPU files by id); configs never repeat ids"
)

from duet.codebook_evaluator import (
    CodebookEvaluator,
    SymmetricEpsilon,
    PositionVaryingEpsilon,
    AsymmetricChannel,
    HammingDistance,
    WeightedHammingDistance,
    SymmetricNLL,
    AsymmetricNLL,
    UniqueMinimum,
    MarginDecoding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N, L, Q = 20, 6, 4
N_SAMPLES = 200
SEED = 42


def _make_codebook(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, Q, size=(N, L), dtype=np.int8)


def _build_symmetric_channel(q, epsilon):
    T = np.full((q, q), epsilon / (q - 1))
    np.fill_diagonal(T, 1.0 - epsilon)
    return T


def _assert_cpu_gpu_cache_match(evaluator):
    """Run CPU (n_jobs=1) and GPU cache init, compare sparse matrices."""
    # CPU path
    cpu_eval = CodebookEvaluator(
        codebook=evaluator.codebook.copy(),
        noise_channel=evaluator.noise_channel,
        decoding_metric=evaluator.decoding_metric,
        decoding_rule=evaluator.decoding_rule,
        n_samples=evaluator.n_samples,
        seed=evaluator.seed,
    )
    cpu_eval.initialize_cache(n_jobs=1, device="cpu")

    # GPU path
    gpu_eval = CodebookEvaluator(
        codebook=evaluator.codebook.copy(),
        noise_channel=evaluator.noise_channel,
        decoding_metric=evaluator.decoding_metric,
        decoding_rule=evaluator.decoding_rule,
        n_samples=evaluator.n_samples,
        seed=evaluator.seed,
    )
    gpu_eval.initialize_cache(device="gpu")

    # Compare sparse matrices
    assert cpu_eval._sparse_matrix.shape == gpu_eval._sparse_matrix.shape
    cpu_dense = cpu_eval._sparse_matrix.toarray()
    gpu_dense = gpu_eval._sparse_matrix.toarray()
    np.testing.assert_array_equal(cpu_dense, gpu_dense)

    # Compare no-error flags
    np.testing.assert_array_equal(
        cpu_eval._no_error_flags,
        gpu_eval._no_error_flags,
    )

    # Compare codeword row indices
    assert len(cpu_eval._codeword_row_indices) == len(gpu_eval._codeword_row_indices)
    for cpu_idx, gpu_idx in zip(cpu_eval._codeword_row_indices,
                                gpu_eval._codeword_row_indices):
        np.testing.assert_array_equal(cpu_idx, gpu_idx)

    # Compare downstream accuracy results
    cpu_acc = cpu_eval.get_codeword_accuracy()
    gpu_acc = gpu_eval.get_codeword_accuracy()
    np.testing.assert_array_equal(cpu_acc, gpu_acc)


# ---------------------------------------------------------------------------
# Core exact-match test
# ---------------------------------------------------------------------------

@requires_gpu
class TestGPUCacheMatchesCPU:
    """GPU and CPU cache init must produce identical sparse matrices."""

    def test_basic_hamming_uniform(self):
        evaluator = CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_cache_match(evaluator)

    def test_weighted_hamming(self):
        weights = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 2.0])
        evaluator = CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=WeightedHammingDistance(weights=weights, alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_cache_match(evaluator)

    def test_symmetric_nll(self):
        evaluator = CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=SymmetricNLL(epsilon=0.1, alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_cache_match(evaluator)

    @pytest.mark.xfail(strict=True, reason=GPU_NLL_REASON)
    def test_asymmetric_nll(self):
        T = _build_symmetric_channel(Q, 0.1)
        evaluator = CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=AsymmetricChannel(channel_matrix=T),
            decoding_metric=AsymmetricNLL(channel_matrix=T),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_cache_match(evaluator)

    def test_margin_decoding(self):
        evaluator = CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=MarginDecoding(k=1.0),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_cache_match(evaluator)

    def test_positional_epsilon(self):
        epsilons = np.array([0.05, 0.1, 0.15, 0.1, 0.05, 0.1])
        evaluator = CodebookEvaluator(
            codebook=_make_codebook(),
            noise_channel=PositionVaryingEpsilon(epsilons=epsilons, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        _assert_cpu_gpu_cache_match(evaluator)


# ---------------------------------------------------------------------------
# Sample batching on GPU
# ---------------------------------------------------------------------------

@requires_gpu
class TestGPUCacheSampleBatching:

    def test_sample_batching_matches_unbatched(self):
        """GPU cache with sample_batch_size must match unbatched."""
        codebook = _make_codebook()

        # Full batch
        eval_full = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        from duet.gpu_cache import initialize_cache_gpu
        initialize_cache_gpu(eval_full, sample_batch_size=N_SAMPLES)

        # Batched
        eval_batched = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        initialize_cache_gpu(eval_batched, sample_batch_size=50)

        # Compare
        np.testing.assert_array_equal(
            eval_full._sparse_matrix.toarray(),
            eval_batched._sparse_matrix.toarray(),
        )
        np.testing.assert_array_equal(
            eval_full._no_error_flags,
            eval_batched._no_error_flags,
        )


@requires_gpu
class TestGPUCacheStoreObserved:
    """Verify store_observed=True retains _observed_sequences on GPU path."""

    def test_store_observed_true_retains_sequences(self):
        codebook = _make_codebook()

        # With store_observed=True
        eval_stored = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            store_observed=True,
        )
        eval_stored.initialize_cache(device="gpu")

        # CPU reference with store_observed=True
        eval_cpu = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            store_observed=True,
        )
        eval_cpu.initialize_cache(n_jobs=1, device="cpu")

        assert eval_stored._observed_sequences is not None
        np.testing.assert_array_equal(
            eval_cpu._observed_sequences,
            eval_stored._observed_sequences,
        )

    def test_store_observed_false_drops_sequences(self):
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            store_observed=False,
        )
        evaluator.initialize_cache(device="gpu")
        assert evaluator._observed_sequences is None
        assert evaluator._no_error_flags is not None


# ---------------------------------------------------------------------------
# Indices-based evaluation with union evaluator
# ---------------------------------------------------------------------------

@requires_gpu
class TestGPUCacheIndices:
    """Verify that subset evaluation via indices works after GPU cache init."""

    def test_indices_subset_accuracy(self):
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        evaluator.initialize_cache(device="gpu")

        # Full codebook accuracy
        full_acc = evaluator.get_codeword_accuracy()
        assert full_acc.shape == (N,)

        # Subset accuracy
        subset = np.array([0, 2, 5, 10, 15])
        subset_acc = evaluator.get_codeword_accuracy(indices=subset)
        assert subset_acc.shape == (len(subset),)


# ---------------------------------------------------------------------------
# Graceful failure without CuPy
# ---------------------------------------------------------------------------

class TestNoCuPyCache:
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
                evaluator.initialize_cache(device="gpu")


# ---------------------------------------------------------------------------
# Multi-GPU support
# ---------------------------------------------------------------------------

@requires_gpu
class TestMultiGPUCache:
    """Multi-GPU device strings must produce identical cache state."""

    def _init_and_compare(self, device_a, device_b):
        """Initialize cache with two device strings, compare results."""
        codebook = _make_codebook()

        eval_a = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        eval_a.initialize_cache(device=device_a)

        eval_b = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
        )
        eval_b.initialize_cache(device=device_b)

        # Compare sparse matrices
        np.testing.assert_array_equal(
            eval_a._sparse_matrix.toarray(),
            eval_b._sparse_matrix.toarray(),
        )

        # Compare no-error flags
        np.testing.assert_array_equal(
            eval_a._no_error_flags,
            eval_b._no_error_flags,
        )

        # Compare codeword row indices
        assert len(eval_a._codeword_row_indices) == len(eval_b._codeword_row_indices)
        for idx_a, idx_b in zip(eval_a._codeword_row_indices,
                                eval_b._codeword_row_indices):
            np.testing.assert_array_equal(idx_a, idx_b)

    def test_single_gpu_explicit_matches_default(self):
        """device='gpu:0' must match device='gpu'."""
        self._init_and_compare("gpu", "gpu:0")

    @pytest.mark.xfail(strict=True, reason=DUPLICATE_GPU_ID_REASON)
    def test_two_threads_same_gpu_matches_single(self):
        """device='gpu:0,0' (two threads, one GPU) must match device='gpu:0'."""
        self._init_and_compare("gpu:0", "gpu:0,0")


# ---------------------------------------------------------------------------
# Streaming cache (mmap_cache=True) vs non-streaming (mmap_cache=False)
# ---------------------------------------------------------------------------

@requires_gpu
class TestStreamingCache:
    """Streaming disk-backed path must produce identical results to in-memory path."""

    def test_streaming_matches_non_streaming(self):
        """mmap_cache=True (streaming) must match mmap_cache=False (non-streaming)."""
        codebook = _make_codebook()

        eval_stream = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=True,
        )
        eval_stream.initialize_cache(device="gpu")

        eval_mem = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=False,
        )
        eval_mem.initialize_cache(device="gpu")

        # Sparse matrices must be identical
        np.testing.assert_array_equal(
            eval_stream._sparse_matrix.toarray(),
            eval_mem._sparse_matrix.toarray(),
        )

        # No-error flags must be identical
        np.testing.assert_array_equal(
            eval_stream._no_error_flags,
            eval_mem._no_error_flags,
        )

        # Downstream accuracy must match
        np.testing.assert_array_equal(
            eval_stream.get_codeword_accuracy(),
            eval_mem.get_codeword_accuracy(),
        )

        # PEP from cache must match
        np.testing.assert_array_equal(
            eval_stream.get_pairwise_error_from_cache(),
            eval_mem.get_pairwise_error_from_cache(),
        )

        # Error correction metrics must match
        ecm_stream = eval_stream.get_error_correction_metrics()
        ecm_mem = eval_mem.get_error_correction_metrics()
        assert ecm_stream.no_error_rate == ecm_mem.no_error_rate
        assert ecm_stream.corrected_rate == ecm_mem.corrected_rate
        assert ecm_stream.failed_rate == ecm_mem.failed_rate

        eval_stream.close()
        eval_mem.close()

    def test_streaming_cleanup(self):
        """evaluator.close() must remove the streaming mmap directory."""
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=True,
        )
        evaluator.initialize_cache(device="gpu")

        import os
        mmap_dir = evaluator._mmap_dir
        assert mmap_dir is not None
        assert os.path.isdir(mmap_dir)

        evaluator.close()
        assert not os.path.isdir(mmap_dir)
        assert evaluator._mmap_dir is None

    def test_streaming_uniform_row_indices(self):
        """Streaming path must use _UniformRowIndices, returning correct ranges."""
        from duet.gpu_cache import _UniformRowIndices

        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=True,
        )
        evaluator.initialize_cache(device="gpu")

        assert isinstance(evaluator._codeword_row_indices, _UniformRowIndices)
        assert len(evaluator._codeword_row_indices) == N

        for idx in [0, N // 2, N - 1]:
            expected = np.arange(idx * N_SAMPLES, (idx + 1) * N_SAMPLES)
            np.testing.assert_array_equal(
                evaluator._codeword_row_indices[idx], expected,
            )

        evaluator.close()

    def test_streaming_store_observed(self):
        """store_observed=True with streaming must produce a memmap."""
        codebook = _make_codebook()

        eval_stream = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            store_observed=True,
            mmap_cache=True,
        )
        eval_stream.initialize_cache(device="gpu")

        assert eval_stream._observed_sequences is not None
        assert isinstance(eval_stream._observed_sequences, np.memmap)

        # Compare with non-streaming store_observed
        eval_mem = CodebookEvaluator(
            codebook=codebook.copy(),
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            store_observed=True,
            mmap_cache=False,
        )
        eval_mem.initialize_cache(device="gpu")

        np.testing.assert_array_equal(
            eval_stream._observed_sequences,
            eval_mem._observed_sequences,
        )

        eval_stream.close()

    def test_streaming_single_codeword(self):
        """Edge case: N=1 codeword with streaming path."""
        rng = np.random.default_rng(99)
        codebook = rng.integers(0, Q, size=(1, L), dtype=np.int8)

        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=True,
        )
        evaluator.initialize_cache(device="gpu")

        assert evaluator._sparse_matrix.shape == (N_SAMPLES, 1)
        acc = evaluator.get_codeword_accuracy()
        assert acc.shape == (1,)
        # Single codeword always decodes to itself
        assert acc[0] == 1.0

        evaluator.close()


# ---------------------------------------------------------------------------
# scratch_dir — mmap files land in the specified directory
# ---------------------------------------------------------------------------

@requires_gpu
class TestScratchDirGPU:
    """Verify that scratch_dir controls where GPU mmap temp files are created."""

    def test_gpu_streaming_scratch_dir(self, tmp_path):
        """GPU streaming path (mmap_cache=True) must create mmap files under scratch_dir."""
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=True,
            scratch_dir=str(tmp_path),
        )
        evaluator.initialize_cache(device="gpu")

        import os
        mmap_dir = evaluator._mmap_dir
        assert mmap_dir is not None
        # mmap_dir should be a subdirectory of the scratch_dir
        assert os.path.dirname(mmap_dir) == str(tmp_path)
        assert os.path.isdir(mmap_dir)

        # Verify results are valid
        assert evaluator._sparse_matrix is not None
        assert evaluator._sparse_matrix.shape == (N * N_SAMPLES, N)

        evaluator.close()
        assert not os.path.isdir(mmap_dir)


class TestScratchDirCPU:
    """Verify that scratch_dir controls where CPU mmap temp files are created."""

    def test_cpu_mmap_scratch_dir(self, tmp_path):
        """CPU mmap path (mmap_cache=True, device='cpu') must create mmap files under scratch_dir."""
        codebook = _make_codebook()
        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=Q),
            decoding_metric=HammingDistance(alphabet_size=Q),
            decoding_rule=UniqueMinimum(),
            n_samples=N_SAMPLES,
            seed=SEED,
            mmap_cache=True,
            scratch_dir=str(tmp_path),
        )
        evaluator.initialize_cache(n_jobs=1)

        import os
        mmap_dir = evaluator._mmap_dir
        assert mmap_dir is not None
        assert os.path.dirname(mmap_dir) == str(tmp_path)
        assert os.path.isdir(mmap_dir)

        assert evaluator._sparse_matrix is not None
        assert evaluator._sparse_matrix.shape == (N * N_SAMPLES, N)

        evaluator.close()
        assert not os.path.isdir(mmap_dir)
