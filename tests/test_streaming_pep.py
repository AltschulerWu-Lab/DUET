"""Tests for streaming PEP cache: output parameter and atomic rename."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from duet.codebook_evaluator import CodebookEvaluator
from duet.providers import PEPMatrixProvider, _save_metadata, _default_metadata
from duet.evaluator_config import EvaluatorConfig, ComponentConfig


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


@pytest.fixture
def tiny_library():
    """Minimal library for provider tests."""
    return ["ATCG", "GCTA", "TTAA"]


@pytest.fixture
def default_config():
    return EvaluatorConfig(
        noise_channel=ComponentConfig(type="symmetric", params={"epsilon": 0.1}),
        decoding_metric=ComponentConfig(type="hamming", params={}),
        decoding_rule=ComponentConfig(type="unique_minimum", params={}),
        num_samples=100,
    )


# =============================================================================
# (a) TestOutputParameter — in compute_pep_matrix()
# =============================================================================


class TestOutputParameter:
    """Tests for the output parameter of compute_pep_matrix()."""

    def test_output_numpy_array_matches(self):
        """compute_pep_matrix(output=np.zeros(...)) produces identical results to output=None."""
        evaluator1 = _make_small_evaluator(n_codewords=15, n_samples=80, seed=123)
        evaluator2 = _make_small_evaluator(n_codewords=15, n_samples=80, seed=123)

        # Without output parameter
        counts_ref, ns_ref = evaluator1.compute_pep_matrix(n_jobs=1)

        # With pre-allocated output
        N = evaluator2.num_codewords
        pre_alloc = np.zeros((N, N), dtype=np.uint16)
        counts_out, ns_out = evaluator2.compute_pep_matrix(n_jobs=1, output=pre_alloc)

        assert ns_ref == ns_out
        np.testing.assert_array_equal(counts_ref, counts_out)
        # Returned array should be the same object as the pre-allocated one
        assert counts_out is pre_alloc

    def test_output_mmap_matches(self, tmp_path):
        """compute_pep_matrix(output=open_memmap(...)) produces identical results."""
        evaluator1 = _make_small_evaluator(n_codewords=15, n_samples=80, seed=123)
        evaluator2 = _make_small_evaluator(n_codewords=15, n_samples=80, seed=123)

        # Reference
        counts_ref, ns_ref = evaluator1.compute_pep_matrix(n_jobs=1)

        # With mmap output
        N = evaluator2.num_codewords
        mmap_path = tmp_path / "test_output.npy"
        output_mmap = np.lib.format.open_memmap(
            str(mmap_path), mode='w+', dtype=np.uint16, shape=(N, N),
        )

        counts_out, ns_out = evaluator2.compute_pep_matrix(n_jobs=1, output=output_mmap)

        assert ns_ref == ns_out
        np.testing.assert_array_equal(counts_ref, counts_out)

        # Verify the .npy file on disk is loadable and matches
        del output_mmap
        loaded = np.load(mmap_path)
        np.testing.assert_array_equal(counts_ref, loaded)

    def test_output_wrong_shape_raises(self):
        """Wrong shape raises ValueError."""
        evaluator = _make_small_evaluator(n_codewords=10, n_samples=50, seed=42)
        wrong_shape = np.zeros((5, 5), dtype=np.uint16)

        with pytest.raises(ValueError, match="output shape"):
            evaluator.compute_pep_matrix(n_jobs=1, output=wrong_shape)

    def test_output_wrong_dtype_raises(self):
        """Wrong dtype raises ValueError."""
        evaluator = _make_small_evaluator(n_codewords=10, n_samples=50, seed=42)
        N = evaluator.num_codewords
        wrong_dtype = np.zeros((N, N), dtype=np.float64)

        with pytest.raises(ValueError, match="output dtype"):
            evaluator.compute_pep_matrix(n_jobs=1, output=wrong_dtype)

    def test_flush_called_on_mmap(self, tmp_path):
        """When output has flush(), verify it's called."""
        evaluator = _make_small_evaluator(n_codewords=10, n_samples=50, seed=42)
        N = evaluator.num_codewords

        mmap_path = tmp_path / "test_flush.npy"
        output_mmap = np.lib.format.open_memmap(
            str(mmap_path), mode='w+', dtype=np.uint16, shape=(N, N),
        )

        # Spy on the flush method
        original_flush = output_mmap.flush
        flush_called = [False]

        def spy_flush():
            flush_called[0] = True
            return original_flush()

        output_mmap.flush = spy_flush

        evaluator.compute_pep_matrix(n_jobs=1, output=output_mmap)

        assert flush_called[0], "flush() was not called on the mmap output"


# =============================================================================
# (b) TestAtomicRenameProvider — in PEPMatrixProvider.get()
# =============================================================================


class TestAtomicRenameProvider:
    """Tests for atomic rename in PEPMatrixProvider.get()."""

    def test_get_writes_npy_and_metadata(self, tmp_path, tiny_library, default_config):
        """After get(), .npy is loadable, .meta.json has expected keys, no .tmp.npy remains."""
        provider = PEPMatrixProvider(cache_dir=tmp_path, enabled=True)

        count_matrix, n_samples = provider.get(
            tiny_library, default_config, alphabet_size=4, n_jobs=1
        )

        # Find the cache files
        fingerprint = provider._compute_fingerprint(
            tiny_library, default_config, alphabet_size=4
        )
        npy_path = provider.cache.get_path(fingerprint, ".npy")
        meta_path = provider.cache.get_path(fingerprint, ".meta.json")
        tmp_path_file = npy_path.parent / (npy_path.stem + ".tmp.npy")

        # .npy is loadable
        assert npy_path.exists()
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(loaded, count_matrix)

        # .meta.json has expected keys
        assert meta_path.exists()
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        assert "n_samples" in meta
        assert "artifact_type" in meta
        assert meta["artifact_type"] == "PEPMatrix"
        assert "shape" in meta

        # No .tmp.npy remains
        assert not tmp_path_file.exists()

    def test_stale_tmp_overwritten(self, tmp_path, tiny_library, default_config):
        """Pre-write a garbage .tmp.npy, call get() -> .npy is valid, .tmp.npy gone."""
        provider = PEPMatrixProvider(cache_dir=tmp_path, enabled=True)

        fingerprint = provider._compute_fingerprint(
            tiny_library, default_config, alphabet_size=4
        )
        npy_path = provider.cache.get_path(fingerprint, ".npy")
        tmp_path_file = npy_path.parent / (npy_path.stem + ".tmp.npy")

        # Write garbage to .tmp.npy
        tmp_path_file.write_bytes(b"garbage data that is not a valid npy file")
        assert tmp_path_file.exists()

        count_matrix, n_samples = provider.get(
            tiny_library, default_config, alphabet_size=4, n_jobs=1
        )

        # .npy is valid
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(loaded, count_matrix)

        # .tmp.npy is gone (renamed to .npy)
        assert not tmp_path_file.exists()

    def test_caching_disabled_no_files(self, tmp_path, tiny_library, default_config):
        """enabled=False -> no .npy, no .tmp.npy, no .meta.json. Correct result returned."""
        provider = PEPMatrixProvider(cache_dir=tmp_path, enabled=False)

        count_matrix, n_samples = provider.get(
            tiny_library, default_config, alphabet_size=4, n_jobs=1
        )

        # Correct result returned
        assert count_matrix.dtype == np.uint16
        assert count_matrix.shape == (3, 3)
        assert isinstance(n_samples, (int, np.integer))

        # No cache files created
        fingerprint = provider._compute_fingerprint(
            tiny_library, default_config, alphabet_size=4
        )
        npy_path = provider.cache.get_path(fingerprint, ".npy")
        meta_path = provider.cache.get_path(fingerprint, ".meta.json")
        tmp_path_file = npy_path.parent / (npy_path.stem + ".tmp.npy")

        assert not npy_path.exists()
        assert not meta_path.exists()
        assert not tmp_path_file.exists()

    def test_result_matches_no_cache(self, tmp_path, tiny_library):
        """Compare get(enabled=True) result vs get(enabled=False) result — must be array_equal."""
        # Use a fixed seed so both providers produce identical Monte Carlo results
        config = EvaluatorConfig(
            noise_channel=ComponentConfig(type="symmetric", params={"epsilon": 0.1}),
            decoding_metric=ComponentConfig(type="hamming", params={}),
            decoding_rule=ComponentConfig(type="unique_minimum", params={}),
            num_samples=100,
            seed=42,
        )

        cache_dir_enabled = tmp_path / "enabled"
        cache_dir_enabled.mkdir()
        cache_dir_disabled = tmp_path / "disabled"
        cache_dir_disabled.mkdir()

        provider_enabled = PEPMatrixProvider(cache_dir=cache_dir_enabled, enabled=True)
        provider_disabled = PEPMatrixProvider(cache_dir=cache_dir_disabled, enabled=False)

        count_enabled, ns_enabled = provider_enabled.get(
            tiny_library, config, alphabet_size=4, n_jobs=1
        )
        count_disabled, ns_disabled = provider_disabled.get(
            tiny_library, config, alphabet_size=4, n_jobs=1
        )

        assert ns_enabled == ns_disabled
        np.testing.assert_array_equal(count_enabled, count_disabled)
