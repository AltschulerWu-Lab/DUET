"""Tests for PEPMatrixProvider cache resilience.

Verifies that missing or incomplete metadata files do not crash cache loading.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from duet.providers import PEPMatrixProvider, _save_metadata, _default_metadata
from duet.evaluator_config import EvaluatorConfig, ComponentConfig


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


class TestPEPMatrixProviderMissingMetadata:
    """Regression tests for missing .meta.json alongside cached .npy / .sym.dat."""

    def test_get_recomputes_when_metadata_missing(
        self, tmp_path, tiny_library, default_config
    ):
        """get() should recompute when .npy exists but .meta.json is missing."""
        provider = PEPMatrixProvider(cache_dir=tmp_path, enabled=True)

        # Pre-populate a fake .npy cache file with NO accompanying .meta.json
        fingerprint = provider._compute_fingerprint(
            tiny_library, default_config, alphabet_size=4
        )
        npy_path = provider.cache.get_path(fingerprint, ".npy")
        fake_matrix = np.zeros((3, 3), dtype=np.uint16)
        np.save(npy_path, fake_matrix)

        # Verify the .meta.json does NOT exist
        meta_path = provider.cache.get_path(fingerprint, ".meta.json")
        assert not meta_path.exists()

        # get() should NOT crash — it should fall through to recomputation
        count_matrix, n_samples = provider.get(
            tiny_library, default_config, alphabet_size=4, n_jobs=1
        )

        # Should have actually computed a real result
        assert count_matrix.dtype == np.uint16
        assert count_matrix.shape == (3, 3)
        assert isinstance(n_samples, (int, np.integer))

        # Metadata should now exist after recomputation + save
        assert meta_path.exists()

    def test_get_recomputes_when_metadata_incomplete(
        self, tmp_path, tiny_library, default_config
    ):
        """get() should recompute when .meta.json exists but lacks n_samples."""
        provider = PEPMatrixProvider(cache_dir=tmp_path, enabled=True)

        fingerprint = provider._compute_fingerprint(
            tiny_library, default_config, alphabet_size=4
        )
        npy_path = provider.cache.get_path(fingerprint, ".npy")
        meta_path = provider.cache.get_path(fingerprint, ".meta.json")

        # Create .npy and a metadata file that is missing 'n_samples'
        fake_matrix = np.zeros((3, 3), dtype=np.uint16)
        np.save(npy_path, fake_matrix)
        _save_metadata(meta_path, {"created_at": "2024-01-01", "artifact_type": "PEPMatrix"})

        count_matrix, n_samples = provider.get(
            tiny_library, default_config, alphabet_size=4, n_jobs=1
        )

        assert count_matrix.dtype == np.uint16
        assert isinstance(n_samples, (int, np.integer))

    def test_get_mmap_recomputes_when_metadata_missing(
        self, tmp_path, tiny_library, default_config
    ):
        """get_mmap() should recompute when .sym.dat exists but .sym.meta.json is missing."""
        provider = PEPMatrixProvider(cache_dir=tmp_path, enabled=True)

        fingerprint = provider._compute_fingerprint(
            tiny_library, default_config, alphabet_size=4
        )
        sym_path = provider.cache.get_path(fingerprint, ".sym.dat")
        sym_meta_path = provider.cache.get_path(fingerprint, ".sym.meta.json")

        # Create a fake .sym.dat file with no metadata
        N = len(tiny_library)
        fake_sym = np.zeros((N, N), dtype=np.uint16)
        fake_sym.tofile(sym_path)

        assert not sym_meta_path.exists()

        # get_mmap() should NOT crash — it should fall through to recomputation
        accessor = provider.get_mmap(
            tiny_library, default_config, alphabet_size=4, n_jobs=1
        )

        # Should return a working accessor
        assert accessor is not None
        row = accessor.get_row(0)
        assert row.dtype == np.float64

        # Metadata should now exist
        assert sym_meta_path.exists()
