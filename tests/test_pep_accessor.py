"""
Tests for PEPAccessor abstraction layer.

Tests cover:
1. InMemoryPEPAccessor - symmetrization, dtype, row/diagonal access
2. MmapPEPAccessor - roundtrip, equivalence with in-memory, pickle support
3. create_symmetric_mmap - factory function
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from duet.pep_accessor import (
    PEPAccessor,
    InMemoryPEPAccessor,
    MmapPEPAccessor,
    _estimate_symmetrize_batch_size,
    create_symmetric_mmap,
    create_symmetric_mmap_streaming,
)


# =============================================================================
# InMemoryPEPAccessor Tests
# =============================================================================


class TestInMemoryPEPAccessor:
    """Tests for InMemoryPEPAccessor."""

    def test_from_pep_matrix_symmetry(self, asymmetric_pep_matrix):
        """Verify X = M + M^T and X = X^T."""
        M = asymmetric_pep_matrix
        accessor = InMemoryPEPAccessor.from_pep_matrix(M)
        X_expected = M + M.T

        # Verify symmetry by checking every row
        for i in range(accessor.N):
            row = accessor.get_row(i)
            np.testing.assert_allclose(row, X_expected[i, :], rtol=1e-6)

        # Verify X is symmetric
        X_full = accessor.get_rows(np.arange(accessor.N))
        np.testing.assert_allclose(X_full, X_full.T, rtol=1e-6)

    def test_from_pep_matrix_dtype(self, asymmetric_pep_matrix):
        """Verify internal storage is float32."""
        accessor = InMemoryPEPAccessor.from_pep_matrix(asymmetric_pep_matrix)
        assert accessor._X.dtype == np.float32

    def test_get_row_matches_direct(self, asymmetric_pep_matrix):
        """accessor.get_row(i) matches X[i, :]."""
        M = asymmetric_pep_matrix
        X = (M + M.T).astype(np.float32)
        accessor = InMemoryPEPAccessor(X)

        for i in range(accessor.N):
            row = accessor.get_row(i)
            np.testing.assert_allclose(row, X[i, :].astype(np.float64), rtol=1e-6)

    def test_get_rows_matches_direct(self, asymmetric_pep_matrix):
        """accessor.get_rows(indices) matches X[indices, :]."""
        M = asymmetric_pep_matrix
        X = (M + M.T).astype(np.float32)
        accessor = InMemoryPEPAccessor(X)

        indices = np.array([0, 2, 3])
        rows = accessor.get_rows(indices)
        np.testing.assert_allclose(rows, X[indices, :].astype(np.float64), rtol=1e-6)

    def test_get_diagonal(self, asymmetric_pep_matrix):
        """accessor.get_diagonal() matches np.diag(X)."""
        M = asymmetric_pep_matrix
        X = (M + M.T).astype(np.float32)
        accessor = InMemoryPEPAccessor(X)

        diag = accessor.get_diagonal()
        np.testing.assert_allclose(diag, np.diag(X).astype(np.float64), rtol=1e-6)

    def test_N_property(self, asymmetric_pep_matrix):
        """Verify pool size."""
        accessor = InMemoryPEPAccessor.from_pep_matrix(asymmetric_pep_matrix)
        assert accessor.N == 4

    def test_get_row_returns_float64(self, asymmetric_pep_matrix):
        """Verify return dtype is float64."""
        accessor = InMemoryPEPAccessor.from_pep_matrix(asymmetric_pep_matrix)
        row = accessor.get_row(0)
        assert row.dtype == np.float64

    def test_get_rows_returns_float64(self, asymmetric_pep_matrix):
        """Verify batch return dtype is float64."""
        accessor = InMemoryPEPAccessor.from_pep_matrix(asymmetric_pep_matrix)
        rows = accessor.get_rows(np.array([0, 1]))
        assert rows.dtype == np.float64


# =============================================================================
# MmapPEPAccessor Tests
# =============================================================================


class TestMmapPEPAccessor:
    """Tests for MmapPEPAccessor."""

    def test_mmap_roundtrip(self, asymmetric_pep_matrix):
        """Write then read, values match."""
        M = asymmetric_pep_matrix
        X_expected = (M + M.T).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            accessor = create_symmetric_mmap(M, path)

            X_read = accessor.get_rows(np.arange(accessor.N))
            np.testing.assert_allclose(X_read, X_expected.astype(np.float64), rtol=1e-6)

    def test_mmap_get_row_matches_inmemory(self, asymmetric_pep_matrix):
        """Same results as InMemoryPEPAccessor."""
        M = asymmetric_pep_matrix
        inmem = InMemoryPEPAccessor.from_pep_matrix(M)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)

            for i in range(mmap_acc.N):
                np.testing.assert_allclose(
                    mmap_acc.get_row(i), inmem.get_row(i), rtol=1e-6
                )

    def test_mmap_get_rows_matches_inmemory(self, asymmetric_pep_matrix):
        """Same results for batch read."""
        M = asymmetric_pep_matrix
        inmem = InMemoryPEPAccessor.from_pep_matrix(M)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)

            indices = np.array([0, 2, 3])
            np.testing.assert_allclose(
                mmap_acc.get_rows(indices), inmem.get_rows(indices), rtol=1e-6
            )

    def test_mmap_get_diagonal_matches_inmemory(self, asymmetric_pep_matrix):
        """Same diagonal."""
        M = asymmetric_pep_matrix
        inmem = InMemoryPEPAccessor.from_pep_matrix(M)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)

            np.testing.assert_allclose(
                mmap_acc.get_diagonal(), inmem.get_diagonal(), rtol=1e-6
            )

    def test_mmap_diagonal_value_constant(self, asymmetric_pep_matrix):
        """Verify diagonal_value=2.0 produces correct constant diagonal."""
        M = asymmetric_pep_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path, diagonal_value=2.0)

            diag = mmap_acc.get_diagonal()
            np.testing.assert_allclose(diag, np.full(4, 2.0))

    def test_mmap_readonly(self, asymmetric_pep_matrix):
        """Verify writes are rejected."""
        M = asymmetric_pep_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M, path)

            with pytest.raises((ValueError, TypeError)):
                mmap_acc._mmap[0, 0] = 999.0

    def test_mmap_pickle_roundtrip(self, asymmetric_pep_matrix):
        """Pickle and unpickle, verify same results (critical for multiprocessing)."""
        M = asymmetric_pep_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            original = create_symmetric_mmap(M, path)

            # Get original values
            original_rows = original.get_rows(np.arange(original.N))

            # Pickle and unpickle
            data = pickle.dumps(original)
            restored = pickle.loads(data)

            # Verify same values
            restored_rows = restored.get_rows(np.arange(restored.N))
            np.testing.assert_allclose(restored_rows, original_rows, rtol=1e-6)
            assert restored.N == original.N

    def test_mmap_N_property(self, asymmetric_pep_matrix):
        """Verify pool size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            acc = create_symmetric_mmap(asymmetric_pep_matrix, path)
            assert acc.N == 4


# =============================================================================
# create_symmetric_mmap Tests
# =============================================================================


class TestCreateSymmetricMmap:
    """Tests for create_symmetric_mmap factory function."""

    def test_creates_file(self, asymmetric_pep_matrix):
        """File is created on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            create_symmetric_mmap(asymmetric_pep_matrix, path)
            assert path.exists()

    def test_file_size_correct(self, asymmetric_pep_matrix):
        """File size matches N*N*sizeof(float32)."""
        N = asymmetric_pep_matrix.shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            create_symmetric_mmap(asymmetric_pep_matrix, path)
            assert path.stat().st_size == N * N * 4  # float32

    def test_returns_accessor(self, asymmetric_pep_matrix):
        """Returns a MmapPEPAccessor instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            result = create_symmetric_mmap(asymmetric_pep_matrix, path)
            assert isinstance(result, MmapPEPAccessor)


# =============================================================================
# Uint16 InMemoryPEPAccessor Tests
# =============================================================================


class TestUint16InMemoryAccessor:
    """Tests for InMemoryPEPAccessor with uint16 count storage."""

    def test_from_count_matrix_symmetry(self, asymmetric_count_matrix):
        """Verify X = M_count + M_count^T and get_row returns X / n_samples."""
        M_counts, n_samples = asymmetric_count_matrix
        accessor = InMemoryPEPAccessor.from_count_matrix(M_counts, n_samples)
        X_counts = M_counts.astype(np.float64) + M_counts.T.astype(np.float64)
        X_expected = X_counts / n_samples

        for i in range(accessor.N):
            row = accessor.get_row(i)
            np.testing.assert_allclose(row, X_expected[i, :], rtol=1e-6)

        # Verify symmetry
        X_full = accessor.get_rows(np.arange(accessor.N))
        np.testing.assert_allclose(X_full, X_full.T, rtol=1e-6)

    def test_count_accessor_matches_float_accessor(self, asymmetric_pep_matrix, asymmetric_count_matrix):
        """uint16 accessor produces same results as float accessor."""
        M_counts, n_samples = asymmetric_count_matrix
        float_accessor = InMemoryPEPAccessor.from_pep_matrix(asymmetric_pep_matrix)
        uint16_accessor = InMemoryPEPAccessor.from_count_matrix(M_counts, n_samples)

        indices = np.arange(float_accessor.N)
        np.testing.assert_allclose(
            uint16_accessor.get_rows(indices),
            float_accessor.get_rows(indices),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            uint16_accessor.get_diagonal(),
            float_accessor.get_diagonal(),
            rtol=1e-5,
        )

    def test_uint16_storage_dtype(self, asymmetric_count_matrix):
        """Verify internal storage is uint16."""
        M_counts, n_samples = asymmetric_count_matrix
        accessor = InMemoryPEPAccessor.from_count_matrix(M_counts, n_samples)
        assert accessor._X.dtype == np.uint16

    def test_n_samples_validation(self):
        """ValueError for n_samples <= 0 or > 32767."""
        X = np.zeros((4, 4), dtype=np.uint16)
        with pytest.raises(ValueError):
            InMemoryPEPAccessor(X, n_samples=0)
        with pytest.raises(ValueError):
            InMemoryPEPAccessor(X, n_samples=-1)
        with pytest.raises(ValueError):
            InMemoryPEPAccessor(X, n_samples=32768)

    def test_overflow_assertion(self):
        """Defense-in-depth assert fires when counts would overflow uint16."""
        # Bypass n_samples validation by using a valid n_samples,
        # but craft M_counts such that M + M^T would overflow
        M_counts = np.full((2, 2), 40000, dtype=np.uint16)
        with pytest.raises(AssertionError):
            InMemoryPEPAccessor.from_count_matrix(M_counts, n_samples=32767)

    def test_from_pep_matrix_uint16_dispatch(self, asymmetric_count_matrix):
        """from_pep_matrix with uint16 input delegates to from_count_matrix."""
        M_counts, n_samples = asymmetric_count_matrix
        accessor = InMemoryPEPAccessor.from_pep_matrix(M_counts, n_samples=n_samples)
        assert accessor._X.dtype == np.uint16

    def test_from_pep_matrix_uint16_requires_n_samples(self, asymmetric_count_matrix):
        """from_pep_matrix with uint16 input raises without n_samples."""
        M_counts, _ = asymmetric_count_matrix
        with pytest.raises(ValueError, match="n_samples required"):
            InMemoryPEPAccessor.from_pep_matrix(M_counts)


# =============================================================================
# Uint16 MmapPEPAccessor Tests
# =============================================================================


class TestUint16MmapAccessor:
    """Tests for MmapPEPAccessor with uint16 count storage."""

    def test_mmap_uint16_roundtrip(self, asymmetric_count_matrix):
        """Write uint16 counts, read back, verify values."""
        M_counts, n_samples = asymmetric_count_matrix
        X_expected = (M_counts.astype(np.float64) + M_counts.T.astype(np.float64)) / n_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            accessor = create_symmetric_mmap(M_counts, path, n_samples=n_samples)

            X_read = accessor.get_rows(np.arange(accessor.N))
            np.testing.assert_allclose(X_read, X_expected, rtol=1e-6)

    def test_mmap_uint16_matches_inmemory(self, asymmetric_count_matrix):
        """Equivalence with in-memory uint16 accessor."""
        M_counts, n_samples = asymmetric_count_matrix
        inmem = InMemoryPEPAccessor.from_count_matrix(M_counts, n_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            mmap_acc = create_symmetric_mmap(M_counts, path, n_samples=n_samples)

            for i in range(mmap_acc.N):
                np.testing.assert_allclose(
                    mmap_acc.get_row(i), inmem.get_row(i), rtol=1e-6
                )

    def test_mmap_uint16_pickle_roundtrip(self, asymmetric_count_matrix):
        """n_samples preserved through pickle/unpickle."""
        M_counts, n_samples = asymmetric_count_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            original = create_symmetric_mmap(M_counts, path, n_samples=n_samples)
            original_rows = original.get_rows(np.arange(original.N))

            data = pickle.dumps(original)
            restored = pickle.loads(data)

            restored_rows = restored.get_rows(np.arange(restored.N))
            np.testing.assert_allclose(restored_rows, original_rows, rtol=1e-6)
            assert restored.N == original.N
            assert restored._n_samples == n_samples

    def test_mmap_file_size_uint16(self, asymmetric_count_matrix):
        """File size = N * N * 2 bytes for uint16."""
        M_counts, n_samples = asymmetric_count_matrix
        N = M_counts.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            create_symmetric_mmap(M_counts, path, n_samples=n_samples)
            assert path.stat().st_size == N * N * 2  # uint16

    def test_mmap_uint16_diagonal_value(self, asymmetric_count_matrix):
        """diagonal_value=2.0 still returns 2.0 in probability space."""
        M_counts, n_samples = asymmetric_count_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.dat"
            accessor = create_symmetric_mmap(
                M_counts, path, diagonal_value=2.0, n_samples=n_samples,
            )
            diag = accessor.get_diagonal()
            np.testing.assert_allclose(diag, np.full(4, 2.0))


# =============================================================================
# create_symmetric_mmap_streaming Tests
# =============================================================================


class TestCreateSymmetricMmapStreaming:
    """Tests for streaming symmetric mmap creation from .npy files."""

    def test_streaming_matches_dense(self, asymmetric_count_matrix):
        """Streaming result matches dense create_symmetric_mmap row-by-row."""
        M_counts, n_samples = asymmetric_count_matrix
        N = M_counts.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Dense reference
            dense_path = Path(tmpdir) / "dense.dat"
            dense_acc = create_symmetric_mmap(
                M_counts, dense_path, n_samples=n_samples,
            )

            # Save M to .npy for streaming
            raw_path = Path(tmpdir) / "raw.npy"
            np.save(raw_path, M_counts)

            # Streaming
            stream_path = Path(tmpdir) / "stream.dat"
            stream_acc = create_symmetric_mmap_streaming(
                raw_path, stream_path, N, n_samples=n_samples,
            )

            for i in range(N):
                np.testing.assert_allclose(
                    stream_acc.get_row(i), dense_acc.get_row(i), rtol=1e-6,
                )

    def test_streaming_various_batch_sizes(self, asymmetric_count_matrix):
        """batch_size=1, 2, 3, N all produce identical results."""
        M_counts, n_samples = asymmetric_count_matrix
        N = M_counts.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.npy"
            np.save(raw_path, M_counts)

            # Reference with default batch_size
            ref_path = Path(tmpdir) / "ref.dat"
            ref_acc = create_symmetric_mmap_streaming(
                raw_path, ref_path, N, n_samples=n_samples,
            )
            ref_rows = ref_acc.get_rows(np.arange(N))

            for bs in [1, 2, 3, N]:
                out_path = Path(tmpdir) / f"bs{bs}.dat"
                acc = create_symmetric_mmap_streaming(
                    raw_path, out_path, N, batch_size=bs, n_samples=n_samples,
                )
                np.testing.assert_allclose(
                    acc.get_rows(np.arange(N)), ref_rows, rtol=1e-6,
                )

    def test_streaming_diagonal_value(self, asymmetric_count_matrix):
        """diagonal_value=2.0 works correctly with streaming."""
        M_counts, n_samples = asymmetric_count_matrix
        N = M_counts.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.npy"
            np.save(raw_path, M_counts)

            stream_path = Path(tmpdir) / "stream.dat"
            acc = create_symmetric_mmap_streaming(
                raw_path, stream_path, N,
                diagonal_value=2.0, n_samples=n_samples,
            )
            diag = acc.get_diagonal()
            np.testing.assert_allclose(diag, np.full(N, 2.0))

    def test_streaming_overflow_raises_valueerror(self):
        """Overflow in M + M^T raises ValueError, not AssertionError."""
        N = 2
        # Craft M where M + M^T > 65535 for at least one element
        M = np.array([[0, 40000], [30000, 0]], dtype=np.uint16)
        # M[0,1] + M[1,0] = 40000 + 30000 = 70000 > 65535

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.npy"
            np.save(raw_path, M)

            stream_path = Path(tmpdir) / "stream.dat"
            with pytest.raises(ValueError, match="Overflow"):
                create_symmetric_mmap_streaming(
                    raw_path, stream_path, N, n_samples=1000,
                )

    def test_streaming_mem_budget_gb(self, asymmetric_count_matrix):
        """mem_budget_gb auto-computes batch_size; result matches explicit batch_size."""
        M_counts, n_samples = asymmetric_count_matrix
        N = M_counts.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.npy"
            np.save(raw_path, M_counts)

            # Auto-computed batch from budget
            auto_path = Path(tmpdir) / "auto.dat"
            auto_acc = create_symmetric_mmap_streaming(
                raw_path, auto_path, N, n_samples=n_samples,
                mem_budget_gb=8.0,
            )

            # Explicit batch_size=1 (smallest possible)
            explicit_path = Path(tmpdir) / "explicit.dat"
            explicit_acc = create_symmetric_mmap_streaming(
                raw_path, explicit_path, N, batch_size=1, n_samples=n_samples,
            )

            np.testing.assert_allclose(
                auto_acc.get_rows(np.arange(N)),
                explicit_acc.get_rows(np.arange(N)),
                rtol=1e-6,
            )

    def test_streaming_batch_size_overrides_budget(self, asymmetric_count_matrix):
        """Explicit batch_size takes precedence over mem_budget_gb."""
        M_counts, n_samples = asymmetric_count_matrix
        N = M_counts.shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.npy"
            np.save(raw_path, M_counts)

            path = Path(tmpdir) / "test.dat"
            acc = create_symmetric_mmap_streaming(
                raw_path, path, N, batch_size=1,
                mem_budget_gb=0.0001,
                n_samples=n_samples,
            )
            assert acc.N == N


class TestEstimateSymmetrizeBatchSize:
    """Tests for _estimate_symmetrize_batch_size."""

    def test_basic_calculation(self):
        """B = floor(budget / (14 * N))."""
        N = 154000
        budget = 2 * 10**9  # 2 GB
        B = _estimate_symmetrize_batch_size(N, budget)
        expected = int(2e9 / (14 * 154000))
        assert B == expected

    def test_clamp_min_1(self):
        """Tiny budget still returns at least 1."""
        B = _estimate_symmetrize_batch_size(100000, 1)
        assert B == 1

    def test_clamp_max_N(self):
        """Huge budget caps at N."""
        B = _estimate_symmetrize_batch_size(10, 10**12)
        assert B == 10

    def test_zero_N(self):
        """N=0 returns 0 (degenerate case)."""
        B = _estimate_symmetrize_batch_size(0, 10**9)
        assert B == 0
