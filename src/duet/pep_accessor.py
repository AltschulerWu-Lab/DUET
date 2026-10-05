"""
PEP Access Abstraction Layer.

Provides a uniform interface for accessing the symmetrized PEP matrix
X = M + M^T, where M is the raw pairwise error probability matrix.
This enables DecodingSwapCache to work identically with in-memory or
disk-backed (memory-mapped) storage.

Key classes:
    PEPAccessor (ABC): Abstract interface for row-based X access
    InMemoryPEPAccessor: Wraps dense ndarray X in memory
    MmapPEPAccessor: Wraps numpy.memmap file (read-only)

Factory functions:
    create_symmetric_mmap: Create a memory-mapped symmetric PEP file
    create_symmetric_mmap_streaming: Streaming variant (constant-memory)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tqdm import tqdm

from duet.utils import log_memory

logger = logging.getLogger(__name__)

MAX_SAMPLES_UINT16 = 32767  # 2 * n_samples must fit in uint16


# =============================================================================
# PEPAccessor ABC
# =============================================================================


class PEPAccessor(ABC):
    """Abstract interface for accessing the symmetrized PEP matrix X = M + M^T.

    All implementations provide:
    - Row access: read a contiguous row X[i, :] as a length-N ndarray
    - Diagonal access: X[i, i] for any i
    - Pool size N

    The key design contract is that callers access X one row at a time,
    enabling efficient sequential I/O for disk-backed implementations.
    """

    @property
    @abstractmethod
    def N(self) -> int:
        """Pool size (number of candidates)."""
        ...

    @abstractmethod
    def get_row(self, i: int) -> np.ndarray:
        """Return X[i, :] as a 1D float64 array of length N."""
        ...

    @abstractmethod
    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        """Return X[indices, :] as a 2D float64 array of shape (len(indices), N)."""
        ...

    @abstractmethod
    def get_diagonal(self) -> np.ndarray:
        """Return diag(X) as a 1D float64 array of length N."""
        ...

    def get_row_by_candidate(self, c: int) -> np.ndarray:
        """Return the X row for candidate c, translating via candidate_to_sequence_idx
        if the accessor was constructed with one (PEP dedup mode). When the
        translation map is absent (U == pool_size), delegates to get_row(c)."""
        if self._c_to_u is None:
            return self.get_row(int(c))
        return self.get_row(int(self._c_to_u[c]))

    def get_rows_by_candidate(self, cs: np.ndarray) -> np.ndarray:
        """Return X rows for an array of candidate indices, translating via
        candidate_to_sequence_idx if present."""
        if self._c_to_u is None:
            return self.get_rows(np.asarray(cs))
        return self.get_rows(self._c_to_u[np.asarray(cs)])


# =============================================================================
# InMemoryPEPAccessor
# =============================================================================


class InMemoryPEPAccessor(PEPAccessor):
    """Wraps a dense ndarray X = M + M^T in memory.

    The matrix X is stored as raw symmetrized PEP values with no modifications
    (no duplicate penalty, no group-level masking). Modifications are handled
    externally by DecodingSwapCache.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_samples: int | None = None,
        duplicate_offset: float = 0.0,
        candidate_to_sequence_idx: np.ndarray | None = None,
    ):
        """Initialize from a precomputed symmetric matrix X.

        Args:
            X: Symmetric 2D ndarray of shape (N, N). Either uint16 counts
               (when n_samples is not None) or float32 probabilities (when n_samples is None).
            n_samples: Monte Carlo sample count. If not None, get_row/get_rows divide
               stored counts by n_samples to return probabilities. If None, values
               are already probabilities (legacy float path).
            duplicate_offset: Scalar overlay applied to the diagonal at row-read
               time. The stored X is NOT mutated; get_diagonal returns raw + offset
               and get_row/get_rows apply offset only at the row's diagonal slot.
               After PEP dedup (one row per unique sequence), duplicates are
               discouraged via this diagonal overlay rather than off-diagonal
               penalties.
            candidate_to_sequence_idx: Optional int32 array of shape (pool_size,)
               mapping candidate index -> sequence (row) index. When provided,
               enables get_row_by_candidate / get_rows_by_candidate to translate
               candidate -> sequence at the call boundary. None means U == pool_size
               (no dedup; candidate-indexed methods fall through to sequence-indexed).
        """
        if n_samples is not None and (n_samples <= 0 or n_samples > MAX_SAMPLES_UINT16):
            raise ValueError(
                f"n_samples must be None (legacy float) or in [1, {MAX_SAMPLES_UINT16}], "
                f"got {n_samples}"
            )
        self._X = X
        self._n_samples = n_samples
        self._duplicate_offset = float(duplicate_offset)
        self._c_to_u = candidate_to_sequence_idx
        if n_samples is not None:
            self._raw_diag = np.diag(X).astype(np.float64) / n_samples
        else:
            self._raw_diag = np.diag(X).astype(np.float64).copy()

    @classmethod
    def from_count_matrix(
        cls,
        M_counts: np.ndarray,
        n_samples: int,
        duplicate_offset: float = 0.0,
        candidate_to_sequence_idx: np.ndarray | None = None,
    ) -> InMemoryPEPAccessor:
        """Construct from raw asymmetric uint16 count matrix.

        Computes X_count = M_count + M_count^T. Since n_samples <= 32767
        (validated), max(X) <= 2 * 32767 = 65534 < 65536, so uint16
        addition cannot overflow.
        """
        # Check in uint32 to detect overflow (uint16 + uint16 wraps silently)
        X_check = M_counts.astype(np.uint32) + M_counts.T.astype(np.uint32)
        assert X_check.max() <= np.iinfo(np.uint16).max, (
            f"Symmetrized count max={X_check.max()} exceeds uint16 range. "
            f"This should not happen if n_samples <= {MAX_SAMPLES_UINT16}."
        )
        X = X_check.astype(np.uint16)
        return cls(
            X, n_samples,
            duplicate_offset=duplicate_offset,
            candidate_to_sequence_idx=candidate_to_sequence_idx,
        )

    @classmethod
    def from_pep_matrix(
        cls,
        M: np.ndarray,
        n_samples: int | None = None,
        duplicate_offset: float = 0.0,
        candidate_to_sequence_idx: np.ndarray | None = None,
    ) -> InMemoryPEPAccessor:
        """Construct from raw asymmetric PEP matrix M.

        If M is uint16, delegates to from_count_matrix (n_samples required).
        If M is float, uses legacy float32 path (backward compatible).
        """
        if M.dtype == np.uint16:
            if n_samples is None:
                raise ValueError("n_samples required when M is uint16 count matrix")
            return cls.from_count_matrix(
                M, n_samples,
                duplicate_offset=duplicate_offset,
                candidate_to_sequence_idx=candidate_to_sequence_idx,
            )
        X = (M + M.T).astype(np.float32)
        return cls(
            X, n_samples=None,
            duplicate_offset=duplicate_offset,
            candidate_to_sequence_idx=candidate_to_sequence_idx,
        )

    @property
    def N(self) -> int:
        return self._X.shape[0]

    def get_row(self, i: int) -> np.ndarray:
        row = self._X[i, :].astype(np.float64)
        if self._n_samples is not None:
            row /= self._n_samples
        row[i] += self._duplicate_offset
        return row

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        rows = self._X[indices, :].astype(np.float64)
        if self._n_samples is not None:
            rows /= self._n_samples
        rows[np.arange(len(indices)), indices] += self._duplicate_offset
        return rows

    def get_diagonal(self) -> np.ndarray:
        return self._raw_diag + self._duplicate_offset


# =============================================================================
# MmapPEPAccessor
# =============================================================================


class MmapPEPAccessor(PEPAccessor):
    """Wraps a memory-mapped file containing X = M + M^T.

    The file stores X as a contiguous row-major float32 array. Since X is
    symmetric, column access X[:, j] = row access X[j, :], so a single file
    suffices.

    The file is opened read-only, making it safe to share across multiple
    worker processes via the OS page cache.
    """

    def __init__(
        self,
        path: str | Path,
        N: int,
        dtype: np.dtype = np.float32,
        diagonal_value: float | None = None,
        n_samples: int | None = None,
        duplicate_offset: float = 0.0,
        candidate_to_sequence_idx: np.ndarray | None = None,
    ):
        """Initialize from a memory-mapped file.

        Args:
            path: Path to the mmap file.
            N: Pool size (number of candidates).
            dtype: Data type of stored values.
            diagonal_value: If provided, use this constant for all diagonal
                entries instead of reading from disk. For k > 0 (the standard
                case), X(s,s) = 2.0, so pass diagonal_value=2.0 to avoid
                N scattered reads during initialization.
            n_samples: Monte Carlo sample count. If not None, get_row/get_rows
                divide by n_samples. If None, values are already probabilities.
            duplicate_offset: Scalar overlay applied to the diagonal at row-read
                time. The stored mmap is NOT mutated; get_diagonal returns
                raw + offset and get_row/get_rows apply offset only at the row's
                diagonal slot.
            candidate_to_sequence_idx: Optional int32 array mapping candidate
                index -> sequence (row) index. Enables candidate-indexed access
                methods on the ABC when set. None means U == pool_size.
        """
        self._path = Path(path)
        self._mmap = np.memmap(path, dtype=dtype, mode='r', shape=(N, N))
        self._N = N
        self._dtype = dtype
        self._diagonal_value = diagonal_value
        self._n_samples = n_samples
        self._duplicate_offset = float(duplicate_offset)
        self._c_to_u = candidate_to_sequence_idx
        self._raw_diag = None  # Lazy-loaded

    @property
    def N(self) -> int:
        return self._N

    def get_row(self, i: int) -> np.ndarray:
        row = self._mmap[i, :].astype(np.float64)
        if self._n_samples is not None:
            row /= self._n_samples
        row[i] += self._duplicate_offset
        return row

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        rows = self._mmap[indices, :].astype(np.float64)
        if self._n_samples is not None:
            rows /= self._n_samples
        rows[np.arange(len(indices)), indices] += self._duplicate_offset
        return rows

    def get_diagonal(self) -> np.ndarray:
        if self._raw_diag is None:
            if self._diagonal_value is not None:
                self._raw_diag = np.full(self._N, self._diagonal_value, dtype=np.float64)
            else:
                raw = np.array(
                    [self._mmap[i, i] for i in range(self._N)], dtype=np.float64
                )
                if self._n_samples is not None:
                    raw /= self._n_samples
                self._raw_diag = raw
        return self._raw_diag + self._duplicate_offset

    def __getstate__(self):
        """Pickle support: serialize path + metadata, not the mmap."""
        return {
            'path': str(self._path),
            'N': self._N,
            'dtype': self._dtype,
            'diagonal_value': self._diagonal_value,
            'n_samples': self._n_samples,
            'duplicate_offset': self._duplicate_offset,
            'candidate_to_sequence_idx': self._c_to_u,
        }

    def __setstate__(self, state):
        """Unpickle: re-open mmap from path (read-only)."""
        self._path = Path(state['path'])
        self._N = state['N']
        self._dtype = state['dtype']
        self._diagonal_value = state.get('diagonal_value')
        self._n_samples = state.get('n_samples', None)
        self._duplicate_offset = float(state.get('duplicate_offset', 0.0))
        self._c_to_u = state.get('candidate_to_sequence_idx', None)
        self._mmap = np.memmap(
            self._path, dtype=self._dtype, mode='r', shape=(self._N, self._N)
        )
        self._raw_diag = None


# =============================================================================
# posix_fadvise helpers
# =============================================================================


def _fadvise_sequential(path):
    """Hint OS to use aggressive read-ahead (FADV_SEQUENTIAL)."""
    if not hasattr(os, 'posix_fadvise'):
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    finally:
        os.close(fd)


def _fadvise_dontneed(path):
    """Advise OS to drop page cache for this file (FADV_DONTNEED)."""
    if not hasattr(os, 'posix_fadvise'):
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# =============================================================================
# Factory Function
# =============================================================================


def create_symmetric_mmap(
    M: np.ndarray,
    path: str | Path,
    dtype: np.dtype = None,
    diagonal_value: float | None = None,
    n_samples: int | None = None,
    duplicate_offset: float = 0.0,
) -> MmapPEPAccessor:
    """Create a memory-mapped symmetric PEP file from raw PEP matrix M.

    Computes X = M + M^T and writes to disk as a flat row-major array.

    Args:
        M: Raw asymmetric PEP matrix of shape (N, N). uint16 counts or float.
        path: Output file path for the mmap file.
        dtype: Data type for storage. If None, inferred from M (uint16 for
            uint16 input, float32 for float input).
        diagonal_value: If provided, passed to MmapPEPAccessor for
            constant diagonal optimization.
        n_samples: Monte Carlo sample count for uint16 count mode.

    Returns:
        MmapPEPAccessor wrapping the created file.
    """
    N = M.shape[0]
    if M.dtype == np.uint16 and n_samples is not None:
        # Check in uint32 to detect overflow (uint16 + uint16 wraps silently)
        X_check = M.astype(np.uint32) + M.T.astype(np.uint32)
        assert X_check.max() <= np.iinfo(np.uint16).max, (
            f"Symmetrized count max={X_check.max()} exceeds uint16 range."
        )
        X = X_check.astype(np.uint16)
        dtype = np.uint16
    else:
        dtype = dtype or np.float32
        X = (M + M.T).astype(dtype)
    fp = np.memmap(path, dtype=dtype, mode='w+', shape=(N, N))
    fp[:] = X
    fp.flush()
    del fp
    return MmapPEPAccessor(
        path, N, dtype,
        diagonal_value=diagonal_value,
        n_samples=n_samples,
        duplicate_offset=duplicate_offset,
    )


def _estimate_symmetrize_batch_size(N: int, mem_budget_bytes: int) -> int:
    """Estimate batch size for streaming symmetrization given a memory budget.

    The dominant cost is Pass 2 (symmetrize), which allocates three (B, N)
    uint32 buffers (rows_M, rows_MT, sym_block) plus a transient (B, N)
    uint16 buffer from ``.astype(np.uint16)``.
    Total = (3×4 + 1×2) × B × N = 14 × B × N bytes.

    Args:
        N: Matrix dimension (number of codewords).
        mem_budget_bytes: Target peak anonymous memory in bytes.

    Returns:
        Batch size B (at least 1, at most N).
    """
    bytes_per_row = 14 * N
    if bytes_per_row <= 0:
        return N
    batch = int(mem_budget_bytes / bytes_per_row)
    return max(1, min(batch, N))


def create_symmetric_mmap_streaming(
    raw_path: str | Path,
    sym_path: str | Path,
    N: int,
    dtype: np.dtype = np.uint16,
    batch_size: int | None = None,
    mem_budget_gb: float = 8.0,
    diagonal_value: float | None = None,
    n_samples: int | None = None,
    duplicate_offset: float = 0.0,
) -> MmapPEPAccessor:
    """Create a symmetric PEP mmap from a raw .npy file using streaming.

    Two-pass block-transpose approach — all I/O is sequential:
      Pass 1: Read M in row strips, scatter-write transposed (B×B) tiles
              to a temporary M^T file on disk.
      Pass 2: Read M and M^T both sequentially, write X = M + M^T.

    Temporary disk cost: N×N×2 bytes (deleted after). Peak RAM is controlled
    by ``batch_size`` / ``mem_budget_gb``.

    Args:
        raw_path: Path to the raw asymmetric .npy file (uint16 counts).
        sym_path: Output path for the symmetric mmap file.
        N: Matrix dimension (number of codewords).
        dtype: Storage dtype for the output (default uint16).
        batch_size: Number of rows per batch.  If None, computed from
            ``mem_budget_gb``.
        mem_budget_gb: Target peak anonymous memory per batch in GB
            (default 8).  Ignored when ``batch_size`` is set.
        diagonal_value: If provided, passed to MmapPEPAccessor for
            constant diagonal optimization.
        n_samples: Monte Carlo sample count for uint16 count mode.

    Returns:
        MmapPEPAccessor wrapping the created file.

    Raises:
        ValueError: If M + M^T overflows uint16 in any batch.
    """
    if batch_size is None:
        batch_size = _estimate_symmetrize_batch_size(N, int(mem_budget_gb * 1e9))
        logger.info(
            "Symmetrize batch_size auto-computed: B=%d (N=%d, budget=%.1f GB)",
            batch_size, N, mem_budget_gb,
        )

    # np.load with mmap_mode='r' handles .npy headers correctly.
    # Do NOT use np.memmap directly — it doesn't understand .npy headers.
    M_mmap = np.load(str(raw_path), mmap_mode='r')
    MT_path = Path(str(sym_path) + '.T.tmp')
    MT_mmap = np.memmap(str(MT_path), dtype=M_mmap.dtype, mode='w+', shape=(N, N))
    X_mmap = None

    _fadvise_sequential(raw_path)

    try:
        # Pass 1: Create M^T on disk via sequential-read block transpose
        n_batches = (N + batch_size - 1) // batch_size
        for i in tqdm(range(0, N, batch_size), desc="Transpose M → M^T", total=n_batches):
            ie = min(i + batch_size, N)
            row_strip = M_mmap[i:ie, :]       # sequential read: (B, N) uint16
            for j in range(0, N, batch_size):
                je = min(j + batch_size, N)
                MT_mmap[j:je, i:ie] = row_strip[:, j:je].T   # scatter-write (B', B) tile
        MT_mmap.flush()
        log_memory("after transpose M → M^T")

        # Pass 2: X = M + M^T using purely sequential reads
        X_mmap = np.memmap(str(sym_path), dtype=dtype, mode='w+', shape=(N, N))
        for start in tqdm(range(0, N, batch_size), desc="Symmetrize M + M^T", total=n_batches):
            end = min(start + batch_size, N)
            rows_M = M_mmap[start:end, :].astype(np.uint32)      # sequential
            rows_MT = MT_mmap[start:end, :].astype(np.uint32)     # sequential
            sym_block = rows_M + rows_MT
            if sym_block.max() > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"Overflow in rows [{start}:{end}]: max={sym_block.max()}"
                )
            X_mmap[start:end, :] = sym_block.astype(np.uint16)

        X_mmap.flush()
        log_memory("after symmetrize M + M^T")
        _fadvise_dontneed(raw_path)
    finally:
        del M_mmap, MT_mmap
        if X_mmap is not None:
            del X_mmap
        if MT_path.exists():
            MT_path.unlink()

    return MmapPEPAccessor(
        sym_path, N, dtype,
        diagonal_value=diagonal_value,
        n_samples=n_samples,
        duplicate_offset=duplicate_offset,
    )
