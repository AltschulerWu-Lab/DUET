"""
Providers for expensive computational artifacts.

This module provides caching infrastructure for CodebookEvaluator and PEP matrices.
Each provider handles fingerprint computation, cache lookup, artifact creation,
and persistence.

Storage formats:
- CodebookEvaluator: pickle (.pkl)
- PEP matrix: numpy (.npy)
- Metadata: JSON (.meta.json)

Example usage:
    # For ground truth evaluation (DNA)
    evaluator_provider = EvaluatorProvider(cache_dir="/path/to/cache")
    evaluator = evaluator_provider.get(library, config, alphabet_size=4, n_jobs=8)
    accuracy = evaluator.get_accuracy(indices)

    # For dual-guide evaluation (hex encoding)
    evaluator = evaluator_provider.get(library, config, alphabet_size=16, n_jobs=8)

    # For DUET optimization
    pep_provider = PEPMatrixProvider(cache_dir="/path/to/cache")
    pep_matrix = pep_provider.get(library, config, alphabet_size=4, n_jobs=8)
    optimizer = DUET(pep_matrix=pep_matrix, ...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from duet.evaluator_config import EvaluatorConfig, create_evaluator
from duet.codebook_evaluator import CodebookEvaluator
from duet.utils import log_memory

logger = logging.getLogger(__name__)


# =============================================================================
# Hashing Utilities
# =============================================================================


def hash_library(library) -> str:
    """Compute SHA256 hash of library sequences.

    Args:
        library: List of string sequences (DNA, hex, etc.) or a 2-D numpy
            integer array of shape (N, L) where each row is an
            integer-encoded sequence.

    Returns:
        First 16 characters of SHA256 hex digest.
    """
    if isinstance(library, np.ndarray):
        # Integer-encoded sequences (e.g. synthetic benchmark).
        # tobytes() is deterministic for a given dtype/shape/values.
        content = library.astype(np.int32).tobytes()
        return hashlib.sha256(content).hexdigest()[:16]
    content = "\n".join(library)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def hash_dict(d: Dict[str, Any]) -> str:
    """Compute SHA256 hash of dictionary (JSON-serialized).

    Args:
        d: Dictionary to hash.

    Returns:
        First 16 characters of SHA256 hex digest.
    """
    content = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Low-level Cache Infrastructure
# =============================================================================


class ArtifactCache:
    """
    Low-level cache manager for computational artifacts.

    Provides fingerprinting and path management. Does not handle serialization
    directly - that responsibility belongs to the providers.

    Directory structure:
        cache_dir/
            {fingerprint}.pkl        # pickled objects
            {fingerprint}.npy        # numpy arrays
            {fingerprint}.meta.json  # metadata
    """

    def __init__(self, cache_dir: Path | str):
        """
        Initialize cache manager.

        Args:
            cache_dir: Root directory for cached artifacts.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_fingerprint(self, payload: Dict[str, Any]) -> str:
        """
        Compute cache fingerprint from payload dictionary.

        Args:
            payload: Dictionary containing all inputs that affect the artifact.

        Returns:
            Hex string fingerprint.
        """
        return hash_dict(payload)

    def get_path(self, fingerprint: str, suffix: str) -> Path:
        """
        Get the cache path for a fingerprint with given suffix.

        Args:
            fingerprint: Cache key.
            suffix: File extension (e.g., ".pkl", ".npy").

        Returns:
            Full path to the cached file.
        """
        return self.cache_dir / f"{fingerprint}{suffix}"

    def exists(self, fingerprint: str, suffix: str) -> bool:
        """Check if a cached artifact exists."""
        return self.get_path(fingerprint, suffix).exists()


# =============================================================================
# Metadata Helpers
# =============================================================================


def _save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    """Save metadata to JSON file."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _load_metadata(path: Path) -> Dict[str, Any] | None:
    """Load metadata from JSON file, returns None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _default_metadata(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create default metadata dict with timestamp."""
    meta = {
        "created_at": datetime.now().isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


# =============================================================================
# EvaluatorProvider
# =============================================================================


class EvaluatorProvider:
    """
    Provider for initialized CodebookEvaluator objects.

    Handles caching of expensive evaluator initialization. The evaluator's
    sparse competitor matrix is computed once and persisted for reuse.

    Supports arbitrary alphabet sizes through the alphabet_size parameter.

    Example:
        # DNA sequences
        provider = EvaluatorProvider(cache_dir="/path/to/cache")
        evaluator = provider.get(library, config, alphabet_size=4, n_jobs=8)
        accuracy = evaluator.get_accuracy(indices)

        # Dual-guide hex sequences
        evaluator = provider.get(hex_library, config, alphabet_size=16, n_jobs=8)
    """

    def __init__(
        self,
        cache_dir: Path | str,
        enabled: bool = True,
        force_rebuild: bool = False,
    ):
        """
        Initialize provider.

        Args:
            cache_dir: Directory for cached artifacts.
            enabled: Whether caching is enabled.
            force_rebuild: If True, always rebuild even if cached.
        """
        self.cache = ArtifactCache(cache_dir)
        self.enabled = enabled
        self.force_rebuild = force_rebuild

    def get(
        self,
        library: List[str],
        config: EvaluatorConfig,
        alphabet_size: int,
        n_jobs: int | None = None,
    ) -> CodebookEvaluator:
        """
        Get or create an initialized CodebookEvaluator.

        The encoder is automatically selected based on alphabet_size:
        - alphabet_size=2: BinaryEncoder
        - alphabet_size=4: DNAEncoder
        - alphabet_size=16: HexEncoder

        Args:
            library: List of sequences (encoding must match alphabet_size).
            config: Evaluator configuration.
            alphabet_size: Alphabet size (4 for DNA, 16 for hex, etc.).
            n_jobs: Number of parallel workers for initialization.

        Returns:
            Initialized CodebookEvaluator ready for use.
        """
        fingerprint = self._compute_fingerprint(library, config, alphabet_size)
        pkl_path = self.cache.get_path(fingerprint, ".pkl")
        meta_path = self.cache.get_path(fingerprint, ".meta.json")

        # Try to load from cache
        if self.enabled and not self.force_rebuild and pkl_path.exists():
            logger.info(f"Loading cached evaluator: {pkl_path}")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

        # Create and initialize evaluator using generic factory
        logger.info(f"Creating and initializing CodebookEvaluator (alphabet_size={alphabet_size})...")
        evaluator = create_evaluator(library, config, alphabet_size=alphabet_size)
        evaluator.initialize_cache(n_jobs=n_jobs)

        # Save to cache
        if self.enabled:
            logger.info(f"Saving evaluator to cache: {pkl_path}")
            with open(pkl_path, "wb") as f:
                pickle.dump(evaluator, f)

            _save_metadata(
                meta_path,
                _default_metadata(
                    {
                        "artifact_type": "CodebookEvaluator",
                        "library_size": len(library),
                        "seq_length": len(library[0]),
                        "alphabet_size": alphabet_size,
                        "config": config.to_dict(),
                    }
                ),
            )

        return evaluator

    def _compute_fingerprint(self, library: List[str], config: EvaluatorConfig, alphabet_size: int) -> str:
        """Compute cache fingerprint from inputs."""
        payload = {
            "artifact_type": "CodebookEvaluator",
            "library_hash": hash_library(library),
            "config": config.to_dict(),
            "alphabet_size": alphabet_size,
        }
        return self.cache.compute_fingerprint(payload)


# =============================================================================
# PEPMatrixProvider
# =============================================================================


class PEPMatrixProvider:
    """
    Provider for Pairwise Error Probability (PEP) matrices.

    The PEP matrix contains P(H(R, s_j) ≤ H(R, s_i) | S = s_i) for all pairs,
    enabling efficient union bound approximations during DUET optimization.

    This provider uses streaming computation (compute_pep_matrix) which is
    memory-efficient for large libraries.

    Supports arbitrary alphabet sizes through the alphabet_size parameter.

    Example:
        # DNA sequences
        provider = PEPMatrixProvider(cache_dir="/path/to/cache")
        pep_matrix = provider.get(library, config, alphabet_size=4, n_jobs=8)
        optimizer = DUET(pep_matrix=pep_matrix, ...)

        # Dual-guide hex sequences
        pep_matrix = provider.get(hex_library, config, alphabet_size=16, n_jobs=8)
    """

    def __init__(
        self,
        cache_dir: Path | str,
        enabled: bool = True,
        force_rebuild: bool = False,
    ):
        """
        Initialize provider.

        Args:
            cache_dir: Directory for cached artifacts.
            enabled: Whether caching is enabled.
            force_rebuild: If True, always rebuild even if cached.
        """
        self.cache = ArtifactCache(cache_dir)
        self.enabled = enabled
        self.force_rebuild = force_rebuild

    def _ensure_raw_npy(
        self,
        library: List[str],
        config: EvaluatorConfig,
        alphabet_size: int,
        n_jobs: int | None = None,
        sample_batch_size: int | None = None,
        device: str = "cpu",
    ) -> tuple[Path, int]:
        """Ensure the raw .npy cache file exists and return its path.

        Returns (npy_path, n_samples) WITHOUT loading the matrix into RAM.
        If the cache file already exists with valid metadata, returns
        immediately. Otherwise computes via mmap-streaming and persists.

        Raises:
            RuntimeError: If caching is disabled (no file to return).
        """
        if not self.enabled:
            raise RuntimeError(
                "_ensure_raw_npy() requires caching to be enabled "
                "(no file path available when caching is disabled)"
            )

        fingerprint = self._compute_fingerprint(library, config, alphabet_size)
        npy_path = self.cache.get_path(fingerprint, ".npy")
        meta_path = self.cache.get_path(fingerprint, ".meta.json")

        # Try to load from cache
        if not self.force_rebuild and npy_path.exists():
            meta = _load_metadata(meta_path)
            if meta is not None and "n_samples" in meta:
                logger.info(f"Raw .npy cache hit: {npy_path}")
                return npy_path, meta["n_samples"]
            logger.warning(
                f"Cache file exists but metadata missing or incomplete: "
                f"{meta_path}. Recomputing."
            )

        if device.startswith("gpu"):
            # Fail before allocating the N x N output file below.
            from duet.gpu_utils import require_cupy
            require_cupy(device)

        # Create evaluator and compute PEP matrix (streaming)
        logger.info(f"Computing PEP matrix (streaming, alphabet_size={alphabet_size})...")
        evaluator = create_evaluator(library, config, alphabet_size=alphabet_size)

        N = evaluator.num_codewords
        # Temp file uses .tmp.npy suffix: valid .npy (has numpy header from
        # open_memmap), but invisible to cache lookup which checks for .npy.
        tmp_path = npy_path.parent / (npy_path.stem + ".tmp.npy")

        # np.lib.format.open_memmap creates a .npy with a proper numpy header
        # and returns a writable memmap. Stable since numpy 1.3; used internally
        # by np.load()/np.save(). Preferred over raw np.memmap because the .npy
        # header makes the file self-describing and loadable by np.load().
        output_mmap = np.lib.format.open_memmap(
            str(tmp_path), mode='w+', dtype=np.uint16, shape=(N, N),
        )

        result, n_samples = evaluator.compute_pep_matrix(
            n_jobs=n_jobs, sample_batch_size=sample_batch_size,
            output=output_mmap, device=device,
        )
        del result  # Drop returned ref (may alias output_mmap on GPU path)
        log_memory("after compute_pep_matrix")

        # Flush and release the mmap before renaming.
        # `del` triggers CPython refcount finalization -> mmap.close().
        # This is a CPython assumption; Linux-only target.
        output_mmap.flush()
        del output_mmap
        log_memory("after flush + release PEP mmap")

        # Atomic rename: POSIX rename is atomic on the same filesystem.
        import os
        os.replace(str(tmp_path), str(npy_path))

        # Write metadata AFTER rename succeeds. If we crash between rename
        # and metadata write, callers see .npy without valid metadata and
        # recompute (existing guard above).
        _save_metadata(
            meta_path,
            _default_metadata(
                {
                    "artifact_type": "PEPMatrix",
                    "storage_format": "uint16_counts",
                    "n_samples": n_samples,
                    "library_size": len(library),
                    "seq_length": len(library[0]),
                    "alphabet_size": alphabet_size,
                    "config": config.to_dict(),
                    "shape": [N, N],
                    # Provenance only: the fingerprint ignores the device, so a
                    # later run on another device reuses these counts.
                    "device": device,
                }
            ),
        )

        return npy_path, n_samples

    def get(
        self,
        library: List[str],
        config: EvaluatorConfig,
        alphabet_size: int,
        n_jobs: int | None = None,
        sample_batch_size: int | None = None,
        device: str = "cpu",
    ) -> tuple[np.ndarray, int]:
        """
        Get or create a PEP matrix using streaming computation.

        The encoder is automatically selected based on alphabet_size.

        Args:
            library: List of sequences (encoding must match alphabet_size).
            config: Evaluator configuration (noise channel, distance metric, etc.).
            alphabet_size: Alphabet size (4 for DNA, 16 for hex, etc.).
            n_jobs: Number of parallel workers for computation.
            sample_batch_size: Number of Monte Carlo samples per mini-batch.
                None means auto. See CodebookEvaluator.compute_pep_matrix.
            device: Computation device ("cpu" or "gpu").

        Returns:
            Tuple of (count_matrix, n_samples) where count_matrix is uint16.
        """
        if self.enabled:
            npy_path, n_samples = self._ensure_raw_npy(
                library, config, alphabet_size, n_jobs, sample_batch_size, device,
            )
            count_matrix = np.load(npy_path)
            return count_matrix, n_samples
        else:
            # Caching disabled: compute in-memory (no file I/O)
            logger.info(f"Computing PEP matrix (in-memory, alphabet_size={alphabet_size})...")
            evaluator = create_evaluator(library, config, alphabet_size=alphabet_size)
            count_matrix, n_samples = evaluator.compute_pep_matrix(
                n_jobs=n_jobs, sample_batch_size=sample_batch_size,
                device=device,
            )
            return count_matrix, n_samples

    def get_mmap(
        self,
        library: List[str],
        config: EvaluatorConfig,
        alphabet_size: int,
        n_jobs: int | None = None,
        sample_batch_size: int | None = None,
        device: str = "cpu",
        sym_mem_budget_gb: float = 8.0,
    ) -> "MmapPEPAccessor":
        """Get or create a memory-mapped symmetric PEP matrix (uint16 counts).

        The mmap file stores raw X = M + M^T with no modifications (no
        duplicate penalty, no group-level masking). These are applied
        separately by DecodingSwapCache.

        Args:
            library: List of sequences.
            config: Evaluator configuration.
            alphabet_size: Alphabet size (4 for DNA, 16 for hex, etc.).
            n_jobs: Number of parallel workers for computation.
            sample_batch_size: Number of Monte Carlo samples per mini-batch.
                None means auto. See CodebookEvaluator.compute_pep_matrix.
            device: Computation device ("cpu" or "gpu").
            sym_mem_budget_gb: Target peak memory per batch in GB for
                streaming symmetrization (default 8).

        Returns:
            MmapPEPAccessor wrapping the symmetric PEP file.
        """
        from duet.pep_accessor import (
            MmapPEPAccessor, create_symmetric_mmap,
            create_symmetric_mmap_streaming,
        )

        fingerprint = self._compute_fingerprint(library, config, alphabet_size)
        sym_path = self.cache.get_path(fingerprint, ".sym.dat")
        sym_meta_path = self.cache.get_path(fingerprint, ".sym.meta.json")

        if self.enabled and not self.force_rebuild and sym_path.exists():
            meta = _load_metadata(sym_meta_path)
            if meta is None or "n_samples" not in meta:
                logger.warning(
                    f"Cache file exists but metadata missing or incomplete: "
                    f"{sym_meta_path}. Recomputing."
                )
            else:
                N = len(library)
                return MmapPEPAccessor(
                    sym_path, N, np.uint16,
                    diagonal_value=2.0, n_samples=meta["n_samples"],
                )

        if self.enabled:
            # Streaming path: raw .npy on disk -> streaming symmetrization
            # Peak RSS: ~553 MB per batch (B=256, N=154K) instead of ~332.5 GB
            log_memory("before _ensure_raw_npy")
            npy_path, n_samples = self._ensure_raw_npy(
                library, config, alphabet_size, n_jobs, sample_batch_size, device,
            )
            log_memory("after _ensure_raw_npy")
            N = len(library)
            log_memory("before streaming symmetrization")
            accessor = create_symmetric_mmap_streaming(
                npy_path, sym_path, N,
                diagonal_value=2.0, n_samples=n_samples,
                mem_budget_gb=sym_mem_budget_gb,
            )
            log_memory("after streaming symmetrization")
        else:
            # Caching disabled: compute in-memory then symmetrize in-memory.
            # This still materializes M + M^T in RAM, which is acceptable
            # since uncached mode implies small runs.
            M, n_samples = self.get(
                library, config, alphabet_size, n_jobs, sample_batch_size, device,
            )
            N = len(library)
            accessor = create_symmetric_mmap(
                M, sym_path, diagonal_value=2.0, n_samples=n_samples,
            )

        if self.enabled:
            # The symmetric file is derived from the raw counts, which may have
            # been a cache hit computed on another device: record that device.
            raw_meta = _load_metadata(self.cache.get_path(fingerprint, ".meta.json"))
            _save_metadata(
                sym_meta_path,
                _default_metadata(
                    {
                        "artifact_type": "SymmetricPEPMmap",
                        "storage_format": "uint16_counts",
                        "n_samples": n_samples,
                        "N": N,
                        "device": (raw_meta or {}).get("device"),
                    }
                ),
            )

        return accessor

    def cache_status(
        self,
        library: List[str],
        config: EvaluatorConfig,
        alphabet_size: int,
    ) -> Dict[str, Any]:
        """Report, without computing anything, what :meth:`get_mmap` would reuse.

        Mirrors the cache-hit tests in :meth:`get_mmap` and
        :meth:`_ensure_raw_npy`. ``device`` is the device recorded in the
        metadata of the file that would be reused (None if nothing is cached
        or the file predates device metadata).

        Returns:
            Dict with keys ``fingerprint``, ``sym_hit``, ``raw_hit`` and
            ``device``.
        """
        fingerprint = self._compute_fingerprint(library, config, alphabet_size)
        reuse = self.enabled and not self.force_rebuild
        sym_meta = _load_metadata(self.cache.get_path(fingerprint, ".sym.meta.json"))
        raw_meta = _load_metadata(self.cache.get_path(fingerprint, ".meta.json"))
        sym_hit = bool(
            reuse
            and self.cache.get_path(fingerprint, ".sym.dat").exists()
            and sym_meta is not None and "n_samples" in sym_meta
        )
        raw_hit = bool(
            reuse
            and self.cache.get_path(fingerprint, ".npy").exists()
            and raw_meta is not None and "n_samples" in raw_meta
        )
        if sym_hit:
            device = sym_meta.get("device")
        elif raw_hit:
            device = raw_meta.get("device")
        else:
            device = None
        return {
            "fingerprint": fingerprint,
            "sym_hit": sym_hit,
            "raw_hit": raw_hit,
            "device": device,
        }

    def _compute_fingerprint(self, library: List[str], config: EvaluatorConfig, alphabet_size: int) -> str:
        """Compute cache fingerprint from inputs."""
        payload = {
            "artifact_type": "PEPMatrix",
            "storage_format": "uint16_counts_dedup_v2",
            "library_hash": hash_library(library),
            "config": config.to_dict(),
            "alphabet_size": alphabet_size,
        }
        return self.cache.compute_fingerprint(payload)