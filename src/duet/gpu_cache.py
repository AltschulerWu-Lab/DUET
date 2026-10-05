"""GPU-accelerated cache initialization via CuPy.

This module provides a GPU alternative to the CPU multiprocessing
pipeline in CodebookEvaluator.initialize_cache(). Supports single-GPU
and multi-GPU execution via Python threading + CuPy device contexts.

Key difference from PEP: instead of summing competitor counts, we
retain the full bool competitor matrix (K, N) per codeword, convert
it to sparse CSR on GPU, and transfer only the sparse data to CPU.

RNG seeding matches the CPU path exactly: SeedSequence.spawn() creates
one seed per codeword (not per batch like PEP).
"""

import atexit
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

from duet.gpu_utils import (
    HAS_CUPY,
    _one_hot_encode_gpu,
    _gpu_auto_sample_batch_size,
    _gpu_auto_sample_batch_size_for_device,
)
from duet.utils import log_memory

if HAS_CUPY:
    import cupy as cp
    import cupyx.scipy.sparse as cusp

logger = logging.getLogger(__name__)


@dataclass
class _CacheWorkerArgs:
    gpu_id: int
    assigned_codewords: list  # list of (codeword_idx: int, seed: int)
    codebook: np.ndarray      # CPU numpy, (N, L)
    psi_tx: np.ndarray        # CPU numpy, (N, L*q_obs)
    q_obs: int
    margin: float
    n_samples: int
    sample_batch_size: int | None  # None = auto per-device
    noise_channel: object


class _UniformRowIndices:
    """O(1) memory stand-in for list-of-arange when all codewords have equal n_samples.

    Behaves like a list of N arrays where ``self[idx]`` returns
    ``np.arange(idx * K, (idx + 1) * K)``.  Avoids storing N separate arrays
    (which would be ~12 GB for N=300K, K=5K).
    """

    __slots__ = ('_n', '_k')

    def __init__(self, n: int, k: int):
        self._n = n
        self._k = k

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += self._n
            if not 0 <= idx < self._n:
                raise IndexError(f"index {idx} out of range [0, {self._n})")
            start = int(idx) * self._k
            return np.arange(start, start + self._k)
        raise TypeError(f"indices must be integers, not {type(idx)}")

    def __len__(self):
        return self._n


def _cache_worker_gpu(args, results_dict, pbar):
    """Process assigned codewords on a single GPU.

    Runs inside a thread. Stores results in ``results_dict`` at
    disjoint keys (no lock needed).
    """
    with cp.cuda.Device(args.gpu_id):
        psi_tx_gpu = cp.asarray(args.psi_tx)

        # Resolve sample_batch_size per-device
        sample_batch_size = args.sample_batch_size
        if sample_batch_size is None:
            N = args.codebook.shape[0]
            sample_batch_size = _gpu_auto_sample_batch_size_for_device(
                args.n_samples, N, args.gpu_id,
            )

        for codeword_idx, seed in args.assigned_codewords:
            codeword = args.codebook[codeword_idx]
            rng = np.random.default_rng(seed)

            # 1. Noise generation on CPU
            observed = args.noise_channel.generate(
                codeword, args.n_samples, rng,
            )  # (K, L)

            if sample_batch_size >= args.n_samples:
                obs_gpu = cp.asarray(observed)
                phi_obs_gpu = _one_hot_encode_gpu(obs_gpu, args.q_obs)
                cost_gpu = phi_obs_gpu @ psi_tx_gpu.T
                tx_cost_gpu = cost_gpu[:, codeword_idx]
                competitors_gpu = cost_gpu <= (tx_cost_gpu[:, None] + args.margin)
                sparse_gpu = cusp.csr_matrix(competitors_gpu)
                sparse_mat = sparse_gpu.get()
            else:
                sparse_batches = []
                for start in range(0, args.n_samples, sample_batch_size):
                    end = min(start + sample_batch_size, args.n_samples)
                    obs_batch_gpu = cp.asarray(observed[start:end])
                    phi_batch = _one_hot_encode_gpu(obs_batch_gpu, args.q_obs)
                    cost_batch = phi_batch @ psi_tx_gpu.T
                    tx_cost_batch = cost_batch[:, codeword_idx]
                    comp_batch = cost_batch <= (tx_cost_batch[:, None] + args.margin)
                    sparse_batch_gpu = cusp.csr_matrix(comp_batch)
                    sparse_batches.append(sparse_batch_gpu.get())
                sparse_mat = vstack(sparse_batches, format='csr')

            no_error = np.all(observed == codeword, axis=1)  # (K,) bool
            results_dict[codeword_idx] = (sparse_mat, observed, no_error)
            pbar.update(1)


@dataclass
class _StreamingCacheWorkerArgs:
    gpu_id: int
    assigned_codewords: list      # list of (codeword_idx: int, seed: int)
    codebook: np.ndarray          # CPU numpy, (N, L)
    psi_tx: np.ndarray            # CPU numpy, (N, L*q_obs)
    q_obs: int
    margin: float
    n_samples: int
    sample_batch_size: int | None
    noise_channel: object
    # Streaming-specific fields
    indices_file_path: str
    nnz_per_row_mmap: np.ndarray      # shared mmap, (N*K,), int32
    no_error_flags_mmap: np.ndarray   # shared mmap, (N*K,), bool
    observed_mmap: np.ndarray | None  # shared mmap, (N*K, L), or None
    store_observed: bool


def _cache_worker_gpu_streaming(args, pbar):
    """Process assigned codewords on a single GPU, streaming results to disk.

    Writes sparse indices to a per-GPU binary file and fixed-size arrays
    to shared mmaps at disjoint slices.  No ``results_dict`` — each
    per-codeword CSR is freed immediately after writing.
    """
    K = args.n_samples

    with cp.cuda.Device(args.gpu_id):
        psi_tx_gpu = cp.asarray(args.psi_tx)

        sample_batch_size = args.sample_batch_size
        if sample_batch_size is None:
            N = args.codebook.shape[0]
            sample_batch_size = _gpu_auto_sample_batch_size_for_device(
                K, N, args.gpu_id,
            )

        with open(args.indices_file_path, 'ab') as idx_file:
            for codeword_idx, seed in args.assigned_codewords:
                codeword = args.codebook[codeword_idx]
                rng = np.random.default_rng(seed)

                observed = args.noise_channel.generate(codeword, K, rng)

                if sample_batch_size >= K:
                    obs_gpu = cp.asarray(observed)
                    phi_obs_gpu = _one_hot_encode_gpu(obs_gpu, args.q_obs)
                    cost_gpu = phi_obs_gpu @ psi_tx_gpu.T
                    tx_cost_gpu = cost_gpu[:, codeword_idx]
                    competitors_gpu = cost_gpu <= (tx_cost_gpu[:, None] + args.margin)
                    sparse_gpu = cusp.csr_matrix(competitors_gpu)
                    sparse_mat = sparse_gpu.get()
                else:
                    sparse_batches = []
                    for start in range(0, K, sample_batch_size):
                        end = min(start + sample_batch_size, K)
                        obs_batch_gpu = cp.asarray(observed[start:end])
                        phi_batch = _one_hot_encode_gpu(obs_batch_gpu, args.q_obs)
                        cost_batch = phi_batch @ psi_tx_gpu.T
                        tx_cost_batch = cost_batch[:, codeword_idx]
                        comp_batch = cost_batch <= (tx_cost_batch[:, None] + args.margin)
                        sparse_batch_gpu = cusp.csr_matrix(comp_batch)
                        sparse_batches.append(sparse_batch_gpu.get())
                    sparse_mat = vstack(sparse_batches, format='csr')

                no_error = np.all(observed == codeword, axis=1)

                # --- Stream to disk ---
                row_start = codeword_idx * K
                row_end = row_start + K

                # Append column indices to per-GPU file
                sparse_mat.indices.astype(np.int32, copy=False).tofile(idx_file)

                # Write nnz-per-row to shared mmap (disjoint slice)
                args.nnz_per_row_mmap[row_start:row_end] = np.diff(
                    sparse_mat.indptr,
                ).astype(np.int32, copy=False)

                # Write no-error flags
                args.no_error_flags_mmap[row_start:row_end] = no_error

                # Write observed sequences if requested
                if args.store_observed and args.observed_mmap is not None:
                    args.observed_mmap[row_start:row_end] = observed

                pbar.update(1)


def _initialize_cache_gpu_streaming(
    evaluator, codebook, psi_tx, q_obs, margin,
    child_seeds, n_samples, N, sample_batch_size, gpu_ids,
):
    """Streaming GPU cache init — writes results directly to disk.

    Called by ``initialize_cache_gpu`` when ``evaluator.mmap_cache=True``.
    Eliminates in-memory accumulation of per-codeword CSR matrices.
    """
    n_gpus = len(gpu_ids)
    K = n_samples
    L = codebook.shape[1]
    total_rows = N * K

    # ------------------------------------------------------------------
    # Phase 1: Setup — create mmap_dir and pre-allocate shared mmaps
    # ------------------------------------------------------------------
    evaluator._cleanup_mmap()
    mmap_dir = tempfile.mkdtemp(prefix="duet_cache_", dir=evaluator.scratch_dir)
    evaluator._mmap_dir = mmap_dir
    atexit.register(evaluator._cleanup_mmap)

    # Pre-allocate shared mmaps (workers write to disjoint slices)
    nnz_per_row_path = os.path.join(mmap_dir, "nnz_per_row.dat")
    nnz_per_row_mmap = np.memmap(
        nnz_per_row_path, dtype=np.int32, mode='w+', shape=(total_rows,),
    )

    no_error_path = os.path.join(mmap_dir, "no_error_flags.dat")
    no_error_mmap = np.memmap(
        no_error_path, dtype=bool, mode='w+', shape=(total_rows,),
    )

    observed_mmap = None
    observed_path = None
    if evaluator.store_observed:
        observed_path = os.path.join(mmap_dir, "observed.dat")
        observed_mmap = np.memmap(
            observed_path, dtype=codebook.dtype, mode='w+',
            shape=(total_rows, L),
        )

    # Log expected disk usage
    disk_bytes = total_rows * 4 + total_rows  # nnz_per_row + no_error_flags
    if evaluator.store_observed:
        disk_bytes += total_rows * L * codebook.dtype.itemsize
    logger.info(
        f"Streaming GPU cache: {N} codewords × {K} samples, {n_gpus} GPU(s), "
        f"pre-allocated {disk_bytes / 1e9:.1f} GB on disk (excluding indices)"
    )
    log_memory("before streaming GPU cache computation")

    # Contiguous block assignment (not round-robin) so each GPU's
    # indices file is already in globally sorted codeword order.
    gpu_assignments = {}
    block_size = N // n_gpus
    remainder = N % n_gpus
    start = 0
    for i, gid in enumerate(gpu_ids):
        count = block_size + (1 if i < remainder else 0)
        gpu_assignments[gid] = [
            (cw_idx, child_seeds[cw_idx])
            for cw_idx in range(start, start + count)
        ]
        start += count

    indices_file_paths = {
        gid: os.path.join(mmap_dir, f"indices_gpu{gid}.bin")
        for gid in gpu_ids
    }

    # ------------------------------------------------------------------
    # Phase 2: Launch streaming workers
    # ------------------------------------------------------------------
    worker_args_list = [
        _StreamingCacheWorkerArgs(
            gpu_id=gid,
            assigned_codewords=gpu_assignments[gid],
            codebook=codebook,
            psi_tx=psi_tx,
            q_obs=q_obs,
            margin=margin,
            n_samples=K,
            sample_batch_size=sample_batch_size,
            noise_channel=evaluator.noise_channel,
            indices_file_path=indices_file_paths[gid],
            nnz_per_row_mmap=nnz_per_row_mmap,
            no_error_flags_mmap=no_error_mmap,
            observed_mmap=observed_mmap,
            store_observed=evaluator.store_observed,
        )
        for gid in gpu_ids
    ]

    pbar = tqdm(total=N, desc=f"Cache init ({N} codewords) [GPU x{n_gpus}] streaming")

    if n_gpus == 1:
        _cache_worker_gpu_streaming(worker_args_list[0], pbar)
    else:
        executor = ThreadPoolExecutor(max_workers=n_gpus)
        try:
            futures = [
                executor.submit(_cache_worker_gpu_streaming, args, pbar)
                for args in worker_args_list
            ]
            for f in futures:
                f.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    pbar.close()
    log_memory("after streaming GPU cache workers")

    # Flush shared mmaps
    nnz_per_row_mmap.flush()
    no_error_mmap.flush()
    if observed_mmap is not None:
        observed_mmap.flush()

    # ------------------------------------------------------------------
    # Phase 3: Assemble CSR from disk
    # ------------------------------------------------------------------

    # 1. Compute indptr (int64 — cumulative nnz can exceed int32 max)
    indptr_path = os.path.join(mmap_dir, "indptr.dat")
    indptr_mmap = np.memmap(
        indptr_path, dtype=np.int64, mode='w+', shape=(total_rows + 1,),
    )
    indptr_mmap[0] = 0
    np.cumsum(nnz_per_row_mmap, out=indptr_mmap[1:])
    indptr_mmap.flush()
    total_nnz = int(indptr_mmap[-1])

    # Reopen as read-only
    del indptr_mmap
    indptr_mmap = np.memmap(
        indptr_path, dtype=np.int64, mode='r', shape=(total_rows + 1,),
    )

    # 2. Validate per-GPU file sizes
    for gid in gpu_ids:
        gpu_path = indices_file_paths[gid]
        if not os.path.exists(gpu_path):
            raise RuntimeError(
                f"GPU {gid} indices file missing: {gpu_path}"
            )
        actual_bytes = os.path.getsize(gpu_path)
        # Sum nnz for all codewords assigned to this GPU
        cw_indices = [cw_idx for cw_idx, _ in gpu_assignments[gid]]
        expected_nnz = sum(
            int(nnz_per_row_mmap[cw_idx * K:(cw_idx + 1) * K].sum())
            for cw_idx in cw_indices
        )
        expected_bytes = expected_nnz * 4  # int32
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"GPU {gid} indices file size mismatch: "
                f"expected {expected_bytes} bytes ({expected_nnz} nnz), "
                f"got {actual_bytes} bytes. File may be truncated."
            )

    # 3. Build indices mmap
    if n_gpus == 1:
        # Single GPU: mmap the file directly (zero copy)
        single_gpu_path = indices_file_paths[gpu_ids[0]]
        indices_mmap = np.memmap(
            single_gpu_path, dtype=np.int32, mode='r', shape=(total_nnz,),
        )
    else:
        # Multi-GPU: chunked concatenation of per-GPU files
        indices_path = os.path.join(mmap_dir, "indices.dat")
        indices_mmap = np.memmap(
            indices_path, dtype=np.int32, mode='w+', shape=(total_nnz,),
        )
        CHUNK_ELEMS = 64 * 1024 * 1024 // 4  # 64 MB in int32 elements
        write_offset = 0
        for gid in gpu_ids:
            gpu_path = indices_file_paths[gid]
            with open(gpu_path, 'rb') as f:
                while True:
                    chunk = np.fromfile(f, dtype=np.int32, count=CHUNK_ELEMS)
                    if chunk.size == 0:
                        break
                    indices_mmap[write_offset:write_offset + chunk.size] = chunk
                    write_offset += chunk.size
            os.remove(gpu_path)
        indices_mmap.flush()
        del indices_mmap
        indices_mmap = np.memmap(
            indices_path, dtype=np.int32, mode='r', shape=(total_nnz,),
        )

    # 3b. Widen indices to int64 if needed.
    #     scipy CSR uses a single idx_dtype for both indices and indptr.
    #     When total_nnz > int32_max, indptr must be int64, so scipy
    #     forces indices to int64 too — creating a ~26 GB in-memory copy.
    #     Pre-widening into a mmap avoids that anonymous allocation.
    if total_nnz > np.iinfo(np.int32).max:
        logger.info(
            f"Widening indices from int32 to int64 "
            f"(total_nnz={total_nnz:,} > int32_max)"
        )
        indices_i64_path = os.path.join(mmap_dir, "indices_int64.dat")
        indices_i64 = np.memmap(
            indices_i64_path, dtype=np.int64, mode='w+', shape=(total_nnz,),
        )
        WIDEN_CHUNK = 64 * 1024 * 1024  # 64M elements per chunk
        for s in range(0, total_nnz, WIDEN_CHUNK):
            e = min(s + WIDEN_CHUNK, total_nnz)
            indices_i64[s:e] = indices_mmap[s:e]
        indices_i64.flush()
        del indices_i64

        # Delete the int32 source file to reclaim disk space
        int32_path = indices_mmap.filename
        del indices_mmap
        if int32_path is not None:
            os.remove(int32_path)

        indices_mmap = np.memmap(
            indices_i64_path, dtype=np.int64, mode='r', shape=(total_nnz,),
        )

    log_memory("after indices build")

    # 4. Build data mmap (all True, filled in chunks)
    data_path = os.path.join(mmap_dir, "data.dat")
    data_mmap = np.memmap(
        data_path, dtype=bool, mode='w+', shape=(total_nnz,),
    )
    DATA_CHUNK = 256 * 1024 * 1024  # 256M elements
    for chunk_start in range(0, total_nnz, DATA_CHUNK):
        chunk_end = min(chunk_start + DATA_CHUNK, total_nnz)
        data_mmap[chunk_start:chunk_end] = True
    data_mmap.flush()
    del data_mmap
    data_mmap = np.memmap(
        data_path, dtype=bool, mode='r', shape=(total_nnz,),
    )

    log_memory("after data fill")

    # 5. Construct CSR matrix
    evaluator._sparse_matrix = csr_matrix(
        (data_mmap, indices_mmap, indptr_mmap), shape=(total_rows, N),
    )
    log_memory("after CSR construction")

    # 6. Reopen no_error_flags as read-only
    del no_error_mmap
    evaluator._no_error_flags = np.memmap(
        no_error_path, dtype=bool, mode='r', shape=(total_rows,),
    )

    # 7. Handle observed sequences
    if evaluator.store_observed and observed_path is not None:
        del observed_mmap
        evaluator._observed_sequences = np.memmap(
            observed_path, dtype=codebook.dtype, mode='r',
            shape=(total_rows, L),
        )
    else:
        evaluator._observed_sequences = None

    # 8. Lazy row indices
    evaluator._codeword_row_indices = _UniformRowIndices(N, K)

    log_memory("after streaming GPU cache assembly")

    # Clean up nnz_per_row mmap reference (file stays for cleanup)
    del nnz_per_row_mmap

    stats = {
        'num_codewords': N,
        'seq_length': evaluator.seq_length,
        'n_samples': K,
        'total_rows': evaluator._sparse_matrix.shape[0],
        'matrix_shape': evaluator._sparse_matrix.shape,
        'nnz': evaluator._sparse_matrix.nnz,
        'density': evaluator._sparse_matrix.nnz / np.prod(evaluator._sparse_matrix.shape),
        'memory_bytes': (
            evaluator._sparse_matrix.data.nbytes +
            evaluator._sparse_matrix.indices.nbytes +
            evaluator._sparse_matrix.indptr.nbytes
        ),
        'no_error_flags_bytes': evaluator._no_error_flags.nbytes,
    }

    logger.info(
        f"Streaming GPU cache initialized: {stats['matrix_shape']}, nnz={stats['nnz']}"
    )

    return stats


def initialize_cache_gpu(evaluator, sample_batch_size=None, gpu_ids=None):
    """GPU-accelerated cache initialization.

    Processes one codeword at a time on GPU. Noise is generated on CPU
    (preserves RNG consistency); cost computation and competitor
    identification happen on GPU.

    Args:
        evaluator: A CodebookEvaluator instance (provides codebook,
            noise_channel, decoding_metric, decoding_rule, n_samples, seed).
        sample_batch_size: Mini-batch size for Monte Carlo samples.
            None (default) applies the auto heuristic targeting ~2 GB
            GPU memory per codeword (per-device for multi-GPU).
        gpu_ids: List of GPU device IDs to use. ``None`` defaults to
            ``[0]`` (single GPU, backward compatible).

    Returns:
        Dictionary with initialization statistics (same format as
        CPU initialize_cache).
    """
    if gpu_ids is None:
        gpu_ids = [0]

    codebook = evaluator.codebook          # (N, L) on CPU
    noise_channel = evaluator.noise_channel
    decoding_metric = evaluator.decoding_metric
    decoding_rule = evaluator.decoding_rule
    n_samples = evaluator.n_samples
    seed = evaluator.seed
    N = evaluator.num_codewords

    # --- Pre-compute psi_tx on CPU (once, shared across GPUs) ---
    precomputed = decoding_metric.precompute_transmitted(codebook)
    psi_tx = precomputed.psi_tx    # numpy (N, L*q_obs)
    q_obs = precomputed.q_obs
    margin = float(decoding_rule.competitor_margin())

    # --- RNG seeding: match CPU path exactly (one seed per codeword) ---
    root_ss = np.random.SeedSequence(seed)
    child_seeds = [ss.generate_state(1)[0] for ss in root_ss.spawn(N)]

    # --- Streaming path: write directly to disk when mmap_cache=True ---
    if evaluator.mmap_cache:
        return _initialize_cache_gpu_streaming(
            evaluator, codebook, psi_tx, q_obs, margin,
            child_seeds, n_samples, N, sample_batch_size, gpu_ids,
        )

    # --- Round-robin assign codewords to GPUs ---
    gpu_assignments = {gid: [] for gid in gpu_ids}
    for cw_idx in range(N):
        target_gpu = gpu_ids[cw_idx % len(gpu_ids)]
        gpu_assignments[target_gpu].append((cw_idx, child_seeds[cw_idx]))

    # --- Build worker args ---
    worker_args_list = [
        _CacheWorkerArgs(
            gpu_id=gid,
            assigned_codewords=gpu_assignments[gid],
            codebook=codebook,
            psi_tx=psi_tx,
            q_obs=q_obs,
            margin=margin,
            n_samples=n_samples,
            sample_batch_size=sample_batch_size,
            noise_channel=noise_channel,
        )
        for gid in gpu_ids
    ]

    results_dict = {}
    n_gpus = len(gpu_ids)

    logger.info(f"GPU cache init: {N} codewords × {n_samples} samples, {n_gpus} GPU(s)")
    log_memory("before GPU cache computation")

    pbar = tqdm(total=N, desc=f"Cache init ({N} codewords) [GPU x{n_gpus}]")

    if n_gpus == 1:
        # Single GPU: run directly, no thread overhead
        _cache_worker_gpu(worker_args_list[0], results_dict, pbar)
    else:
        # Multi-GPU: one thread per GPU
        executor = ThreadPoolExecutor(max_workers=n_gpus)
        try:
            futures = [
                executor.submit(_cache_worker_gpu, args, results_dict, pbar)
                for args in worker_args_list
            ]
            for f in futures:
                f.result()  # re-raises any worker exception
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    pbar.close()

    # --- Assemble results deterministically by codeword index ---
    sparse_matrices = []
    observed_list = []
    no_error_list = []
    codeword_row_indices = []
    current_row = 0

    for cw_idx in sorted(results_dict.keys()):
        sparse_mat, observed, no_error = results_dict[cw_idx]
        sparse_matrices.append(sparse_mat)
        observed_list.append(observed)
        no_error_list.append(no_error)

        num_rows = observed.shape[0]
        codeword_row_indices.append(np.arange(current_row, current_row + num_rows))
        current_row += num_rows

    # Assemble into evaluator state
    evaluator._sparse_matrix = vstack(sparse_matrices, format='csr')
    evaluator._no_error_flags = np.concatenate(no_error_list)
    if evaluator.store_observed:
        evaluator._observed_sequences = np.vstack(observed_list)
    evaluator._codeword_row_indices = codeword_row_indices
    log_memory("after GPU cache assembly (vstack)")

    stats = {
        'num_codewords': N,
        'seq_length': evaluator.seq_length,
        'n_samples': n_samples,
        'total_rows': evaluator._sparse_matrix.shape[0],
        'matrix_shape': evaluator._sparse_matrix.shape,
        'nnz': evaluator._sparse_matrix.nnz,
        'density': evaluator._sparse_matrix.nnz / np.prod(evaluator._sparse_matrix.shape),
        'memory_bytes': (
            evaluator._sparse_matrix.data.nbytes +
            evaluator._sparse_matrix.indices.nbytes +
            evaluator._sparse_matrix.indptr.nbytes
        ),
        'no_error_flags_bytes': evaluator._no_error_flags.nbytes,
    }

    logger.info(f"GPU cache initialized: {stats['matrix_shape']}, nnz={stats['nnz']}")

    return stats
