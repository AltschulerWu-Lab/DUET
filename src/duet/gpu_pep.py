"""GPU-accelerated PEP matrix computation via CuPy.

This module provides a drop-in GPU alternative to the CPU multiprocessing
pipeline in CodebookEvaluator.compute_pep_matrix(). Supports single-GPU
and multi-GPU execution via Python threading + CuPy device contexts.

Design decisions:
- Noise generated on CPU (reuses all 4 noise channel classes, preserves RNG)
- psi_tx built on CPU then transferred (reuses all 6 decoding metric classes)
- Dense competitor matrix on GPU (no sparse — dense is faster for this shape)
- One codeword at a time (saturates GPU SMs without excessive memory)
- Margin obtained via DecodingRule.competitor_margin() (Strategy Pattern)
- RNG seeding matches CPU path exactly (SeedSequence.spawn per batch)
- Multi-GPU: threading (GIL released during GPU kernels), shared numpy output
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from duet.gpu_utils import (
    HAS_CUPY,
    has_gpu,
    _one_hot_encode_gpu,
    _gpu_auto_sample_batch_size,
    _gpu_auto_sample_batch_size_for_device,
)

if HAS_CUPY:
    import cupy as cp


@dataclass
class _PEPWorkerArgs:
    gpu_id: int
    assigned_batches: list  # list of (batch_indices: np.ndarray, batch_seed: int)
    codebook: np.ndarray      # CPU numpy, (N, L)
    psi_tx: np.ndarray        # CPU numpy, (N, L*q_obs)
    q_obs: int
    margin: float
    n_samples: int
    sample_batch_size: int | None  # None = auto per-device
    noise_channel: object


def _pep_worker_gpu(args, pep_counts, pbar):
    """Process assigned batches on a single GPU.

    Runs inside a thread. Writes to disjoint rows of the shared
    ``pep_counts`` numpy array (no lock needed).
    """
    N = pep_counts.shape[0]

    with cp.cuda.Device(args.gpu_id):
        psi_tx_gpu = cp.asarray(args.psi_tx)

        # Resolve sample_batch_size per-device
        sample_batch_size = args.sample_batch_size
        if sample_batch_size is None:
            sample_batch_size = _gpu_auto_sample_batch_size_for_device(
                args.n_samples, N, args.gpu_id,
            )

        for batch_indices, batch_seed in args.assigned_batches:
            rng = np.random.default_rng(batch_seed)

            for codeword_idx in batch_indices:
                codeword = args.codebook[codeword_idx]

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
                    counts_gpu = competitors_gpu.sum(axis=0).astype(cp.uint16)
                    pep_counts[codeword_idx, :] = cp.asnumpy(counts_gpu)
                else:
                    counts_accum = cp.zeros(N, dtype=cp.uint32)
                    for start in range(0, args.n_samples, sample_batch_size):
                        end = min(start + sample_batch_size, args.n_samples)
                        obs_batch_gpu = cp.asarray(observed[start:end])
                        phi_batch = _one_hot_encode_gpu(obs_batch_gpu, args.q_obs)
                        cost_batch = phi_batch @ psi_tx_gpu.T
                        tx_cost_batch = cost_batch[:, codeword_idx]
                        comp_batch = cost_batch <= (tx_cost_batch[:, None] + args.margin)
                        counts_accum += comp_batch.sum(axis=0).astype(cp.uint32)

                    pep_counts[codeword_idx, :] = cp.asnumpy(
                        counts_accum.astype(cp.uint16)
                    )

                pbar.update(1)


def compute_pep_matrix_gpu(
    evaluator,
    batch_size: int = 100,
    sample_batch_size=None,
    output=None,
    gpu_ids: list | None = None,
):
    """GPU-accelerated PEP matrix computation.

    Processes one codeword at a time on GPU. Noise is generated on CPU
    and transferred; all other work stays on-device.

    RNG seeding matches the CPU path exactly: codeword indices are split
    into batches of ``batch_size``, each batch gets a seed from
    SeedSequence.spawn(), and codewords within a batch share one RNG.

    Args:
        evaluator: A CodebookEvaluator instance (provides codebook,
            noise_channel, decoding_metric, decoding_rule, n_samples, seed).
        batch_size: Number of codewords per batch for RNG seed splitting.
            Must match the CPU path's batch_size for identical results.
        sample_batch_size: Mini-batch size for Monte Carlo samples.
            None (default) applies the auto heuristic targeting ~2 GB
            GPU memory per codeword (per-device for multi-GPU).
        output: Optional pre-allocated buffer (e.g. mmap) to write results
            into. Must be uint16 with shape (N, N). When provided, avoids
            allocating a 47.5 GB dense array for large N.
        gpu_ids: List of GPU device IDs to use. ``None`` defaults to
            ``[0]`` (single GPU, backward compatible).

    Returns:
        (pep_count_matrix, n_samples) matching the CPU signature.
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

    if output is not None:
        if output.shape != (N, N):
            raise ValueError(f"output shape {output.shape} != expected ({N}, {N})")
        if output.dtype != np.uint16:
            raise ValueError(f"output dtype {output.dtype} != expected uint16")
        pep_counts = output
    else:
        pep_counts = np.zeros((N, N), dtype=np.uint16)

    # --- RNG seeding: replicate CPU batch/seed structure ---
    indices = np.arange(N)
    batches = [indices[i:i + batch_size] for i in range(0, N, batch_size)]
    root_ss = np.random.SeedSequence(seed)
    batch_seeds = [ss.generate_state(1)[0] for ss in root_ss.spawn(len(batches))]

    # --- Round-robin assign batches to GPUs ---
    gpu_assignments = {gid: [] for gid in gpu_ids}
    for batch_idx, (batch_indices, batch_seed) in enumerate(zip(batches, batch_seeds)):
        target_gpu = gpu_ids[batch_idx % len(gpu_ids)]
        gpu_assignments[target_gpu].append((batch_indices, batch_seed))

    # --- Build worker args ---
    worker_args_list = [
        _PEPWorkerArgs(
            gpu_id=gid,
            assigned_batches=gpu_assignments[gid],
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

    n_gpus = len(gpu_ids)
    pbar = tqdm(total=N, desc=f"PEP matrix ({N} codewords) [GPU x{n_gpus}]")

    if n_gpus == 1:
        # Single GPU: run directly, no thread overhead
        _pep_worker_gpu(worker_args_list[0], pep_counts, pbar)
    else:
        # Multi-GPU: one thread per GPU
        executor = ThreadPoolExecutor(max_workers=n_gpus)
        try:
            futures = [
                executor.submit(_pep_worker_gpu, args, pep_counts, pbar)
                for args in worker_args_list
            ]
            for f in futures:
                f.result()  # re-raises any worker exception
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    pbar.close()
    return pep_counts, n_samples
