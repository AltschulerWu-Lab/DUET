"""Shared GPU utility functions for CuPy-based acceleration.

This module centralizes GPU helpers used by both gpu_pep.py (PEP matrix
computation) and gpu_cache.py (cache initialization). Keeping them in one
place avoids duplication and ensures consistent behaviour.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def has_gpu() -> bool:
    """Return True if CuPy is installed and a GPU is available."""
    if not HAS_CUPY:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        # CUDARuntimeError when no device is visible; other CUDA-stack errors
        # (for example a missing driver library) also mean "no usable GPU".
        return False


def require_cupy(device: str) -> None:
    """Raise one clear ImportError if a GPU device is requested without CuPy.

    CuPy is an optional dependency (the ``gpu`` extra). Call this before any
    GPU work or file allocation on a GPU path.
    """
    if not HAS_CUPY:
        raise ImportError(
            f"CuPy is required for GPU devices (device={device!r}). Install the "
            'GPU extra with: pip install "duet-codebook[gpu]" (CuPy built for '
            "CUDA 12.x; see the README for other CUDA versions), or use "
            'device="cpu".'
        )


def get_gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    if not HAS_CUPY:
        return 0
    try:
        return cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        return 0


def parse_device(device: str) -> list:
    """Parse a device string into a list of GPU IDs.

    Device string semantics:
    - ``"gpu"``     → ``[0]`` (backward compatible)
    - ``"gpu:all"`` → all available GPUs
    - ``"gpu:0"``   → ``[0]``
    - ``"gpu:0,2"`` → ``[0, 2]``

    Raises:
        ValueError: If the device string is malformed, references
            non-existent GPU IDs, or no GPUs are available.
    """
    if not device.startswith("gpu"):
        raise ValueError(
            f"Device string must start with 'gpu', got '{device}'"
        )

    suffix = device[3:]  # everything after "gpu"

    if suffix == "":
        # "gpu" → single GPU 0
        count = get_gpu_count()
        if count == 0:
            raise ValueError(
                "device='gpu' requested but no GPUs are available"
            )
        gpu_ids = [0]
    elif suffix.startswith(":"):
        spec = suffix[1:]  # strip leading colon
        if spec == "all":
            count = get_gpu_count()
            if count == 0:
                raise ValueError(
                    "device='gpu:all' requested but no GPUs are available"
                )
            gpu_ids = list(range(count))
        else:
            try:
                gpu_ids = [int(x.strip()) for x in spec.split(",")]
            except ValueError:
                raise ValueError(
                    f"Invalid GPU ID(s) in device string '{device}'. "
                    f"Expected comma-separated integers after 'gpu:'."
                )
    else:
        raise ValueError(
            f"Invalid device string '{device}'. "
            f"Expected 'gpu', 'gpu:all', 'gpu:<id>', or 'gpu:<id1>,<id2>'."
        )

    # Validate GPU IDs exist
    count = get_gpu_count()
    invalid = [gid for gid in gpu_ids if gid < 0 or gid >= count]
    if invalid:
        raise ValueError(
            f"GPU ID(s) {invalid} out of range. "
            f"{count} GPU(s) available (IDs 0..{count - 1})."
        )

    return gpu_ids


def _one_hot_encode_gpu(seqs_gpu, q: int):
    """GPU one-hot encode. seqs_gpu: (M, L) int -> (M, L*q) float32."""
    M, L = seqs_gpu.shape
    phi = cp.zeros((M, L * q), dtype=cp.float32)
    rows = cp.repeat(cp.arange(M), L)
    cols = (cp.tile(cp.arange(L), M) * q + seqs_gpu.ravel()).astype(cp.int64)
    phi[rows, cols] = 1.0
    return phi


def _build_psi_tx_gpu(codebook_gpu, decoding_metric):
    """Build the (N, L*q_obs) psi_tx feature matrix on GPU.

    Uses the CPU precompute_transmitted() to handle all 6 decoding metric
    types correctly, then transfers the result to GPU once.

    Returns:
        (psi_tx_gpu, q_obs)
    """
    precomputed = decoding_metric.precompute_transmitted(cp.asnumpy(codebook_gpu))
    psi_tx_gpu = cp.asarray(precomputed.psi_tx)   # (N, L*q_obs) float32
    return psi_tx_gpu, precomputed.q_obs


def _gpu_auto_sample_batch_size(n_samples: int, N: int) -> int:
    """Compute auto sample_batch_size for GPU.

    Targets ~2 GB peak GPU memory per codeword. The dominant arrays are:
    - phi_obs: (K, L*q) float32 ~ K * L*q * 4 bytes
    - cost:    (K, N)   float32 ~ K * N * 4 bytes
    - competitors: (K, N) bool  ~ K * N bytes

    The cost + competitors dominate: K * N * 5 bytes.
    """
    target_bytes = 2 * 1024**3
    auto = min(n_samples, max(1, target_bytes // (N * 5)))
    return auto


def _gpu_auto_sample_batch_size_for_device(n_samples: int, N: int, gpu_id: int) -> int:
    """Compute auto sample_batch_size using actual free memory on a specific GPU.

    Must be called inside a ``with cp.cuda.Device(gpu_id):`` context.
    Uses 80% of free device memory (capped at 2 GB) as the target.
    Falls back to the fixed 2 GB heuristic on failure.
    """
    try:
        free_bytes, _ = cp.cuda.Device(gpu_id).mem_info
        target_bytes = int(min(free_bytes * 0.8, 2 * 1024**3))
    except Exception:
        target_bytes = 2 * 1024**3
    return min(n_samples, max(1, target_bytes // (N * 5)))
