"""BLAS thread-pool management for multiprocessing workers.

In a multiprocessing.Pool, each worker is a separate process. Without an
explicit BLAS thread limit, OpenBLAS (and MKL) default to spawning
`nproc` threads per process at every BLAS dispatch. With N workers, that
yields N * nproc OS-level threads competing for nproc cores -- severe
oversubscription, manifesting as elevated load average and per-call
thread-setup overhead that dominates the math on small matmuls.

Use _worker_init_blas as the `initializer=` argument of any Pool that
runs BLAS-heavy work in its workers.
"""
from __future__ import annotations


def _worker_init_blas() -> None:
    """Restrict BLAS to 1 thread per worker to prevent oversubscription.

    With multiprocessing, total parallelism = n_jobs worker processes.
    Each worker should use single-threaded BLAS so that CPU usage is
    predictable and equals exactly n_jobs.
    """
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
