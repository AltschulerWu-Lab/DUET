"""Tests for BLAS thread pinning in multiprocessing workers."""

import multiprocessing
import os

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from duet.utils import _worker_init_blas


def _get_blas_thread_counts(_=None):
    """Return list of BLAS thread counts in the current process.

    Module-level function so it is picklable for multiprocessing.
    Accepts a dummy argument for compatibility with pool.map.
    """
    # Trigger BLAS library loading
    a = np.random.randn(4, 4)
    np.dot(a, a.T)

    return [
        lib["num_threads"]
        for lib in threadpool_info()
        if lib["user_api"] == "blas"
    ]


def _fresh_pool(processes, initializer=None):
    """Return a Pool whose workers start from a fresh interpreter.

    The default (fork) context copies the parent's BLAS thread limit into
    every worker. Other tests call _worker_init_blas() in the pytest process
    itself (e.g. test_blas_decoding_metric.py::TestWorkerInitBlas), and that
    limit of 1 persists for the rest of the session. Forked workers would then
    report 1 thread whether or not an initializer ran, making the sanity check
    below fail and the initializer test pass vacuously. Spawned workers start
    from the library's default thread count, so both tests measure what they
    claim regardless of test order.
    """
    return multiprocessing.get_context("spawn").Pool(
        processes, initializer=initializer
    )


def test_worker_has_one_blas_thread():
    """Workers initialized with _worker_init_blas should have 1 BLAS thread."""
    with _fresh_pool(2, initializer=_worker_init_blas) as pool:
        results = pool.map(_get_blas_thread_counts, range(2))

    for worker_counts in results:
        assert len(worker_counts) > 0, "No BLAS libraries detected"
        for count in worker_counts:
            assert count == 1, f"Expected 1 BLAS thread, got {count}"


def test_without_init_has_multiple_threads():
    """Sanity check: without the initializer, default BLAS threads > 1."""
    cpu_count = os.cpu_count() or 1
    if cpu_count < 2:
        pytest.skip("Single-core machine; default threads would be 1")

    with _fresh_pool(1) as pool:
        results = pool.map(_get_blas_thread_counts, range(1))

    thread_counts = results[0]
    assert len(thread_counts) > 0, "No BLAS libraries detected"
    assert any(
        c > 1 for c in thread_counts
    ), f"Expected default BLAS threads > 1 on {cpu_count}-core machine, got {thread_counts}"


def test_run_duet_core_passes_blas_initializer():
    """The lambda-sweep Pool in _run_duet_core must pass initializer=_worker_init_blas.

    Without this, each of N workers would spawn `nproc` BLAS threads at every
    matmul dispatch, producing N * nproc threads contending for nproc cores --
    severe oversubscription. See the num_cpus note in the DuetOptimizerConfig
    docstring (src/duet/runner/core.py) and _worker_init_blas in
    src/duet/utils/threading.py.

    This is a source-level check rather than a runtime check. The runtime
    behavior of (Pool + initializer=_worker_init_blas) is already covered by
    test_worker_has_one_blas_thread above. What this test enforces is that
    _run_duet_core actually wires that pattern into its Pool construction.
    If someone refactors the Pool construction (e.g. extracting a helper),
    they must keep the initializer in place; this test catches accidental
    removal.
    """
    import inspect

    from duet.runner.core import _run_duet_core

    source = inspect.getsource(_run_duet_core)
    assert "initializer=_worker_init_blas" in source, (
        "_run_duet_core must construct its lambda-sweep Pool with "
        "initializer=_worker_init_blas, so that each lambda worker runs "
        "single-threaded BLAS (see the DuetOptimizerConfig docstring)."
    )
