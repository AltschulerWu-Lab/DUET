"""Skip tests whose code needs an optional extra that is not installed.

The default test job installs only ``duet-codebook[test]``. Tests of the
benchmark layer (``duet.ops_benchmark``, ``duet.merfish_benchmark``,
``scripts/benchmark/``) need the ``benchmark`` extra and are skipped without
it; CI runs them in the job that installs ``[test,benchmark]``.
"""

import importlib.util

import pytest

# Import names of the packages in the `benchmark` extra (pyproject.toml).
BENCHMARK_MODULES = ("pyarrow", "pymoo", "sklearn", "seaborn", "yaml")

_REASON = 'needs the benchmark extra: pip install "duet-codebook[benchmark]"'

HAS_BENCHMARK_EXTRA = all(importlib.util.find_spec(m) is not None for m in BENCHMARK_MODULES)

# Decorator for single tests or classes, e.g. @benchmark_extra
benchmark_extra = pytest.mark.skipif(not HAS_BENCHMARK_EXTRA, reason=_REASON)


def require_benchmark_extra() -> None:
    """Skip the calling test module or test unless the benchmark extra is installed.

    Call it at module level, before imports that need the extra.
    """
    for name in BENCHMARK_MODULES:
        pytest.importorskip(name, reason=_REASON)
