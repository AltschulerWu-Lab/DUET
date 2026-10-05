"""Refactor regression: post-refactor enumeration runner must match the
pre-refactor golden parquet on the tiny regression config.

The tiny config (1 group × 4 candidates × quota 2; 2 trials; 1 noise channel;
n_samples=100) runs in well under a second, so this regression check fits
cleanly into the unit-test suite. A multi-hour bit-for-bit check against
the full benchmark output was a one-off manual step, not a checked-in test.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import pandas as pd
import pytest

from duet.benchmark.baseline_objectives import BaselineObjectivesConfig
from scripts.benchmark.synthetic.run_duet_vs_baseline_objectives import (
    run_baseline_objectives,
)


DATA = Path(__file__).parent / "data"


def test_baseline_objectives_refactor_parity(tmp_path):
    """Run the refactored enumeration runner against the tiny config and
    assert its output frame equals the pre-refactor golden frame."""
    config = BaselineObjectivesConfig.from_yaml(
        str(DATA / "baseline_objectives_tiny_config.yaml")
    )
    # Redirect the runner output to tmp_path so we don't write to /tmp on
    # CI runners that don't have it.
    config.outdir = str(tmp_path / "out")

    df_new = run_baseline_objectives(config)

    golden_path = DATA / "baseline_objectives_golden.parquet"
    df_golden = pd.read_parquet(golden_path)

    # Sort both frames into a canonical order before comparison. The runner
    # iterates noise channels in config order and codebooks in enumeration
    # order; both should be deterministic, but the sort makes the test
    # robust to any future reordering change.
    sort_keys = ["trial", "noise_channel", "codebook_index"]
    df_new = df_new.sort_values(sort_keys).reset_index(drop=True)
    df_golden = df_golden.sort_values(sort_keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_new,
        df_golden,
        rtol=1e-12,
        check_dtype=True,
        check_column_type=True,
    )
