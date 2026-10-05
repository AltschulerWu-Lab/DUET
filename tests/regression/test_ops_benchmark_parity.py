# tests/regression/test_ops_benchmark_parity.py
"""OPS index-level parity against pre-refactor snapshot."""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import json
from pathlib import Path

import pandas as pd
import pytest

from duet.ops_benchmark import OpsBenchmarkConfig, run_ops_benchmark


DATA = Path(__file__).parent / "data"


def _load_snapshot():
    return json.loads((DATA / "ops_snapshot.json").read_text())


@pytest.mark.slow
def test_ops_indices_exact_parity(tmp_path):
    snapshot = _load_snapshot()
    config = OpsBenchmarkConfig.from_yaml(DATA / "ops_config_post.yaml")
    config.outdir = tmp_path / "out"

    results_df = run_ops_benchmark(config, config_path=None)

    duet_rows = results_df[results_df["Method"].str.startswith("DUET")]
    per_lambda_new = {}
    for method in duet_rows["Method"].unique():
        rows = duet_rows[duet_rows["Method"] == method]
        lambda_ = float(method.split("lambda=")[1].rstrip(")"))
        per_lambda_new[str(lambda_)] = sorted([int(i) for i in rows["Index"].tolist()])

    for lambda_str, expected in snapshot["per_lambda"].items():
        expected_sorted = sorted(expected["best_indices"])
        actual_sorted = per_lambda_new[lambda_str]
        assert actual_sorted == expected_sorted, (
            f"OPS index mismatch at lambda={lambda_str}: "
            f"expected {expected_sorted}, got {actual_sorted}"
        )
