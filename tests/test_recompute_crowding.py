"""Unit tests for scripts/benchmark/recompute_crowding.py orchestration.

The heavy crowding simulation, config parsing, and expression loading are mocked
so these tests exercise only the script's own logic: the --n-trials override, the
reproduction gate, the n_transcripts / se_* backfills, the summary.yaml sync, and
the seed resolution (prefer the recorded run seed; error if none is available).
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from duet.merfish_benchmark.config import CrowdingConfig
from duet.merfish_benchmark.evaluation import CrowdingEvaluation

# recompute_crowding.py is a script (not an installed module); load it directly.
_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "recompute_crowding", _REPO / "scripts" / "benchmark" / "recompute_crowding.py"
)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def _write_inputs(tmp_path, *, stored_n_trials=100, seed_in_summary=42, cfg_seed=42):
    """Two methods x two codewords of fake persisted results, + a summary.yaml."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    pd.DataFrame({
        "Method": ["M", "M", "B", "B"], "Index": [0, 1, 0, 1],
        "Gene": ["GeneA", "GeneB", "GeneA", "GeneB"],
        "Sequence": ["1100", "0011", "1100", "0011"],
        "Decode accuracy": [0.90, 0.92, 0.90, 0.92],
        "Identified fraction": [0.80, 0.70, 0.80, 0.70],
        "No_error": [0.9] * 4, "Corrected": [0.05] * 4, "Failed": [0.05] * 4,
        "Trial": [1] * 4, "Valid": [True] * 4,
    }).to_csv(outdir / "results.csv", index=False)
    pd.DataFrame({
        "Method": ["M", "B"],
        "Mean decode accuracy": [0.91, 0.91], "Std decode accuracy": [0.01, 0.01],
        "mean_identified_fraction": [0.75, 0.75], "std_identified_fraction": [0.02, 0.02],
        "mean_conflict_fraction": [0.25, 0.25], "std_conflict_fraction": [0.02, 0.02],
    }).to_csv(outdir / "metrics.csv", index=False)
    summary = {"n_crowding_trials": stored_n_trials}
    if seed_in_summary is not None:
        summary["seed"] = seed_in_summary
    (outdir / "summary.yaml").write_text(yaml.safe_dump(summary))
    cfg = SimpleNamespace(
        outdir=outdir,
        crowding=CrowdingConfig(n_trials=stored_n_trials),
        expression=SimpleNamespace(path="ignored", expression_col="e", gene_col="g"),
        seed=cfg_seed,
    )
    return outdir, cfg


def _fake_ce(gene_names):
    n = len(gene_names)
    return CrowdingEvaluation(
        mean_conflict_fraction=0.10, std_conflict_fraction=0.01,
        mean_identified_fraction=0.90, std_identified_fraction=0.05,
        per_codeword_identified_fraction=np.full(n, 0.90),
        per_codeword_count=np.full(n, 7, dtype=np.int64),
    )


def _patches(cfg, eval_side_effect):
    return (
        patch.object(rc.MerfishBenchmarkConfig, "from_yaml", return_value=cfg),
        patch.object(rc, "load_expression", return_value=pd.DataFrame()),
        patch.object(rc, "evaluate_crowding_simulation", side_effect=eval_side_effect),
    )


def test_backfills_columns_overrides_trials_and_syncs_summary(tmp_path):
    outdir, cfg = _write_inputs(tmp_path, stored_n_trials=100)
    p1, p2, p3 = _patches(cfg, lambda **kw: _fake_ce(kw["gene_names"]))
    with p1, p2, p3:
        rc.main(["--config", "dummy.yaml", "--n-trials", "500"])

    res = pd.read_csv(outdir / "results.csv")
    met = pd.read_csv(outdir / "metrics.csv")
    assert (res["n_transcripts"] == 7).all()                       # backfilled count
    # SE uses the OVERRIDE n_trials=500, not the stored 100; decode SE uses K=2.
    np.testing.assert_allclose(met["se_identified_fraction"], 0.05 / np.sqrt(500))
    np.testing.assert_allclose(met["se_decode_accuracy"], 0.01 / np.sqrt(2))
    # gate OFF (500 != stored 100) -> fractions overwritten toward the re-estimate.
    np.testing.assert_allclose(res["Identified fraction"], 0.90)
    assert yaml.safe_load((outdir / "summary.yaml").read_text())["n_crowding_trials"] == 500


def test_uses_recorded_seed_when_config_seed_is_null(tmp_path):
    # config seed null, but summary recorded the realized seed -> use it, no crash.
    _outdir, cfg = _write_inputs(tmp_path, seed_in_summary=7, cfg_seed=None)
    seen = {}

    def fake_eval(**kw):
        seen["seed"] = kw["seed"]
        return _fake_ce(kw["gene_names"])

    p1, p2, p3 = _patches(cfg, fake_eval)
    with p1, p2, p3:
        rc.main(["--config", "dummy.yaml", "--n-trials", "10"])
    assert seen["seed"] == 7


def test_errors_when_no_seed_available(tmp_path):
    # neither summary nor config supplies a seed -> SystemExit, not a None+int crash.
    _outdir, cfg = _write_inputs(tmp_path, seed_in_summary=None, cfg_seed=None)
    p1, p2, p3 = _patches(cfg, lambda **kw: _fake_ce(kw["gene_names"]))
    with p1, p2, p3, pytest.raises(SystemExit):
        rc.main(["--config", "dummy.yaml"])
