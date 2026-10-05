"""Tests for the DUET objective-gap runner and visualizer."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark.synthetic.run_objective_gap import compute_duet_objective
from duet.pareto_optimization import DecodingSwapCache


def test_objective_matches_decoding_swap_cache():
    """The runner's DUET objective equals DUET's compute_objective on a raw M."""
    # Raw ASYMMETRIC PEP matrix M[i,j] = P(j decoded | i transmitted).
    # Must NOT be pre-symmetrized: DecodingSwapCache.from_pep_matrix expects
    # raw M and symmetrizes internally (X = M + M^T, then divides by 2 in
    # compute_objective). compute_duet_objective works directly on raw M and
    # does NOT symmetrize.
    rng = np.random.default_rng(0)
    n = 6
    pep_matrix = rng.random((n, n)) * 0.1
    np.fill_diagonal(pep_matrix, 0.85)

    group_to_candidates = {"group_0": list(range(n))}
    codebook = np.array([0, 2, 4])

    runner_objective = compute_duet_objective(pep_matrix, codebook)

    cache = DecodingSwapCache.from_pep_matrix(
        pep_matrix,
        group_to_candidates,
        codeword_to_group=["group_0"] * len(codebook),
    )
    cache.build_cache(codebook)
    duet_objective = cache.compute_objective(codebook)

    assert runner_objective == pytest.approx(duet_objective)


def test_config_from_yaml(tmp_path: Path):
    """ObjectiveGapConfig.from_yaml parses the experiment config shape."""
    from scripts.benchmark.synthetic.run_objective_gap import ObjectiveGapConfig

    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent("""\
        outdir: results/benchmark/05-18-2026/duet_objective_gap
        seed: 42
        trials: 5

        pool:
          num_groups: 1
          candidates_per_group: 15
          seq_length: 8
          alphabet_size: 2
          quota: 5

        evaluator:
          num_samples: 10000
          num_cpus: 20

        noise_channel: symmetric
        error_rate: 0.1
    """))

    cfg = ObjectiveGapConfig.from_yaml(str(config_path))

    assert cfg.outdir == "results/benchmark/05-18-2026/duet_objective_gap"
    assert cfg.seed == 42
    assert cfg.trials == 5
    assert cfg.pool.candidates_per_group == 15
    assert cfg.pool.quota == 5
    assert cfg.evaluator.num_samples == 10000
    assert cfg.evaluator.num_cpus == 20
    assert cfg.noise_channel == "symmetric"
    assert cfg.error_rate == 0.1


def test_runner_smoke(tmp_path: Path):
    """run_objective_gap produces a parquet with the expected schema."""
    from scripts.benchmark.synthetic.run_objective_gap import (
        ObjectiveGapConfig,
        run_objective_gap,
    )
    from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
        EvaluatorRunConfig,
        PoolConfig,
    )
    from duet.benchmark.exhaustive import compute_total_codebooks
    from duet.benchmark.synthetic_pool import create_2d_synthetic_pool

    outdir = tmp_path / "run"
    config = ObjectiveGapConfig(
        outdir=str(outdir),
        seed=7,
        trials=2,
        pool=PoolConfig(
            num_groups=1,
            candidates_per_group=6,
            seq_length=6,
            alphabet_size=2,
            quota=3,
        ),
        evaluator=EvaluatorRunConfig(num_samples=200, num_cpus=1),
        noise_channel="symmetric",
        error_rate=0.1,
    )

    df = run_objective_gap(config)

    # Expected row count: trials * C(6, 3), asserted via compute_total_codebooks.
    pool = create_2d_synthetic_pool(
        num_groups=1, candidates_per_group=6, seq_length=6,
        alphabet_size=2, quota=3, seed=7,
    )
    total = compute_total_codebooks(pool.group_to_candidates, pool.quotas)
    assert total == 20
    assert len(df) == config.trials * total

    assert set(df.columns) == {
        "trial", "seed", "noise_channel", "error_rate",
        "codebook_index", "codebook_members",
        "decode_accuracy", "duet_objective",
    }
    assert sorted(df["trial"].unique().tolist()) == [1, 2]
    assert df["decode_accuracy"].between(0.0, 1.0).all()
    # duet_objective is bounded above by 1 (total_pep ≥ 0) but *not* below
    # by 0 — the objective is a union bound, and at higher noise it can go
    # negative (as seen at an error rate of 0.2). At this smoke test's
    # parameters it happens to stay non-negative, but the assertion does
    # not rely on that.
    assert (df["duet_objective"] <= 1.0).all()
    assert np.isfinite(df["duet_objective"]).all()
    assert (outdir / "objective_gap.parquet").exists()


def test_visualizer_smoke(tmp_path: Path):
    """render_figures writes one SVG per trial, including coincident optima."""
    from scripts.benchmark.synthetic.visualize_objective_gap import render_figures

    rng = np.random.default_rng(1)
    rows = []
    # Trial 1: distinct optima (accuracy-optimal != objective-optimal).
    for i in range(20):
        rows.append({
            "trial": 1,
            "decode_accuracy": float(rng.uniform(0.5, 0.9)),
            "duet_objective": float(rng.uniform(0.5, 0.9)),
        })
    # Trial 2: coincident optima — one row is the max of BOTH metrics.
    for i in range(19):
        rows.append({
            "trial": 2,
            "decode_accuracy": float(rng.uniform(0.5, 0.8)),
            "duet_objective": float(rng.uniform(0.5, 0.8)),
        })
    rows.append({"trial": 2, "decode_accuracy": 0.99, "duet_objective": 0.99})
    df = pd.DataFrame(rows)

    render_figures(df, tmp_path)

    assert (tmp_path / "figures" / "trial_01.svg").exists()
    assert (tmp_path / "figures" / "trial_02.svg").exists()

    svg_01 = (tmp_path / "figures" / "trial_01.svg").read_text()
    assert "Decode accuracy" in svg_01
    assert "DUET decode objective" in svg_01
    # Distinct-optima trial: both leader-line callouts present, the DUET
    # callout reports the accuracy rank, and the old Δ-accuracy annotation
    # is gone.
    assert "Accuracy-optimal" in svg_01
    assert "decode-accuracy rank" in svg_01
    assert "Δ accuracy" not in svg_01

    # Coincident-optima trial: the accuracy- and DUET-optimal codebooks are
    # the same one, but each still gets its own callout — two leader lines,
    # no merged label.
    svg_02 = (tmp_path / "figures" / "trial_02.svg").read_text()
    assert "Accuracy-optimal" in svg_02
    assert "DUET-optimal" in svg_02
    assert "decode-accuracy rank" in svg_02
    assert "Accuracy- &amp; DUET-optimal" not in svg_02


def test_compute_duet_objective_importable_from_duet_metrics():
    """compute_duet_objective must be importable from duet.benchmark.metrics.

    The function was hoisted from scripts.benchmark.synthetic.run_objective_gap
    so other runners (e.g. the 2-D synthetic benchmark) can import it without
    pulling in the objective-gap CLI module. Both import paths must continue
    to resolve to the same callable.
    """
    from duet.benchmark.metrics import compute_duet_objective as from_metrics
    from scripts.benchmark.synthetic.run_objective_gap import (
        compute_duet_objective as from_runner,
    )
    assert from_metrics is from_runner
