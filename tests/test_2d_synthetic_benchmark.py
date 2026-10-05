"""Integration smoke test for the 2-D synthetic benchmark."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
    TwoDSyntheticBenchmarkConfig,
    run_benchmark,
)


@pytest.fixture
def tiny_small_config(tmp_path: Path) -> TwoDSyntheticBenchmarkConfig:
    """A minimal small-regime config that runs end-to-end in seconds.

    Exercises both `uniform` (the simplest path) and `positional_channel`
    (the path with the most structural change in the noise-model redesign:
    [L, q, q] matrices, per-position × per-symbol eps). Two error rates
    so the parametrization isn't degenerate.
    """
    yaml_text = textwrap.dedent(f"""\
        outdir: {tmp_path}/run
        seed: 7
        trials: 1

        pool:
          num_groups: 3
          candidates_per_group: 3
          seq_length: 6
          alphabet_size: 2
          quota: 1

        evaluator:
          num_samples: 200
          num_cpus: 1

        noise_channels: ["symmetric", "position_varying_asymmetric"]
        error_rates: [0.1, 0.2]

        duet:
          optimizer:
            lambda: [0.0, 0.5, 1.0]
            temperature: 0.0
            max_iter: 1000
            max_patience: 100
            num_cpus: 1
            store_all_solutions: true

        greedy_hamming_mo:
          lambda: [0.0, 0.5, 1.0]

        greedy_nll_mo:
          lambda: [0.0, 0.5, 1.0]
    """)
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml_text)
    return TwoDSyntheticBenchmarkConfig.from_yaml(str(cfg_path))


def test_smoke_run_writes_expected_outputs(tiny_small_config):
    """End-to-end: run the benchmark on a 3^3 = 27-codebook pool across
    4 cells (2 noise × 2 error_rate). Verifies both Parquet outputs
    exist with the documented columns; cells are auto-classified as
    regime='small'; the exhaustive front is non-empty; and all four
    methods (DUET, Greedy-NLL-MO, Greedy-Hamming-MO, exhaustive) appear.
    """
    results_df, hv_df = run_benchmark(tiny_small_config)

    outdir = Path(tiny_small_config.outdir)
    assert (outdir / "results.parquet").exists()
    assert (outdir / "hv_summary.parquet").exists()

    expected_results_cols = {
        "regime", "trial", "seed", "noise_channel", "error_rate",
        "method", "lambda", "decode_accuracy", "mean_score", "codebook_id",
    }
    assert expected_results_cols.issubset(set(results_df.columns))

    expected_hv_cols = {
        "regime", "trial", "seed", "noise_channel", "error_rate", "method",
        "hv", "normalized_hv", "hvr", "n_pareto_points",
        "decode_acc_at_lambda_one", "max_decode_accuracy",
    }
    assert expected_hv_cols.issubset(set(hv_df.columns))

    assert (results_df["regime"] == "small").all()
    assert (hv_df["regime"] == "small").all()

    # HVR sanity checks (small regime → all cells have an exhaustive ref):
    # - Exhaustive method always has hvr == 1.0 by construction.
    # - For each cell whose exhaustive HV is positive (i.e. the row's
    #   hvr is not NaN due to the runner's degenerate-cell guard at
    #   exhaustive_hv == 0), every non-exhaustive method's hvr is finite
    #   in (0, 1.0 + 1e-9]. Cells where the guard fires (every method
    #   gets NaN, including exhaustive) are not asserted on — that path
    #   is a defensive log, not a correctness assertion.
    exh_hv = hv_df[hv_df["method"] == "exhaustive"]
    assert len(exh_hv) > 0

    cell_keys = ["trial", "noise_channel", "error_rate"]
    valid_cells = exh_hv[exh_hv["hvr"].notna()][cell_keys]
    assert len(valid_cells) > 0  # at least one cell where the guard did NOT fire
    valid_exh = exh_hv.merge(valid_cells, on=cell_keys, how="inner")
    assert np.allclose(valid_exh["hvr"].to_numpy(), 1.0)

    valid_non_exh = (
        hv_df[hv_df["method"] != "exhaustive"]
        .merge(valid_cells, on=cell_keys, how="inner")
    )
    assert valid_non_exh["hvr"].notna().all()
    assert (valid_non_exh["hvr"] > 0.0).all()
    assert (valid_non_exh["hvr"] <= 1.0 + 1e-9).all()

    # Parquet round-trip: the in-memory hv_df is what run_benchmark
    # returned; the parquet was written separately. Re-read it and
    # confirm the new column survives (pandas + parquet handle float
    # NaN cleanly, but the column has to be present in the schema).
    hv_df_roundtrip = pd.read_parquet(outdir / "hv_summary.parquet")
    assert "hvr" in hv_df_roundtrip.columns
    assert len(hv_df_roundtrip) == len(hv_df)

    # Both noise channels appear (verifies position_varying_asymmetric actually ran).
    assert set(results_df["noise_channel"].unique()) == {"symmetric", "position_varying_asymmetric"}

    exh = results_df[results_df["method"] == "exhaustive"]
    assert len(exh) > 0

    # All four methods appear in hv_summary.
    assert set(hv_df["method"].unique()) == {
        "duet", "greedy_hamming_mo", "greedy_nll_mo", "exhaustive",
    }

    # Codebook IDs use the new prefixes.
    assert results_df[results_df["method"] == "greedy_hamming_mo"]["codebook_id"].iloc[0].count("greedy_hamming_lambda_") == 1
    assert results_df[results_df["method"] == "greedy_nll_mo"]["codebook_id"].iloc[0].count("greedy_nll_lambda_") == 1

    # Visualizer smoke: call plot_aggregate_hvr on the in-memory hv_df
    # and confirm the SVG was written. Folds in what would otherwise be
    # a separate end-to-end run.
    from duet.plotting import apply_style
    from scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark import (
        plot_aggregate_hvr,
    )
    apply_style()
    plot_aggregate_hvr(hv_df, outdir)
    svg = outdir / "aggregate_hvr.svg"
    assert svg.exists() and svg.stat().st_size > 0


def test_visualizer_writes_per_trial_dirs_only_with_debug_plots(
    tiny_small_config, monkeypatch, capsys,
):
    """By default the visualizer writes the five aggregate figures and no
    per-trial directory; --debug-plots adds raw_scatter_per_trial/ and
    pareto_per_trial/ (debug plots, duet.plotting.tiers).

    The default run renders for real (this visualizer's end-to-end run). The
    --debug-plots run only has to show which files main asks for, so its
    plotters are fakes that write empty files where the real ones would.
    """
    import sys

    import scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark as viz

    run_benchmark(tiny_small_config)
    outdir = Path(tiny_small_config.outdir)
    per_trial_dirs = ("raw_scatter_per_trial", "pareto_per_trial")

    monkeypatch.setattr(sys, "argv", ["prog", "--indir", str(outdir)])
    viz.main()
    assert {p.name for p in outdir.glob("*.svg")} == {
        "recovered_fronts.svg", "recovered_fronts_per_trial.svg",
        "aggregate_hv.svg", "aggregate_hvr.svg", "aggregate_igd.svg",
    }
    for name in per_trial_dirs:
        assert not (outdir / name).exists()
    out = capsys.readouterr().out
    assert "Skipped 2 per-trial debug plots" in out and "--debug-plots" in out

    def fake_aggregate(filename):
        return lambda df, outdir: (outdir / filename).touch()

    def fake_per_trial(subdir_name):
        # Like the real ones: make the subdirectory, then one file per trial.
        def plot(results_df, outdir):
            subdir = outdir / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)
            for trial in sorted(results_df["trial"].unique()):
                (subdir / f"trial_{int(trial):02d}.svg").touch()
        return plot

    for fn, filename in [
        ("plot_recovered_fronts_aggregated", "recovered_fronts.svg"),
        ("plot_recovered_fronts_per_trial", "recovered_fronts_per_trial.svg"),
        ("plot_aggregate_hv", "aggregate_hv.svg"),
        ("plot_aggregate_hvr", "aggregate_hvr.svg"),
        ("plot_aggregate_igd", "aggregate_igd.svg"),
    ]:
        monkeypatch.setattr(viz, fn, fake_aggregate(filename))
    monkeypatch.setattr(viz, "plot_per_trial_raw_scatters", fake_per_trial("raw_scatter_per_trial"))
    monkeypatch.setattr(viz, "plot_per_trial_pareto_fronts", fake_per_trial("pareto_per_trial"))

    monkeypatch.setattr(sys, "argv", ["prog", "--indir", str(outdir), "--debug-plots"])
    viz.main()
    for name in per_trial_dirs:
        assert [p.name for p in (outdir / name).iterdir()] == ["trial_01.svg"]


def test_hv_summary_has_objective_columns_with_correct_rowfill(tiny_small_config):
    """hv_summary.parquet contains duet_objective_at_lambda_one and
    max_duet_objective, populated only on the duet and exhaustive rows
    respectively. Greedy and other-method rows have both as NaN.
    """
    _, hv_df = run_benchmark(tiny_small_config)

    assert "duet_objective_at_lambda_one" in hv_df.columns
    assert "max_duet_objective" in hv_df.columns

    duet_rows = hv_df[hv_df["method"] == "duet"]
    exh_rows = hv_df[hv_df["method"] == "exhaustive"]
    greedy_rows = hv_df[hv_df["method"].isin(
        ["greedy_hamming_mo", "greedy_nll_mo"]
    )]

    # DUET row populates objective_at_lambda_one, leaves max NaN.
    assert duet_rows["duet_objective_at_lambda_one"].notna().all()
    assert duet_rows["max_duet_objective"].isna().all()

    # Exhaustive row populates max, leaves objective_at_lambda_one NaN.
    assert exh_rows["max_duet_objective"].notna().all()
    assert exh_rows["duet_objective_at_lambda_one"].isna().all()

    # Greedy rows have both NaN — the objective is a DUET-specific metric.
    assert greedy_rows["duet_objective_at_lambda_one"].isna().all()
    assert greedy_rows["max_duet_objective"].isna().all()

    # The objective is a real-valued union bound (<= 1; can be negative).
    assert (duet_rows["duet_objective_at_lambda_one"] <= 1.0 + 1e-9).all()
    assert (exh_rows["max_duet_objective"] <= 1.0 + 1e-9).all()

    # Max objective >= DUET's objective within each cell (DUET's pick is one
    # of the enumerated codebooks). Join on the cell keys.
    cell_keys = ["trial", "noise_channel", "error_rate"]
    paired = duet_rows[cell_keys + ["duet_objective_at_lambda_one"]].merge(
        exh_rows[cell_keys + ["max_duet_objective"]], on=cell_keys
    )
    assert len(paired) > 0
    assert (
        paired["duet_objective_at_lambda_one"]
        <= paired["max_duet_objective"] + 1e-9
    ).all()


def test_objective_at_lambda_one_agrees_with_decoding_swap_cache(
    tiny_small_config,
):
    """duet_objective_at_lambda_one (stored from the evaluator's cached PEP)
    matches DecodingSwapCache.compute_objective on the same codebook and
    matrix to within 1e-8 — the two paths share input but use different
    summation orders, so float64 accumulation differs by O(1e-9).

    The benchmark gate inside hv_summary.parquet uses 1e-9 because both
    sides come from the same compute_duet_objective call; this cross-path
    test legitimately needs more headroom. If this assertion fails,
    either:
      - the runner is using DUET's pep_count_matrix somewhere it shouldn't,
        or
      - compute_duet_objective disagrees with DecodingSwapCache.compute_objective
        on the chosen codebook beyond float64 rounding.
    Either resolution is a precondition for trusting the gate.
    """
    import numpy as np

    from duet.benchmark.metrics import compute_duet_objective
    from duet.benchmark.synthetic_pool import create_2d_synthetic_pool
    from duet.codebook_evaluator import CodebookEvaluator, UniqueMinimum
    from duet.runner.ops import run_duet_ops
    from duet.pareto_optimization import DecodingSwapCache
    from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
        _build_evaluator_config,
        _create_noise_and_decoding_metric,
    )

    cfg = tiny_small_config
    trial_seed = cfg.seed
    pool = create_2d_synthetic_pool(
        num_groups=cfg.pool.num_groups,
        candidates_per_group=cfg.pool.candidates_per_group,
        seq_length=cfg.pool.seq_length,
        alphabet_size=cfg.pool.alphabet_size,
        quota=cfg.pool.quota,
        seed=trial_seed,
    )

    seq_length = pool.sequences.shape[1]
    alphabet_size = max(int(pool.sequences.max()) + 1, 2)
    noise_type = cfg.noise_channels[0]  # "symmetric"
    error_rate = cfg.error_rates[0]   # 0.1

    noise_channel, decoding_metric = _create_noise_and_decoding_metric(
        noise_type, error_rate, seq_length, alphabet_size,
    )
    evaluator = CodebookEvaluator(
        codebook=pool.sequences,
        noise_channel=noise_channel,
        decoding_metric=decoding_metric,
        decoding_rule=UniqueMinimum(),
        n_samples=cfg.evaluator.num_samples,
        seed=trial_seed,
    )
    evaluator.initialize_cache(n_jobs=cfg.evaluator.num_cpus)

    pep_config = _build_evaluator_config(
        noise_type=noise_type,
        error_rate=error_rate,
        seq_length=seq_length,
        alphabet_size=alphabet_size,
        num_samples=cfg.evaluator.num_samples,
        num_cpus=cfg.evaluator.num_cpus,
        seed=trial_seed,
    )
    init = pool.sample_initial_selection(strategy="random", seed=trial_seed)

    import tempfile
    with tempfile.TemporaryDirectory() as cache_dir:
        duet_out = run_duet_ops(
            candidates=pool,
            pep_config=pep_config,
            optimizer_config=cfg.duet,
            init=init,
            alphabet_size=alphabet_size,
            seed=trial_seed,
            cache_dir=cache_dir,
            use_mmap=True,
        )
        # Pick the lambda=1 codebook.
        S_lambda_1 = None
        for lambda_, indices in zip(duet_out["lambdas"], duet_out["best_indices"]):
            if float(lambda_) == 1.0:
                S_lambda_1 = np.asarray(indices, dtype=int)
                break
        assert S_lambda_1 is not None, "DUET config must include lambda=1"

        M_eval = evaluator.get_pairwise_error_from_cache()
        runner_value = compute_duet_objective(M_eval, S_lambda_1)

        group_to_candidates = {
            f"group_{i}": list(g) for i, g in enumerate(pool.group_to_candidates.values())
        }
        codeword_to_group_list = [
            f"group_{i}"
            for i, q in enumerate(pool.quotas.values())
            for _ in range(q)
        ]
        cache = DecodingSwapCache.from_pep_matrix(
            M_eval,
            group_to_candidates,
            codeword_to_group=codeword_to_group_list,
        )
        cache.build_cache(S_lambda_1)
        cache_value = cache.compute_objective(S_lambda_1)

        # 1e-8 rather than 1e-9: both sides use the same PEP values but
        # different summation paths (direct off-diagonal sum vs. symmetrized
        # row-sum identity), which introduces O(N^2) float64 rounding
        # differences. 1e-8 is tight enough to catch real invariant
        # violations while tolerating benign fp accumulation order.
        assert abs(runner_value - cache_value) <= 1e-8, (
            f"compute_duet_objective ({runner_value}) disagrees with "
            f"DecodingSwapCache.compute_objective ({cache_value}) on the "
            f"same evaluator-cached PEP matrix — gate invariant violated."
        )

        pep = duet_out.get("pep_count_matrix")
        if hasattr(pep, "close"):
            pep.close()

    evaluator.close()


@pytest.mark.parametrize(
    "device_line, expected",
    [
        ("", "cpu"),                    # key absent -> CPU default
        ("device: null\n", "cpu"),      # explicit null -> CPU default
        ('device: "gpu:all"\n', "gpu:all"),
    ],
)
def test_top_level_device_key(tmp_path: Path, device_line: str, expected: str):
    """`device` is an optional top-level key (same convention as the OPS and
    MERFISH benchmark configs). Absent or null resolves to "cpu"; a string is
    passed through verbatim for the evaluator cache and DUET PEP steps."""
    yaml_text = textwrap.dedent(f"""\
        outdir: {tmp_path}/run
        seed: 7
        trials: 1
        {device_line}
        pool:
          num_groups: 3
          candidates_per_group: 3
          seq_length: 6
          alphabet_size: 2
          quota: 1

        evaluator:
          num_samples: 200
          num_cpus: 1

        noise_channels: ["symmetric"]
        error_rates: [0.1]

        duet:
          optimizer:
            lambda: [0.0, 1.0]
            num_cpus: 1
    """)
    cfg_path = tmp_path / "device.yaml"
    cfg_path.write_text(yaml_text)
    cfg = TwoDSyntheticBenchmarkConfig.from_yaml(str(cfg_path))
    assert cfg.device == expected
