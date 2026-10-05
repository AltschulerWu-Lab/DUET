"""Output tiers of the synthetic and rounds-sweep visualizers (duet.plotting.tiers).

By default a visualizer writes its paper panels and diagnostics only; the debug
plots need --debug-plots. Covers the rounds x epsilon sweep heatmaps (Fig 3c)
and the objective-correlation per-trial scatters (Fig 2a) and summaries.
The 2-D synthetic benchmark's per-trial directories are tested in
test_2d_synthetic_benchmark.py, beside the fixture that runs it.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "experiments" / "ops_crispri_rounds_error_sweep" / "visualize_sweep.py"

NOISE_CHANNELS = ["symmetric", "position_varying", "asymmetric", "position_varying_asymmetric"]


@pytest.fixture(scope="module")
def sweep():
    """experiments/ops_crispri_rounds_error_sweep/visualize_sweep.py as a module."""
    spec = importlib.util.spec_from_file_location("rounds_error_visualize_sweep", SWEEP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _sweep_summary(sweep, baselines: list[str]) -> pd.DataFrame:
    """A 2 x 2 (epsilon x rounds) sweep summary, 2 trials, every metric."""
    rng = np.random.default_rng(0)
    rows = []
    for rounds in (8, 10):
        for eps in (0.05, 0.10):
            stem = f"sr{rounds:02d}_eps{eps:.2f}"
            for trial in (1, 2):
                for baseline in baselines:
                    for spec in sweep.SWEEP_METRICS:
                        sign = 1.0 if spec.direction == "max" else -1.0
                        rows.append({
                            "stem": stem, "rounds": rounds, "epsilon": eps,
                            "trial": trial, "baseline": baseline,
                            "metric": spec.name, "kind": spec.kind,
                            "direction": spec.direction,
                            "duet_lambda": "DUET (lambda=0.50)",
                            "delta": sign * rng.uniform(0.01, 0.2),
                        })
    return sweep.summarize_deltas(pd.DataFrame(rows))


def test_sweep_default_is_fig3c_plus_mean_decode_accuracy(sweep):
    baselines = [sweep.MAX_ACTIVITY_METHOD] + [f"Baseline {k}" for k in range(5)]
    default = {
        (b, spec.name)
        for b in baselines
        for spec in sweep.SWEEP_METRICS
        if sweep.heatmap_is_default(b, spec.name)
    }
    fig3c = {(b, m) for _, b, m in sweep.FIG3C_PANELS}
    assert fig3c <= default
    assert default - fig3c == {(b, "Mean decode accuracy") for b in baselines}
    assert len(default) == 8  # of 60


def test_sweep_render_heatmaps_default_and_debug(sweep, tmp_path, monkeypatch):
    other = "Sivanandan et al. (ED=1)"
    summary = _sweep_summary(sweep, [sweep.MAX_ACTIVITY_METHOD, other])
    n_all = 2 * len(sweep.SWEEP_METRICS)

    default_dir = tmp_path / "default"
    assert sweep.render_heatmaps(summary, default_dir) == (4, n_all - 4)
    expected = {sweep.heatmap_name(b, m) for _, b, m in sweep.FIG3C_PANELS} | {
        sweep.heatmap_name(b, "Mean decode accuracy")
        for b in (sweep.MAX_ACTIVITY_METHOD, other)
    }
    assert {p.name for p in default_dir.iterdir()} == expected

    # The default run above renders its four heatmaps for real. The debug run
    # only has to show that every heatmap is asked for, so the renderer is a
    # fake that, like the real one, makes the parent directory and the file.
    def fake_plot(summary, baseline, metric, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        return True

    monkeypatch.setattr(sweep, "plot_delta_heatmap", fake_plot)
    debug_dir = tmp_path / "debug"
    assert sweep.render_heatmaps(summary, debug_dir, debug_plots=True) == (n_all, 0)
    assert len(list(debug_dir.glob("heatmap_*.svg"))) == n_all


@pytest.mark.parametrize("flag, debug", [([], False), (["--debug-plots"], True)])
def test_sweep_main_passes_debug_plots_to_render(sweep, tmp_path, monkeypatch, flag, debug):
    # Stub the loaders so main reaches the render step, then stop it there.
    summary = _sweep_summary(sweep, [sweep.MAX_ACTIVITY_METHOD])
    monkeypatch.setattr(sweep, "load_sweep", lambda base, stems: {stems[0]: None})
    monkeypatch.setattr(sweep, "compute_sweep_deltas", lambda loaded: summary)
    monkeypatch.setattr(sweep, "summarize_deltas", lambda deltas: deltas)
    calls = []

    def fake_render(summary, figures_dir, debug_plots=False):
        calls.append(debug_plots)
        raise SystemExit("stopped after render")

    monkeypatch.setattr(sweep, "render_heatmaps", fake_render)
    with pytest.raises(SystemExit, match="stopped after render"):
        sweep.main([*flag, "--results-base", str(tmp_path),
                    "--figures-dir", str(tmp_path / "figures"), "--no-reference-check"])
    assert calls == [debug]


def _objective_parquet(indir: Path, trials) -> None:
    """An enumeration-shaped objective-correlation parquet for `trials`."""
    rng = np.random.default_rng(1)
    rows = []
    for trial in trials:
        for noise in NOISE_CHANNELS:
            acc = rng.uniform(0.3, 0.95, size=12)
            for idx, a in enumerate(acc):
                rows.append({
                    "trial": trial, "seed": 41 + trial, "noise_channel": noise,
                    "error_rate": 0.2, "codebook_index": idx, "codebook_members": "0,1",
                    "decode_accuracy": float(a),
                    "duet_objective": float(a + rng.normal(0, 0.05)),
                    "mean_pairwise_nll": float(2 * a + rng.normal(0, 0.1)),
                    "min_pairwise_nll": float(1.5 * a + rng.normal(0, 0.1)),
                    "mean_pairwise_hamming": float(4 * a + rng.normal(0, 0.5)),
                    "min_pairwise_hamming": int(round(3 * a + rng.normal(0, 0.3))),
                })
    indir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(indir / "objective_correlation.parquet", index=False)


def _run(main, monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    main()


def test_objective_scatter_default_keeps_only_the_fig2a_trial(tmp_path, monkeypatch, capsys):
    import scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives as viz

    _objective_parquet(tmp_path, trials=(1, 2, 3))
    _run(viz.main, monkeypatch, "--indir", str(tmp_path))
    per_trial = tmp_path / "figures" / "baseline_comparison_per_trial"
    assert [p.name for p in per_trial.iterdir()] == ["trial_01.svg"]
    out = capsys.readouterr().out
    assert "Skipped 2 per-trial debug plots" in out and "--debug-plots" in out
    assert "WARNING" not in out

    # The default run above renders trial 1 for real. The debug run only has to
    # show which trials main renders, so render_figure is a fake; like the real
    # one it leaves creating the directory to main.
    monkeypatch.setattr(viz, "render_figure", lambda trial_df, trial, out_path: out_path.touch())
    _run(viz.main, monkeypatch, "--indir", str(tmp_path), "--debug-plots")
    assert sorted(p.name for p in per_trial.iterdir()) == [
        "trial_01.svg", "trial_02.svg", "trial_03.svg",
    ]


def test_objective_scatter_without_trial_1_writes_nothing_and_warns(tmp_path, monkeypatch, capsys):
    from scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives import main

    _objective_parquet(tmp_path, trials=(2, 3))
    _run(main, monkeypatch, "--indir", str(tmp_path))
    # No empty output folder on a default run.
    assert not (tmp_path / "figures").exists()
    out = capsys.readouterr().out
    assert "WARNING: paper panel 'baseline_comparison_per_trial/trial_01'" in out


@pytest.mark.parametrize("flag", [[], ["--debug-plots"]])
def test_baseline_summary_writes_the_same_figures_with_or_without_flag(tmp_path, monkeypatch, flag):
    from scripts.benchmark.synthetic.visualize_baseline_summary import main

    _objective_parquet(tmp_path, trials=(1, 2, 3))
    _run(main, monkeypatch, "--indir", str(tmp_path), *flag)
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {
        "summary_spearman.svg", "summary_regret.svg",
    }
