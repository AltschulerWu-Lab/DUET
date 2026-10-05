"""Output tiers of the MERFISH visualizers (duet.plotting.tiers).

By default ``scripts/benchmark/visualize_merfish.py`` writes its paper panels
and diagnostics only, plus any debug plot the config lists under
``visualization.paper_panels`` (the Fig 4b/4c jointplot insets); the debug
plots need --debug-plots. ``experiments/merfish_2000_genes/make_figures.py``
hard-codes its two paper panels (Fig 4d,e) and writes its three alternates
only with --debug-plots.

The visualize_merfish.py and sweep tests check which files a run writes, so
they swap the figure renders for fakes (``fake_renders``), except one default
run that renders for real. The make_figures.py figures are cheap and render for
real.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from duet.merfish_benchmark import visualization as viz
from duet.merfish_benchmark.visualization import (
    MIN_TRANSCRIPTS_FOR_JOINTPLOT,
    metric_barplot_stem,
    summary_metric_columns,
    sweep_decode_accuracy_histogram,
    sweep_decode_accuracy_histogram_by_hamming_weight,
    sweep_decode_vs_identified_jointplot,
)
from duet.plotting import apply_style

REPO_ROOT = Path(__file__).resolve().parents[1]
VISUALIZE_SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "visualize_merfish.py"
MAKE_FIGURES_SCRIPT = REPO_ROOT / "experiments" / "merfish_2000_genes" / "make_figures.py"

LAMBDAS = (0.5, 0.8, 1.0)
BASELINES = ("Bostrom et al. (Hamming weight 5)", "MERFISH MHD4 (Hamming weight 5)")
N_GENES = 20

SUMMARY_STEMS = [
    "mean_decode_accuracy", "std_decode_accuracy", "5th_percentile_decode_accuracy",
    "10th_percentile_decode_accuracy", "95th_percentile_decode_accuracy",
    "mean_decode_accuracy_le_5th_pct", "mean_decode_accuracy_le_10th_pct",
    "95th_over_5th_percentile_ratio",
]
DEFAULT_STEMS = {
    "figures/mean_decode_accuracy",
    "figures/5th_percentile_decode_accuracy",
    "figures/decode_accuracy_histogram",
    "figures/hamming_weight_distribution",
    "figures/crowding_pareto_front",
    "figures/crowding_pareto_front_lambda_labeled",
    "figures/crowding_bar_chart",
    "figures/decode_accuracy_histogram_by_hamming_weight",
}
INSET = "figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def visualize_merfish():
    return _load_script("visualize_merfish", VISUALIZE_SCRIPT)


@pytest.fixture(scope="module")
def make_figures():
    return _load_script("merfish_2000_genes_make_figures", MAKE_FIGURES_SCRIPT)


@pytest.fixture(autouse=True)
def _style():
    apply_style()


def _methods() -> list[str]:
    return [f"DUET (lambda={lam:.2f})" for lam in LAMBDAS] + list(BASELINES)


def _results_df() -> pd.DataFrame:
    """Per-codeword rows: HW4 and HW5 genes with spread in both accuracies."""
    rng = np.random.default_rng(0)
    rows = []
    for method in _methods():
        for i in range(N_GENES):
            hw = 4 if i % 2 else 5
            rows.append({
                "Method": method, "Index": i, "Gene": f"G{i}",
                "Sequence": "1" * hw + "0" * (16 - hw),
                "Decode accuracy": rng.uniform(0.85, 0.99),
                "Identified fraction": rng.uniform(0.80, 0.95),
                "n_transcripts": MIN_TRANSCRIPTS_FOR_JOINTPLOT + 5 * i,
                "No_error": 0.9, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    return pd.DataFrame(rows)


def _metrics_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """metrics.csv columns: the eight summary stats, lambda and crowding."""
    rows = []
    for method, g in results_df.groupby("Method", sort=False):
        acc = g["Decode accuracy"]
        p5, p10, p95 = np.percentile(acc, [5, 10, 95])
        lam = float(method.split("=")[1].rstrip(")")) if method.startswith("DUET") else np.nan
        idf = float(g["Identified fraction"].mean())
        rows.append({
            "Method": method,
            "Mean decode accuracy": acc.mean(),
            "Std decode accuracy": acc.std(),
            "5th percentile decode accuracy": p5,
            "10th percentile decode accuracy": p10,
            "95th percentile decode accuracy": p95,
            "Mean decode accuracy (<= 5th pct)": acc[acc <= p5].mean(),
            "Mean decode accuracy (<= 10th pct)": acc[acc <= p10].mean(),
            "95th/5th percentile ratio": p95 / p5,
            "se_decode_accuracy": 0.001,
            "lambda": lam,
            "mean_conflict_fraction": 1 - idf,
            "std_conflict_fraction": 0.02,
            "mean_identified_fraction": idf,
            "std_identified_fraction": 0.02,
            "se_identified_fraction": 0.001,
        })
    return pd.DataFrame(rows)


def _config(tmp_path: Path, outdir: Path, paper_panels: list[str] | None) -> Path:
    """A minimal MERFISH config whose outdir holds the run's CSVs."""
    expression = tmp_path / "expression.csv"
    pd.DataFrame({
        "gene_symbol": [f"G{i}" for i in range(N_GENES)],
        "mean_cpm": [10.0 * (i + 1) for i in range(N_GENES)],
    }).to_csv(expression, index=False)
    evaluator = {
        "noise_channel": {"type": "symmetric", "epsilon": 0.05},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 100, "num_cpus": 1, "seed": 0,
    }
    d = {
        "outdir": str(outdir),
        "candidates": {"seq_rounds": 16, "codebook_size": N_GENES, "hamming_weights": [4, 5]},
        "evaluator": evaluator,
        "expression": {"path": str(expression), "gene_col": "gene_symbol",
                       "expression_col": "mean_cpm"},
        "duet": {
            "use_mmap": False, "device": "cpu", "force_rebuild": False,
            "lambda": [1.0], "temperature": 0.0, "max_iter": 10,
            "max_patience": 5, "avoid_duplicates": False, "num_cpus": 1,
            "pep": dict(evaluator),
        },
        "seed": 0,
    }
    if paper_panels is not None:
        d["visualization"] = {"paper_panels": paper_panels}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A finished run: results.csv and metrics.csv."""
    run = tmp_path / "run"
    run.mkdir()
    results = _results_df()
    results.to_csv(run / "results.csv", index=False)
    _metrics_df(results).to_csv(run / "metrics.csv", index=False)
    return run


def _stems(root: Path) -> set[str]:
    """Figure stems under ``root``; every one must have its SVG and PNG."""
    svgs = {p.relative_to(root).with_suffix("").as_posix() for p in root.rglob("*.svg")}
    pngs = {p.relative_to(root).with_suffix("").as_posix() for p in root.rglob("*.png")}
    assert svgs == pngs
    return svgs


def _run_visualizer(module, monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["visualize_merfish.py", *argv])
    module.main()


# The plotters visualize_merfish.py calls that write one fixed stem each.
FIXED_STEM_PLOTTERS = {
    "plot_hamming_weight_distribution": "hamming_weight_distribution",
    "plot_crowding_pareto_front": "crowding_pareto_front",
    "plot_crowding_pareto_front_lambda_labeled": "crowding_pareto_front_lambda_labeled",
    "plot_crowding_bar_chart": "crowding_bar_chart",
    "plot_identified_fraction_vs_count": "identified_fraction_vs_count",
}
# The renderers behind the canonical and per-λ histograms and jointplots; each
# writes the ``stem`` it is given.
RENDERERS = (
    "_render_decode_accuracy_histogram",
    "_render_decode_vs_identified_jointplot",
    "_render_decode_accuracy_histogram_by_hw",
)


def _fake(real, stems):
    """A stand-in for ``real`` that creates ``output_dir/<stem>.{svg,png}`` for
    each stem ``stems(arguments)`` names. Like savefig it makes no folder, and
    the call must still bind to ``real``'s signature."""
    signature = inspect.signature(real)

    def fake(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for stem in stems(bound.arguments):
            for suffix in (".svg", ".png"):
                (Path(bound.arguments["output_dir"]) / f"{stem}{suffix}").touch()

    return fake


def _barplot_stems(arguments) -> list[str]:
    metrics = arguments["metrics"]
    if metrics is None:
        metrics = summary_metric_columns(arguments["metrics_df"])
    return [metric_barplot_stem(m) for m in metrics]


@pytest.fixture
def fake_renders(monkeypatch, visualize_merfish):
    """Swap the figure renders for ``_fake``s, so a test checks which files a
    run writes without drawing them. What picks the figures stays real: main(),
    the sweeps (which λ, which folder) and the canonical histogram and jointplot
    wrappers (which methods)."""
    monkeypatch.setattr(visualize_merfish, "plot_metrics_barplots",
                        _fake(viz.plot_metrics_barplots, _barplot_stems))
    for name, stem in FIXED_STEM_PLOTTERS.items():
        monkeypatch.setattr(visualize_merfish, name,
                            _fake(getattr(viz, name), lambda _, stem=stem: [stem]))
    for name in RENDERERS:
        monkeypatch.setattr(viz, name, _fake(getattr(viz, name), lambda a: [a["stem"]]))


# -------------------- visualize_merfish.py --------------------


@pytest.mark.usefixtures("fake_renders")
def test_default_run_writes_paper_panels_and_diagnostics(
    visualize_merfish, run_dir, tmp_path, monkeypatch, capsys,
):
    """Default output: the diagnostics plus the one sweep λ the config lists."""
    out = tmp_path / "out"
    config = _config(tmp_path, run_dir, [INSET])
    _run_visualizer(visualize_merfish, monkeypatch,
                    "--config", str(config), "--output-dir", str(out))

    assert _stems(out) == DEFAULT_STEMS | {INSET}
    # Sweep folders exist only where something was written.
    assert sorted(p.name for p in (out / "figures").iterdir() if p.is_dir()) == ["jointplot_sweep"]
    # 6 metric plots, 2 matched-λ checks, 2 + 3 + 3 sweep panels.
    stdout = capsys.readouterr().out
    assert "Skipped 16 debug plots; pass --debug-plots to write them" in stdout
    assert "WARNING" not in stdout


def test_default_run_without_listed_panels_writes_no_sweep(
    visualize_merfish, run_dir, tmp_path, monkeypatch,
):
    """With --input (no config, so no paper_panels) no sweep panel is written.

    Unstubbed: the default suite's end-to-end render of visualize_merfish.py.
    """
    out = tmp_path / "out"
    _run_visualizer(visualize_merfish, monkeypatch,
                    "--input", str(run_dir / "results.csv"), "--output-dir", str(out))
    assert _stems(out) == DEFAULT_STEMS
    assert not [p for p in (out / "figures").iterdir() if p.is_dir()]


@pytest.mark.usefixtures("fake_renders")
def test_debug_plots_adds_every_debug_plot(
    visualize_merfish, run_dir, tmp_path, monkeypatch, capsys,
):
    out = tmp_path / "out"
    config = _config(tmp_path, run_dir, [INSET])
    _run_visualizer(visualize_merfish, monkeypatch,
                    "--config", str(config), "--output-dir", str(out), "--debug-plots")

    sweeps = {
        f"figures/{folder}/{prefix}_lambda{lam:.2f}"
        for folder, prefix in [
            ("jointplot_sweep", "decode_vs_identified_jointplot"),
            ("decode_hist_sweep", "decode_accuracy_histogram"),
            ("decode_hist_by_hw_sweep", "decode_accuracy_histogram_by_hamming_weight"),
        ]
        for lam in LAMBDAS
    }
    expected = (
        DEFAULT_STEMS
        | {f"figures/{s}" for s in SUMMARY_STEMS}
        | {"figures/decode_vs_identified_jointplot", "figures/identified_fraction_vs_count"}
        | sweeps
    )
    assert _stems(out) == expected
    assert "Skipped" not in capsys.readouterr().out


@pytest.mark.usefixtures("fake_renders")
def test_listed_panel_that_is_not_written_is_reported(
    visualize_merfish, run_dir, tmp_path, monkeypatch, capsys,
):
    """A listed λ the run does not have is named in a warning, not dropped silently."""
    out = tmp_path / "out"
    missing = "figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.30"
    config = _config(tmp_path, run_dir, [missing])
    _run_visualizer(visualize_merfish, monkeypatch,
                    "--config", str(config), "--output-dir", str(out))
    assert f"WARNING: paper panel {missing!r}" in capsys.readouterr().out


def test_experiment_configs_list_the_jointplot_inset():
    """Both MERFISH experiments list the λ = 0.80 inset under the stem the
    sweep writes (λ formatted :.2f)."""
    from duet.plotting.tiers import paper_panels_from_config

    for exp in ("merfish_2000_genes", "merfish_zhang2023_v2"):
        config = yaml.safe_load((REPO_ROOT / "experiments" / exp / "config.yaml").read_text())
        assert paper_panels_from_config(config) == {INSET}


# -------------------- per-λ sweeps --------------------


@pytest.mark.parametrize("sweep, folder, prefix", [
    (sweep_decode_vs_identified_jointplot, "jointplot_sweep", "decode_vs_identified_jointplot"),
    (sweep_decode_accuracy_histogram, "decode_hist_sweep", "decode_accuracy_histogram"),
    (sweep_decode_accuracy_histogram_by_hamming_weight, "decode_hist_by_hw_sweep",
     "decode_accuracy_histogram_by_hamming_weight"),
])
@pytest.mark.usefixtures("fake_renders")
def test_sweep_include_picks_panels_by_svg_path(tmp_path, sweep, folder, prefix):
    results = _results_df()
    metrics = _metrics_df(results)
    keep = tmp_path / folder / f"{prefix}_lambda0.80.svg"

    assert sweep(results, metrics, tmp_path, include=lambda p: p == keep) == 2
    assert sorted(p.name for p in (tmp_path / folder).iterdir()) == [
        f"{prefix}_lambda0.80.png", f"{prefix}_lambda0.80.svg",
    ]

    # Nothing kept: nothing written and no empty folder.
    other = tmp_path / "none"
    assert sweep(results, metrics, other, include=lambda p: False) == len(LAMBDAS)
    assert not other.exists()


# -------------------- make_figures.py (Fig 4d,e) --------------------


def _make_figures_config(tmp_path: Path) -> Path:
    """A run with DUET at the script's default λ = 0.90 and one baseline."""
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for method in ("DUET (lambda=0.90)", BASELINES[0]):
        for i in range(N_GENES):
            hw = 4 if i % 2 else 5
            rows.append({"Method": method, "Gene": f"G{i}",
                         "Sequence": "1" * hw + "0" * (16 - hw)})
    pd.DataFrame(rows).to_csv(run / "results.csv", index=False)
    pd.DataFrame({
        "Method": ["DUET (lambda=0.90)", BASELINES[0]],
        "lambda": [0.9, np.nan],
        "Mean decode accuracy": [0.95, 0.93],
        "mean_identified_fraction": [0.90, 0.88],
    }).to_csv(run / "metrics.csv", index=False)
    return _config(tmp_path, run, None)


PAPER_FIG4DE = {"round_expression_uniformity_y0_k2000", "expr_by_hw_violin_k2000"}
DEBUG_FIG4DE = {
    "round_expression_uniformity_k2000", "round_expression_sorted_k2000", "expr_by_hw_box_k2000",
}


def test_make_figures_default_writes_fig4de_panels_only(make_figures, tmp_path, capsys):
    out = tmp_path / "fig4de"
    make_figures.main(["--config", str(_make_figures_config(tmp_path)), "--outdir", str(out)])
    assert _stems(out) == PAPER_FIG4DE
    stdout = capsys.readouterr().out
    assert "wrote 2 figures" in stdout
    assert "skipped 3 debug plots; pass --debug-plots to write them" in stdout


def test_make_figures_debug_plots_adds_the_alternates(make_figures, tmp_path, capsys):
    out = tmp_path / "fig4de"
    make_figures.main(["--config", str(_make_figures_config(tmp_path)), "--outdir", str(out),
                       "--debug-plots"])
    assert _stems(out) == PAPER_FIG4DE | DEBUG_FIG4DE
    stdout = capsys.readouterr().out
    assert "wrote 5 figures" in stdout
    assert "skipped" not in stdout
