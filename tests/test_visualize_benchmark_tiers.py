"""Tests for visualize_benchmark.py's output tiers and the aggregated Pareto panel.

A default run writes the hypervolume bar plot and the aggregated Pareto fronts
only; every per-trial figure is a debug plot, written with --debug-plots or
when the --config YAML lists it under visualization.paper_panels. The CRISPRi
symmetric config's list must name files the script writes. The aggregated
Pareto panel draws single-trial runs without error bars and with square legend
markers, gives a group with nothing to plot an "N/A" row, and gives the
diagnostic metrics a wider right margin than the paper row.

Tests that check only which files a run writes stub the plotters
(_stub_plotters); test_default_run_writes_only_the_diagnostics renders for real.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.text import Text

from duet.plotting import MM_PER_INCH, save_panel, style_context
from duet.plotting.legend import SINGLE_TRIAL_MARKERSIZE
from duet.plotting.tiers import PlotTiers, paper_panels_from_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

import visualize_benchmark  # noqa: E402
from visualize_benchmark import (  # noqa: E402
    MAX_ACTIVITY_METHOD,
    METRICS,
    PLOTTED_METHOD_GROUPS,
    plot_metric_scatterplot_aggregated,
    sanitize_filename,
)

# The files a default run writes, relative to the output directory.
DIAGNOSTICS = {
    "hypervolume/normalized_hypervolume_by_method_group.svg",
    *(f"pareto_fronts/{sanitize_filename(x)}_aggregated.svg" for x, _ in METRICS),
}

# (method, mean and spread of decode accuracy, mean activity score). DUET
# lambda=0.50 clears both activity fractions of the max-activity arm and
# Feldman's own activity, so every comparison resolves to it. The spreads
# differ so the standard-deviation panel spans a normal range; equal spreads
# put its last x tick on the right edge (see
# test_diagnostic_margin_fits_a_tick_label_on_the_right_edge).
METHODS = [
    ("Maximum activity", 0.70, 0.12, 1.00),
    ("DUET (lambda=0.00)", 0.71, 0.11, 0.995),
    ("DUET (lambda=0.50)", 0.80, 0.07, 0.985),
    ("DUET (lambda=1.00)", 0.85, 0.05, 0.90),
    ("Feldman et al. (ED=1)", 0.72, 0.10, 0.97),
]
# Dropped as invalid at load, like Sivanandan ED=3 in the real runs.
INVALID_METHOD = "Sivanandan et al. (ED=3)"
N_GUIDES = 40


def _write_results(path: Path, trials, spread: float | None = None) -> Path:
    """A small results.csv with every column visualize_benchmark.py reads.

    ``spread``, if given, replaces every method's own decode-accuracy spread.
    """
    rng = np.random.default_rng(0)
    frames = []
    for trial in trials:
        for method, decode, own_spread, activity in [*METHODS, (INVALID_METHOD, 0.7, 0.1, 0.9)]:
            acc = np.clip(
                rng.normal(decode, own_spread if spread is None else spread, N_GUIDES),
                0.01,
                1.0,
            )
            frames.append(pd.DataFrame({
                "Method": method,
                "Index": np.arange(N_GUIDES),
                "Decode accuracy": acc,
                "Activity score": rng.normal(activity, 0.01, N_GUIDES),
                "No_error": 0.1,
                "Corrected": acc - 0.01,
                "Failed": 1.0 - acc,
                "Trial": trial,
                "Valid": method != INVALID_METHOD,
            }))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _run_main(monkeypatch, *argv: str) -> None:
    """Run the script's main() with ``argv``, keeping its style out of other tests."""
    monkeypatch.setattr(sys, "argv", ["visualize_benchmark.py", *argv])
    with matplotlib.rc_context():
        visualize_benchmark.main()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _dirs(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}


def _stub_plotters(monkeypatch, keep=()) -> list[Path]:
    """Replace the script's plotters with fakes that only touch their files.

    For tests of which files a run writes: rendering every per-trial figure of
    one trial takes most of a minute. Each fake touches the file(s) its plotter
    would write and, like the real ones, does not make their folder, so a run
    that leaves a folder unmade still fails. Plotters named in ``keep`` stay
    real. Returns the list the fakes append each path to.
    """
    written: list[Path] = []

    def touch(path) -> None:
        Path(path).touch()
        written.append(Path(path))

    def comparison(
        results_df, method_to_hue, trial, output_dir, name_prefix,
        kinds=visualize_benchmark.COMPARISON_KINDS,
    ):
        # The file names plot_comparison writes. The paper-panels test below
        # keeps the real one, which checks them.
        for kind in kinds:
            touch(Path(output_dir) / f"{name_prefix}_trial_{trial}_{kind}.svg")

    def stack(results_df, method_to_color, trial, output_path, ranges, emphasize=None):
        touch(output_path)
        return visualize_benchmark.DistributionStackInfo({}, False, (0.0, 1.0), ())

    fakes = {
        # Diagnostics.
        "plot_hypervolume_barplot": lambda hv_df, output_path: touch(output_path),
        "plot_metric_scatterplot_aggregated": (
            lambda agg_results_df, x_col, y_col, output_path: touch(output_path)
        ),
        # Debug plots.
        "plot_pareto_hypervolume_single_trial": (
            lambda agg_results_df, trial, output_path: touch(output_path)
        ),
        "plot_pareto_hypervolume_aggregated": (
            lambda agg_results_df, output_path: touch(output_path)
        ),
        "plot_metric_scatterplot_single_trial": (
            lambda agg_results_df, trial, x_col, y_col, output_path: touch(output_path)
        ),
        "plot_comparison": comparison,
        "plot_jointplot": (
            lambda results_df, method_to_color, trial, output_path: touch(output_path)
        ),
        "plot_distribution_stack": stack,
        "plot_error_metrics_pie_comparison": (
            lambda results_df, baseline, duet, trial, output_path: touch(output_path)
        ),
        "plot_error_metrics_delta_histogram": (
            lambda results_df, baseline, duet, trial, output_path: touch(output_path)
        ),
        "plot_error_metrics_bar_comparison": (
            lambda results_df, method_pairs, trial, output_path: touch(output_path)
        ),
    }
    for name, fake in fakes.items():
        if name not in keep:
            monkeypatch.setattr(visualize_benchmark, name, fake)
    return written


# =============================================================================
# Tiers
# =============================================================================


def test_default_run_writes_only_the_diagnostics(tmp_path, monkeypatch, capsys):
    # Unstubbed: the default suite's end-to-end render of this script.
    results = _write_results(tmp_path / "run" / "results.csv", trials=[1, 2])
    out = tmp_path / "figures"
    _run_main(monkeypatch, "--input", str(results), "--output-dir", str(out))

    assert _files(out) == DIAGNOSTICS
    # No empty debug folders either.
    assert _dirs(out) == {"hypervolume", "pareto_fronts"}
    log = capsys.readouterr().out
    assert "debug plots; pass --debug-plots" in log


@pytest.mark.parametrize(
    "rendered",
    [False, pytest.param(True, marks=pytest.mark.slow)],
    ids=["stubbed", "rendered"],
)
def test_debug_plots_adds_the_per_trial_figures(tmp_path, monkeypatch, capsys, rendered):
    """The rendered run draws some 60 figures (about 50 s): slow."""
    if not rendered:
        _stub_plotters(monkeypatch)
    results = _write_results(tmp_path / "run" / "results.csv", trials=[1])
    out = tmp_path / "figures"
    _run_main(
        monkeypatch, "--input", str(results), "--output-dir", str(out), "--debug-plots"
    )

    files = _files(out)
    assert DIAGNOSTICS <= files
    assert {
        "hypervolume_visualization/pareto_fronts_trial_1.svg",
        "hypervolume_visualization/pareto_fronts_aggregated.svg",
        "pareto_fronts/Mean_decode_accuracy_trial_1.svg",
        "guide_level_comparisons/max_activity_97p5pct_trial_1_decode_accuracy.svg",
        "guide_level_comparisons/max_activity_97p5pct_trial_1_activity_score.svg",
        "guide_level_comparisons/max_activity_97p5pct_trial_1_dual_objectives.svg",
        "guide_level_comparisons/max_activity_97p5pct_trial_1_jointplot.svg",
        "guide_level_comparisons/Feldman_et_al_ED1_trial_1_distributions.svg",
        "guide_level_comparisons/all_baselines_vs_duet_97p5pct_trial_1_distributions.svg",
        "error_metrics/Feldman_et_al_ED1_vs_duet_95pct_trial_1_pie.svg",
        "error_metrics/Feldman_et_al_ED1_vs_duet_95pct_trial_1_delta_hist.svg",
        "error_metrics/corrected_rate_comparison_trial_1.svg",
        "error_metrics/corrected_rate_comparison_duet_97p5pct_trial_1.svg",
    } <= files
    assert len(files) > len(DIAGNOSTICS) + 40
    assert "Skipped" not in capsys.readouterr().out


def test_config_paper_panels_are_written_by_default(tmp_path, monkeypatch, capsys):
    # plot_comparison stays real: it names its files itself, and this is the
    # default-suite check that its names match the paths main() gates.
    _stub_plotters(monkeypatch, keep={"plot_comparison"})
    outdir = tmp_path / "run"
    _write_results(outdir / "results.csv", trials=[1, 2])
    listed = [
        "guide_level_comparisons/max_activity_97p5pct_trial_2_jointplot",
        "guide_level_comparisons/all_baselines_vs_duet_97p5pct_trial_1_distributions.svg",
        # One of plot_comparison's three files, without the other two.
        "guide_level_comparisons/max_activity_95pct_trial_1_decode_accuracy",
        # Not produced by this run (there is no trial 9): reported, not fatal.
        "guide_level_comparisons/max_activity_97p5pct_trial_9_jointplot",
    ]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "outdir": str(outdir),
        "visualization": {"paper_panels": listed},
    }))
    _run_main(monkeypatch, "--config", str(config))

    written = _files(outdir) - {"results.csv"}
    assert written == DIAGNOSTICS | {
        "guide_level_comparisons/max_activity_97p5pct_trial_2_jointplot.svg",
        "guide_level_comparisons/all_baselines_vs_duet_97p5pct_trial_1_distributions.svg",
        "guide_level_comparisons/max_activity_95pct_trial_1_decode_accuracy.svg",
    }
    assert "error_metrics" not in _dirs(outdir)
    assert "hypervolume_visualization" not in _dirs(outdir)
    log = capsys.readouterr().out
    assert "max_activity_97p5pct_trial_9_jointplot" in log
    assert "WARNING" in log


def test_listed_distribution_stack_keeps_the_trial_wide_ranges(tmp_path, monkeypatch):
    """A default run still takes the stacks' axis ranges over every comparison.

    Otherwise a listed panel (Fig 3b right) would change scale with --debug-plots.
    """
    outdir = tmp_path / "run"
    results = _write_results(outdir / "results.csv", trials=[1])
    panel = "Feldman_et_al_ED1_trial_1_distributions"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "outdir": str(outdir),
        "visualization": {"paper_panels": [f"guide_level_comparisons/{panel}"]},
    }))

    ranges_by_run = []

    def record(results_df, colors, trial, path, ranges, emphasize=None):
        ranges_by_run[-1][Path(path).stem] = ranges
        return visualize_benchmark.DistributionStackInfo({}, False, (0.0, 1.0), ())

    # The other figures do not matter here; skip drawing them.
    _stub_plotters(monkeypatch)
    monkeypatch.setattr(visualize_benchmark, "plot_distribution_stack", record)
    ranges_by_run.append({})
    _run_main(monkeypatch, "--config", str(config))
    ranges_by_run.append({})
    _run_main(
        monkeypatch, "--input", str(results), "--output-dir", str(tmp_path / "debug"),
        "--debug-plots",
    )

    default, debug = ranges_by_run
    assert list(default) == [panel]
    assert default[panel] == debug[panel]
    # The panel's own two methods alone would give other ranges, so the check
    # above can fail.
    results_df, _ = visualize_benchmark.load_and_process_results(results)
    own = visualize_benchmark.shared_axis_ranges(
        results_df, 1, ["Feldman et al. (ED=1)", "DUET (lambda=0.50)"]
    )
    assert default[panel] != own


# =============================================================================
# The CRISPRi symmetric config's paper panels
# =============================================================================

CRISPRI_CONFIG = _REPO_ROOT / "experiments" / "ops_crispri_symmetric" / "config.yaml"

# Baseline arms as (method, mean decode accuracy, mean activity as a fraction
# of the maximum, valid), roughly as in the real run, which drops Feldman ED=2
# and Sivanandan ED=3 as invalid.
CRISPRI_BASELINES = [
    (MAX_ACTIVITY_METHOD, 0.70, 1.0, True),
    ("Feldman et al. (ED=1)", 0.72, 0.99, True),
    ("Feldman et al. (ED=2)", 0.75, 0.93, False),
    ("Sivanandan et al. (ED=1)", 0.71, 0.999, True),
    ("Sivanandan et al. (ED=2)", 0.73, 0.985, True),
    ("Sivanandan et al. (ED=3)", 0.78, 0.90, False),
]
CRISPRI_MAX_ACTIVITY = 0.9


def _write_crispri_results(path: Path, trials, lambdas) -> Path:
    """A per-guide results.csv shaped like the CRISPRi symmetric run's.

    One DUET arm per lambda, labelled as duet.ops_benchmark.runner labels them.
    DUET's mean activity falls by 15% from lambda=0 to lambda=1 while its decode
    accuracy rises, so the 97.5% reference resolves to an interior lambda, as
    in the real run. The activity noise is centred, so each arm's mean activity
    is exact and the lambda picked does not depend on the draw.
    """
    arms = [*CRISPRI_BASELINES, *(
        (f"DUET (lambda={lam:.2f})", 0.70 + 0.15 * lam, 1.0 - 0.15 * lam, True)
        for lam in lambdas
    )]
    rng = np.random.default_rng(2)
    frames = []
    for trial in trials:
        for method, decode, activity, valid in arms:
            acc = np.clip(rng.normal(decode, 0.05, N_GUIDES), 0.01, 1.0)
            noise = rng.normal(0, 0.05, N_GUIDES)
            frames.append(pd.DataFrame({
                "Method": method,
                "Index": np.arange(N_GUIDES),
                "Decode accuracy": acc,
                "Activity score": CRISPRI_MAX_ACTIVITY * activity + noise - noise.mean(),
                "No_error": 0.1,
                "Corrected": acc - 0.01,
                "Failed": 1.0 - acc,
                "Trial": trial,
                "Valid": valid,
            }))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def test_crispri_config_paper_panels_name_files_the_script_writes(
    tmp_path, monkeypatch, capsys
):
    """Every visualization.paper_panels entry of the CRISPRi symmetric config is
    a debug-tier file of this script, so a default run writes it.

    A misspelt entry would otherwise surface only as a warning at the end of
    the real run.
    """
    config = yaml.safe_load(CRISPRI_CONFIG.read_text())
    panels = paper_panels_from_config(config)
    assert panels
    trials = range(1, config["trials"] + 1)
    outdir = tmp_path / "run"
    _write_crispri_results(
        outdir / "results.csv", trials, config["duet"]["optimizer"]["lambda"]
    )
    # The experiment's config, with its outdir moved to tmp_path.
    run_config = tmp_path / "config.yaml"
    run_config.write_text(yaml.safe_dump({**config, "outdir": str(outdir)}))
    diagnostics = {d.removesuffix(".svg") for d in DIAGNOSTICS}
    written = _stub_plotters(monkeypatch)

    debug_dir = tmp_path / "debug"
    _run_main(
        monkeypatch, "--config", str(run_config), "--output-dir", str(debug_dir),
        "--debug-plots",
    )
    debug_tier = {PlotTiers(debug_dir).key(p) for p in written} - diagnostics
    # Every trial has the 97.5% group the listed panels belong to:
    # select_duet_lambda found a DUET arm at 97.5% of the maximum activity.
    assert {
        f"guide_level_comparisons/max_activity_97p5pct_trial_{trial}_jointplot"
        for trial in trials
    } <= debug_tier
    assert panels <= debug_tier, sorted(panels - debug_tier)

    # And a default run from the config writes exactly the listed ones.
    written.clear()
    capsys.readouterr()
    _run_main(monkeypatch, "--config", str(run_config))
    assert {PlotTiers(outdir).key(p) for p in written} - diagnostics == panels
    assert "WARNING: paper panel" not in capsys.readouterr().out


# =============================================================================
# Aggregated Pareto panel: single-trial markers and N/A entries
# =============================================================================

X_COL, Y_COL = "Mean decode accuracy", "Mean activity score"

# (method, group, x, y) per trial, one method per group.
AGG_ROWS = [
    ("Maximum activity", "Maximum activity", 0.70, 1.00),
    ("DUET (lambda=0.50)", "DUET", 0.85, 0.95),
    ("Feldman et al. (ED=1)", "Feldman et al.", 0.80, 0.85),
    ("Sivanandan et al. (ED=2)", "Sivanandan et al.", 0.78, 0.92),
]


def _agg_frame(n_trials: int, drop=(), inf_x=()) -> pd.DataFrame:
    """Aggregated rows for ``n_trials`` trials.

    Groups in ``drop`` are absent; groups in ``inf_x`` have x = inf in every
    trial, as the 95th/5th ratio is when the 5th percentile is 0.
    """
    rng = np.random.default_rng(1)
    rows = []
    for method, group, x, y in AGG_ROWS:
        if group in drop:
            continue
        for trial in range(1, n_trials + 1):
            rows.append({
                "Method": method, "Method group": group, "Trial": trial,
                X_COL: np.inf if group in inf_x else x + rng.normal(0, 0.01),
                Y_COL: y + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def _legend_entries(ax):
    """(label, kind) per legend row, kind in {"errorbar", "square", "N/A"}."""
    legend = ax.get_legend()
    kinds = []
    for handle in legend.legend_handles:
        if isinstance(handle, Text) and handle.get_text() == "N/A":
            kinds.append("N/A")
        elif isinstance(handle, Line2D) and handle.get_marker() == "s":
            kinds.append("square")
        elif isinstance(handle, LineCollection):  # an errorbar entry's first artist
            kinds.append("errorbar")
        else:
            kinds.append(type(handle).__name__)
    return list(zip((t.get_text() for t in legend.get_texts()), kinds))


def _plot(tmp_path, agg: pd.DataFrame, x_col: str = X_COL, name: str = "agg.svg"):
    with style_context():
        return plot_metric_scatterplot_aggregated(agg, x_col, Y_COL, tmp_path / name)


def test_single_trial_panel_uses_square_legend_and_na_for_a_missing_group(tmp_path):
    ax = _plot(tmp_path, _agg_frame(1, drop={"Sivanandan et al."}))

    assert _legend_entries(ax) == [
        ("Maximum activity", "square"),
        ("DUET", "square"),
        ("Feldman et al.", "square"),
        ("Sivanandan et al.", "N/A"),
    ]
    assert ax.get_legend().get_title().get_text() == "Method Group"
    # The plotted points stay circles, larger, with no error bars.
    assert not ax.containers
    points = [line for line in ax.get_lines() if line.get_marker() == "o"]
    assert len(points) == 3
    assert {line.get_markersize() for line in points} == {SINGLE_TRIAL_MARKERSIZE}
    assert "N/A" in (tmp_path / "agg.svg").read_text()


def test_multi_trial_panel_keeps_errorbar_handles(tmp_path):
    ax = _plot(tmp_path, _agg_frame(3))

    assert _legend_entries(ax) == [(g, "errorbar") for g in PLOTTED_METHOD_GROUPS]
    assert len(ax.containers) == 4
    assert all(isinstance(c, ErrorbarContainer) for c in ax.containers)
    assert {c.lines[0].get_markersize() for c in ax.containers} == {1.5}


def test_multi_trial_panel_shows_na_for_a_missing_group(tmp_path):
    ax = _plot(tmp_path, _agg_frame(3, drop={"Feldman et al."}))

    assert _legend_entries(ax) == [
        ("Maximum activity", "errorbar"),
        ("DUET", "errorbar"),
        ("Feldman et al.", "N/A"),
        ("Sivanandan et al.", "errorbar"),
    ]
    assert len(ax.containers) == 3


@pytest.mark.parametrize("n_trials, present", [(1, "square"), (3, "errorbar")])
def test_group_with_only_infinite_values_gets_na(tmp_path, n_trials, present):
    ax = _plot(tmp_path, _agg_frame(n_trials, inf_x={"Maximum activity", "Feldman et al."}))

    assert _legend_entries(ax) == [
        ("Maximum activity", "N/A"),
        ("DUET", present),
        ("Feldman et al.", "N/A"),
        ("Sivanandan et al.", present),
    ]
    assert len(ax.containers) == (0 if n_trials == 1 else 2)


# =============================================================================
# Aggregated Pareto panel: margins
# =============================================================================

SD_COL = "Standard deviation decode accuracy"


def _margins_mm(ax) -> tuple[float, ...]:
    """The axes' (left, right, bottom, top) margins in mm, as slot_figure takes them."""
    width, height = ax.figure.get_size_inches() * MM_PER_INCH
    box = ax.get_position()
    return tuple(
        round(float(mm), 6)
        for mm in (box.x0 * width, (1 - box.x1) * width, box.y0 * height, (1 - box.y1) * height)
    )


ROW_MARGINS_MM = (16.0, 2.0, 12.0, 2.0)


def test_every_metric_keeps_the_row_margins(tmp_path, monkeypatch):
    """All seven panels share the row's margins, whatever their tick labels."""
    results = _write_results(tmp_path / "results.csv", trials=[1, 2], spread=0.05)
    _, agg = visualize_benchmark.load_and_process_results(results)
    margins = {}

    def record_then_save(fig, *paths):
        margins[len(margins)] = _margins_mm(fig.axes[0])
        save_panel(fig, *paths)

    monkeypatch.setattr(visualize_benchmark, "save_panel", record_then_save)
    for x, _ in METRICS:
        _plot(tmp_path, agg, x, f"{sanitize_filename(x)}.svg")

    assert list(margins.values()) == [ROW_MARGINS_MM] * len(METRICS)


def test_edge_tick_label_widens_the_x_range_not_the_margin(tmp_path, monkeypatch):
    """The standard-deviation panel's last label fits by a wider x range.

    With every method at the same decode-accuracy spread, the panel's last x
    tick lands on the axes' right edge and half its label runs past the 2 mm
    right margin, as on the CRISPRi symmetric data. Without the fit the save
    fails; with it the panel saves with the same margins and ticks, and only
    the upper x limit moves out.
    """
    results = _write_results(tmp_path / "results.csv", trials=[1, 2], spread=0.05)
    _, agg = visualize_benchmark.load_and_process_results(results)
    seen = []

    def record_then_save(fig, *paths):
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        seen.append(((lo, hi), [t for t in ax.get_xticks() if lo <= t <= hi], _margins_mm(ax)))
        try:
            save_panel(fig, *paths)
        finally:
            plt.close(fig)  # a failed save skips the plotter's own close

    monkeypatch.setattr(visualize_benchmark, "save_panel", record_then_save)
    with monkeypatch.context() as m:
        m.setattr(visualize_benchmark, "fit_last_xtick_label", lambda fig, ax: None)
        with pytest.raises(ValueError, match="right by"):
            _plot(tmp_path, agg, SD_COL, "unfitted.svg")
    assert not (tmp_path / "unfitted.svg").exists()

    _plot(tmp_path, agg, SD_COL, "fitted.svg")
    assert (tmp_path / "fitted.svg").exists()
    (unfitted_lim, unfitted_ticks, unfitted_margins), (lim, ticks, margins) = seen
    assert margins == unfitted_margins == ROW_MARGINS_MM
    assert ticks == unfitted_ticks
    assert lim[0] == unfitted_lim[0] and lim[1] > unfitted_lim[1]
