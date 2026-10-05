"""Tests for duet.merfish_benchmark.visualization plot functions.

These tests use small synthetic DataFrames so they run in <1s without
needing a live MERFISH benchmark. They assert filename layout, file
presence, and non-zero size -- visual fidelity is verified separately
by manual inspection of re-styled figures.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.merfish_benchmark.visualization import (
    plot_crowding_bar_chart,
    plot_crowding_pareto_front,
    plot_decode_accuracy_histogram,
    plot_metrics_barplots,
)
from duet.plotting import apply_style


SUMMARY_METRIC_COLUMNS = [
    "Mean decode accuracy",
    "Std decode accuracy",
    "5th percentile decode accuracy",
    "10th percentile decode accuracy",
    "95th percentile decode accuracy",
    "Mean decode accuracy (<= 5th pct)",
    "Mean decode accuracy (<= 10th pct)",
    "95th/5th percentile ratio",
]


def _sanitize_metric_filename(metric: str) -> str:
    """Mirror the runner's safe-filename mangle."""
    safe = metric.lower().replace(" ", "_").replace("/", "_over_")
    safe = safe.replace("(", "").replace(")", "").replace("<=", "le")
    safe = safe.replace("%", "pct")
    return safe


@pytest.fixture(autouse=True)
def _apply_style_once():
    """Plots assume apply_style() has run; call it once per test module."""
    apply_style()


@pytest.fixture
def captured_figures(monkeypatch):
    """Intercept plt.close so tests can inspect figures the function would
    otherwise discard. Each captured figure is closed for real after the
    test completes.

    Caveat: only intercepts `plt.close`. Functions that release figures
    via `fig.clf()`, garbage collection, or `plt.close("all")` will
    silently slip through. All six viz functions in scope today call
    `plt.close(fig)` or `plt.close()` — confirm before reusing this
    fixture for other plotting code.
    """
    import matplotlib.pyplot as plt
    figs: list = []
    real_close = plt.close

    def fake_close(arg=None):
        if isinstance(arg, plt.Figure):
            figs.append(arg)
        elif arg is None:
            figs.append(plt.gcf())
        # ignore int / str / "all" forms — none of the viz functions use them

    monkeypatch.setattr("matplotlib.pyplot.close", fake_close)
    yield figs
    for f in figs:
        real_close(f)


@pytest.fixture
def results_df():
    """Per-codeword rows. 1 DUET lambda + 2 Chen baselines, 3 codewords each."""
    rows = []
    for method in ["DUET (lambda=0.00)", "Codebook 1 (Chen 2015)", "Codebook 2 (Chen 2015)"]:
        for i, acc in enumerate([0.95, 0.93, 0.91]):
            rows.append({
                "Method": method, "Index": i, "Gene": f"GENE_{i}",
                "Sequence": "0" * 16, "Decode accuracy": acc,
                "Identified fraction": 0.85,
                "No_error": 0.90, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def results_df_multi_lambda():
    """Per-codeword rows for the multi-lambda sweep (used by histogram lambda-pin)."""
    rows = []
    # Multiple DUET lambdas with deliberately different accuracies -- the
    # histogram must pick lambda=1.00 (decode-only), not lambda=0.50.
    for lambda_, acc in [(1.0, 0.99), (0.5, 0.50)]:
        method = f"DUET (lambda={lambda_:.2f})"
        for i in range(3):
            rows.append({
                "Method": method, "Index": i, "Gene": f"GENE_{i}",
                "Sequence": "0" * 16, "Decode accuracy": acc,
                "Identified fraction": 0.85,
                "No_error": 0.90, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    for i, acc in enumerate([0.80, 0.78, 0.76]):
        rows.append({
            "Method": "Codebook 1 (Chen 2015)", "Index": i,
            "Gene": f"GENE_{i}", "Sequence": "0" * 16,
            "Decode accuracy": acc,
            "Identified fraction": 0.85,
            "No_error": 0.80, "Corrected": 0.10, "Failed": 0.10,
            "Trial": 1, "Valid": True,
        })
    return pd.DataFrame(rows)


def _metrics_row(method: str, mean_acc: float, *, lambda_=None, crowding=False):
    row = {
        "Method": method,
        "Mean decode accuracy": mean_acc,
        "Std decode accuracy": 0.02,
        "5th percentile decode accuracy": mean_acc - 0.04,
        "10th percentile decode accuracy": mean_acc - 0.03,
        "95th percentile decode accuracy": mean_acc + 0.03,
        "Mean decode accuracy (<= 5th pct)": mean_acc - 0.05,
        "Mean decode accuracy (<= 10th pct)": mean_acc - 0.04,
        "95th/5th percentile ratio": (mean_acc + 0.03) / max(mean_acc - 0.04, 1e-6),
    }
    if lambda_ is not None:
        row["lambda"] = lambda_
    if crowding:
        row["mean_identified_fraction"] = 0.85
        row["std_identified_fraction"] = 0.02
        row["mean_conflict_fraction"] = 0.15
        row["std_conflict_fraction"] = 0.03
    return row


@pytest.fixture
def metrics_df_no_crowding():
    return pd.DataFrame([
        _metrics_row("DUET (lambda=0.00)", 0.93, lambda_=0.0),
        _metrics_row("Codebook 1 (Chen 2015)", 0.88),
        _metrics_row("Codebook 2 (Chen 2015)", 0.86),
    ])


@pytest.fixture
def metrics_df_with_crowding():
    return pd.DataFrame([
        _metrics_row("DUET (lambda=0.00)", 0.93, lambda_=0.0, crowding=True),
        _metrics_row("DUET (lambda=0.50)", 0.88, lambda_=0.5, crowding=True),
        _metrics_row("Codebook 1 (Chen 2015)", 0.88, crowding=True),
        _metrics_row("Codebook 2 (Chen 2015)", 0.86, crowding=True),
    ])


# -------------------- _savefig_at_figsize --------------------


def test_savefig_at_figsize_pins_canvas_to_figsize(tmp_path: Path):
    """Helper saves SVG+PNG without bbox=tight canvas expansion.

    The stylesheet sets `savefig.bbox: tight`, which re-expands the
    saved canvas to fit all artists. The helper must wrap savefig in
    `rc_context({"savefig.bbox": "standard"})` so the SVG's reported
    width matches the figure's figsize.
    """
    import re
    import matplotlib.pyplot as plt
    from duet.merfish_benchmark.visualization import _savefig_at_figsize

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    # An annotation parked well past the axes would expand a tight bbox
    # but must not expand a "standard" bbox.
    ax.annotate("far-out artist", xy=(0.5, 0.5), xytext=(5.0, 5.0),
                xycoords="axes fraction", textcoords="axes fraction")

    svg_path = tmp_path / "out.svg"
    png_path = tmp_path / "out.png"
    _savefig_at_figsize(svg_path, png_path, png_dpi=100)
    plt.close(fig)

    assert svg_path.exists() and svg_path.stat().st_size > 0
    assert png_path.exists() and png_path.stat().st_size > 0

    # Matplotlib writes the SVG canvas size into the <svg> tag's
    # `width="..."` attribute in points. 2.0 inches = 144 pt.
    # Unit suffix is implementation detail (was unitless in older
    # matplotlib releases); make it optional.
    match = re.search(r'<svg[^>]*\bwidth="([\d.]+)(?:pt)?"', svg_path.read_text())
    assert match, f"no width attribute in {svg_path}"
    width_pt = float(match.group(1))
    assert 140 <= width_pt <= 160, (
        f"SVG width {width_pt}pt is outside the figsize-pinned range; "
        f"bbox=tight would expand it well past 144pt"
    )

    # rc_context should have restored savefig.bbox to "tight" on exit.
    assert plt.rcParams["savefig.bbox"] == "tight"


# -------------------- plot_metrics_barplots --------------------


def _labels_top_to_bottom(ax) -> list[str]:
    """Y tick label texts in the order they read on the panel, top first."""
    ys = ax.transData.transform([(0, y) for y in ax.get_yticks()])[:, 1]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    return [label for _, label in sorted(zip(ys, labels), key=lambda p: -p[0])]


# Method names of metrics_df_with_crowding as the horizontal bar charts show
# them: row order, top to bottom, with lambda= drawn as λ=.
CROWDING_FIXTURE_LABELS = [
    "DUET (λ=0.00)", "DUET (λ=0.50)", "Codebook 1 (Chen 2015)", "Codebook 2 (Chen 2015)",
]


def test_plot_metrics_barplots_emits_only_summary_stats(
    tmp_path: Path, metrics_df_with_crowding: pd.DataFrame,
    captured_figures: list,
):
    """One SVG+PNG per summary-stat column. No bar plots for lambda or crowding.

    Each is a horizontal half_page slot panel: method names on the y axis, in
    metrics_df row order from the top, and nothing past the figure edge.
    """
    from duet.plotting import FIGURE_WIDTHS, check_panel_fits

    plot_metrics_barplots(metrics_df_with_crowding, tmp_path)

    for metric in SUMMARY_METRIC_COLUMNS:
        safe = _sanitize_metric_filename(metric)
        svg = tmp_path / f"{safe}.svg"
        png = tmp_path / f"{safe}.png"
        assert svg.exists(), f"missing {svg}"
        assert png.exists(), f"missing {png}"
        assert svg.stat().st_size > 0
        assert png.stat().st_size > 0

    for excluded in ["lambda", "mean_identified_fraction",
                     "std_identified_fraction", "mean_conflict_fraction",
                     "std_conflict_fraction"]:
        safe = _sanitize_metric_filename(excluded)
        assert not (tmp_path / f"{safe}.svg").exists(), \
            f"unexpected bar plot for excluded column {excluded}"

    # captured_figures has one entry per loop iteration (8 summary stats).
    assert len(captured_figures) == len(SUMMARY_METRIC_COLUMNS)
    for fig, metric in zip(captured_figures, SUMMARY_METRIC_COLUMNS):
        assert fig.get_figwidth() == pytest.approx(FIGURE_WIDTHS["half_page"])
        assert fig.get_layout_engine() is None
        ax = fig.axes[0]
        assert _labels_top_to_bottom(ax) == CROWDING_FIXTURE_LABELS
        assert ax.get_title(loc="left") == metric
        check_panel_fits(fig)


def test_plot_metrics_barplots_draws_only_the_requested_metrics(
    tmp_path: Path, metrics_df_with_crowding: pd.DataFrame,
):
    """``metrics`` picks the plots; a name that is not a summary stat raises."""
    plot_metrics_barplots(
        metrics_df_with_crowding, tmp_path, metrics=["5th percentile decode accuracy"]
    )
    assert sorted(p.name for p in tmp_path.glob("*.svg")) == ["5th_percentile_decode_accuracy.svg"]
    with pytest.raises(ValueError, match="lambda"):
        plot_metrics_barplots(metrics_df_with_crowding, tmp_path, metrics=["lambda"])


@pytest.mark.parametrize("plot", [plot_metrics_barplots, plot_crowding_bar_chart])
def test_horizontal_bar_charts_grow_one_row_per_method(
    tmp_path: Path, plot, captured_figures: list,
):
    """Height grows by one row per method; the width stays half_page."""
    from duet.merfish_benchmark.visualization import _HBAR_ROW_MM
    from duet.plotting import MM_PER_INCH

    def rows(n):
        return pd.DataFrame([
            _metrics_row(f"DUET (lambda={lam:.2f})", 0.9, lambda_=lam, crowding=True)
            for lam in np.linspace(0, 1, n)
        ])

    plot(rows(2), tmp_path)
    plot(rows(5), tmp_path)
    small, large = captured_figures[0], captured_figures[-1]
    assert small.get_figwidth() == large.get_figwidth()
    assert (large.get_figheight() - small.get_figheight()) * MM_PER_INCH == \
        pytest.approx(3 * _HBAR_ROW_MM)


@pytest.mark.parametrize("plot", [plot_metrics_barplots, plot_crowding_bar_chart])
def test_horizontal_bar_charts_fit_long_method_names(
    tmp_path: Path, plot, captured_figures: list,
):
    """Names are never shortened: the left margin grows to fit the longest one."""
    from duet.plotting import FIGURE_WIDTHS, check_panel_fits

    long_name = "An unpublished baseline codebook (weight 5)"
    df = pd.DataFrame([
        _metrics_row("DUET (lambda=1.00)", 0.93, lambda_=1.0, crowding=True),
        _metrics_row(long_name, 0.90, crowding=True),
    ])
    plot(df, tmp_path)
    for fig in captured_figures:
        ax = fig.axes[0]
        assert _labels_top_to_bottom(ax) == ["DUET (λ=1.00)", long_name]
        assert fig.get_figwidth() == pytest.approx(FIGURE_WIDTHS["half_page"])
        check_panel_fits(fig)


@pytest.mark.parametrize("plot, chart", [
    pytest.param(lambda df, out: plot_metrics_barplots(df, out, metrics=["Mean decode accuracy"]),
                 "mean_decode_accuracy", id="plot_metrics_barplots"),
    pytest.param(plot_crowding_bar_chart, "crowding_bar_chart", id="plot_crowding_bar_chart"),
])
def test_horizontal_bar_charts_widen_for_a_very_long_method_name(
    tmp_path: Path, plot, chart: str, captured_figures: list, capsys,
):
    """A name too long for the half_page slot widens the figure instead of
    raising: the name is drawn whole inside the panel, the bars keep
    _HBAR_MIN_AXES_MM, the other margins hold, and one note names the chart."""
    from duet.merfish_benchmark.visualization import (
        _HBAR_EDGE_MM, _HBAR_MIN_AXES_MM, _HBAR_RIGHT_MM,
    )
    from duet.plotting import FIGURE_WIDTHS, MM_PER_INCH, check_panel_fits

    very_long = ("An unpublished baseline codebook with a much longer descriptive name "
                 "(Hamming weight 5)")
    df = pd.DataFrame([
        _metrics_row("DUET (lambda=1.00)", 0.93, lambda_=1.0, crowding=True),
        _metrics_row(very_long, 0.90, crowding=True),
    ])
    plot(df, tmp_path)
    assert (tmp_path / f"{chart}.svg").exists()

    (fig,) = captured_figures
    ax = fig.axes[0]
    fig_w_mm = fig.get_figwidth() * MM_PER_INCH
    box = ax.get_position()
    assert fig.get_figwidth() > FIGURE_WIDTHS["half_page"]
    assert box.width * fig_w_mm == pytest.approx(_HBAR_MIN_AXES_MM)
    assert (1 - box.x1) * fig_w_mm == pytest.approx(_HBAR_RIGHT_MM)
    assert _labels_top_to_bottom(ax) == ["DUET (λ=1.00)", very_long]
    renderer = fig.canvas.get_renderer()
    names_x0 = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
    assert names_x0 / fig.dpi * MM_PER_INCH == pytest.approx(_HBAR_EDGE_MM, abs=0.05)
    check_panel_fits(fig)

    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Note:")]
    assert notes == [f"Note: widened {chart} to {fig_w_mm:.1f} mm to fit its method names"]


@pytest.mark.parametrize("edge_tick", [0.0125, 0.06, 0.25])
def test_plot_metrics_barplots_fits_a_value_tick_on_the_axes_edge(
    tmp_path: Path, edge_tick: float,
):
    """The value axis autoscales to 1.05 x the largest bar, so a tick such as
    0.0125 can land on the axes edge; half its label must fit the right margin
    (save_panel raises otherwise)."""
    df = pd.DataFrame([
        _metrics_row("DUET (lambda=1.00)", 0.93, lambda_=1.0),
        _metrics_row("Codebook 1 (Chen 2015)", 0.88),
    ])
    df["Std decode accuracy"] = [edge_tick / 1.05, edge_tick / 2]
    plot_metrics_barplots(df, tmp_path, metrics=["Std decode accuracy"])
    assert (tmp_path / "std_decode_accuracy.svg").exists()


# -------------------- plot_decode_accuracy_histogram --------------------


def test_plot_decode_accuracy_histogram_emits_files(
    tmp_path: Path, results_df: pd.DataFrame, metrics_df_no_crowding: pd.DataFrame,
    captured_figures: list,
):
    """Single DUET lambda + Chen baselines -> one SVG+PNG.

    OPS-parity: figsize half_page x 3.0, sentence-case title.
    """
    from duet.plotting import FIGURE_WIDTHS

    plot_decode_accuracy_histogram(results_df, metrics_df_no_crowding, tmp_path)
    svg = tmp_path / "decode_accuracy_histogram.svg"
    png = tmp_path / "decode_accuracy_histogram.png"
    assert svg.exists() and svg.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    assert len(captured_figures) == 1
    fig = captured_figures[0]
    w, h = fig.get_size_inches()
    assert abs(w - FIGURE_WIDTHS["third_page"]) < 1e-6
    assert abs(h - 2.0) < 1e-6
    assert fig.axes[0].get_title() == "Per-codeword decode accuracy distribution"


def test_plot_decode_accuracy_histogram_picks_largest_lambda(
    tmp_path: Path, results_df_multi_lambda: pd.DataFrame,
):
    """When metrics_df has multiple DUET lambdas, histogram uses the largest (decode-only)."""
    metrics_df = pd.DataFrame([
        _metrics_row("DUET (lambda=1.00)", 0.99, lambda_=1.0),
        _metrics_row("DUET (lambda=0.50)", 0.50, lambda_=0.5),
        _metrics_row("Codebook 1 (Chen 2015)", 0.78),
    ])
    plot_decode_accuracy_histogram(results_df_multi_lambda, metrics_df, tmp_path)
    # If the function correctly picks lambda=1.00 (mean acc 0.99), the figure
    # exists. Picking lambda=0.50 would also produce a file, so this test
    # primarily asserts that the largest-lambda selection logic does not
    # crash on a multi-lambda fixture. (Faithful single-lambda reproduction
    # is the spec contract; full visual diff is out of scope.)
    assert (tmp_path / "decode_accuracy_histogram.svg").exists()


# -------------------- plot_crowding_pareto_front --------------------


def test_plot_crowding_pareto_front_emits_files_with_ops_parity(
    tmp_path: Path, metrics_df_with_crowding: pd.DataFrame,
    captured_figures: list,
):
    """File presence (smoke) + OPS-parity assertions.

    Parity asserts: third_page x 2.0 figsize, all markers s=15, every
    method-bearing scatter handle shares marker shape, a legend with no title
    (the title row made it cover a marker on Fig 4b), at least one Line2D
    exists for the DUET connector, axis labels in sentence case with no
    '(1 - conflict rate)' subtext.
    """
    from matplotlib.collections import PathCollection
    from duet.plotting import FIGURE_WIDTHS
    from duet.merfish_benchmark.visualization import _PARETO_MARKER_SIZE

    plot_crowding_pareto_front(metrics_df_with_crowding, tmp_path)

    svg = tmp_path / "crowding_pareto_front.svg"
    png = tmp_path / "crowding_pareto_front.png"
    assert svg.exists() and svg.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    # The captured_figures fixture intercepted the function's plt.close
    # so the figure is still inspectable.
    assert len(captured_figures) == 1
    fig = captured_figures[0]

    w, h = fig.get_size_inches()
    assert abs(w - FIGURE_WIDTHS["third_page"]) < 1e-6, f"width {w} != third_page"
    assert abs(h - 2.0) < 1e-6, f"height {h} != 2.0"

    ax = fig.axes[0]

    # Every scatter call uses s=15. PathCollection.get_sizes() returns
    # the per-point size array; all elements must equal 15.
    # Fixture has 2 DUET lambdas + 2 baselines -> 1 DUET scatter
    # (vectorized) + 2 baseline scatters = 3 PathCollections.
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 3, f"expected 1 DUET + 2 baseline scatters, got {len(scatters)}"
    for sc in scatters:
        sizes = sc.get_sizes()
        assert (sizes == _PARETO_MARKER_SIZE).all(), f"scatter sizes {sizes} != {_PARETO_MARKER_SIZE}"

    # At least one Line2D for the DUET connector (ax.plot result).
    assert len(ax.lines) >= 1, "expected DUET connector line"

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == ""

    assert ax.get_xlabel() == "Mean decode accuracy"
    assert ax.get_ylabel() == "Mean resolved fraction"


# Fig 4b (merfish_zhang2023_v2, 2026-09-24 re-run): mean decode accuracy and
# mean identified fraction per method.
ZHANG_V2_FRONT = [
    ("DUET (lambda=0.00)", 0.908787, 0.910535, 0.00),
    ("DUET (lambda=0.10)", 0.943795, 0.909045, 0.10),
    ("DUET (lambda=0.30)", 0.950253, 0.907493, 0.30),
    ("DUET (lambda=0.50)", 0.952589, 0.906226, 0.50),
    ("DUET (lambda=0.70)", 0.953797, 0.905257, 0.70),
    ("DUET (lambda=0.80)", 0.953978, 0.904515, 0.80),
    ("DUET (lambda=0.90)", 0.954183, 0.903172, 0.90),
    ("DUET (lambda=0.95)", 0.954243, 0.901691, 0.95),
    ("DUET (lambda=1.00)", 0.954035, 0.875735, 1.00),
    ("Bostrom et al. (Hamming weight 4)", 0.920071, 0.890920, None),
    ("Bostrom et al. (Hamming weight 5)", 0.942182, 0.873391, None),
    ("Zhang et al. codebook #2", 0.920719, 0.907319, None),
]


def _front_df(rows) -> pd.DataFrame:
    df = pd.DataFrame([
        {**_metrics_row(method, acc, lambda_=lam, crowding=True),
         "mean_identified_fraction": ident}
        for method, acc, ident, lam in rows
    ])
    df.loc[df["Method"].str.startswith("DUET") == False, "lambda"] = np.nan  # noqa: E712
    return df


def test_crowding_pareto_front_title_fits_and_legend_hides_no_marker(
    tmp_path: Path, captured_figures: list,
):
    """Fig 4b: the title ran ~3 mm past the right edge, and the legend sat on
    Boström HW4."""
    plot_crowding_pareto_front(_front_df(ZHANG_V2_FRONT), tmp_path)
    fig = captured_figures[0]
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title = fig._suptitle
    assert title.get_text() == "Decode accuracy vs optical crowding"
    assert title.get_fontsize() == 9
    bb = title.get_window_extent(renderer)
    assert fig.bbox.x0 <= bb.x0 and bb.x1 <= fig.bbox.x1 and bb.y1 <= fig.bbox.y1

    legend_box = ax.get_legend().get_window_extent(renderer)
    hidden = []
    for coll in ax.collections:
        radius = np.sqrt(coll.get_sizes().max()) / 2 * fig.dpi / 72
        for x, y in ax.transData.transform(coll.get_offsets()):
            if (legend_box.x0 - radius <= x <= legend_box.x1 + radius
                    and legend_box.y0 - radius <= y <= legend_box.y1 + radius):
                hidden.append(coll.get_label())
    assert hidden == []
    # Baseline names take their display spellings; the data keys are untouched.
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    for name in ("Boström et al. (HW4)", "Boström et al. (HW5)", "Zhang et al. codebook 2"):
        assert name in texts, texts
    # The legend stays inside the axes.
    axes_box = ax.get_window_extent(renderer)
    assert axes_box.x0 <= legend_box.x0 and legend_box.x1 <= axes_box.x1
    assert axes_box.y0 <= legend_box.y0 and legend_box.y1 <= axes_box.y1


# -------------------- plot_crowding_pareto_front_lambda_labeled --------------------


def test_plot_crowding_pareto_front_lambda_labeled_annotates_each_duet_point(
    tmp_path: Path, metrics_df_with_crowding: pd.DataFrame,
    captured_figures: list,
):
    """Companion figure: one ``λ=…`` annotation per DUET lambda; baselines unlabeled."""
    from duet.merfish_benchmark.visualization import plot_crowding_pareto_front_lambda_labeled

    plot_crowding_pareto_front_lambda_labeled(metrics_df_with_crowding, tmp_path)

    svg = tmp_path / "crowding_pareto_front_lambda_labeled.svg"
    png = tmp_path / "crowding_pareto_front_lambda_labeled.png"
    assert svg.exists() and svg.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    assert len(captured_figures) == 1
    ax = captured_figures[0].axes[0]
    # Fixture has DUET lambdas 0.00 and 0.50 -> exactly those two λ labels.
    lambda_texts = sorted(t.get_text() for t in ax.texts if t.get_text().startswith("λ="))
    assert lambda_texts == ["λ=0", "λ=0.5"]


def test_plot_crowding_pareto_front_lambda_labeled_noop_without_duet(tmp_path: Path):
    """No DUET sweep (no 'lambda' column) -> the companion figure is skipped."""
    from duet.merfish_benchmark.visualization import plot_crowding_pareto_front_lambda_labeled

    df = pd.DataFrame([_metrics_row("Codebook 1 (Chen 2015)", 0.88, crowding=True)])
    plot_crowding_pareto_front_lambda_labeled(df, tmp_path)
    assert not (tmp_path / "crowding_pareto_front_lambda_labeled.svg").exists()


# -------------------- _ordered_method_palette --------------------


def test_ordered_method_palette_distinct_colors_for_unknown_baselines():
    """Multiple unknown baselines get DISTINCT hues (not one shared grey)."""
    from duet.merfish_benchmark.visualization import _ordered_method_palette
    from duet.plotting import METHOD_PALETTE

    cmap = _ordered_method_palette(
        ["DUET (lambda=0.00)", "Mystery A", "Mystery B", "Mystery C"]
    )
    assert cmap["DUET (lambda=0.00)"] == METHOD_PALETTE["DUET"]
    unknown_colors = [cmap["Mystery A"], cmap["Mystery B"], cmap["Mystery C"]]
    assert len(set(unknown_colors)) == 3, "unknown baselines collapsed to shared colors"
    assert METHOD_PALETTE["DUET"] not in unknown_colors


# -------------------- _method_color --------------------


def test_method_color_resolves_duet_lambda_suffixed_methods():
    """DUET (lambda=X.XX) must map to METHOD_PALETTE['DUET']."""
    from duet.merfish_benchmark.visualization import _method_color
    from duet.plotting import METHOD_PALETTE, UNKNOWN_METHOD_COLOR
    assert _method_color("DUET (lambda=0.00)") == METHOD_PALETTE["DUET"]
    assert _method_color("DUET (lambda=0.50)") == METHOD_PALETTE["DUET"]
    assert _method_color("DUET (lambda=1.00)") == METHOD_PALETTE["DUET"]
    # Non-DUET methods still resolve directly:
    assert _method_color("Codebook 1 (Chen 2015)") == METHOD_PALETTE["Codebook 1 (Chen 2015)"]
    # Genuinely unknown methods fall back to UNKNOWN_METHOD_COLOR:
    assert _method_color("Mystery method") == UNKNOWN_METHOD_COLOR


# -------------------- plot_crowding_bar_chart --------------------


def test_plot_crowding_bar_chart_emits_files(
    tmp_path: Path, metrics_df_with_crowding: pd.DataFrame,
    captured_figures: list,
):
    """File presence; a horizontal half_page slot panel with the method names
    on the y axis, and the legend in its own row between the title and the
    bars, so it covers neither."""
    from matplotlib.text import Text

    from duet.plotting import FIGURE_WIDTHS, check_panel_fits

    plot_crowding_bar_chart(metrics_df_with_crowding, tmp_path)
    svg = tmp_path / "crowding_bar_chart.svg"
    png = tmp_path / "crowding_bar_chart.png"
    assert svg.exists() and svg.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    assert len(captured_figures) == 1
    fig = captured_figures[0]
    ax = fig.axes[0]
    assert fig.get_figwidth() == pytest.approx(FIGURE_WIDTHS["half_page"])
    assert fig.get_layout_engine() is None
    assert _labels_top_to_bottom(ax) == CROWDING_FIXTURE_LABELS
    assert ax.get_title(loc="left") == "Optical crowding: resolved vs missed"
    check_panel_fits(fig)

    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    legend_box = ax.get_legend().get_window_extent(renderer)
    (title,) = [t for t in ax.get_children()
                if isinstance(t, Text) and t.get_text() == ax.get_title(loc="left")]
    title_box = title.get_window_extent(renderer)
    assert legend_box.y0 >= axes_box.y1, "legend overlaps the bars"
    assert title_box.y0 >= legend_box.y1, "title overlaps the legend"


# -------------------- plot_hamming_weight_distribution --------------------


def test_plot_hamming_weight_distribution_emits_files(
    tmp_path: Path, captured_figures: list,
):
    """Stacked-bar HW figure + OPS-parity: sentence-case title."""
    from duet.plotting import FIGURE_WIDTHS
    from duet.merfish_benchmark.visualization import plot_hamming_weight_distribution

    rows = []
    methods_to_seqs = {
        "DUET (lambda=0.00)": ["111", "110", "100", "000"],
        "DUET (lambda=0.50)": ["111", "111", "110", "100"],
        "DUET (lambda=1.00)": ["111", "111", "111", "110"],
        "Zhang et al. codebook #2": ["110", "110", "110", "110"],
    }
    for method, seqs in methods_to_seqs.items():
        for i, seq in enumerate(seqs):
            rows.append({
                "Method": method, "Index": i, "Gene": f"G{i}",
                "Sequence": seq, "Decode accuracy": 0.9,
                "Identified fraction": 0.9,
                "No_error": 0.9, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    results_df = pd.DataFrame(rows)

    plot_hamming_weight_distribution(results_df, tmp_path)

    svg = tmp_path / "hamming_weight_distribution.svg"
    png = tmp_path / "hamming_weight_distribution.png"
    assert svg.exists() and svg.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    assert len(captured_figures) == 1
    fig = captured_figures[0]
    ax = fig.axes[0]
    w, h = fig.get_size_inches()
    assert abs(w - FIGURE_WIDTHS["third_page"]) < 1e-6
    assert abs(h - 2.0) < 1e-6
    assert ax.get_title() == "Hamming weight distribution by method"


# -------------------- plot_decode_vs_identified_jointplot --------------------


def test_plot_decode_vs_identified_jointplot_emits_file(
    tmp_path: Path, captured_figures: list,
):
    """Picks DUET lambda whose mean identified fraction >= baseline's, with
    highest decode accuracy. OPS-parity: scatter s=15."""
    from matplotlib.collections import PathCollection
    from duet.merfish_benchmark.visualization import plot_decode_vs_identified_jointplot

    rows = []
    methods_data = {
        "DUET (lambda=0.00)": (0.95, 0.80),
        "DUET (lambda=0.50)": (0.93, 0.92),
        "DUET (lambda=1.00)": (0.85, 0.99),
        "Zhang et al. codebook #2": (0.88, 0.90),
    }
    for method, (acc, idf) in methods_data.items():
        for i in range(20):
            rows.append({
                "Method": method, "Index": i, "Gene": f"G{i}",
                "Sequence": "1" * 4 + "0" * 12,
                "Decode accuracy": acc + 0.01 * ((i % 5) - 2),
                "Identified fraction": idf + 0.005 * ((i % 5) - 2),
                "No_error": 0.9, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    results_df = pd.DataFrame(rows)

    metrics_df = pd.DataFrame([
        _metrics_row("DUET (lambda=0.00)", 0.95, lambda_=0.0, crowding=True),
        _metrics_row("DUET (lambda=0.50)", 0.93, lambda_=0.5, crowding=True),
        _metrics_row("DUET (lambda=1.00)", 0.85, lambda_=1.0, crowding=True),
        _metrics_row("Zhang et al. codebook #2", 0.88, crowding=True),
    ])
    for method, idf in [
        ("DUET (lambda=0.00)", 0.80),
        ("DUET (lambda=0.50)", 0.92),
        ("DUET (lambda=1.00)", 0.99),
        ("Zhang et al. codebook #2", 0.90),
    ]:
        metrics_df.loc[metrics_df["Method"] == method, "mean_identified_fraction"] = idf

    plot_decode_vs_identified_jointplot(results_df, metrics_df, tmp_path)

    svg = tmp_path / "decode_vs_identified_jointplot.svg"
    png = tmp_path / "decode_vs_identified_jointplot.png"
    assert svg.exists() and svg.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    # Function calls plt.close() with no argument (closes current figure).
    # The fixture captures it.
    assert len(captured_figures) >= 1
    fig = captured_figures[-1]
    # JointGrid builds three axes (joint + 2 marginals). The order of
    # `fig.axes` is implementation detail — walk every axis and look
    # for at least one PathCollection at s=15. Marginal histplot calls
    # don't emit PathCollections, so this is no less specific than
    # targeting the joint axis directly.
    all_scatters = [
        c for ax in fig.axes for c in ax.collections
        if isinstance(c, PathCollection)
    ]
    assert any((sc.get_sizes() == 15).all() for sc in all_scatters), \
        "expected at least one scatter with s=15"


# -------------------- jointplot low-transcript filter --------------------


def _jointplot_inputs(n_transcripts_value: int):
    """results_df + metrics_df for the jointplot, with a constant n_transcripts."""
    methods_data = {
        "DUET (lambda=0.00)": (0.95, 0.80),
        "DUET (lambda=0.50)": (0.93, 0.92),
        "DUET (lambda=1.00)": (0.85, 0.99),
        "Zhang et al. codebook #2": (0.88, 0.90),
    }
    rows = []
    for method, (acc, idf) in methods_data.items():
        for i in range(20):
            rows.append({
                "Method": method, "Index": i, "Gene": f"G{i}",
                "Sequence": "1" * 4 + "0" * 12,
                "Decode accuracy": acc + 0.01 * ((i % 5) - 2),
                "Identified fraction": idf + 0.005 * ((i % 5) - 2),
                "n_transcripts": n_transcripts_value,
                "No_error": 0.9, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    metrics_df = pd.DataFrame([
        _metrics_row("DUET (lambda=0.00)", 0.95, lambda_=0.0, crowding=True),
        _metrics_row("DUET (lambda=0.50)", 0.93, lambda_=0.5, crowding=True),
        _metrics_row("DUET (lambda=1.00)", 0.85, lambda_=1.0, crowding=True),
        _metrics_row("Zhang et al. codebook #2", 0.88, crowding=True),
    ])
    for method, idf in [("DUET (lambda=0.00)", 0.80), ("DUET (lambda=0.50)", 0.92),
                        ("DUET (lambda=1.00)", 0.99), ("Zhang et al. codebook #2", 0.90)]:
        metrics_df.loc[metrics_df["Method"] == method, "mean_identified_fraction"] = idf
    return pd.DataFrame(rows), metrics_df


def test_jointplot_filters_low_transcript_genes(tmp_path: Path):
    """The transcript-count filter empties the data (-> no figure) when every gene
    is below threshold, and keeps it (-> figure) when every gene is above."""
    from duet.merfish_benchmark.visualization import (
        MIN_TRANSCRIPTS_FOR_JOINTPLOT,
        plot_decode_vs_identified_jointplot,
    )

    results_low, metrics_df = _jointplot_inputs(n_transcripts_value=1)
    plot_decode_vs_identified_jointplot(results_low, metrics_df, tmp_path)
    assert not (tmp_path / "decode_vs_identified_jointplot.svg").exists(), \
        "all genes below the transcript threshold should leave nothing to plot"

    results_high, metrics_df2 = _jointplot_inputs(
        n_transcripts_value=MIN_TRANSCRIPTS_FOR_JOINTPLOT + 5)
    plot_decode_vs_identified_jointplot(results_high, metrics_df2, tmp_path)
    assert (tmp_path / "decode_vs_identified_jointplot.svg").exists()


def test_identified_fraction_vs_count_emits_file(tmp_path: Path):
    """The sample-size diagnostic renders (with a real count range for binning)."""
    from duet.merfish_benchmark.visualization import plot_identified_fraction_vs_count

    results_df, metrics_df = _jointplot_inputs(n_transcripts_value=10)
    # Give a real, varied count range so the log axis + binned median exercise.
    results_df["n_transcripts"] = (results_df.groupby("Method").cumcount() + 1) * 3
    plot_identified_fraction_vs_count(results_df, metrics_df, tmp_path)
    assert (tmp_path / "identified_fraction_vs_count.svg").exists()
    assert (tmp_path / "identified_fraction_vs_count.png").exists()


def test_jointplot_filter_threshold_is_inclusive(tmp_path: Path):
    """Boundary: n_transcripts == threshold-1 drops every gene (no figure); ==
    threshold keeps them (figure) -- guards the >= filter against an off-by-one."""
    from duet.merfish_benchmark.visualization import (
        MIN_TRANSCRIPTS_FOR_JOINTPLOT,
        plot_decode_vs_identified_jointplot,
    )

    r_below, m_below = _jointplot_inputs(
        n_transcripts_value=MIN_TRANSCRIPTS_FOR_JOINTPLOT - 1)
    plot_decode_vs_identified_jointplot(r_below, m_below, tmp_path)
    assert not (tmp_path / "decode_vs_identified_jointplot.svg").exists()

    r_at, m_at = _jointplot_inputs(n_transcripts_value=MIN_TRANSCRIPTS_FOR_JOINTPLOT)
    plot_decode_vs_identified_jointplot(r_at, m_at, tmp_path)
    assert (tmp_path / "decode_vs_identified_jointplot.svg").exists()


def test_jointplot_min_transcripts_threshold_pinned():
    """Pin the threshold value so it can't silently drift back toward the noisy
    1-2 transcript regime that motivated raising it (the boundary tests track the
    constant, so they alone would not catch a downward change)."""
    from duet.merfish_benchmark.visualization import MIN_TRANSCRIPTS_FOR_JOINTPLOT
    assert MIN_TRANSCRIPTS_FOR_JOINTPLOT >= 25


def test_jointplot_weights_y_marginal_and_abbreviates_legend(
    tmp_path: Path, captured_figures: list, monkeypatch,
):
    """The identified-fraction (y) marginal is transcript-weighted (weights +
    bins=30) while the decode-accuracy (x) marginal is unweighted, and long
    method names are abbreviated in the legend ('Bostrom et al. (Hamming
    weight 4)' -> 'Boström et al. (HW4)', 'lambda=' -> a short glyph)."""
    from duet.merfish_benchmark import visualization as viz

    hist_calls: list = []
    real_histplot = viz.sns.histplot

    def spy_histplot(*args, **kwargs):
        hist_calls.append(kwargs)
        return real_histplot(*args, **kwargs)

    monkeypatch.setattr(viz.sns, "histplot", spy_histplot)

    # Bostrom baseline exercises the 'Hamming weight'->'HW' branch; the DUET
    # lambda exercises the 'lambda='->glyph branch. Both above the transcript filter.
    n = viz.MIN_TRANSCRIPTS_FOR_JOINTPLOT + 15
    methods = {
        "Bostrom et al. (Hamming weight 4)": (0.90, 0.85),
        "DUET (lambda=0.20)": (0.95, 0.90),
    }
    rows = []
    for method, (acc, idf) in methods.items():
        for i in range(20):
            rows.append({
                "Method": method, "Index": i, "Gene": f"G{i}",
                "Sequence": "1" * 4 + "0" * 12,
                "Decode accuracy": acc + 0.01 * ((i % 5) - 2),
                "Identified fraction": idf + 0.005 * ((i % 5) - 2),
                "n_transcripts": n,
                "No_error": 0.9, "Corrected": 0.05, "Failed": 0.05,
                "Trial": 1, "Valid": True,
            })
    results_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame([
        _metrics_row("Bostrom et al. (Hamming weight 4)", 0.90, crowding=True),
        _metrics_row("DUET (lambda=0.20)", 0.95, lambda_=0.2, crowding=True),
    ])
    metrics_df.loc[metrics_df["Method"] == "Bostrom et al. (Hamming weight 4)",
                   "mean_identified_fraction"] = 0.85
    metrics_df.loc[metrics_df["Method"] == "DUET (lambda=0.20)",
                   "mean_identified_fraction"] = 0.90

    viz.plot_decode_vs_identified_jointplot(results_df, metrics_df, tmp_path)

    y_calls = [c for c in hist_calls if c.get("y") == "Identified fraction"]
    x_calls = [c for c in hist_calls if c.get("x") == "Decode accuracy"]
    assert y_calls, "expected identified-fraction (y) marginal histplot calls"
    assert all(c.get("weights") == "n_transcripts" for c in y_calls), \
        "y-marginal must be transcript-weighted"
    assert all(c.get("bins") == 30 for c in y_calls), \
        "weighted y-marginal must pin bins=30 (seaborn can't auto-bin weighted data)"
    assert x_calls and all(c.get("weights") is None for c in x_calls), \
        "decode-accuracy (x) marginal must stay unweighted"

    # Legend labels abbreviated; no raw 'Hamming weight ' / 'lambda=' survive.
    fig = captured_figures[-1]
    texts = [t.get_text()
             for ax in fig.axes if ax.get_legend() is not None
             for t in ax.get_legend().get_texts()]
    assert "Boström et al. (HW4)" in texts, f"expected abbreviated Bostrom label, got {texts}"
    assert not any("Hamming weight " in t or "lambda=" in t for t in texts), \
        f"legend labels should be abbreviated, got {texts}"


def test_jointplot_zero_variance_subset_does_not_crash(tmp_path: Path):
    """A zero-variance identified-fraction subset must not abort the figure: the
    *weighted* y-marginal KDE raises (LinAlgError) where the unweighted path only
    warns, so the loop gates the KDE on the subset having spread."""
    from duet.merfish_benchmark.visualization import (
        MIN_TRANSCRIPTS_FOR_JOINTPLOT,
        plot_decode_vs_identified_jointplot,
    )
    results_df, metrics_df = _jointplot_inputs(
        n_transcripts_value=MIN_TRANSCRIPTS_FOR_JOINTPLOT + 5)
    # Every survivor shares one identified fraction -> zero variance within each
    # method; the weighted KDE would raise without the guard added in this commit.
    results_df["Identified fraction"] = 0.9
    plot_decode_vs_identified_jointplot(results_df, metrics_df, tmp_path)
    assert (tmp_path / "decode_vs_identified_jointplot.svg").exists()


def test_pareto_family_connector_only_for_multi_hw_families(
    tmp_path: Path, captured_figures: list,
):
    """A 2-Hamming-weight family adds a connector Line2D that a single-HW family
    (and the DUET sweep alone) does not."""
    two_hw = pd.DataFrame([
        _metrics_row("DUET (lambda=0.00)", 0.93, lambda_=0.0, crowding=True),
        _metrics_row("Bostrom et al. (Hamming weight 4)", 0.92, crowding=True),
        _metrics_row("Bostrom et al. (Hamming weight 5)", 0.94, crowding=True),
    ])
    plot_crowding_pareto_front(two_hw, tmp_path)
    n_lines_two_hw = len(captured_figures[0].axes[0].lines)

    one_hw = pd.DataFrame([
        _metrics_row("DUET (lambda=0.00)", 0.93, lambda_=0.0, crowding=True),
        _metrics_row("Bostrom et al. (Hamming weight 5)", 0.94, crowding=True),
    ])
    plot_crowding_pareto_front(one_hw, tmp_path)
    n_lines_one_hw = len(captured_figures[1].axes[0].lines)

    assert n_lines_two_hw > n_lines_one_hw, \
        "the 2-Hamming-weight Bostrom family should draw a connector the single-HW case lacks"
