"""Tests for visualize_comparison.py: output tiers, the horizontal hypervolume
bar chart, and the single-trial / "N/A" legend of the aggregated Pareto panel.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.text import Text

from duet.plotting import FIGURE_WIDTHS, MM_PER_INCH, check_panel_fits, style_context
from duet.plotting.legend import SINGLE_TRIAL_MARKERSIZE

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

import visualize_comparison as vc  # noqa: E402

X_COL = "Mean decode accuracy"
Y_COL = "Mean activity score"
RATIO_COL = "95th over 5th percentile decode accuracy"

# Config order, as in the S4b config: the longest label first.
LABELS = [
    "DUET (Position-varying asymmetric error)",
    "DUET (Symmetric error)",
    "Feldman et al.",
]
METRIC_NAMES = [vc.sanitize_filename(x) for x, _ in vc.METRICS]


def _agg_frame(n_trials: int, labels=LABELS) -> pd.DataFrame:
    """aggregated_metrics.csv rows: 3 DUET lambdas per DUET label, Feldman ED1
    plus the Maximum-activity anchor under the Feldman label."""
    rng = np.random.default_rng(0)
    rows = []
    for li, label in enumerate(labels):
        if label.startswith("DUET"):
            methods = [(f"DUET (lambda={lam:.2f})", 0.80 + 0.1 * lam - 0.02 * li, 1.0 - 0.1 * lam)
                       for lam in (0.2, 0.5, 0.9)]
        else:
            methods = [("Feldman et al. (ED=1)", 0.70, 0.95), ("Maximum activity", 0.60, 1.0)]
        for method, x, y in methods:
            for trial in range(1, n_trials + 1):
                row = {"Label": label, "Method": method, "Trial": trial}
                for x_col, _ in vc.METRICS:
                    row[x_col] = x + rng.normal(0, 0.005)
                row[Y_COL] = y + rng.normal(0, 0.005)
                rows.append(row)
    return pd.DataFrame(rows)


def _palette(labels=LABELS):
    return {label: f"C{i}" for i, label in enumerate(labels)}


@pytest.fixture
def captured_figures(monkeypatch):
    """Keep the figures the plot functions close, to inspect them."""
    import matplotlib.pyplot as plt

    figs: list = []
    real_close = plt.close
    monkeypatch.setattr("matplotlib.pyplot.close", lambda fig=None: figs.append(fig))
    yield figs
    for f in figs:
        real_close(f)


# =============================================================================
# Tiers (main)
# =============================================================================


def _write_run(tmp_path: Path, n_trials: int, paper_panels=None) -> tuple[Path, Path]:
    outdir = tmp_path / "out"
    outdir.mkdir()
    _agg_frame(n_trials).to_csv(outdir / "aggregated_metrics.csv", index=False)
    config = {
        "outdir": str(outdir),
        "cache_dir": None,
        "seed": 42,
        "reference_dir": str(tmp_path / "ref"),
        "evaluator": {
            "noise_channel": {"type": "symmetric"},
            "decoding_metric": {"type": "hamming"},
            "decoding_rule": {"type": "unique_minimum"},
        },
        "methods": [
            {"path": str(tmp_path / "results.csv"), "regex": "X.*", "label": label}
            for label in LABELS
        ],
    }
    if paper_panels is not None:
        config["visualization"] = {"paper_panels": paper_panels}
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, outdir


def _run_main(monkeypatch, config_path: Path, *extra: str) -> None:
    argv = ["visualize_comparison.py", "--config", str(config_path), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with style_context():  # main() applies the stylesheet; restore rcParams after
        vc.main()


def _svgs(outdir: Path) -> set[str]:
    return {p.relative_to(outdir).as_posix() for p in outdir.rglob("*.svg")}


AGGREGATED = {"hypervolume/hypervolume_barplot.svg"} | {
    f"pareto_fronts/{m}_aggregated.svg" for m in METRIC_NAMES
}

AGGREGATED_PLOTTERS = ("plot_hypervolume_barplot", "plot_pareto_aggregated")
PER_TRIAL_PLOTTERS = ("plot_hypervolume_single_trial", "plot_pareto_single_trial")


def _fake_plotters(monkeypatch, names) -> None:
    """Swap main()'s plotters ``names`` for fakes that only create the file
    they are given, so a tiers test checks which files are written without
    rendering them. Like savefig, a fake makes no parent directory; the call
    must still bind to the real signature."""
    for name in names:
        signature = inspect.signature(getattr(vc, name))

        def fake(*args, _signature=signature, **kwargs):
            Path(_signature.bind(*args, **kwargs).arguments["output_path"]).touch()

        monkeypatch.setattr(vc, name, fake)


@pytest.fixture
def fake_plotters(monkeypatch):
    _fake_plotters(monkeypatch, AGGREGATED_PLOTTERS + PER_TRIAL_PLOTTERS)


# The one unstubbed end-to-end render of main().
def test_default_run_writes_only_the_aggregated_figures(tmp_path, monkeypatch, capsys):
    config_path, outdir = _write_run(tmp_path, n_trials=2)
    _run_main(monkeypatch, config_path)

    assert _svgs(outdir) == AGGREGATED
    out = capsys.readouterr().out
    # 2 hypervolume + 7 metrics x 2 per-trial Pareto panels.
    assert "Skipped 16 per-trial debug plots" in out
    assert "--debug-plots" in out


def test_debug_plots_adds_the_per_trial_figures(tmp_path, monkeypatch, capsys, fake_plotters):
    config_path, outdir = _write_run(tmp_path, n_trials=2)
    _run_main(monkeypatch, config_path, "--debug-plots")

    per_trial = {f"hypervolume/hypervolume_trial_{k}.svg" for k in (1, 2)} | {
        f"pareto_fronts/{m}_trial_{k}.svg" for m in METRIC_NAMES for k in (1, 2)
    }
    assert _svgs(outdir) == AGGREGATED | per_trial
    assert "Skipped" not in capsys.readouterr().out


def test_paper_panel_listed_in_config_is_written_by_default(tmp_path, monkeypatch, capsys):
    # Only the aggregated plotters are faked (the default run renders them), so
    # the two listed per-trial panels are the per-trial plotters' real render.
    _fake_plotters(monkeypatch, AGGREGATED_PLOTTERS)
    config_path, outdir = _write_run(
        tmp_path, n_trials=2,
        paper_panels=[
            "pareto_fronts/Mean_decode_accuracy_trial_2",
            "hypervolume/hypervolume_trial_1",
            "pareto_fronts/typo",
        ],
    )
    _run_main(monkeypatch, config_path)

    assert _svgs(outdir) == AGGREGATED | {
        "pareto_fronts/Mean_decode_accuracy_trial_2.svg",
        "hypervolume/hypervolume_trial_1.svg",
    }
    out = capsys.readouterr().out
    assert "Skipped 14 per-trial debug plots" in out
    # Only the typo is reported as missing.
    assert out.count("WARNING: paper panel") == 1
    assert "'pareto_fronts/typo'" in out


# =============================================================================
# Hypervolume bar chart
# =============================================================================


def test_hypervolume_barplot_is_horizontal_in_label_order_and_fits(tmp_path, captured_figures):
    with style_context():
        vc.plot_hypervolume_barplot(
            _agg_frame(3), _palette(), LABELS, tmp_path / "hv.svg"
        )
        (fig,) = captured_figures
        check_panel_fits(fig)  # nothing runs past the figure edge

        ax = fig.axes[0]
        ticks = ax.get_yticklabels()
        # Read top to bottom: highest on the page first.
        renderer = fig.canvas.get_renderer()
        top_to_bottom = sorted(ticks, key=lambda t: -t.get_window_extent(renderer).y0)
        assert [t.get_text() for t in top_to_bottom] == LABELS
        assert all(t.get_rotation() == 0 for t in ticks)
        assert ax.get_xlabel() == "Normalized Hypervolume"
        # Every label sits left of the axes, clear of the figure edge.
        assert min(t.get_window_extent(renderer).x0 for t in ticks) >= 0
        assert fig.get_figwidth() == pytest.approx(FIGURE_WIDTHS["half_page"])
    assert (tmp_path / "hv.svg").exists()


def test_hypervolume_barplot_height_scales_with_label_count(tmp_path, captured_figures):
    labels = LABELS + ["DUET (Asymmetric error)", "DUET (Position-varying error)"]
    with style_context():
        vc.plot_hypervolume_barplot(_agg_frame(2), _palette(), LABELS, tmp_path / "a.svg")
        vc.plot_hypervolume_barplot(
            _agg_frame(2, labels), _palette(labels), labels, tmp_path / "b.svg"
        )
    small, large = (f.get_figheight() * MM_PER_INCH for f in captured_figures)
    assert large - small == pytest.approx(2 * vc.HV_BAR_ROW_MM)


@pytest.mark.parametrize(
    "long_label",
    [
        # Leaves the bars some width, under the minimum.
        "DUET (Position-varying asymmetric error, first 10 rounds)",
        # Wider than the whole half_page panel.
        "DUET (" + ", ".join(["Position-varying asymmetric error"] * 3) + ")",
    ],
    ids=["squeezes_bars", "wider_than_panel"],
)
def test_hypervolume_barplot_widens_for_long_labels(
    tmp_path, captured_figures, capsys, long_label
):
    labels = [long_label, *LABELS[1:]]
    with style_context():
        vc.plot_hypervolume_barplot(
            _agg_frame(2, labels), _palette(labels), labels, tmp_path / "hv.svg"
        )
        (fig,) = captured_figures
        check_panel_fits(fig)

        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        width_mm = fig.get_figwidth() * MM_PER_INCH
        pos = ax.get_position()
        assert width_mm > FIGURE_WIDTHS["half_page"] * MM_PER_INCH
        # Wider just enough: the bars get the minimum width, the right margin
        # is kept and the labels start the 1 mm pad from the left edge.
        assert pos.width * width_mm == pytest.approx(vc.HV_BAR_MIN_WIDTH_MM)
        assert (1 - pos.x1) * width_mm == pytest.approx(vc.HV_BAR_MARGINS_MM[0])
        px_per_mm = fig.dpi / MM_PER_INCH
        assert ax.get_tightbbox(renderer).x0 / px_per_mm == pytest.approx(1.0, abs=0.05)
        # The labels are kept whole.
        assert sorted(t.get_text() for t in ax.get_yticklabels()) == sorted(labels)
    assert (tmp_path / "hv.svg").exists()
    assert capsys.readouterr().out.count("Note: widened hv.svg") == 1


@pytest.mark.parametrize(
    "label_order",
    [["DUET (typo)", "Feldman (typo)"], [LABELS[0], "Feldman (typo)"]],
    ids=["no_label_in_data", "one_label_in_data"],
)
def test_hypervolume_barplot_is_skipped_without_two_labels_to_compare(
    tmp_path, capsys, label_order
):
    # No label in the data used to crash slot_figure (a zero-height axes). One
    # label is not enough either: the reference point comes from all of a
    # trial's fronts, so a trial with fewer than two gives no hypervolume.
    with style_context():
        vc.plot_hypervolume_barplot(
            _agg_frame(2), _palette(label_order), label_order, tmp_path / "hv.svg"
        )
    assert not (tmp_path / "hv.svg").exists()
    assert capsys.readouterr().out.count("Note: skipped hv.svg") == 1


# =============================================================================
# Aggregated Pareto panel legend
# =============================================================================


def _pareto(tmp_path, agg_df, captured_figures, x_col=X_COL, labels=LABELS):
    with style_context():
        vc.plot_pareto_aggregated(
            agg_df, x_col, Y_COL, _palette(labels), labels, tmp_path / "p.svg"
        )
    fig = captured_figures[-1]
    ax = fig.axes[0]
    legend = ax.get_legend()
    fig.canvas.draw()
    return ax, legend


def _legend_markers(legend) -> list[str]:
    markers = [line.get_marker() for line in legend.findobj(Line2D)]
    return [m for m in markers if m not in (None, "None", "")]


def _legend_texts(legend) -> list[str]:
    return [t.get_text() for t in legend.get_texts()]


def _na_rows(legend) -> list[str]:
    """Legend labels whose row shows "N/A" in place of a marker."""
    renderer = legend.figure.canvas.get_renderer()

    def y_mid(text):
        bb = text.get_window_extent(renderer)
        return (bb.y0 + bb.y1) / 2

    labels = legend.get_texts()
    return [
        min(labels, key=lambda lab: abs(y_mid(lab) - y_mid(na))).get_text()
        for na in legend.findobj(Text) if na.get_text() == "N/A"
    ]


def test_single_trial_pareto_uses_big_circles_and_square_legend(tmp_path, captured_figures):
    ax, legend = _pareto(tmp_path, _agg_frame(1), captured_figures)

    assert not any(isinstance(c, ErrorbarContainer) for c in ax.containers)
    points = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(points) == len(LABELS)
    assert all(p.get_markersize() == SINGLE_TRIAL_MARKERSIZE for p in points)
    assert _legend_markers(legend) == ["s"] * len(LABELS)
    assert not legend.findobj(LineCollection)
    assert _legend_texts(legend) == [textwrap.fill(label, vc.LEGEND_WRAP_WIDTH) for label in LABELS]
    # The title is kept (user decision), single trial or not.
    assert ax.figure.get_suptitle().startswith("Aggregated Across Trials (± SE)")


def test_multi_trial_pareto_keeps_errorbar_handles(tmp_path, captured_figures):
    ax, legend = _pareto(tmp_path, _agg_frame(3), captured_figures)

    assert sum(isinstance(c, ErrorbarContainer) for c in ax.containers) == len(LABELS)
    assert "s" not in _legend_markers(legend)
    assert legend.findobj(LineCollection)  # the s.e.m. bars in each handle
    assert _na_rows(legend) == []


@pytest.mark.parametrize("n_trials", [1, 3])
def test_label_absent_from_data_gets_na(tmp_path, captured_figures, n_trials):
    agg = _agg_frame(n_trials)
    agg = agg[agg["Label"] != "DUET (Symmetric error)"]
    ax, legend = _pareto(tmp_path, agg, captured_figures)

    # The row is kept in its usual place, with "N/A" for the marker.
    assert _legend_texts(legend) == [textwrap.fill(label, vc.LEGEND_WRAP_WIDTH) for label in LABELS]
    assert _na_rows(legend) == ["DUET (Symmetric error)"]
    if n_trials == 1:
        assert _legend_markers(legend) == ["s", "s"]
    else:
        assert sum(isinstance(c, ErrorbarContainer) for c in ax.containers) == 2


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
@pytest.mark.parametrize("n_trials", [1, 3])
def test_label_with_no_finite_point_gets_na(tmp_path, captured_figures, n_trials):
    # The 95th/5th ratio is inf when the 5th percentile is 0 (Feldman, S4c).
    agg = _agg_frame(n_trials)
    agg.loc[agg["Label"] == "Feldman et al.", RATIO_COL] = np.inf
    ax, legend = _pareto(tmp_path, agg, captured_figures, x_col=RATIO_COL)

    assert _na_rows(legend) == ["Feldman et al."]
    if n_trials == 1:
        assert _legend_markers(legend) == ["s", "s"]
    else:
        assert sum(isinstance(c, ErrorbarContainer) for c in ax.containers) == 2


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
@pytest.mark.parametrize("x_col", [x for x, _ in vc.METRICS])
@pytest.mark.parametrize("n_trials", [1, 3])
def test_pareto_title_fits_the_third_page_panel(tmp_path, captured_figures, x_col, n_trials):
    # With "Mean decode accuracy vs Mean activity score" appended, the S4b / S4c
    # title was ~112 mm on one line, about twice the 57.33 mm slot.
    ax, _ = _pareto(tmp_path, _agg_frame(n_trials), captured_figures, x_col=x_col)
    fig = ax.figure

    title = fig._suptitle
    assert title.get_text() == "Aggregated Across Trials (± SE)"  # one line
    assert title.get_fontsize() == 9  # shortened, not shrunk
    bb = title.get_window_extent(fig.canvas.get_renderer())
    assert bb.x0 >= fig.bbox.x0 and bb.x1 <= fig.bbox.x1
    assert bb.y1 <= fig.bbox.y1
    # The title sits above the axes, not over them.
    assert bb.y0 >= ax.get_window_extent(fig.canvas.get_renderer()).y1
