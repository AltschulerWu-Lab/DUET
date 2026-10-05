"""Layout of the rounds x epsilon sweep heatmaps (Fig 3c).

The heatmaps are third-page slot panels with fixed millimetre margins, so the
two Fig 3c panels drop into the right-hand third of Fig 3 unscaled and line up
when stacked. Titles wrap to two lines, cell labels carry no percent sign, and
the unit sits on the colour bar.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.plotting import FIGURE_WIDTHS, MM_PER_INCH, PANEL_GAP_MM

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "experiments" / "ops_crispri_rounds_error_sweep" / "visualize_sweep.py"

# Fig 3c cells from the 2026-09-26 re-run (DUET 97.5% arm minus maximum
# activity, in %): epsilon 0.10, 0.05, 0.03 (rows) x rounds 8-13 (columns).
FIG3C_TOP = [[45, 29, 19, 13, 9.0, 7.4], [65, 35, 19, 11, 6.7, 4.7], [75, 36, 18, 8.7, 4.9, 3.3]]
FIG3C_BOTTOM = [[-63, -62, -58, -58, -54, -60], [-78, -74, -75, -77, -75, -80],
                [-85, -83, -85, -87, -86, -86]]


@pytest.fixture(scope="module")
def sweep():
    """experiments/ops_crispri_rounds_error_sweep/visualize_sweep.py as a module."""
    spec = importlib.util.spec_from_file_location("rounds_error_sweep_heatmaps", SWEEP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def captured_figures(monkeypatch):
    """Keep the figures plot_delta_heatmap closes, to inspect them."""
    import matplotlib.pyplot as plt

    figs: list = []
    real_close = plt.close
    monkeypatch.setattr("matplotlib.pyplot.close", lambda fig=None: figs.append(fig))
    yield figs
    for f in figs:
        real_close(f)


def _summary(sweep, metric: str, cells_pct, baseline=None) -> pd.DataFrame:
    """sweep_summary rows for one (baseline, metric) heatmap from a % grid."""
    spec = next(s for s in sweep.SWEEP_METRICS if s.name == metric)
    rows = []
    for eps, row in zip((0.10, 0.05, 0.03), cells_pct):
        for rounds, value in zip(sweep.SEQ_ROUNDS, row):
            rows.append({
                "stem": f"sr{rounds:02d}_eps{eps:.2f}", "rounds": rounds, "epsilon": eps,
                "baseline": baseline or sweep.MAX_ACTIVITY_METHOD, "metric": metric,
                "kind": spec.kind, "direction": spec.direction,
                "mean": value / 100, "sd": 0.01, "se": 0.005, "n": 5,
                "lambdas": "DUET (lambda=0.12)",
            })
    return pd.DataFrame(rows)


def _render(sweep, captured_figures, tmp_path, metric, cells_pct):
    out = tmp_path / f"{len(captured_figures)}.svg"
    assert sweep.plot_delta_heatmap(_summary(sweep, metric, cells_pct),
                                    sweep.MAX_ACTIVITY_METHOD, metric, out)
    fig = captured_figures[-1]
    fig.canvas.draw()
    return fig, out


def _box_mm(fig, ax):
    """An axes' box (x0, y0, x1, y1) in mm from the figure's lower-left corner."""
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return tuple(round(v / fig.dpi * MM_PER_INCH, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1))


def _fig3c_metrics(sweep):
    return {position: metric for position, _, metric in sweep.FIG3C_PANELS}


def test_fig3c_titles_wrap_to_two_lines(sweep):
    metrics = _fig3c_metrics(sweep)

    def title(metric):
        spec = next(s for s in sweep.SWEEP_METRICS if s.name == metric)
        return sweep.panel_title(spec.name, spec.kind, spec.direction)

    assert title(metrics["top"]) == "Absolute gain in mean decode\naccuracy (≤ 10th percentile)"
    assert title(metrics["bottom"]) == "Relative reduction in standard\ndeviation of decode accuracy"


@pytest.mark.parametrize("value, label", [
    (74.86, "+75"), (45.2, "+45"), (9.0, "+9.0"), (7.44, "+7.4"), (3.26, "+3.3"),
    (9.96, "+10"), (0.38, "+0.4"), (-87.49, "-87"), (-4.71, "-4.7"), (np.nan, ""),
])
def test_cell_label_has_at_most_one_decimal_and_no_percent_sign(sweep, value, label):
    assert sweep.cell_label(value) == label


def test_colorbar_unit(sweep):
    assert sweep.colorbar_unit("absolute") == "pp"
    assert sweep.colorbar_unit("relative") == "%"
    assert sweep.colorbar_unit("relative_over_one") == "%"


def test_fig3c_panels_are_third_page_and_line_up(sweep, captured_figures, tmp_path):
    metrics = _fig3c_metrics(sweep)
    top, top_svg = _render(sweep, captured_figures, tmp_path, metrics["top"], FIG3C_TOP)
    bottom, bottom_svg = _render(sweep, captured_figures, tmp_path, metrics["bottom"], FIG3C_BOTTOM)

    width_mm = FIGURE_WIDTHS["third_page"] * MM_PER_INCH
    for fig, svg in ((top, top_svg), (bottom, bottom_svg)):
        w_in, h_in = fig.get_size_inches()
        assert w_in * MM_PER_INCH == pytest.approx(width_mm)
        assert h_in * MM_PER_INCH == pytest.approx(sweep.HEATMAP_HEIGHT_MM)
        head = svg.read_text()[:2000]  # saved at exactly that size
        assert float(re.search(r'width="([\d.]+)pt"', head).group(1)) == pytest.approx(w_in * 72, abs=0.01)
        assert float(re.search(r'height="([\d.]+)pt"', head).group(1)) == pytest.approx(h_in * 72, abs=0.01)
    # Stacked with the panel gap, the pair spans Fig 3b right.
    assert 2 * sweep.HEATMAP_HEIGHT_MM + PANEL_GAP_MM == pytest.approx(
        sweep.STACK_HEIGHT_OVERLAY * MM_PER_INCH
    )
    # Cells and colour bars sit at the same place in both panels.
    assert _box_mm(top, top.axes[0]) == _box_mm(bottom, bottom.axes[0])
    assert _box_mm(top, top.axes[1]) == _box_mm(bottom, bottom.axes[1])


def test_fig3c_text_sizes_and_units(sweep, captured_figures, tmp_path):
    metrics = _fig3c_metrics(sweep)
    for metric, cells, unit in ((metrics["top"], FIG3C_TOP, "pp"),
                                (metrics["bottom"], FIG3C_BOTTOM, "%")):
        fig, _ = _render(sweep, captured_figures, tmp_path, metric, cells)
        ax, cax = fig.axes
        renderer = fig.canvas.get_renderer()

        assert ax.title.get_fontsize() == 9
        assert ax.title.get_text().count("\n") == 1
        assert cax.title.get_text() == unit

        labels = [t for t in ax.texts if t.get_text()]
        assert len(labels) == 18
        assert all("%" not in t.get_text() for t in labels)
        assert {t.get_fontsize() for t in labels} == {7}


def _arial_metric_font():
    """Arial, or Liberation Sans (the same glyph widths), if installed; else None."""
    from matplotlib import font_manager

    for name in ("Arial", "Liberation Sans"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            return name
        except ValueError:
            continue
    return None


def test_fig3c_cell_labels_fit_their_cells_in_arial(sweep, captured_figures, tmp_path):
    # The paper is set in Arial, where every label clears its ~5.8 mm cell by
    # at least 0.65 mm. The fallback DejaVu Sans is wider ('+7.4' runs ~0.3 mm
    # over), so the check needs Arial's glyph widths.
    import matplotlib.pyplot as plt

    font = _arial_metric_font()
    if font is None:
        pytest.skip("needs Arial or Liberation Sans: label widths are font-specific")
    metrics = _fig3c_metrics(sweep)
    with plt.rc_context({"font.sans-serif": [font]}):
        for metric, cells in ((metrics["top"], FIG3C_TOP), (metrics["bottom"], FIG3C_BOTTOM)):
            fig, _ = _render(sweep, captured_figures, tmp_path, metric, cells)
            ax = fig.axes[0]
            renderer = fig.canvas.get_renderer()
            cell_px = ax.get_window_extent(renderer).width / 6
            widths = [t.get_window_extent(renderer).width for t in ax.texts if t.get_text()]
            assert max(widths) < cell_px


def test_a_longer_title_makes_the_panel_taller_not_the_cells_shorter(
    sweep, captured_figures, tmp_path
):
    two_lines, _ = _render(sweep, captured_figures, tmp_path,
                           "Standard deviation decode accuracy", FIG3C_BOTTOM)
    three_lines, _ = _render(sweep, captured_figures, tmp_path,
                             "Median absolute deviation decode accuracy", FIG3C_BOTTOM)
    assert three_lines.axes[0].title.get_text().count("\n") == 2

    def cells_height(fig):
        x0, y0, x1, y1 = _box_mm(fig, fig.axes[0])
        return y1 - y0

    assert cells_height(three_lines) == pytest.approx(cells_height(two_lines), abs=0.01)
    extra_mm = (three_lines.get_size_inches()[1] - two_lines.get_size_inches()[1]) * MM_PER_INCH
    assert extra_mm == pytest.approx(9 * 1.2 / 72 * MM_PER_INCH)  # one 9 pt line
