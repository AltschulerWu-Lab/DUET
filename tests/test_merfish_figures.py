"""Tests for the Figure-4 load & Hamming-weight plotting helpers."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import matplotlib
matplotlib.use("Agg")  # headless; no display

import numpy as np
import pandas as pd
import pytest

from duet.merfish_benchmark.visualization import (
    codebook_from_results,
    compute_round_expression_load,
    select_best_lambda,
)


def test_compute_round_expression_load_hand_computed():
    # Gene G0 lit in rounds 0,1 (expr 2); G1 lit in rounds 2,3 (expr 3).
    cb = pd.DataFrame({"Gene": ["G0", "G1"], "Sequence": ["1100", "0011"]})
    expr_map = {"G0": 2.0, "G1": 3.0}
    beta = compute_round_expression_load(cb, expr_map)
    assert np.allclose(beta, [2.0, 2.0, 3.0, 3.0])


def test_compute_round_expression_load_missing_gene_is_zero():
    cb = pd.DataFrame({"Gene": ["G0", "Gmissing"], "Sequence": ["10", "11"]})
    beta = compute_round_expression_load(cb, {"G0": 5.0})
    # Gmissing -> 0 expression; only G0 contributes to round 0.
    assert np.allclose(beta, [5.0, 0.0])


def test_select_best_lambda_picks_max_objective_sum():
    metrics = pd.DataFrame({
        "Method": ["DUET (lambda=0.00)", "DUET (lambda=0.20)", "Bostrom et al. (Hamming weight 5)"],
        "lambda": [0.0, 0.2, np.nan],
        "Mean decode accuracy": [0.95, 0.96, 0.92],
        "mean_identified_fraction": [0.80, 0.90, 0.78],
    })
    assert select_best_lambda(metrics) == 0.2  # 0.96+0.90 > 0.95+0.80; baseline ignored


def test_select_best_lambda_tiebreak_largest_lambda():
    metrics = pd.DataFrame({
        "Method": ["DUET (lambda=0.10)", "DUET (lambda=0.90)"],
        "lambda": [0.1, 0.9],
        "Mean decode accuracy": [0.90, 0.80],
        "mean_identified_fraction": [0.80, 0.90],  # both sum to 1.70
    })
    assert select_best_lambda(metrics) == 0.9  # tie -> more weight on decoding


def test_select_best_lambda_no_duet_rows_raises():
    metrics = pd.DataFrame({
        "Method": ["Bostrom et al. (Hamming weight 5)"],
        "lambda": [np.nan],
        "Mean decode accuracy": [0.92],
        "mean_identified_fraction": [0.78],
    })
    with pytest.raises(ValueError):
        select_best_lambda(metrics)


def test_codebook_from_results_dedups_per_gene():
    results = pd.DataFrame({
        "Method": ["DUET (lambda=0.20)"] * 3 + ["Bostrom et al. (Hamming weight 5)"],
        "Gene": ["G0", "G0", "G1", "G0"],   # G0 appears twice for DUET (per-trial)
        "Sequence": ["1100", "1100", "0011", "1010"],
    })
    cb = codebook_from_results(results, "DUET (lambda=0.20)")
    assert list(cb["Gene"]) == ["G0", "G1"]
    assert list(cb["Sequence"]) == ["1100", "0011"]


def test_codebook_from_results_unknown_method_raises():
    results = pd.DataFrame({"Method": ["DUET (lambda=0.20)"], "Gene": ["G0"], "Sequence": ["1100"]})
    with pytest.raises(ValueError):
        codebook_from_results(results, "nope")


from duet.merfish_benchmark.visualization import (
    _method_color,
    plot_round_expression_load,
)


@pytest.mark.parametrize("sort", [False, True])
def test_plot_round_expression_load_returns_figure(sort):
    import matplotlib.figure
    from matplotlib.colors import same_color
    loads = {
        "Bostrom et al. (Hamming weight 4)": np.array([3.0, 1.0, 2.0, 0.0]),
        "Bostrom et al. (Hamming weight 5)": np.array([1.0, 1.0, 1.0, 1.0]),
        "DUET (lambda=0.20)": np.array([1.0, 1.1, 0.9, 1.0]),
    }
    fig = plot_round_expression_load(loads, sort=sort)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    lines = ax.lines  # insertion order: Bostrom HW4, Bostrom HW5, DUET
    assert len(lines) == 3  # one line per method
    from matplotlib.colors import to_rgb
    # DUET keeps its base palette color (no Hamming weight -> no shading).
    assert same_color(lines[2].get_color(), _method_color("DUET (lambda=0.20)"))
    # Hamming weight is now encoded by SHADE, not linestyle: every line is solid,
    # the higher HW keeps the family base color, and the lower HW is a lighter
    # white-tint of it (so the two variants are distinguishable in the legend).
    assert all(ln.get_linestyle() == "-" for ln in lines)
    base_orange = _method_color("Bostrom et al. (Hamming weight 5)")
    assert same_color(lines[1].get_color(), base_orange)                 # HW5 -> base
    assert not same_color(lines[0].get_color(), base_orange)             # HW4 -> lighter
    assert sum(to_rgb(lines[0].get_color())) > sum(to_rgb(base_orange))  # HW4 is lighter
    # Circle markers on ALL series, small enough to leave the connecting line
    # visible between points (not a solid chain of circles), and the connecting
    # line itself is thin so the markers read as the anchors.
    assert all(ln.get_marker() == "o" for ln in lines)
    assert all(ln.get_markersize() <= 2.0 for ln in lines)
    assert all(ln.get_linewidth() <= 1.1 for ln in lines)
    # Axis labels: standardized to "Total expression" / "Readout bit" (or the
    # sorted-rank variant), matching the duet.plotting label conventions.
    assert ax.get_ylabel() == "Total expression"
    assert ax.get_xlabel() == ("Round rank (sorted by load)" if sort else "Readout bit")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert all("CV=" in lbl for lbl in labels)
    # Legend labels are abbreviated to fit the third-page panel: no verbose
    # "Hamming weight" token survives (it is compacted to "HW").
    assert not any("Hamming weight" in lbl for lbl in labels)
    assert any("Boström et al. (HW4)" in lbl for lbl in labels)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_round_expression_load_y_from_zero():
    """y_from_zero pins the y-axis floor at 0; the default autoscales above 0."""
    loads = {
        "Bostrom et al. (Hamming weight 4)": np.array([300.0, 100.0, 200.0, 250.0]),
        "DUET (lambda=0.20)": np.array([210.0, 220.0, 205.0, 215.0]),
    }
    fig0 = plot_round_expression_load(loads, sort=False, y_from_zero=True)
    assert fig0.axes[0].get_ylim()[0] == 0
    # Default keeps matplotlib's autoscaled floor, which sits above 0 for these
    # all-positive loads (min 100, so the margined floor is well clear of 0).
    fig_auto = plot_round_expression_load(loads, sort=False)
    assert fig_auto.axes[0].get_ylim()[0] > 0
    import matplotlib.pyplot as plt
    plt.close(fig0)
    plt.close(fig_auto)


def test_plot_round_expression_load_legend_puts_cv_on_its_own_line():
    """The wrapped legend fits the slot panel; one line per label ran 3.5 mm over."""
    loads = {
        "Bostrom et al. (Hamming weight 4)": np.array([3.0, 1.0, 2.0, 0.0]),
        "DUET (lambda=0.20)": np.array([1.0, 1.1, 0.9, 1.0]),
    }
    fig = plot_round_expression_load(loads, sort=False)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[1] == "DUET (λ=0.20)\n(CV=0.071)"
    assert all("\n(CV=" in lbl for lbl in labels)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _assert_third_page_slot_panel(fig, height_in):
    from duet.plotting import DEFAULT_MARGINS_MM, FIGURE_WIDTHS, MM_PER_INCH

    assert fig.get_layout_engine() is None
    w, h = fig.get_size_inches()
    assert w == pytest.approx(FIGURE_WIDTHS["third_page"])
    assert h == pytest.approx(height_in)
    left, right, bottom, top = DEFAULT_MARGINS_MM
    box = fig.axes[0].get_position()
    assert box.x0 * w * MM_PER_INCH == pytest.approx(left)
    assert box.y0 * h * MM_PER_INCH == pytest.approx(bottom)
    assert (1 - box.x1) * w * MM_PER_INCH == pytest.approx(right)
    assert (1 - box.y1) * h * MM_PER_INCH == pytest.approx(top)


def test_plot_round_expression_load_default_figure_is_a_slot_panel():
    loads = {"DUET (lambda=0.20)": np.array([1.0, 1.1, 0.9, 1.0])}
    fig = plot_round_expression_load(loads, sort=False)
    _assert_third_page_slot_panel(fig, 2.2)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_round_expression_load_draws_on_a_given_axes():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    out = plot_round_expression_load(
        {"DUET (lambda=0.20)": np.array([1.0, 1.1, 0.9, 1.0])}, sort=True, ax=ax
    )
    assert out is fig and len(ax.lines) == 1
    plt.close(fig)


from duet.merfish_benchmark.visualization import (
    plot_expression_by_hamming_weight,
    _hamming_weight_colors,
)
from duet.plotting import OKABE_ITO, sequential_shades


@pytest.mark.parametrize("kind", ["violin", "box"])
def test_plot_expression_by_hamming_weight_returns_figure(kind):
    import matplotlib.figure
    import matplotlib.pyplot as plt
    # 4 HW4 codewords + 4 HW5 codewords (need >=2 per group for the violin KDE).
    cb = pd.DataFrame({
        "Gene": [f"G{i}" for i in range(8)],
        "Sequence": ["11110000", "11110000", "11110000", "11110000",
                     "11111000", "11111000", "11111000", "11111000"],
    })
    expr_map = {f"G{i}": float(10 * (i + 1)) for i in range(8)}
    fig = plot_expression_by_hamming_weight(cb, expr_map, kind=kind)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    # one x tick per HW present (HW4, HW5)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["HW4", "HW5"]
    plt.close(fig)


def test_plot_expression_by_hamming_weight_default_figure_is_a_slot_panel():
    import matplotlib.pyplot as plt
    cb = pd.DataFrame({"Gene": ["G0", "G1", "G2", "G3"],
                       "Sequence": ["1100", "1010", "1110", "0111"]})
    fig = plot_expression_by_hamming_weight(cb, {f"G{i}": 10.0 * (i + 1) for i in range(4)})
    _assert_third_page_slot_panel(fig, 2.2)
    plt.close(fig)


def test_plot_expression_by_hamming_weight_bad_kind_raises():
    cb = pd.DataFrame({"Gene": ["G0", "G1"], "Sequence": ["1100", "0110"]})
    with pytest.raises(ValueError):
        plot_expression_by_hamming_weight(cb, {"G0": 1.0, "G1": 2.0}, kind="nope")


def test_hamming_weight_colors_purple_sequential_ramp():
    # HW colors are now a purple sequential ramp (light HW4 -> dark HW5), not
    # sky_blue/yellow -- keeps HW distinct from DUET-blue / baseline hues.
    df = pd.DataFrame({"Sequence": ["1111", "1111", "11111", "11111"]})  # HW4, HW5
    colors = _hamming_weight_colors(df)
    expected = sequential_shades(OKABE_ITO["reddish_purple"], 2)
    assert colors[4] == expected[0]
    assert colors[5] == expected[1]
