"""Tests for the guide-level distribution stack (Figure 3 density panels).

One 2x1 panel per baseline-vs-DUET comparison: activity density on top,
decode-accuracy density below, each with a 5th-95th percentile strip. Guides
with decode accuracy exactly 0 (duplicate codewords) are a point mass that the
kernel density cannot represent; they are split off, the curve is fitted on
the non-zero guides and scaled by their fraction, and the zeros are drawn as a
histogram bin of one bandwidth in a broken-axis slot, only on panels where
such guides exist.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

from duet.plotting import FIGURE_WIDTHS, METHOD_PALETTE, OVERLAY_VARIANT_PALETTE  # noqa: E402
from visualize_benchmark import (  # noqa: E402
    AxisRanges,
    OverlaySpec,
    _draw_zero_bin,
    continuous_density,
    display_method_label,
    overlay_colors,
    plot_distribution_stack,
    shared_axis_ranges,
    zero_bin_bar,
)

DUET = "DUET (lambda=0.88)"
MAX_ACT = "Maximum activity"
FELDMAN1 = "Feldman et al. (ED=1)"
SIVANANDAN1 = "Sivanandan et al. (ED=1)"
SIVANANDAN2 = "Sivanandan et al. (ED=2)"


def _guides(method: str, accuracy, activity, trial: int = 1) -> pd.DataFrame:
    accuracy = np.asarray(accuracy, dtype=float)
    activity = np.asarray(activity, dtype=float)
    assert len(accuracy) == len(activity)
    return pd.DataFrame(
        {
            "Method": method,
            "Trial": trial,
            "Decode accuracy": accuracy,
            "Activity score": activity,
        }
    )


def _frame(baseline_zeros: int = 0, trial: int = 1) -> pd.DataFrame:
    """Two methods, 400 guides each, with an optional pile of zeros in the baseline."""
    rng = np.random.default_rng(0)
    n = 400
    duet_acc = np.clip(rng.normal(0.79, 0.04, n), 0.55, 0.95)
    base_acc = np.clip(rng.normal(0.74, 0.07, n), 0.50, 0.95)
    base_acc[:baseline_zeros] = 0.0
    act = np.clip(rng.normal(0.9, 0.2, n), 0.3, 1.7)
    return pd.concat(
        [
            _guides(DUET, duet_acc, act, trial),
            _guides(MAX_ACT, base_acc, act, trial),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------- labels


def test_alpha_spelling_renders_as_lambda_symbol():
    # The 04-24-2026 results spell the weight "alpha"; the figure shows λ.
    assert display_method_label("DUET (alpha=0.88)") == "DUET (λ=0.88)"
    assert display_method_label("DUET (lambda=0.88)") == "DUET (λ=0.88)"


def test_sivanandan_reads_hamming_distance_and_feldman_keeps_edit_distance():
    # Sivanandan et al. enforce a Hamming-distance floor; Feldman et al. an
    # edit distance. results.csv says ED for both.
    assert display_method_label(SIVANANDAN1) == "Sivanandan et al. (HD = 1)"
    assert display_method_label(SIVANANDAN2) == "Sivanandan et al. (HD = 2)"
    assert display_method_label(FELDMAN1) == "Feldman et al. (ED=1)"
    assert display_method_label(MAX_ACT) == MAX_ACT


def test_overlay_legend_shows_sivanandan_as_hd(tmp_path):
    # Fig 3b right: the all-baselines overlay legend.
    df = _overlay_frame()
    colors = overlay_colors(_overlay_spec())
    out = tmp_path / "overlay.svg"
    plot_distribution_stack(
        df, colors, trial=1, output_path=out,
        ranges=shared_axis_ranges(df, 1, list(colors)), emphasize=DUET,
    )
    svg = out.read_text()
    assert "Sivanandan et al. (HD = 1)" in svg
    assert "Sivanandan et al. (HD = 2)" in svg
    assert "Feldman et al. (ED=1)" in svg
    assert "Sivanandan et al. (ED=" not in svg


# ---------------------------------------------------------------- axis ranges


def test_accuracy_lower_bound_ignores_exact_zeros():
    df = _frame(baseline_zeros=20)
    ranges = shared_axis_ranges(df, trial=1, methods=[DUET, MAX_ACT])
    nonzero_min = df.loc[df["Decode accuracy"] > 0, "Decode accuracy"].min()
    assert isinstance(ranges, AxisRanges)
    assert ranges.accuracy[0] > 0.0
    assert ranges.accuracy[0] <= nonzero_min
    assert ranges.accuracy[0] == pytest.approx(np.floor(nonzero_min * 20) / 20)


def test_accuracy_upper_bound_is_capped_at_one():
    df = _frame()
    ranges = shared_axis_ranges(df, trial=1, methods=[DUET, MAX_ACT])
    assert ranges.accuracy[1] <= 1.0
    assert ranges.accuracy[1] >= df["Decode accuracy"].max()


def test_activity_upper_bound_is_the_pooled_99p5_percentile_rounded_up():
    df = _frame()
    ranges = shared_axis_ranges(df, trial=1, methods=[DUET, MAX_ACT])
    q = df["Activity score"].quantile(0.995)
    assert ranges.activity[1] == pytest.approx(np.ceil(q * 10) / 10)
    assert ranges.activity[0] <= df["Activity score"].min()


def test_axis_ranges_are_scoped_to_the_requested_methods_and_trial():
    df = pd.concat(
        [
            _frame(trial=1),
            _guides("DUET (lambda=0.00)", [0.99], [5.0], trial=1),  # not requested
            _guides(MAX_ACT, [0.99], [5.0], trial=2),  # other trial
        ],
        ignore_index=True,
    )
    ranges = shared_axis_ranges(df, trial=1, methods=[DUET, MAX_ACT])
    assert ranges.activity[1] < 5.0
    assert ranges.accuracy[1] < 0.99


# ---------------------------------------------------------------- densities


def test_continuous_density_is_scaled_by_the_nonzero_fraction():
    rng = np.random.default_rng(1)
    nonzero = rng.normal(0.75, 0.05, 500)
    with_zeros = np.concatenate([nonzero, np.zeros(125)])  # p = 0.2
    grid = np.linspace(0.5, 0.95, 200)

    y_ref, _ = continuous_density(nonzero, grid)
    y, _ = continuous_density(with_zeros, grid)

    assert np.allclose(y, 0.8 * y_ref)


def test_zeros_do_not_inflate_the_bandwidth():
    rng = np.random.default_rng(2)
    nonzero = rng.normal(0.75, 0.05, 500)
    with_zeros = np.concatenate([nonzero, np.zeros(125)])
    grid = np.linspace(0.5, 0.95, 50)

    _, bw_ref = continuous_density(nonzero, grid)
    _, bw = continuous_density(with_zeros, grid)

    kde = gaussian_kde(nonzero)
    assert bw == pytest.approx(kde.factor * nonzero.std(ddof=1))
    assert bw == pytest.approx(bw_ref)


def test_zero_bin_bar_area_equals_the_zero_fraction():
    height, width = zero_bin_bar(zero_fraction=0.0118, bandwidth=0.0175)
    assert width == pytest.approx(0.0175)
    assert height * width == pytest.approx(0.0118)


# ---------------------------------------------------------------- the panel


def test_stack_draws_zero_bin_only_when_a_method_has_zeros(tmp_path):
    colors = {MAX_ACT: "#999999", DUET: "#0072B2"}

    df_zeros = _frame(baseline_zeros=20)
    out_zeros = tmp_path / "with_zeros.svg"
    info = plot_distribution_stack(
        df_zeros, colors, trial=1, output_path=out_zeros,
        ranges=shared_axis_ranges(df_zeros, 1, list(colors)),
    )
    assert info.zero_bin is True
    assert info.zero_fractions[MAX_ACT] == pytest.approx(20 / 400)
    assert info.zero_fractions[DUET] == 0.0
    assert out_zeros.stat().st_size > 0

    df_clean = _frame(baseline_zeros=0)
    out_clean = tmp_path / "no_zeros.svg"
    info = plot_distribution_stack(
        df_clean, colors, trial=1, output_path=out_clean,
        ranges=shared_axis_ranges(df_clean, 1, list(colors)),
    )
    assert info.zero_bin is False
    assert out_clean.stat().st_size > 0


def test_stack_zero_slot_keeps_the_continuous_range_intact(tmp_path):
    colors = {MAX_ACT: "#999999", DUET: "#0072B2"}
    df = _frame(baseline_zeros=20)
    ranges = shared_axis_ranges(df, 1, list(colors))
    info = plot_distribution_stack(
        df, colors, trial=1, output_path=tmp_path / "p.svg", ranges=ranges,
    )
    # The slot is added to the LEFT of the shared range; the range itself is
    # unchanged so the curves stay comparable with panels that have no slot.
    assert info.accuracy_xlim[1] == pytest.approx(ranges.accuracy[1])
    assert info.accuracy_xlim[0] < ranges.accuracy[0]


def test_zero_bin_drops_the_tick_that_would_crowd_the_slot():
    import matplotlib.pyplot as plt

    # A range starting on a round number gets an auto tick at its left edge,
    # which would sit right beside the "0" slot and the break marks.
    fig, ax = plt.subplots()
    lims = (0.4, 0.95)
    ax.set_xlim(*lims)
    assert 0.4 in ax.get_xticks()  # precondition: the crowding tick exists

    _draw_zero_bin(
        ax, {MAX_ACT: "#999999"}, {MAX_ACT: 0.02}, {MAX_ACT: 0.02}, lims, ymax=5.0,
    )
    labels = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)

    assert labels[0] == "0"
    assert "0.4" not in labels
    assert "0.5" in labels


def _overlay_frame(trial: int = 1) -> pd.DataFrame:
    """Five methods, 300 guides each; only maximum activity has zeros."""
    rng = np.random.default_rng(3)
    n = 300
    act = np.clip(rng.normal(0.9, 0.2, n), 0.3, 1.7)
    frames = []
    for method, mu, sd, zeros in [
        (MAX_ACT, 0.73, 0.07, 6),
        (FELDMAN1, 0.74, 0.07, 0),
        (SIVANANDAN1, 0.74, 0.07, 0),
        (SIVANANDAN2, 0.75, 0.06, 0),
        (DUET, 0.79, 0.04, 0),
    ]:
        acc = np.clip(rng.normal(mu, sd, n), 0.5, 0.95)
        acc[:zeros] = 0.0
        frames.append(_guides(method, acc, act, trial))
    return pd.concat(frames, ignore_index=True)


def _overlay_spec() -> OverlaySpec:
    return OverlaySpec(
        baselines=(MAX_ACT, FELDMAN1, SIVANANDAN1, SIVANANDAN2),
        duet=DUET,
        prefix="all_baselines_vs_duet_97p5pct",
        policy="97p5pct",
    )


def test_overlay_colors_tint_sivanandan_ed1_and_put_duet_last():
    colors = overlay_colors(_overlay_spec())
    assert list(colors) == [MAX_ACT, FELDMAN1, SIVANANDAN1, SIVANANDAN2, DUET]
    assert colors[SIVANANDAN1] == OVERLAY_VARIANT_PALETTE[SIVANANDAN1]
    assert colors[SIVANANDAN2] == METHOD_PALETTE["Sivanandan et al."]
    assert colors[FELDMAN1] == METHOD_PALETTE["Feldman et al."]
    assert colors[MAX_ACT] == METHOD_PALETTE["Maximum activity"]
    assert colors[DUET] == METHOD_PALETTE["DUET"]


def test_stack_fills_every_method_by_default(tmp_path):
    colors = {MAX_ACT: "#999999", DUET: "#0072B2"}
    df = _frame()
    info = plot_distribution_stack(
        df, colors, trial=1, output_path=tmp_path / "pair.svg",
        ranges=shared_axis_ranges(df, 1, list(colors)),
    )
    assert info.filled == (MAX_ACT, DUET)


def test_stack_fills_only_the_emphasized_method_in_an_overlay(tmp_path):
    df = _overlay_frame()
    colors = overlay_colors(_overlay_spec())
    out = tmp_path / "overlay.svg"
    info = plot_distribution_stack(
        df, colors, trial=1, output_path=out,
        ranges=shared_axis_ranges(df, 1, list(colors)), emphasize=DUET,
    )
    assert info.filled == (DUET,)
    assert info.zero_bin is True
    assert info.zero_fractions[MAX_ACT] == pytest.approx(6 / 300)
    assert out.stat().st_size > 0


def _svg_size_pt(path: Path) -> tuple[float, float]:
    head = path.read_text()[:2000]
    width = float(re.search(r'width="([\d.]+)pt"', head).group(1))
    height = float(re.search(r'height="([\d.]+)pt"', head).group(1))
    return width, height


def test_pairwise_stack_is_third_page_wide_and_2p52_inches_tall(tmp_path):
    colors = {MAX_ACT: "#999999", DUET: "#0072B2"}
    df = _frame()
    out = tmp_path / "pair.svg"
    plot_distribution_stack(
        df, colors, trial=1, output_path=out, ranges=shared_axis_ranges(df, 1, list(colors)),
    )
    width, height = _svg_size_pt(out)
    assert width == pytest.approx(FIGURE_WIDTHS["third_page"] * 72, abs=0.01)
    assert height == pytest.approx(2.52 * 72, abs=0.01)


def test_overlay_stack_is_2p70_inches_tall(tmp_path):
    df = _overlay_frame()
    colors = overlay_colors(_overlay_spec())
    out = tmp_path / "overlay.svg"
    plot_distribution_stack(
        df, colors, trial=1, output_path=out,
        ranges=shared_axis_ranges(df, 1, list(colors)), emphasize=DUET,
    )
    _, height = _svg_size_pt(out)
    assert height == pytest.approx(2.70 * 72, abs=0.01)


def test_zero_bin_bars_for_several_methods_stay_inside_the_slot():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Overlay ordering: the second zero-mass method is the LAST of five, so an
    # offset by legend position (not by position among zero-mass methods) would
    # push its bar past the break onto the continuous axis.
    colors = {MAX_ACT: "#999999", FELDMAN1: "#009E73", SIVANANDAN1: "#F0C060",
              SIVANANDAN2: "#E69F00", DUET: "#0072B2"}
    zero_fractions = {MAX_ACT: 0.012, FELDMAN1: 0.0, SIVANANDAN1: 0.0,
                      SIVANANDAN2: 0.0, DUET: 0.013}
    bandwidths = {m: 0.02 for m in colors}
    fig, ax = plt.subplots()
    lims = (0.45, 0.95)
    ax.set_xlim(*lims)

    left, _ = _draw_zero_bin(ax, colors, zero_fractions, bandwidths, lims, ymax=5.0)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    plt.close(fig)

    assert len(bars) == 2
    for bar in bars:
        x0, x1 = bar.get_x(), bar.get_x() + bar.get_width()
        assert left <= x0 < x1 < lims[0], (x0, x1)
    # Side by side, not overlapping.
    (a0, a1), (b0, b1) = sorted((b.get_x(), b.get_x() + b.get_width()) for b in bars)
    assert a1 <= b0 + 1e-9


def test_zero_bin_labels_stack_clear_of_every_bar():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Two bars of very different height: a percent label centred on each narrow
    # bar would overlap its neighbour's, since the text is far wider than the
    # bar. Labels must stack vertically above the tallest bar instead.
    colors = {MAX_ACT: "#999999", FELDMAN1: "#009E73", DUET: "#0072B2"}
    zero_fractions = {MAX_ACT: 0.020, FELDMAN1: 0.003, DUET: 0.0}
    bandwidths = {m: 0.0175 for m in colors}
    fig, ax = plt.subplots()
    lims = (0.45, 0.95)
    ax.set_xlim(*lims)

    _draw_zero_bin(ax, colors, zero_fractions, bandwidths, lims, ymax=5.0)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    labels = [t for t in ax.texts if t.get_text().endswith("%")]
    plt.close(fig)

    assert len(bars) == 2 and len(labels) == 2
    tallest = max(b.get_height() for b in bars)
    ys = sorted(t.get_position()[1] for t in labels)
    # Every label clears the tallest bar, so none can sit against a bar.
    assert ys[0] >= tallest
    # And they are separated from each other by a real gap.
    assert ys[1] - ys[0] >= 0.05 * 5.0
    # Sharing one x means the stack reads as a column, not a collision.
    assert len({round(t.get_position()[0], 9) for t in labels}) == 1


def test_stack_only_plots_the_requested_trial(tmp_path):
    colors = {MAX_ACT: "#999999", DUET: "#0072B2"}
    df = pd.concat([_frame(trial=1), _frame(baseline_zeros=50, trial=2)], ignore_index=True)
    info = plot_distribution_stack(
        df, colors, trial=1, output_path=tmp_path / "t1.svg",
        ranges=shared_axis_ranges(df, 1, list(colors)),
    )
    assert info.zero_bin is False
