"""Validate color palette constants in duet.plotting.palette."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import re

import pytest

from duet.plotting import palette


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_okabe_ito_has_eight_entries():
    assert len(palette.OKABE_ITO) == 8


def test_okabe_ito_values_are_hex():
    for key, value in palette.OKABE_ITO.items():
        assert HEX_RE.match(value), f"{key}={value!r} is not a 6-digit hex"


def test_okabe_ito_named_keys():
    expected_keys = {
        "orange", "sky_blue", "bluish_green", "yellow",
        "blue", "vermillion", "reddish_purple", "black",
    }
    assert set(palette.OKABE_ITO.keys()) == expected_keys


def test_method_palette_has_expected_methods():
    # The palette is intentionally extensible: new MERFISH/2-D-synthetic
    # baselines register their own hues in palette.py over time.
    # Assert the canonical methods are PRESENT (subset) rather than pinning an
    # exact set, which goes stale every time a baseline hue is added.
    expected = {
        "DUET", "Feldman et al.", "Sivanandan et al.", "Maximum activity",
        "Greedy-Hamming-MO", "Exhaustive",
    }
    assert expected <= set(palette.METHOD_PALETTE.keys())


def test_method_palette_values_are_hex():
    for label, color in palette.METHOD_PALETTE.items():
        assert HEX_RE.match(color), f"{label}={color!r} is not a valid hex"


def test_overlay_variant_tint_is_a_lighter_sivanandan_orange():
    # The all-baselines overlay draws both Sivanandan floors on one panel; the
    # weaker one (ED=1) takes the middle step of a three-step ramp to white.
    import matplotlib.colors as mcolors

    from duet.plotting import OVERLAY_VARIANT_PALETTE

    tint = OVERLAY_VARIANT_PALETTE["Sivanandan et al. (ED=1)"]
    base = palette.METHOD_PALETTE["Sivanandan et al."]
    assert HEX_RE.match(tint)
    assert tint.lower() != base.lower()
    assert tint.lower() == palette.sequential_shades(base, 3)[1].lower()
    hue_t, _, val_t = mcolors.rgb_to_hsv(mcolors.to_rgb(tint))
    hue_b, _, val_b = mcolors.rgb_to_hsv(mcolors.to_rgb(base))
    assert hue_t == pytest.approx(hue_b, abs=0.02)
    assert val_t >= val_b


def test_error_category_palette_has_three_entries():
    assert set(palette.ERROR_CATEGORY_PALETTE.keys()) == {
        "No error", "Corrected", "Failed",
    }


def test_figure_widths_expected_keys():
    assert set(palette.FIGURE_WIDTHS.keys()) == {
        "third_page", "half_page", "two_thirds_page", "full_page",
    }


def test_figure_widths_are_in_inches_and_ordered():
    # Range assertions catch unit errors (cm / mm / m) and gross conversion
    # mistakes without tautologically re-deriving the arithmetic from palette.py.
    # Slot widths of the 180 mm page with 4 mm gaps, in inches: ~2.26 / ~3.46 /
    # ~4.67 / ~7.09 (duet.plotting.layout).
    # half_page / full_page bounds exclude Nature's 89 / 183 mm (3.50" / 7.20").
    assert 2.2 < palette.FIGURE_WIDTHS["third_page"] < 2.5
    assert 3.4 < palette.FIGURE_WIDTHS["half_page"] < 3.5
    assert 4.6 < palette.FIGURE_WIDTHS["two_thirds_page"] < 4.9
    assert 7.0 < palette.FIGURE_WIDTHS["full_page"] < 7.15
    # Ordering invariant — catches accidental key/value swaps.
    assert (
        palette.FIGURE_WIDTHS["third_page"]
        < palette.FIGURE_WIDTHS["half_page"]
        < palette.FIGURE_WIDTHS["two_thirds_page"]
        < palette.FIGURE_WIDTHS["full_page"]
    )


# ---------------------------------------------------------------------------
# Tests for figsize kwarg and new defaults in visualization helpers
# ---------------------------------------------------------------------------

import numpy as np

from duet.visualization import (
    plot_activity_scores,
    plot_codeword_probabilities,
    plot_dual_objectives,
)


def _make_groups_for_hist():
    rng = np.random.default_rng(0)
    return [
        ("A", rng.standard_normal(50), "tab:blue"),
        ("B", rng.standard_normal(50), "tab:orange"),
    ]


def _make_groups_for_dual():
    rng = np.random.default_rng(0)
    return [
        ("A", rng.standard_normal(50), rng.standard_normal(50), "tab:blue"),
        ("B", rng.standard_normal(50), rng.standard_normal(50), "tab:orange"),
    ]


def test_plot_codeword_probabilities_uses_default_half_page():
    fig, _ax = plot_codeword_probabilities(_make_groups_for_hist())
    width, height = fig.get_size_inches()
    assert abs(width - palette.FIGURE_WIDTHS["half_page"]) < 0.01
    assert abs(height - 2.2) < 0.01


def test_plot_codeword_probabilities_honors_figsize_kwarg():
    fig, _ax = plot_codeword_probabilities(
        _make_groups_for_hist(), figsize=(5.0, 4.0)
    )
    width, height = fig.get_size_inches()
    assert abs(width - 5.0) < 0.01
    assert abs(height - 4.0) < 0.01


def test_plot_activity_scores_uses_default_half_page():
    fig, _ax = plot_activity_scores(_make_groups_for_hist())
    width, height = fig.get_size_inches()
    assert abs(width - palette.FIGURE_WIDTHS["half_page"]) < 0.01
    assert abs(height - 2.2) < 0.01


def test_plot_activity_scores_honors_figsize_kwarg():
    fig, _ax = plot_activity_scores(
        _make_groups_for_hist(), figsize=(5.0, 4.0)
    )
    width, height = fig.get_size_inches()
    assert abs(width - 5.0) < 0.01
    assert abs(height - 4.0) < 0.01


def test_plot_dual_objectives_uses_default_half_page():
    fig, _ax = plot_dual_objectives(_make_groups_for_dual())
    width, height = fig.get_size_inches()
    assert abs(width - palette.FIGURE_WIDTHS["half_page"]) < 0.01
    assert abs(height - 2.6) < 0.01


def test_plot_dual_objectives_honors_figsize_kwarg():
    fig, _ax = plot_dual_objectives(
        _make_groups_for_dual(), figsize=(5.0, 4.0)
    )
    width, height = fig.get_size_inches()
    assert abs(width - 5.0) < 0.01
    assert abs(height - 4.0) < 0.01


def test_delta_diverging_cmap_anchors_on_okabe_ito():
    from matplotlib.colors import to_hex

    from duet.plotting import OKABE_ITO, delta_diverging_cmap

    cmap = delta_diverging_cmap()
    assert to_hex(cmap(0.0)).lower() == OKABE_ITO["blue"].lower()
    assert to_hex(cmap(1.0)).lower() == OKABE_ITO["vermillion"].lower()


def test_delta_diverging_cmap_is_white_at_the_midpoint():
    from duet.plotting import delta_diverging_cmap

    r, g, b, _ = delta_diverging_cmap()(0.5)
    assert min(r, g, b) > 0.98, "zero delta must read as white"


def test_delta_diverging_cmap_accepts_a_custom_name():
    from duet.plotting import delta_diverging_cmap

    assert delta_diverging_cmap("custom_name").name == "custom_name"
