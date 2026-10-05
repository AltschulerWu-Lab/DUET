"""Verify apply_style() loads the DUET publication stylesheet correctly."""
from __future__ import annotations

import matplotlib as mpl
import pytest

from duet.plotting import apply_style


@pytest.fixture(autouse=True)
def _reset_rcparams():
    """Each test starts from matplotlib defaults and restores afterwards."""
    original = mpl.rcParams.copy()
    mpl.rcdefaults()
    yield
    mpl.rcParams.update(original)


def test_apply_style_runs_without_error():
    apply_style()


def test_apply_style_sets_font_family_to_sans_serif():
    apply_style()
    # matplotlib parses `font.family: sans-serif` from an mplstyle into a
    # single-element list. If a future matplotlib ever emits a bare string,
    # this assertion needs updating.
    assert mpl.rcParams["font.family"] == ["sans-serif"]


def test_apply_style_prefers_arial_in_sans_serif_stack():
    apply_style()
    assert mpl.rcParams["font.sans-serif"][0] == "Arial"


def test_apply_style_sets_base_font_size_to_8pt():
    apply_style()
    assert mpl.rcParams["font.size"] == 8


def test_apply_style_sets_svg_fonttype_to_none_for_inkscape():
    apply_style()
    assert mpl.rcParams["svg.fonttype"] == "none"


def test_apply_style_hides_top_and_right_spines():
    apply_style()
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False


def test_apply_style_ticks_point_outward():
    apply_style()
    assert mpl.rcParams["xtick.direction"] == "out"
    assert mpl.rcParams["ytick.direction"] == "out"


def test_apply_style_disables_default_grid():
    apply_style()
    assert mpl.rcParams["axes.grid"] is False


def test_apply_style_uses_okabe_ito_prop_cycle():
    apply_style()
    cycle_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    # Okabe-Ito hexes as emitted by matplotlib (prefixed with #)
    assert cycle_colors[0].lower() == "#e69f00"
    assert cycle_colors[4].lower() == "#0072b2"


def test_apply_style_sets_savefig_format_to_svg():
    apply_style()
    assert mpl.rcParams["savefig.format"] == "svg"
