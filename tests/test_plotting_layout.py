"""Slot-sized panels: slot widths, fixed mm margins, exact-size saves, clipping check."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pytest

from duet.plotting import (
    DEFAULT_MARGINS_MM,
    FIGURE_WIDTHS,
    MM_PER_INCH,
    PAGE_WIDTH_MM,
    PANEL_GAP_MM,
    check_panel_fits,
    fit_last_xtick_label,
    fit_xlabel,
    save_panel,
    slot_figure,
    slot_width_mm,
    style_context,
)

PT_PER_MM = 72 / MM_PER_INCH


@pytest.fixture(autouse=True)
def _publication_style():
    """Draw under the stylesheet, whose savefig.bbox: tight save_panel must beat."""
    with style_context():
        yield
    plt.close("all")


def _svg_size_mm(path) -> tuple[float, float]:
    head = path.read_text()[:2000]
    width = re.search(r'<svg[^>]*\bwidth="([\d.]+)(?:pt)?"', head)
    height = re.search(r'<svg[^>]*\bheight="([\d.]+)(?:pt)?"', head)
    assert width and height, f"no width/height on the <svg> tag of {path}"
    return float(width.group(1)) / PT_PER_MM, float(height.group(1)) / PT_PER_MM


def _labelled_panel(margins_mm=DEFAULT_MARGINS_MM):
    fig, ax = slot_figure(slot_width_mm(3), 55.88, margins_mm=margins_mm)
    ax.plot([0, 1, 2], [0, 50, 100], label="series")
    ax.set_xlabel("Imaging round")
    ax.set_ylabel("Total expression")
    return fig, ax


# -------------------- slot widths --------------------


def test_two_halves_and_a_gap_fill_the_page():
    assert slot_width_mm(2) == pytest.approx(88.0)
    assert 2 * slot_width_mm(2) + PANEL_GAP_MM == pytest.approx(PAGE_WIDTH_MM)


def test_three_thirds_and_two_gaps_fill_the_page():
    assert 3 * slot_width_mm(3) + 2 * PANEL_GAP_MM == pytest.approx(PAGE_WIDTH_MM)


def test_a_slot_spanning_columns_includes_the_gaps_it_covers():
    assert slot_width_mm(3, span=2) == pytest.approx(2 * slot_width_mm(3) + PANEL_GAP_MM)
    assert slot_width_mm(3, span=2) == pytest.approx(118.667, abs=1e-3)
    assert slot_width_mm(4) == pytest.approx(42.0)
    assert slot_width_mm(1) == pytest.approx(PAGE_WIDTH_MM)
    assert slot_width_mm(3, span=3) == pytest.approx(PAGE_WIDTH_MM)


@pytest.mark.parametrize("n_per_row, span", [(0, 1), (3, 0), (3, 4)])
def test_slot_width_rejects_impossible_slots(n_per_row, span):
    with pytest.raises(ValueError):
        slot_width_mm(n_per_row, span=span)


def test_figure_widths_are_the_slot_widths():
    expected_mm = {
        "third_page": slot_width_mm(3),
        "half_page": slot_width_mm(2),
        "two_thirds_page": slot_width_mm(3, span=2),
        "full_page": slot_width_mm(1),
    }
    for key, mm in expected_mm.items():
        assert FIGURE_WIDTHS[key] * MM_PER_INCH == pytest.approx(mm), key


# -------------------- slot_figure --------------------


def test_slot_figure_places_axes_at_the_mm_margins():
    width_mm, height_mm = slot_width_mm(3), 50.8
    fig, ax = slot_figure(width_mm, height_mm, margins_mm=(16.0, 2.0, 12.0, 2.0))

    assert fig.get_layout_engine() is None
    w_in, h_in = fig.get_size_inches()
    assert w_in * MM_PER_INCH == pytest.approx(width_mm)
    assert h_in * MM_PER_INCH == pytest.approx(height_mm)
    box = ax.get_position()
    assert box.x0 * width_mm == pytest.approx(16.0)
    assert (1 - box.x1) * width_mm == pytest.approx(2.0)
    assert box.y0 * height_mm == pytest.approx(12.0)
    assert (1 - box.y1) * height_mm == pytest.approx(2.0)


def test_slot_figure_rejects_margins_that_leave_no_axes():
    with pytest.raises(ValueError, match="no room"):
        slot_figure(40.0, 30.0, margins_mm=(25.0, 20.0, 10.0, 2.0))


# -------------------- save_panel --------------------


def test_saved_svg_is_exactly_the_requested_size(tmp_path):
    fig, _ax = _labelled_panel()
    path = tmp_path / "panel.svg"
    save_panel(fig, path)

    width_mm, height_mm = _svg_size_mm(path)
    assert width_mm == pytest.approx(slot_width_mm(3), abs=0.01)
    assert height_mm == pytest.approx(55.88, abs=0.01)
    # The save is local: the stylesheet's tight bbox is back in force afterwards.
    assert plt.rcParams["savefig.bbox"] == "tight"


def test_save_panel_writes_every_path(tmp_path):
    from matplotlib.image import imread

    fig, _ax = _labelled_panel()
    svg, png = tmp_path / "panel.svg", tmp_path / "panel.png"
    save_panel(fig, svg, png, dpi=100)

    assert _svg_size_mm(svg)[0] == pytest.approx(slot_width_mm(3), abs=0.01)
    rows, cols = imread(png).shape[:2]
    assert abs(cols - slot_width_mm(3) / MM_PER_INCH * 100) < 1
    assert abs(rows - 55.88 / MM_PER_INCH * 100) < 1


def test_save_panel_rejects_bbox_inches(tmp_path):
    fig, _ax = _labelled_panel()
    with pytest.raises(TypeError, match="bbox_inches"):
        save_panel(fig, tmp_path / "panel.svg", bbox_inches="tight")


# -------------------- clipping check --------------------


def test_default_margins_hold_a_labelled_panel():
    fig, _ax = _labelled_panel()
    check_panel_fits(fig)


def test_too_small_margins_raise_naming_side_and_overflow(tmp_path):
    fig, _ax = _labelled_panel(margins_mm=(3.0, 2.0, 3.0, 2.0))
    with pytest.raises(ValueError, match=r"left by \d+\.\d\d mm.*bottom by \d+\.\d\d mm"):
        save_panel(fig, tmp_path / "panel.svg")
    assert not (tmp_path / "panel.svg").exists()


def test_a_legend_wider_than_the_panel_raises():
    fig, ax = _labelled_panel()
    ax.plot([0, 1], [0, 1], label="a legend label far too long for a third-page slot panel")
    ax.legend(loc="upper left")
    with pytest.raises(ValueError, match="right by"):
        check_panel_fits(fig)


# -------------------- last x tick label at the right edge --------------------


def _edge_tick_panel(xmax: float):
    """A third-page panel with x ticks every 0.025 up to 0.625 and its x range
    ending at ``xmax``. With 0.6255 the 0.625 tick sits about 0.2 mm from the
    axes' right end and its label runs past the 2 mm right margin."""
    fig, ax = slot_figure(FIGURE_WIDTHS["third_page"] * MM_PER_INCH, 50.0)
    ax.plot([0.51, 0.62], [0.7, 0.9], "o")
    ax.set_xticks([0.525, 0.55, 0.575, 0.6, 0.625])
    ax.set_xlim(0.5, xmax)
    ax.set_xlabel("Mean decode accuracy")
    ax.set_ylabel("Mean activity score")
    return fig, ax


def _drawn_xticks(ax):
    lo, hi = ax.get_xlim()
    return [t for t in ax.get_xticks() if lo <= t <= hi]


def test_fit_last_xtick_label_widens_the_x_range_to_fit_the_label():
    fig, ax = _edge_tick_panel(0.6255)
    with pytest.raises(ValueError, match="right by"):
        check_panel_fits(fig)
    position, ticks = ax.get_position().bounds, _drawn_xticks(ax)

    fit_last_xtick_label(fig, ax)

    check_panel_fits(fig)
    assert ax.get_xlim()[0] == 0.5 and ax.get_xlim()[1] > 0.6255
    assert _drawn_xticks(ax) == ticks  # same ticks, now further from the end
    assert ax.get_position().bounds == position  # margins unchanged


def test_fit_last_xtick_label_leaves_a_panel_that_fits_alone():
    fig, ax = _edge_tick_panel(0.64)
    check_panel_fits(fig)
    locator = ax.xaxis.get_major_locator()

    fit_last_xtick_label(fig, ax)

    assert ax.get_xlim() == (0.5, 0.64)
    assert ax.xaxis.get_major_locator() is locator


def test_fit_last_xtick_label_freezes_locator_ticks_before_widening():
    """With the default locator, the ticks it chose stay; it is not re-run."""
    fig, ax = slot_figure(FIGURE_WIDTHS["third_page"] * MM_PER_INCH, 50.0)
    ax.plot([0.51, 0.62], [0.7, 0.9], "o")
    ax.set_xlim(0.5, 0.6255)
    fig.canvas.draw()
    ticks = _drawn_xticks(ax)

    fit_last_xtick_label(fig, ax)

    check_panel_fits(fig)
    assert _drawn_xticks(ax) == ticks


def test_fit_last_xtick_label_handles_a_tick_on_the_rounded_axis_end():
    """A tick a float epsilon past the x limit is still drawn, so still fitted."""
    fig, ax = _edge_tick_panel(0.6255)
    ax.set_xticks([0.5, 0.525, 0.55, 0.575, 0.6, 0.625])
    ax.set_xlim(0.5, 0.625 - 1e-15)  # e.g. 1.05 * (0.625 / 1.05) in floating point
    with pytest.raises(ValueError, match="right by"):
        check_panel_fits(fig)

    fit_last_xtick_label(fig, ax)

    check_panel_fits(fig)
    assert 0.625 in _drawn_xticks(ax)


def test_fit_last_xtick_label_needs_a_linear_increasing_axis():
    fig, ax = _edge_tick_panel(0.6255)
    ax.invert_xaxis()
    with pytest.raises(ValueError, match="linear, non-inverted"):
        fit_last_xtick_label(fig, ax)


# -------------------- x label at the bottom edge --------------------


def _two_line_xlabel_panel(bottom_mm: float):
    """A third-page panel with a two-line x label over a ``bottom_mm`` margin.

    The label needs about 11.5 mm under matplotlib 3.10 and 12.2 mm under 3.11,
    so 11 mm overflows by less than the label's 4 pt (1.41 mm) pad in both,
    13 mm fits in both, and 9 mm overflows by more than the pad.
    """
    fig, ax = slot_figure(
        FIGURE_WIDTHS["third_page"] * MM_PER_INCH, 50.0, margins_mm=(16.0, 2.0, bottom_mm, 2.0)
    )
    ax.plot([0.5, 0.6], [0.7, 0.9], "o")
    ax.set_xlabel("95th over 5th percentile\ndecode accuracy")
    ax.set_ylabel("Mean activity score")
    return fig, ax


def test_fit_xlabel_narrows_the_label_gap_to_fit_the_bottom():
    fig, ax = _two_line_xlabel_panel(11.0)
    with pytest.raises(ValueError, match="bottom by"):
        check_panel_fits(fig)
    position, pad = ax.get_position().bounds, ax.xaxis.labelpad

    fit_xlabel(fig, ax)

    check_panel_fits(fig)
    assert 0 < ax.xaxis.labelpad < pad
    assert ax.get_position().bounds == position  # margins unchanged


def test_fit_xlabel_leaves_a_label_that_fits_alone():
    fig, ax = _two_line_xlabel_panel(13.0)
    pad = ax.xaxis.labelpad

    fit_xlabel(fig, ax)

    assert ax.xaxis.labelpad == pad


def test_fit_xlabel_leaves_an_unfixable_overflow_for_save_panel():
    fig, ax = _two_line_xlabel_panel(9.0)

    fit_xlabel(fig, ax)

    assert ax.xaxis.labelpad == 0
    with pytest.raises(ValueError, match="bottom by"):
        check_panel_fits(fig)
