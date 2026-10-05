"""Slot-sized figure panels with fixed millimetre margins.

Manuscript figures are composed in Inkscape on a 180 mm page, with panels saved
as separate files. A panel made here is exactly its slot's size and its axes
sit at fixed millimetre margins, so it drops into the slot without resizing and
its plot area lines up with its neighbours. Constrained layout cannot give
that: it sizes each file on its own, so separately saved panels end up with
different axes boxes.

Margins are in mm, so the same numbers work at any slot width. A row that needs
more room (a wide tick label, a two-line axis label) takes one override, applied
to every panel in that row so their axes still line up.
"""
from __future__ import annotations

MM_PER_INCH = 25.4

PAGE_WIDTH_MM = 180.0
PANEL_GAP_MM = 4.0

# (left, right, bottom, top). Left holds a one-line y label with tick labels up
# to "100,000" (15.1 mm), or a two-line y label (14.0 mm); bottom holds tick
# labels plus a one-line x label (8.5 mm); top and right only hold half an edge
# tick label.
DEFAULT_MARGINS_MM: tuple[float, float, float, float] = (16.0, 2.0, 10.0, 2.0)

# Slack allowed before save_panel calls a panel clipped.
_CLIP_TOLERANCE_MM = 0.1


def slot_width_mm(n_per_row: int, span: int = 1) -> float:
    """Width of a slot spanning ``span`` of ``n_per_row`` equal columns.

    Columns share the page with a ``PANEL_GAP_MM`` gap between neighbours; a
    slot spanning several columns includes the gaps it covers. For example,
    ``slot_width_mm(2)`` is 88 mm and ``slot_width_mm(3, span=2)`` is 118.67 mm.
    """
    if n_per_row < 1:
        raise ValueError(f"n_per_row must be >= 1, got {n_per_row}")
    if not 1 <= span <= n_per_row:
        raise ValueError(f"span must be between 1 and {n_per_row}, got {span}")
    column_mm = (PAGE_WIDTH_MM - (n_per_row - 1) * PANEL_GAP_MM) / n_per_row
    return span * column_mm + (span - 1) * PANEL_GAP_MM


def slot_figure(
    width_mm: float,
    height_mm: float,
    margins_mm: tuple[float, float, float, float] = DEFAULT_MARGINS_MM,
):
    """Return ``(fig, ax)`` for a panel of exactly ``width_mm`` x ``height_mm``.

    The figure has no layout engine; the axes are placed at ``margins_mm``
    (left, right, bottom, top). Save it with :func:`save_panel`, which checks
    that nothing drawn runs past the figure edge.
    """
    import matplotlib.pyplot as plt

    left, right, bottom, top = margins_mm
    axes_w = width_mm - left - right
    axes_h = height_mm - bottom - top
    if axes_w <= 0 or axes_h <= 0:
        raise ValueError(
            f"margins {margins_mm} mm leave no room for axes in a "
            f"{width_mm:.2f} x {height_mm:.2f} mm panel"
        )
    fig = plt.figure(
        figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), layout="none"
    )
    ax = fig.add_axes(
        [left / width_mm, bottom / height_mm, axes_w / width_mm, axes_h / height_mm]
    )
    return fig, ax


def check_panel_fits(fig) -> None:
    """Raise ``ValueError`` if anything drawn runs past the figure edge.

    Compares the figure's tight bounding box (tick labels, axis labels, legend,
    titles) to the figure size and names each side that overflows, with the
    overflow in mm.
    """
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width_in, height_in = fig.get_size_inches()
    overflow_mm = {
        "left": -bbox.x0 * MM_PER_INCH,
        "right": (bbox.x1 - width_in) * MM_PER_INCH,
        "bottom": -bbox.y0 * MM_PER_INCH,
        "top": (bbox.y1 - height_in) * MM_PER_INCH,
    }
    clipped = {side: mm for side, mm in overflow_mm.items() if mm > _CLIP_TOLERANCE_MM}
    if clipped:
        detail = ", ".join(f"{side} by {mm:.2f} mm" for side, mm in clipped.items())
        raise ValueError(
            f"panel content runs past the figure edge ({detail}); widen that "
            f"margin (for every panel in the row) or shrink the content"
        )


def fit_last_xtick_label(fig, ax, slack_mm: float = 0.05) -> None:
    """Widen ``ax``'s x range so its last tick label ends inside ``fig``.

    A panel from :func:`slot_figure` has fixed margins, so its axes cannot
    shrink to make room for a tick label. A tick label is centred on its tick
    and reaches half its width past it; when the locator puts the last tick
    close to the axes' right end, that half can be wider than the right margin
    and the label runs off the panel, which :func:`save_panel` refuses.

    This raises the upper x limit until the label fits, ``slack_mm`` inside the
    edge. The axes box and margins stay put; the axes span a little more data,
    so every point and tick sits slightly further left. The tick positions are
    fixed first, so the locator does not choose new ticks for the wider range.
    A panel whose last label already fits is not changed.

    Call it after drawing and before :func:`save_panel`. The x axis must be
    linear, increasing to the right.
    """
    from matplotlib.ticker import FixedLocator

    if ax.get_xscale() != "linear" or ax.xaxis_inverted():
        raise ValueError("fit_last_xtick_label needs a linear, non-inverted x axis")
    fig.canvas.draw()  # places the ticks and sets their label text
    renderer = fig.canvas.get_renderer()
    xmin, xmax = ax.get_xlim()
    # Matplotlib draws a tick that lies within a relative 1e-10 of the view
    # limits, e.g. a 0.0125 tick on an axis that ends at 0.012499999999999999.
    tol = 1e-10 * (xmax - xmin)

    def in_view(loc):
        return xmin - tol <= loc <= xmax + tol

    drawn = [
        tick for tick in ax.xaxis.get_major_ticks()
        if in_view(tick.get_loc()) and tick.label1.get_visible() and tick.label1.get_text()
    ]
    if not drawn:
        return
    last = drawn[-1]
    px_per_mm = fig.dpi / MM_PER_INCH
    overflow_mm = (last.label1.get_window_extent(renderer).x1 - fig.bbox.x1) / px_per_mm
    if overflow_mm <= 0:
        return

    # Move the last tick that much further from the axes' right end: its
    # distance, as a fraction f of the axes width, must satisfy
    # (new_xmax - tick) = f * (new_xmax - xmin).
    axes_mm = ax.get_window_extent(renderer).width / px_per_mm
    tick = last.get_loc()
    needed_mm = (xmax - tick) / (xmax - xmin) * axes_mm + overflow_mm + slack_mm
    f = needed_mm / axes_mm
    if f >= 1:
        raise ValueError(
            f"the last x tick label needs {needed_mm:.1f} mm, more than the "
            f"{axes_mm:.1f} mm axes"
        )
    ax.xaxis.set_major_locator(FixedLocator([t for t in ax.get_xticks() if in_view(t)]))
    ax.set_xlim(xmin, (tick - f * xmin) / (1 - f))


def fit_xlabel(fig, ax, slack_mm: float = 0.05) -> None:
    """Move ``ax``'s x label up so it ends inside the bottom of ``fig``.

    A panel from :func:`slot_figure` has a fixed bottom margin sized for its
    x label. Text heights depend on the matplotlib version: from 3.11 a line's
    height comes from the font's ascent, descent and line gap, so a two-line
    label is about 0.65 mm taller than under 3.10 and can end below the panel.

    This narrows the gap between the tick labels and the x label (the axis
    ``labelpad``) until the label ends ``slack_mm`` inside the edge. The axes,
    margins and label text stay put. A label that already fits is not
    changed. If closing the gap entirely is not enough, the label is left for
    :func:`save_panel` to report.

    Call it after drawing and before :func:`save_panel`.
    """
    label = ax.xaxis.label
    if not label.get_text() or not label.get_visible() or ax.xaxis.get_label_position() != "bottom":
        return
    fig.canvas.draw()  # places the label below the tick labels
    px_per_mm = fig.dpi / MM_PER_INCH
    overflow_mm = (fig.bbox.y0 - label.get_window_extent(fig.canvas.get_renderer()).y0) / px_per_mm
    if overflow_mm <= 0:
        return
    pt_per_mm = 72 / MM_PER_INCH
    ax.xaxis.labelpad = max(ax.xaxis.labelpad - (overflow_mm + slack_mm) * pt_per_mm, 0.0)


def save_panel(fig, *paths, **savefig_kwargs) -> None:
    """Check that ``fig`` fits, then save it to each path at exactly its size.

    The stylesheet's ``savefig.bbox: tight`` would crop each file to its drawn
    content, so the save runs under ``savefig.bbox: "standard"``. Passing
    ``bbox_inches=None`` would not help: None means "use rcParams".
    ``savefig_kwargs`` (e.g. ``dpi``) go to every save.
    """
    import matplotlib.pyplot as plt

    if not paths:
        raise TypeError("save_panel needs at least one output path")
    if "bbox_inches" in savefig_kwargs:
        raise TypeError("save_panel saves at the figure size; do not pass bbox_inches")
    check_panel_fits(fig)
    with plt.rc_context({"savefig.bbox": "standard"}):
        for path in paths:
            fig.savefig(path, **savefig_kwargs)
