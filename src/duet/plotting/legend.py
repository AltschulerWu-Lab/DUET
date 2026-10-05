"""Legend entries for method panels: square markers and "N/A" entries.

Single-trial Pareto panels (Fig 3d-f, Supp S4c) have no error bars, so their
points are drawn larger and their legends use solid squares, as in the
manuscript's Fig 3d. A method with nothing to plot (no valid codebook, or a
metric that is undefined for it, like the 95th/5th ratio when the 5th
percentile is 0) keeps its legend row with a grey "N/A" in place of the marker.
"""
from __future__ import annotations

from typing import Sequence

from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.text import Text

from duet.plotting.palette import NOT_AVAILABLE_COLOR

# Point size (pt) for Pareto panels built from a single trial. Chosen
# 2026-09-30 from 1.5-5 pt proofs of Fig 3d-f and S4c; multi-trial panels keep
# 1.5 pt points with s.e.m. bars.
SINGLE_TRIAL_MARKERSIZE = 4.0

# Legend squares are drawn at the point size or this, whichever is larger, so
# they stay legible next to small points (Fig 3d's squares measure about this).
LEGEND_SQUARE_MIN_SIZE = 3.5

# The "N/A" text is this fraction of the legend font size.
_NOT_AVAILABLE_FONT_SCALE = 0.8


class NotAvailable:
    """Legend proxy for a method with nothing plotted."""


class NotAvailableHandler(HandlerBase):
    """Draws a grey "N/A" where the legend marker would be."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        return [Text(
            x=width / 2 - xdescent,
            y=height / 2 - ydescent,
            text="N/A",
            ha="center",
            va="center",
            fontsize=fontsize * _NOT_AVAILABLE_FONT_SCALE,
            color=NOT_AVAILABLE_COLOR,
            transform=trans,
        )]


# Pass as ``ax.legend(..., handler_map=LEGEND_HANDLER_MAP)``.
LEGEND_HANDLER_MAP = {NotAvailable: NotAvailableHandler()}


def square_handle(color: str, markersize: float, alpha: float = 0.7) -> Line2D:
    """A solid square legend marker at ``max(markersize, LEGEND_SQUARE_MIN_SIZE)``."""
    return Line2D(
        [], [],
        linestyle="none",
        marker="s",
        color=color,
        alpha=alpha,
        markersize=max(markersize, LEGEND_SQUARE_MIN_SIZE),
        markeredgewidth=0,
    )


def square_legend_handles(
    entries: Sequence[tuple[str, str | None]],
    markersize: float,
    alpha: float = 0.7,
) -> tuple[list, list[str]]:
    """Handles and labels for ``(label, color)`` entries, in order.

    A ``None`` color marks a method with nothing plotted; it gets "N/A".
    """
    handles = [
        NotAvailable() if color is None else square_handle(color, markersize, alpha)
        for _, color in entries
    ]
    return handles, [label for label, _ in entries]
