"""Shared matplotlib style infrastructure for DUET publication figures."""
from __future__ import annotations

from duet.plotting.labels import (
    METRIC_AXIS_LABELS,
    metric_axis_label,
    metric_inline_label,
)
from duet.plotting.layout import (
    DEFAULT_MARGINS_MM,
    MM_PER_INCH,
    PAGE_WIDTH_MM,
    PANEL_GAP_MM,
    check_panel_fits,
    fit_last_xtick_label,
    fit_xlabel,
    save_panel,
    slot_figure,
    slot_width_mm,
)
from duet.plotting.palette import (
    BASELINE_OBJECTIVE_PALETTE,
    DNA_BASE_PALETTE,
    ERROR_CATEGORY_PALETTE,
    FIGURE_WIDTHS,
    METHOD_MARKERS,
    METHOD_PALETTE,
    NOT_AVAILABLE_COLOR,
    OKABE_ITO,
    OVERLAY_VARIANT_PALETTE,
    UNKNOWN_METHOD_COLOR,
    delta_diverging_cmap,
    gain_sequential_cmap,
    reduction_sequential_cmap,
    sequential_shades,
)
from duet.plotting.style import apply_style, style_context

__all__ = [
    "apply_style",
    "BASELINE_OBJECTIVE_PALETTE",
    "check_panel_fits",
    "DEFAULT_MARGINS_MM",
    "delta_diverging_cmap",
    "DNA_BASE_PALETTE",
    "ERROR_CATEGORY_PALETTE",
    "FIGURE_WIDTHS",
    "fit_last_xtick_label",
    "fit_xlabel",
    "gain_sequential_cmap",
    "METHOD_MARKERS",
    "METHOD_PALETTE",
    "METRIC_AXIS_LABELS",
    "metric_axis_label",
    "metric_inline_label",
    "MM_PER_INCH",
    "NOT_AVAILABLE_COLOR",
    "OKABE_ITO",
    "OVERLAY_VARIANT_PALETTE",
    "PAGE_WIDTH_MM",
    "PANEL_GAP_MM",
    "reduction_sequential_cmap",
    "save_panel",
    "sequential_shades",
    "slot_figure",
    "slot_width_mm",
    "style_context",
    "UNKNOWN_METHOD_COLOR",
]
