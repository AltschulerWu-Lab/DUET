"""Tests for duet.plotting.labels: metric column name -> display label mapping.

Guards the rule that display labels use the ``≤`` glyph (never ASCII ``<=``)
and wrap long names, while the underlying metric *column names* stay untouched.
"""
from duet.plotting import metric_axis_label, metric_inline_label
from duet.plotting.labels import METRIC_AXIS_LABELS


def test_axis_label_uses_le_glyph_and_wraps_bracketed_metrics():
    assert (
        metric_axis_label("Mean decode accuracy (<= 10th percentile)")
        == "Mean decode accuracy\n(≤ 10th percentile)"
    )
    assert (
        metric_axis_label("Mean decode accuracy (<= 5th percentile)")
        == "Mean decode accuracy\n(≤ 5th percentile)"
    )


def test_axis_label_wraps_long_metrics_without_inequality():
    assert (
        metric_axis_label("95th over 5th percentile decode accuracy")
        == "95th over 5th percentile\ndecode accuracy"
    )
    assert (
        metric_axis_label("10th percentile decode accuracy")
        == "10th percentile\ndecode accuracy"
    )


def test_axis_label_passthrough_for_short_or_unknown_columns():
    assert metric_axis_label("Mean decode accuracy") == "Mean decode accuracy"
    assert metric_axis_label("Mean activity score") == "Mean activity score"
    assert metric_axis_label("Some unmapped column") == "Some unmapped column"


def test_inline_label_is_single_line_with_le_glyph():
    assert (
        metric_inline_label("Mean decode accuracy (<= 10th percentile)")
        == "Mean decode accuracy (≤ 10th percentile)"
    )
    assert "\n" not in metric_inline_label("95th over 5th percentile decode accuracy")
    # Unknown columns fall through unchanged, single line.
    assert metric_inline_label("Mean activity score") == "Mean activity score"


def test_no_ascii_le_in_any_display_label():
    """Every registered display label must use the ≤ glyph, never ASCII '<='."""
    for display in METRIC_AXIS_LABELS.values():
        assert "<=" not in display
