"""Display-label helpers for DUET publication figures.

Benchmark metric *column names* (e.g. ``"Mean decode accuracy (<= 10th
percentile)"``) double as DataFrame keys, CSV headers, and output-filename
seeds, so they must stay ASCII and stable. This module maps those stable
column names to their typeset axis labels: ``<=`` becomes the ``≤`` glyph and
long names wrap onto a second line so narrow Pareto panels (third-page width)
stay readable. Keep this separate from the data layer — only axis/label
rendering should route through here, never data access.
"""
from __future__ import annotations


# Maps a benchmark metric column name -> its typeset, line-wrapped axis label.
# Columns absent from this map (already-short labels such as
# ``"Mean decode accuracy"`` and ``"Mean activity score"``) fall through
# unchanged via ``metric_axis_label``.
METRIC_AXIS_LABELS: dict[str, str] = {
    "Standard deviation decode accuracy": "Standard deviation\ndecode accuracy",
    "5th percentile decode accuracy": "5th percentile\ndecode accuracy",
    "10th percentile decode accuracy": "10th percentile\ndecode accuracy",
    "95th percentile decode accuracy": "95th percentile\ndecode accuracy",
    "95th over 5th percentile decode accuracy": "95th over 5th percentile\ndecode accuracy",
    "Mean decode accuracy (<= 5th percentile)": "Mean decode accuracy\n(≤ 5th percentile)",
    "Mean decode accuracy (<= 10th percentile)": "Mean decode accuracy\n(≤ 10th percentile)",
}


def metric_axis_label(column: str) -> str:
    """Return the typeset axis label for a benchmark metric column name.

    Falls back to the column name unchanged when no special formatting is
    registered, so short labels (``"Mean decode accuracy"``,
    ``"Mean activity score"``) render as-is.
    """
    return METRIC_AXIS_LABELS.get(column, column)


def metric_inline_label(column: str) -> str:
    """Single-line variant of :func:`metric_axis_label`.

    Same ``<=`` -> ``≤`` substitution, but with the second-line wrap collapsed
    back to a space. Use for plot titles or legends where a one-line label
    reads better than a wrapped axis label. Like :func:`metric_axis_label`,
    columns with no registered label fall through unchanged.
    """
    return metric_axis_label(column).replace("\n", " ")
