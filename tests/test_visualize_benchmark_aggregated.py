"""Tests for the aggregated metric scatter (Figure 3d-f), a slot-sized panel."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.plotting import FIGURE_WIDTHS, MM_PER_INCH, style_context

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

from visualize_benchmark import plot_metric_scatterplot_aggregated  # noqa: E402

X_COL = "Mean decode accuracy"
Y_COL = "Mean activity score"


def _agg_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method, group, x, y in [
        ("Maximum activity", "Maximum activity", 0.70, 1.00),
        ("DUET (lambda=0.50)", "DUET", 0.85, 0.95),
        ("DUET (lambda=0.90)", "DUET", 0.88, 0.90),
        ("Feldman et al. (ED=2)", "Feldman et al.", 0.80, 0.85),
        ("Sivanandan et al. (ED=2)", "Sivanandan et al.", 0.78, 0.92),
    ]:
        for trial in range(3):
            rows.append({
                "Method": method, "Method group": group, "Trial": trial,
                X_COL: x + rng.normal(0, 0.01), Y_COL: y + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_aggregated_scatter_saves_a_third_page_slot_panel_without_title(tmp_path):
    path = tmp_path / "agg.svg"
    with style_context():
        plot_metric_scatterplot_aggregated(_agg_frame(), X_COL, Y_COL, path)

    svg = path.read_text()
    pt_per_mm = 72 / MM_PER_INCH
    width = float(re.search(r'<svg[^>]*\bwidth="([\d.]+)(?:pt)?"', svg).group(1))
    height = float(re.search(r'<svg[^>]*\bheight="([\d.]+)(?:pt)?"', svg).group(1))
    assert width / pt_per_mm == pytest.approx(FIGURE_WIDTHS["third_page"] * MM_PER_INCH, abs=0.01)
    assert height / pt_per_mm == pytest.approx(2.0 * MM_PER_INCH, abs=0.01)
    # The title does not fit a 2 mm top margin and was cut from the shipped figure.
    assert "Aggregated Across Trials" not in svg
