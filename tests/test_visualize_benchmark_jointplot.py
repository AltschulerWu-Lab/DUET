"""Tests for the guide-level jointplot (Figure 3b right, Supplementary Fig. S4d-f)."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

from visualize_benchmark import plot_jointplot  # noqa: E402

DUET = "DUET (lambda=0.88)"
MAX_ACT = "Maximum activity"


def _frame(n: int = 300, baseline: str = MAX_ACT) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method, mu in [(baseline, 0.73), (DUET, 0.79)]:
        rows.append(
            pd.DataFrame(
                {
                    "Method": method,
                    "Trial": 1,
                    "Decode accuracy": np.clip(rng.normal(mu, 0.05, n), 0.5, 0.95),
                    "Activity score": np.clip(rng.normal(0.9, 0.2, n), 0.3, 1.7),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_jointplot_legend_uses_square_swatches(tmp_path):
    colors = {MAX_ACT: "#999999", DUET: "#0072B2"}
    grid = plot_jointplot(_frame(), colors, trial=1, output_path=tmp_path / "j.svg")

    legend = grid.ax_joint.get_legend()
    handles = legend.legend_handles
    assert len(handles) == 2
    assert all(isinstance(h, Rectangle) for h in handles)
    # A square, not the default 2:0.7 bar.
    assert legend.handlelength == legend.handleheight
    assert {mcolors.to_hex(h.get_facecolor()) for h in handles} == {"#999999", "#0072b2"}
    assert [t.get_text() for t in legend.get_texts()] == [
        "Maximum activity", "DUET (λ=0.88)",
    ] or [t.get_text() for t in legend.get_texts()] == [
        "DUET (λ=0.88)", "Maximum activity",
    ]
    assert (tmp_path / "j.svg").stat().st_size > 0


def test_jointplot_legend_shows_sivanandan_as_hd(tmp_path):
    # S2f,g: the runner writes "Sivanandan et al. (ED=k)"; the method enforces a
    # Hamming-distance floor, so the legend reads HD.
    sivanandan = "Sivanandan et al. (ED=2)"
    colors = {sivanandan: "#E69F00", DUET: "#0072B2"}
    grid = plot_jointplot(
        _frame(baseline=sivanandan), colors, trial=1, output_path=tmp_path / "j.svg"
    )
    texts = {t.get_text() for t in grid.ax_joint.get_legend().get_texts()}
    assert texts == {"Sivanandan et al. (HD = 2)", "DUET (λ=0.88)"}
