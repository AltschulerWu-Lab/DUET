#!/usr/bin/env python3
"""
Visualize the DUET-vs-baseline-objectives experiment, one figure per trial.

Reads the run directory's baseline-objectives parquet — either
`objective_correlation.parquet` (enumeration runner) or
`objective_correlation_sampled.parquet` (sampling runner; falls back to
`objective_correlation_sampled.partial.parquet` if the sampling run is
mid-stream). Renders a 5x4 small-multiples SVG per unique `trial` value
under <indir>/figures/baseline_comparison_per_trial/trial_<id>.svg
(zero-padded to two digits). The per-trial figures are debug plots
(duet.plotting.tiers): by default only trial_01.svg, the paper panel, is
written; --debug-plots writes every trial.

  - Rows = objective (DUET objective, mean NLL, min NLL, mean Hamming, min Hamming).
  - Columns = noise channel (symmetric, position_varying, asymmetric, position_varying_asymmetric).
  - Each cell: scatter of (objective on y) vs (decode_accuracy on x) for
    THAT trial only. Per-cell annotation is the single Spearman rho for
    that trial's data, rendered upper-left as `ρ = value`.

Cross-trial summary statistics are the responsibility of
visualize_baseline_summary.py — this figure deliberately shows one trial
at a time so the visual and the annotation describe the same data.

Usage (from the repository root; experiments/synthetic_objective_correlation/run.sh
runs it on that experiment's output folder):
    PYTHONPATH=. python -m scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives \
        --indir results/experiments/synthetic_objective_correlation \
        [--debug-plots]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from duet.benchmark.baseline_objectives import OBJECTIVES, round_for_ranking
from duet.plotting import BASELINE_OBJECTIVE_PALETTE, FIGURE_WIDTHS, apply_style
from duet.plotting.tiers import PlotTiers, add_debug_plots_arg, count_skipped

logger = logging.getLogger(__name__)


# (noise_channel DataFrame value, column title) — the DataFrame stores the
# underscore form `position_varying_asymmetric`; the display version hyphenates
# "position-varying" and breaks the fourth column onto two lines for publication.
NOISE_COL_ORDER: List[Tuple[str, str]] = [
    ("symmetric", "symmetric"),
    ("position_varying", "position-varying"),
    ("asymmetric", "asymmetric"),
    ("position_varying_asymmetric", "position-varying,\nasymmetric"),
]

# Scatter visualizer ignores OBJECTIVES' `direction` field — only column
# names and display labels are needed here.
OBJECTIVE_ROWS: List[Tuple[str, str]] = [(col, label) for col, label, _ in OBJECTIVES]

# The paper panel is trial 1 (seed 42), picked by hand as the
# "representative trial" (trial_01.svg). It is written by default, the other
# trials only with --debug-plots. This visualizer takes no config
# (experiments/synthetic_objective_correlation/run.sh passes only --indir), so
# the panel is named here, relative to <indir>/figures/, instead of under the
# config's visualization.paper_panels.
PAPER_PANELS = frozenset({"baseline_comparison_per_trial/trial_01"})


def _cell_spearman(cell_df: pd.DataFrame, y_col: str, trial: int, noise_key: str) -> float:
    """Spearman rho for one (trial, noise) cell. Objective values are
    passed through `round_for_ranking` first so that codebooks whose
    objective is mathematically equal but differs in the last bits are
    ranked as the tie they are. Emits a warning and returns NaN if
    either column has fewer than 3 distinct values after rounding
    (matches the convention in visualize_baseline_summary.py).
    """
    x = cell_df["decode_accuracy"].to_numpy()
    y = round_for_ranking(cell_df[y_col].to_numpy())
    if np.unique(x).size < 3 or np.unique(y).size < 3:
        logger.warning(
            "Spearman emitted NaN for trial=%s, noise=%s, y=%s: "
            "fewer than 3 distinct values in x (%d) or y (%d).",
            trial, noise_key, y_col, np.unique(x).size, np.unique(y).size,
        )
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def render_figure(trial_df: pd.DataFrame, trial: int, out_path: Path) -> None:
    """Render one trial's 5x4 baseline-comparison SVG to `out_path`.

    `trial_df` must already be filtered to a single trial. `out_path.parent`
    must already exist — the caller owns directory creation. `apply_style()`
    must have been called before invoking this function (the caller does
    this once before looping rather than per-trial).

    sharey='row' places each objective on a single scale across noise
    channels; sharex='col' zooms each noise channel's decode_accuracy range
    independently.
    """
    n_rows = len(OBJECTIVE_ROWS)
    n_cols = len(NOISE_COL_ORDER)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGURE_WIDTHS["full_page"], 5.4),
        sharex="col",
        sharey="row",
    )

    for r, (y_col, row_title) in enumerate(OBJECTIVE_ROWS):
        is_duet_row = y_col == "duet_objective"
        # Keep the DUET row visually dominant: same palette but higher alpha
        # than the baselines, since DUET is the headline method.
        scatter_color = BASELINE_OBJECTIVE_PALETTE[y_col]
        scatter_alpha = 0.40 if is_duet_row else 0.25
        for c, (noise_key, noise_title) in enumerate(NOISE_COL_ORDER):
            ax = axes[r, c]
            cell_df = trial_df[trial_df["noise_channel"] == noise_key]
            ax.scatter(
                cell_df["decode_accuracy"], cell_df[y_col],
                s=4, alpha=scatter_alpha, color=scatter_color, edgecolors="none",
                rasterized=True,
            )
            rho = _cell_spearman(cell_df, y_col, trial, noise_key)
            rho_str = "NaN" if np.isnan(rho) else f"{rho:.3f}"
            ax.text(
                0.04, 0.96,
                f"ρ = {rho_str}",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8,
            )

            if r == 0:
                ax.set_title(noise_title, fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Decode accuracy")

    # Reserve left margin for the spelled-out row labels. Set explicit
    # margins so the labels render at predictable figure-coord positions
    # rather than relying on tight_layout to back-fit a placement
    # for axes-coord text that extends far left of the spines.
    fig.subplots_adjust(
        left=0.22, right=0.98, top=0.95, bottom=0.07,
        wspace=0.20, hspace=0.30,
    )

    # Row labels in figure coordinates, centered at the midpoint of the
    # left margin (x = 0.11 = left/2). ha="center" + ma="center" so
    # multi-line labels read with both lines center-aligned with each
    # other rather than left-justified.
    for r, (_, row_title) in enumerate(OBJECTIVE_ROWS):
        bbox = axes[r, 0].get_position()
        fig.text(
            0.11, (bbox.y0 + bbox.y1) / 2, row_title,
            va="center", ha="center",
            multialignment="center",
            fontsize=10,
        )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the DUET-vs-baseline-objectives experiment, one figure per trial.",
    )
    parser.add_argument(
        "--indir", required=True,
        help="Run directory containing the baseline-objectives parquet.",
    )
    add_debug_plots_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    indir = Path(args.indir)
    enum_path = indir / "objective_correlation.parquet"
    sampled_path = indir / "objective_correlation_sampled.parquet"
    partial_path = indir / "objective_correlation_sampled.partial.parquet"

    if enum_path.exists():
        df = pd.read_parquet(enum_path)
    elif sampled_path.exists():
        df = pd.read_parquet(sampled_path)
    elif partial_path.exists():
        df = pd.read_parquet(partial_path)
    else:
        raise FileNotFoundError(
            f"No baseline-objectives parquet found in {indir}. "
            "Expected one of: objective_correlation.parquet, "
            "objective_correlation_sampled.parquet, "
            "objective_correlation_sampled.partial.parquet."
        )

    figures_dir = indir / "figures"
    tiers = PlotTiers(figures_dir, args.debug_plots, PAPER_PANELS)
    per_trial_dir = figures_dir / "baseline_comparison_per_trial"
    out_paths = {
        int(trial): per_trial_dir / f"trial_{int(trial):02d}.svg"
        for trial in sorted(df["trial"].unique())
    }

    # apply_style() once before the loop — rcParams are global, so doing
    # this per-trial would be 20 redundant resets.
    apply_style()

    for trial, out_path in out_paths.items():
        if not tiers.writes_debug(out_path):
            continue
        per_trial_dir.mkdir(parents=True, exist_ok=True)
        render_figure(df[df["trial"] == trial], trial, out_path)
        print(f"Wrote {out_path}")
    if not args.debug_plots:
        print(f"Skipped {count_skipped(out_paths.values(), tiers)} per-trial debug "
              "plots (pass --debug-plots to write them).")
    tiers.report_missing_paper_panels()


if __name__ == "__main__":
    main()
