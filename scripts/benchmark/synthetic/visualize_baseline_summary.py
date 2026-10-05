#!/usr/bin/env python3
"""
Visualize the baseline-objectives summary.

Reads either `objective_correlation.parquet` (enumeration runner) or
`objective_correlation_sampled.parquet` (sampling runner) and renders
horizontal-boxplot summary SVGs:

- `summary_spearman.svg`: always emitted. 1 × N_noise grid. x = per-trial
  Spearman correlation between an objective and decode accuracy. All
  five objectives are max-direction (DUET objective = `1 - error_bound`
  per `compute_duet_objective`), so no sign flip is needed.
- `summary_regret.svg`: emitted only when the parquet is from the
  enumeration runner (detected by presence of `codebook_index`). x =
  argmax-regret = `decode_accuracy(optimum) - decode_accuracy(top by
  objective)`, with mean-over-ties when an objective has multiple
  argmax codebooks.

Figure margins (`left`, `right`, `wspace`) match the scatter visualizer
`visualize_duet_vs_baseline_objectives.py` so when the two figures are
stacked vertically in Inkscape, the noise-model columns line up.

Both figures are trial-aggregated, so this visualizer has no debug plots; it
accepts `--debug-plots` (with no effect) so every visualizer takes the flag.

Usage:
    python -m scripts.benchmark.synthetic.visualize_baseline_summary \
        --indir <run-directory>
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
from duet.plotting import (
    BASELINE_OBJECTIVE_PALETTE,
    FIGURE_WIDTHS,
    OKABE_ITO,
    apply_style,
)
from duet.plotting.tiers import add_debug_plots_arg

logger = logging.getLogger(__name__)


# (DataFrame value, display column title) — same order as the scatter
# visualizer for cross-figure consistency.
NOISE_COL_ORDER: List[Tuple[str, str]] = [
    ("symmetric", "symmetric"),
    ("position_varying", "position-varying"),
    ("asymmetric", "asymmetric"),
    ("position_varying_asymmetric", "position-varying,\nasymmetric"),
]


def compute_per_trial_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one Spearman per (trial, noise_channel, objective).

    All five objectives are max-direction, so the raw correlation is
    used as-is. Objective values are passed through `round_for_ranking`
    first: two codebooks whose objective is mathematically equal can
    differ in the last bits when the value is a sum of floats, and
    `spearmanr` would rank those pseudo-ties instead of averaging them.
    NaN is emitted (with a warning) when the objective or accuracy column
    has fewer than 3 distinct values within the group, counted after
    rounding.
    """
    out: List[dict] = []
    for (trial, noise_channel), group in df.groupby(["trial", "noise_channel"]):
        for obj_col, _label, _direction in OBJECTIVES:
            x = group["decode_accuracy"].to_numpy()
            y = round_for_ranking(group[obj_col].to_numpy())
            if np.unique(x).size < 3 or np.unique(y).size < 3:
                logger.warning(
                    "Spearman emitted NaN for trial=%s, noise=%s, objective=%s: "
                    "fewer than 3 distinct values in x (%d) or y (%d).",
                    trial, noise_channel, obj_col,
                    np.unique(x).size, np.unique(y).size,
                )
                rho = float("nan")
            else:
                rho, _ = spearmanr(x, y)
                rho = float(rho)
            out.append({
                "trial": trial,
                "noise_channel": noise_channel,
                "objective": obj_col,
                "spearman": rho,
            })
    return pd.DataFrame(out)


def _per_box_color(obj_col: str) -> str:
    """Each objective gets its dedicated palette color so the boxplot
    rows match the scatter rows in visualize_duet_vs_baseline_objectives."""
    return BASELINE_OBJECTIVE_PALETTE[obj_col]


def render_spearman_figure(df: pd.DataFrame, out_path: Path):
    """Render the 1 × N_noise Spearman summary SVG. Returns the figure
    object so tests can introspect the axes."""
    apply_style()
    spear = compute_per_trial_spearman(df)

    # Limit noise-model column set to those actually present.
    present_noise = [
        (key, title)
        for key, title in NOISE_COL_ORDER
        if key in df["noise_channel"].unique()
    ]
    n_cols = len(present_noise)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(FIGURE_WIDTHS["full_page"], 2.7),
        sharey=True,
        squeeze=False,
    )
    axes_row = axes[0]

    # Build per-objective box positions: DUET at top of the y-axis, going
    # down to min_pairwise_hamming at the bottom. Matplotlib's horizontal
    # boxplot lays positions on a numeric axis; we map objective index
    # (0..N-1) to position (N..1) so the visual top-to-bottom matches
    # the OBJECTIVES list order.
    n_obj = len(OBJECTIVES)
    positions = list(range(n_obj, 0, -1))

    for c, (noise_key, noise_title) in enumerate(present_noise):
        ax = axes_row[c]
        cell = spear[spear["noise_channel"] == noise_key]

        box_data = []
        box_colors = []
        for obj_col, _label, _direction in OBJECTIVES:
            obj_rows = cell[cell["objective"] == obj_col]["spearman"].dropna()
            box_data.append(obj_rows.to_numpy())
            box_colors.append(_per_box_color(obj_col))

        bp = ax.boxplot(
            box_data,
            positions=positions,
            vert=False,
            whis=(5, 95),
            patch_artist=True,
            widths=0.6,
            flierprops=dict(marker="o", markersize=2, alpha=0.5),
            medianprops=dict(color="black", linewidth=1.0),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("0.3")
        # Color every row's outlier markers to match its box face. boxplot
        # accepts only a single flierprops for all positions, so the per-row
        # color is applied after construction.
        for flier, color in zip(bp["fliers"], box_colors):
            flier.set(markerfacecolor=color, markeredgecolor=color)

        # "Optimal" reference at x = 1 (perfect rank agreement). Labelled
        # over every column so the meaning is unambiguous regardless of
        # which panel a reader's eye lands on first.
        ax.axvline(
            1.0, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1,
        )
        ax.text(
            1.0, n_obj + 0.5, "Optimal",
            color=OKABE_ITO["vermillion"],
            ha="center", va="bottom", fontsize=8,
        )

        ax.set_title(noise_title, fontsize=10)
        ax.set_xlabel("Spearman ρ")
        ax.set_xlim(0.0, 1.05)

    # Set y-tick labels on the shared y-axis. Atomic `labels=` form binds
    # the FixedLocator and FixedFormatter together so matplotlib's
    # default integer formatter does not replace them on draw.
    yticklabels = [label for _col, label, _direction in OBJECTIVES]
    axes_row[0].set_yticks(positions, labels=yticklabels, fontsize=10)

    # Match the scatter visualizer's left/right/wspace so the four
    # noise-model columns line up when the two figures stack vertically.
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.18, wspace=0.20)
    fig.savefig(out_path, dpi=300)
    return fig


def compute_argmax_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one mean-over-ties regret per (trial, noise_channel, objective).

    Argmax-regret = optimal_accuracy − accuracy_of_codebook_chosen_by_objective.
    When an objective has multiple argmax codebooks within a (trial,
    noise_channel) group, the regret is the mean of those tied codebooks'
    individual regrets — "expected regret under uniform tiebreak."

    Objective values are passed through `round_for_ranking` before the
    exact-equality comparison that finds the tie set: without it, two
    codebooks whose objective is mathematically equal but differs in the
    last bits are treated as distinct, under-reporting `n_tied` and
    returning one arbitrary member's regret instead of the mean.

    Returns:
        DataFrame with columns: trial, noise_channel, objective, regret,
        n_tied (the size of the tied argmax/argmin set).
    """
    out: List[dict] = []
    for (trial, noise_channel), group in df.groupby(["trial", "noise_channel"]):
        optimal_acc = group["decode_accuracy"].max()
        for obj_col, _label, direction in OBJECTIVES:
            obj_values = round_for_ranking(group[obj_col].to_numpy())
            if direction == "max":
                target = obj_values.max()
            else:
                target = obj_values.min()
            tied = group[obj_values == target]
            regret = (optimal_acc - tied["decode_accuracy"]).mean()
            out.append({
                "trial": trial,
                "noise_channel": noise_channel,
                "objective": obj_col,
                "regret": float(regret),
                "n_tied": int(len(tied)),
            })
    return pd.DataFrame(out)


def render_regret_figure(df: pd.DataFrame, out_path: Path):
    """Render the 1 × N_noise argmax-regret summary SVG. Returns the
    figure object so tests can introspect the axes."""
    apply_style()
    regret = compute_argmax_regret(df)

    present_noise = [
        (key, title)
        for key, title in NOISE_COL_ORDER
        if key in df["noise_channel"].unique()
    ]
    n_cols = len(present_noise)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(FIGURE_WIDTHS["full_page"], 2.7),
        sharey=True,
        squeeze=False,
    )
    axes_row = axes[0]

    n_obj = len(OBJECTIVES)
    positions = list(range(n_obj, 0, -1))

    for c, (noise_key, noise_title) in enumerate(present_noise):
        ax = axes_row[c]
        cell = regret[regret["noise_channel"] == noise_key]

        box_data = []
        box_colors = []
        for obj_col, _label, _direction in OBJECTIVES:
            obj_rows = cell[cell["objective"] == obj_col]["regret"].dropna()
            box_data.append(obj_rows.to_numpy())
            box_colors.append(_per_box_color(obj_col))

        bp = ax.boxplot(
            box_data,
            positions=positions,
            vert=False,
            whis=(5, 95),
            patch_artist=True,
            widths=0.6,
            flierprops=dict(marker="o", markersize=2, alpha=0.5),
            medianprops=dict(color="black", linewidth=1.0),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("0.3")
        for flier, color in zip(bp["fliers"], box_colors):
            flier.set(markerfacecolor=color, markeredgecolor=color)

        ax.axvline(
            0.0, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1,
        )
        ax.text(
            0.0, n_obj + 0.5, "Optimal",
            color=OKABE_ITO["vermillion"],
            ha="center", va="bottom", fontsize=8,
        )

        ax.set_title(noise_title, fontsize=10)
        ax.set_xlabel("Δ accuracy from optimum")
        # x-axis range left to matplotlib's autoscale so the boxes fill
        # the available horizontal space at any data scale.

    yticklabels = [label for _col, label, _direction in OBJECTIVES]
    axes_row[0].set_yticks(positions, labels=yticklabels, fontsize=10)

    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.18, wspace=0.20)
    fig.savefig(out_path, dpi=300)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the baseline-objectives summary boxplot SVGs.",
    )
    parser.add_argument(
        "--indir", required=True,
        help="Run directory containing the parquet.",
    )
    add_debug_plots_arg(
        parser,
        help="Accepted so every visualizer takes the same flag; this one has no "
             "debug plots and writes the same figures either way.",
    )
    args = parser.parse_args()

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

    figdir = indir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    spearman_path = figdir / "summary_spearman.svg"
    render_spearman_figure(df, spearman_path)
    print(f"Wrote {spearman_path}")

    # Regret figure: only when this is enumeration data (has codebook_index).
    if "codebook_index" in df.columns:
        regret_path = figdir / "summary_regret.svg"
        render_regret_figure(df, regret_path)
        print(f"Wrote {regret_path}")
    else:
        print("Skipping regret figure: parquet lacks `codebook_index` (sampled data).")


if __name__ == "__main__":
    main()
