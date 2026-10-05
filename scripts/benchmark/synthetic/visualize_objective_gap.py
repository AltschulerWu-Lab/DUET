#!/usr/bin/env python3
"""
Visualize the DUET objective-gap experiment.

Reads objective_gap.parquet from a run directory and renders one scatter
SVG per trial to <indir>/figures/trial_NN.svg. Each scatter plots, for every
enumerated codebook, its decode accuracy (x) against DUET's union-bound
decode objective (y). The accuracy-optimal and DUET-optimal codebooks are
marked with leader-line callouts; the DUET-optimal callout reports that
codebook's rank in the decode-accuracy ordering.

Usage:
    python -m scripts.benchmark.synthetic.visualize_objective_gap \
        --indir results/benchmark/05-18-2026/duet_objective_gap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from duet.plotting import FIGURE_WIDTHS, METHOD_PALETTE, OKABE_ITO, apply_style


def _annotate_optimum(ax, label: str, point, text_xy, color: str) -> None:
    """Draw a leader-line callout.

    Places `label` at `text_xy` (axes-fraction coordinates) and draws a thin
    line from the label to `point` (data coordinates). Replaces a legend:
    each marked codebook is identified in place rather than by a colour key.
    """
    ax.annotate(
        label,
        xy=point, xycoords="data",
        xytext=text_xy, textcoords="axes fraction",
        color=color, fontsize="small", va="center", ha="left",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8,
                        shrinkA=2, shrinkB=4),
        zorder=5,
    )


def plot_objective_gap_trial(ax, trial_df: pd.DataFrame, trial: int) -> None:
    """Render one trial's objective-gap scatter onto `ax`.

    Marks the accuracy-optimal and DUET-optimal codebooks with star markers,
    each identified by its own leader-line callout. The DUET-optimal callout
    reports that codebook's rank in the decode-accuracy ordering. When the
    same codebook is optimal for both objectives the two stars coincide, but
    both callouts are still drawn.
    """
    acc = trial_df["decode_accuracy"].to_numpy()
    obj = trial_df["duet_objective"].to_numpy()
    n_codebooks = len(acc)

    # All codebooks — small, low-alpha, neutral.
    ax.scatter(acc, obj, s=4, alpha=0.25, color="0.55",
               edgecolors="none", zorder=1)

    # y = x reference line — a visual reference only, not a bound.
    lo = float(min(acc.min(), obj.min()))
    hi = float(max(acc.max(), obj.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.6, color="0.7", zorder=0)

    acc_opt = int(np.argmax(acc))  # np.argmax picks the first on a tie
    obj_opt = int(np.argmax(obj))

    # Rank of the DUET-optimal codebook in the decode-accuracy ordering
    # (1 = most accurate). Strict-greater count, so ties take the best rank.
    duet_acc_rank = int((acc > acc[obj_opt]).sum()) + 1

    acc_color = OKABE_ITO["bluish_green"]
    duet_color = METHOD_PALETTE["DUET"]

    # Accuracy-optimal star + callout.
    ax.scatter(
        [acc[acc_opt]], [obj[acc_opt]],
        marker="*", s=60, color=acc_color,
        edgecolors="black", linewidths=0.4, zorder=4,
    )
    _annotate_optimum(
        ax, "Accuracy-optimal",
        point=(acc[acc_opt], obj[acc_opt]),
        text_xy=(0.05, 0.93),
        color=acc_color,
    )

    # DUET-optimal star + callout. If this is also the accuracy-optimal
    # codebook the two stars coincide, but both callouts are still drawn.
    ax.scatter(
        [acc[obj_opt]], [obj[obj_opt]],
        marker="*", s=60, color=duet_color,
        edgecolors="black", linewidths=0.4, zorder=4,
    )
    _annotate_optimum(
        ax,
        f"DUET-optimal\n"
        f"(decode-accuracy rank {duet_acc_rank} / {n_codebooks})",
        point=(acc[obj_opt], obj[obj_opt]),
        text_xy=(0.05, 0.75),
        color=duet_color,
    )

    ax.set_xlabel("Decode accuracy")
    ax.set_ylabel("DUET decode objective")
    ax.set_title(f"Trial {trial}")


def render_figures(df: pd.DataFrame, outdir: Path) -> None:
    """Render one objective-gap scatter SVG per trial into <outdir>/figures/."""
    figdir = Path(outdir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    apply_style()

    for trial, trial_df in df.groupby("trial"):
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTHS["half_page"], 3.2),
        )
        plot_objective_gap_trial(ax, trial_df, int(trial))
        fig.tight_layout()
        fig.savefig(figdir / f"trial_{int(trial):02d}.svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the DUET objective-gap experiment.",
    )
    parser.add_argument(
        "--indir", required=True,
        help="Run directory containing objective_gap.parquet.",
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    df = pd.read_parquet(indir / "objective_gap.parquet")
    render_figures(df, indir)
    n_trials = df["trial"].nunique()
    print(f"Wrote {n_trials} figures to {indir / 'figures'}")


if __name__ == "__main__":
    main()
