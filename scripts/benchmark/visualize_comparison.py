#!/usr/bin/env python3
"""
Generate comparison plots from aggregated_metrics.csv produced by compare_benchmarks.py.

This is the plotting half of the compare_benchmarks.py / visualize_comparison.py
pipeline. Run compare_benchmarks.py first to produce aggregated_metrics.csv, then
run this script with the same YAML config to (re)generate plots without paying the
re-evaluation cost.

By default it writes the trial-aggregated figures: the hypervolume bar chart
and one Pareto panel per metric. The per-trial copies of both are debug plots,
written only with --debug-plots, or when listed under the config's
`visualization: {paper_panels: [...]}` (see duet.plotting.tiers).

Usage:
    python visualize_comparison.py --config /path/to/compare_config.yaml
    python visualize_comparison.py --config /path/to/compare_config.yaml --debug-plots
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from comparison import ComparisonConfig, MethodSource
from duet.benchmark.metrics import compare_pareto_fronts
from duet.plotting import (
    FIGURE_WIDTHS,
    METHOD_PALETTE,
    MM_PER_INCH,
    OKABE_ITO,
    apply_style,
    fit_last_xtick_label,
    metric_axis_label,
    save_panel,
    slot_figure,
)
from duet.plotting.legend import (
    LEGEND_HANDLER_MAP,
    SINGLE_TRIAL_MARKERSIZE,
    NotAvailable,
    square_legend_handles,
)
from duet.plotting.tiers import DEBUG_PLOTS_FLAG, PlotTiers, add_debug_plots_arg


# =============================================================================
# Constants
# =============================================================================

# Static mapping for named baselines (substring-matched in generate_palette).
BASE_PALETTE = {
    "Feldman et al.":    METHOD_PALETTE["Feldman et al."],
    "Sivanandan et al.": METHOD_PALETTE["Sivanandan et al."],
    "Maximum activity":  METHOD_PALETTE["Maximum activity"],
}

# Cycle for DUET variants. First entry is the canonical DUET blue from
# METHOD_PALETTE; remaining entries are Okabe-Ito hues not used by BASE_PALETTE.
# Order puts the most dissimilar hues first so 2- and 3-variant comparisons are
# legible without effort (sky_blue lands at position 5 to avoid blue-on-blue).
DUET_COLORS = [
    METHOD_PALETTE["DUET"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["yellow"],
    OKABE_ITO["sky_blue"],
]

# Character width at which Pareto-panel legend labels are wrapped onto
# additional lines. Pareto panels are FIGURE_WIDTHS["third_page"] (60 mm) with
# a `fontsize="small"` legend, which leaves ~1.8 in of axes width. Unwrapped
# comparison labels ("DUET (Position-varying asymmetric error)") render ~2.1 in
# wide — wider than the axes — and constrained_layout charges that overflow to
# the margin, shrinking the axes to ~1.3 in and clipping the legend at the
# canvas edge. Wrapping at 24 keeps the legend inside the axes on every panel.
#
# Applied to the legend text only: `label_order` and the palette keys keep the
# unwrapped labels, so the Label column in aggregated_metrics.csv and the
# hypervolume barplot's category axis are unaffected.
LEGEND_WRAP_WIDTH = 24

# Hypervolume bar chart: horizontal bars, one row per label, with the full
# config labels read horizontally on the y axis. Labels like "DUET
# (Position-varying asymmetric error)" are ~46 mm wide at the tick font, so the
# chart is half_page wide; its left margin is measured from the labels at draw
# time. The top margin holds the title, set over the whole figure because it
# is wider than the bars.
HV_BAR_ROW_MM = 5.0
HV_BAR_MARGINS_MM = (2.0, 10.0, 7.0)  # right, bottom, top
# Narrowest bar area the chart keeps. Labels too long for half_page widen the
# figure rather than squeeze the bars below this: 30 mm holds the centred
# "Normalized Hypervolume" x label (~32 mm) within the 2 mm right margin. The
# real S4b and S4c labels (the same wording since 2026-10-04) leave 33 mm, so
# neither is widened.
HV_BAR_MIN_WIDTH_MM = 30.0

METRICS = [
    ("Mean decode accuracy", "Mean activity score"),
    ("Standard deviation decode accuracy", "Mean activity score"),
    ("5th percentile decode accuracy", "Mean activity score"),
    ("10th percentile decode accuracy", "Mean activity score"),
    ("95th over 5th percentile decode accuracy", "Mean activity score"),
    ("Mean decode accuracy (<= 5th percentile)", "Mean activity score"),
    ("Mean decode accuracy (<= 10th percentile)", "Mean activity score"),
]


# =============================================================================
# Helpers
# =============================================================================


def _compute_se(x: pd.Series) -> float:
    """Compute standard error, returning 0 for single observations."""
    n = len(x)
    if n <= 1:
        return 0.0
    return x.std() / np.sqrt(n)


def _savefig_at_figsize(output_path: Path) -> None:
    """Save the current figure at exactly its figsize.

    The project mplstyle sets `savefig.bbox: tight`, which re-expands the
    saved canvas to fit all artists even when constrained_layout keeps
    content inside figsize. The documented escape hatch is to flip the
    rcParam to `"standard"` (validates to None) for the savefig call.
    """
    with plt.rc_context({"savefig.bbox": "standard"}):
        plt.savefig(output_path)


def _set_pareto_title(fig, title: str) -> None:
    """Set a Pareto panel's title over the whole figure, at the axes-title size.

    The title used to be "Aggregated Across Trials (± SE): Mean decode
    accuracy vs Mean activity score", 112 mm at 9 pt, about twice the
    57.33 mm slot, so it ran past both edges (S4b, S4c). Its "x vs y" half
    repeats the two axis labels word for word, so it is dropped; wrapping it
    instead took three title lines, and the shorter axes left the S4b legend
    on top of the fronts. Centred on the axes, the remaining ~45 mm still ran
    past the right edge whenever wide y tick labels pushed the axes right;
    centred on the figure it has about 6 mm either side. Constrained layout
    keeps room for it above the axes.
    """
    fig.suptitle(title, fontsize=plt.rcParams["axes.titlesize"])


def _fit_left_margin(
    fig: plt.Figure, ax: plt.Axes, min_width_mm: float, pad_mm: float = 1.0
) -> bool:
    """Move the axes' left edge so the y tick labels and y label just fit.

    The right edge stays put; the plot area gets whatever width is left. If
    that is under ``min_width_mm``, the figure is widened instead, just enough
    to leave ``min_width_mm``; the labels are never shortened. Returns whether
    the figure was widened.
    """
    renderer = fig.canvas.get_renderer()
    overhang_px = ax.get_window_extent(renderer).x0 - ax.get_tightbbox(renderer).x0
    fig_width_px = fig.get_figwidth() * fig.dpi
    fig_width_mm = fig.get_figwidth() * MM_PER_INCH
    left = overhang_px / fig_width_px + pad_mm / fig_width_mm
    pos = ax.get_position()
    if (pos.x1 - left) * fig_width_mm >= min_width_mm:
        ax.set_position([left, pos.y0, pos.x1 - left, pos.height])
        return False

    # Text is sized in points, so both margins keep their mm in a wider figure.
    left_mm = left * fig_width_mm
    right_mm = (1 - pos.x1) * fig_width_mm
    width_mm = left_mm + min_width_mm + right_mm
    fig.set_figwidth(width_mm / MM_PER_INCH)
    ax.set_position([left_mm / width_mm, pos.y0, min_width_mm / width_mm, pos.height])
    return True


def generate_palette(methods: List[MethodSource]) -> Dict[str, str]:
    """Generate color palette for all methods.

    Each MethodSource may set an explicit `color` (matplotlib-parseable
    string) which wins. For methods without an override, the existing
    behavior applies: BASE_PALETTE substring match for known baselines,
    then DUET_COLORS cycle for everything else (in input order).

    Args:
        methods: List of MethodSource configurations from
            ComparisonConfig.methods.

    Returns:
        Dict mapping label to color.
    """
    palette: Dict[str, str] = {}
    duet_idx = 0

    for m in methods:
        if m.color is not None:
            palette[m.label] = m.color
            continue

        # Existing logic: BASE_PALETTE substring match → DUET_COLORS cycle
        matched = False
        for base_label, color in BASE_PALETTE.items():
            if base_label.lower() in m.label.lower():
                palette[m.label] = color
                matched = True
                break

        if not matched:
            palette[m.label] = DUET_COLORS[duet_idx % len(DUET_COLORS)]
            duet_idx += 1

    return palette


def sanitize_filename(s: str) -> str:
    """Convert a string to a safe filename."""
    return (
        s.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "")
        .replace("<", "lt")
        .replace("%", "pct")
        .replace("/", "_")
    )


# =============================================================================
# Plotting Functions (copied verbatim from compare_benchmarks.py)
# =============================================================================


def plot_pareto_single_trial(
    agg_df: pd.DataFrame,
    trial: int,
    x_col: str,
    y_col: str,
    palette: Dict[str, str],
    label_order: List[str],
    output_path: Path,
) -> None:
    """Plot Pareto fronts for a single trial.

    Mirrors ``plot_pareto_aggregated``'s styling — a Maximum-activity-anchored
    connector line plus small markers — so single-trial panels match the
    aggregated frontier panels. A single trial carries no across-trial spread,
    so error bars are omitted (markers only).
    """
    df_t = agg_df[agg_df["Trial"] == trial]

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
        constrained_layout=True,
    )

    # Anchor connector lines at the per-trial point whose underlying Method is
    # "Maximum activity" (folded into Feldman / Sivanandan labels by the regex
    # array). Falls back to sort-by-y within label (no anchor prepend) if absent.
    ma_mask = df_t["Method"] == "Maximum activity"
    ma_x = df_t.loc[ma_mask, x_col].iloc[0] if ma_mask.any() else None
    ma_y = df_t.loc[ma_mask, y_col].iloc[0] if ma_mask.any() else None

    for label in label_order:
        if label not in df_t["Label"].values:
            continue
        label_df = df_t[df_t["Label"] == label]
        color = palette.get(label, "#333333")

        sorted_df = label_df.sort_values(y_col, ascending=False)
        if ma_x is not None:
            line_x = [ma_x, *sorted_df[x_col].tolist()]
            line_y = [ma_y, *sorted_df[y_col].tolist()]
        else:
            line_x = sorted_df[x_col].tolist()
            line_y = sorted_df[y_col].tolist()
        ax.plot(line_x, line_y, color=color, linewidth=0.8, alpha=0.4, zorder=1)

        ax.errorbar(
            label_df[x_col],
            label_df[y_col],
            fmt="o",
            color=color,
            label=textwrap.fill(label, width=LEGEND_WRAP_WIDTH),
            alpha=0.7,
            markersize=1.5,
            capsize=0,
        )

    ax.set_xlabel(metric_axis_label(x_col))
    ax.set_ylabel(metric_axis_label(y_col))
    _set_pareto_title(fig, f"Trial {trial}")
    ax.legend(title="Method", loc="best", fontsize="small")

    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_pareto_aggregated(
    agg_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    palette: Dict[str, str],
    label_order: List[str],
    output_path: Path,
) -> None:
    """Plot aggregated Pareto fronts with SE bars.

    With a single trial (the genome-wide S4c) there is no spread to show, so
    each label's points are drawn as larger plain markers and the legend uses
    solid squares. A label with nothing to plot (absent from the data, or no
    finite point, like the 95th/5th ratio when the 5th percentile is 0) keeps
    its legend row, with "N/A" in place of the marker.
    """
    single_trial = agg_df["Trial"].nunique() == 1
    stats = (
        agg_df.groupby(["Label", "Method"])
        .agg(
            **{
                f"{x_col} mean": (x_col, "mean"),
                f"{x_col} se": (x_col, _compute_se),
                f"{y_col} mean": (y_col, "mean"),
                f"{y_col} se": (y_col, _compute_se),
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
        constrained_layout=True,
    )

    # Anchor connector lines at the per-label point whose underlying Method
    # is "Maximum activity". Comparison configs typically fold those rows
    # into the Feldman / Sivanandan labels via the regex array, so the
    # anchor coordinates are taken from any matching row. Falls back to
    # sort-by-x within label (no anchor prepend) if absent.
    ma_mask = stats["Method"] == "Maximum activity"
    ma_x = stats.loc[ma_mask, f"{x_col} mean"].iloc[0] if ma_mask.any() else None
    ma_y = stats.loc[ma_mask, f"{y_col} mean"].iloc[0] if ma_mask.any() else None

    # Per label in label_order: its color, or None when nothing is plotted
    # ("N/A" in the legend). Multi-trial panels keep each errorbar container
    # as its legend handle.
    legend_colors = []
    errorbar_handles = []
    for label in label_order:
        label_df = stats[stats["Label"] == label]
        finite = np.isfinite(label_df[f"{x_col} mean"]) & np.isfinite(label_df[f"{y_col} mean"])
        if not finite.any():
            legend_colors.append(None)
            errorbar_handles.append(NotAvailable())
            continue
        color = palette.get(label, "#333333")
        legend_colors.append(color)

        sorted_df = label_df.sort_values(f"{y_col} mean", ascending=False)
        if ma_x is not None:
            line_x = [ma_x, *sorted_df[f"{x_col} mean"].tolist()]
            line_y = [ma_y, *sorted_df[f"{y_col} mean"].tolist()]
        else:
            line_x = sorted_df[f"{x_col} mean"].tolist()
            line_y = sorted_df[f"{y_col} mean"].tolist()
        ax.plot(line_x, line_y, color=color, linewidth=0.8, alpha=0.4, zorder=1)

        if single_trial:
            ax.plot(
                label_df[f"{x_col} mean"],
                label_df[f"{y_col} mean"],
                linestyle="none",
                marker="o",
                color=color,
                alpha=0.7,
                markersize=SINGLE_TRIAL_MARKERSIZE,
                markeredgewidth=0,
            )
            continue

        errorbar_handles.append(ax.errorbar(
            label_df[f"{x_col} mean"],
            label_df[f"{y_col} mean"],
            xerr=label_df[f"{x_col} se"],
            yerr=label_df[f"{y_col} se"],
            fmt="o",
            color=color,
            label=textwrap.fill(label, width=LEGEND_WRAP_WIDTH),
            alpha=0.7,
            markersize=1.5,
            capsize=0,
        ))

    ax.set_xlabel(metric_axis_label(x_col))
    ax.set_ylabel(metric_axis_label(y_col))
    _set_pareto_title(fig, "Aggregated Across Trials (± SE)")
    legend_labels = [textwrap.fill(label, width=LEGEND_WRAP_WIDTH) for label in label_order]
    if single_trial:
        # Squares in the legend only; the points stay circles.
        legend_handles, _ = square_legend_handles(
            list(zip(legend_labels, legend_colors)), markersize=SINGLE_TRIAL_MARKERSIZE
        )
    else:
        legend_handles = errorbar_handles
    ax.legend(
        legend_handles,
        legend_labels,
        title="Method",
        loc="best",
        fontsize="small",
        handler_map=LEGEND_HANDLER_MAP,
    )

    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_hypervolume_single_trial(
    agg_df: pd.DataFrame,
    trial: int,
    palette: Dict[str, str],
    label_order: List[str],
    output_path: Path,
) -> None:
    """Plot Pareto fronts with hypervolume polygons for a single trial."""
    df_t = agg_df[agg_df["Trial"] == trial]
    labels = [label for label in label_order if label in df_t["Label"].values]

    fronts = []
    for label in labels:
        front_data = df_t[df_t["Label"] == label][
            ["Mean decode accuracy", "Mean activity score"]
        ].to_numpy()
        fronts.append(front_data)

    results = compare_pareto_fronts(fronts, maximize=True)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["third_page"], 2.5))

    for label in labels:
        label_df = df_t[df_t["Label"] == label]
        color = palette.get(label, "#333333")

        ax.scatter(
            label_df["Mean decode accuracy"],
            label_df["Mean activity score"],
            c=color,
            s=50,
            alpha=0.3,
        )

    ref_point = results.reference_point

    for i, label in enumerate(labels):
        color = palette.get(label, "#333333")
        front = results.fronts[i].pareto_front
        front = front[front[:, 0].argsort()]

        polygon_vertices = []
        polygon_vertices.append([ref_point[0], front[0, 1]])

        for j in range(len(front) - 1):
            polygon_vertices.append(front[j])
            polygon_vertices.append([front[j, 0], front[j + 1, 1]])

        polygon_vertices.append(front[-1])
        polygon_vertices.append([front[-1, 0], ref_point[1]])
        polygon_vertices.append(ref_point)

        polygon = patches.Polygon(
            np.array(polygon_vertices),
            closed=True,
            alpha=0.2,
            facecolor=color,
        )
        ax.add_patch(polygon)

        ax.plot(
            front[:, 0],
            front[:, 1],
            marker="o",
            linestyle="--",
            color=color,
            label=f"{label} (HV={results.fronts[i].normalized_hv:.3f})",
            ms=8,
            mew=1.5,
            mfc="none",
        )

    ax.plot(
        ref_point[0],
        ref_point[1],
        "D",
        markerfacecolor="none",
        markeredgecolor="#404040",
        ms=7,
        mew=1,
        label="Reference Point",
    )

    ax.set_xlabel("Mean decode accuracy")
    ax.set_ylabel("Mean activity score")
    ax.set_title(f"Pareto Fronts with Hypervolume (Trial {trial})")
    ax.legend(title="Method", loc="best", fontsize="x-small")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hypervolume_barplot(
    agg_df: pd.DataFrame,
    palette: Dict[str, str],
    label_order: List[str],
    output_path: Path,
) -> None:
    """Plot normalized hypervolume bar plot summarized across trials.

    Horizontal bars, one row per label in label_order from top to bottom, so
    the full labels read horizontally. The figure grows by HV_BAR_ROW_MM per
    label, and past half_page only if the labels would leave the bars less
    than HV_BAR_MIN_WIDTH_MM. Skipped, with a note, if no trial has two labels
    of label_order to compare (none at all if the config's labels match
    nothing in the data).
    """
    labels = [label for label in label_order if label in agg_df["Label"].values]
    trials = sorted(agg_df["Trial"].unique())

    hv_data = defaultdict(list)

    for trial in trials:
        df_t = agg_df[agg_df["Trial"] == trial]

        fronts = []
        present_labels = []
        for label in labels:
            label_df = df_t[df_t["Label"] == label]
            if label_df.empty:
                continue
            front_data = label_df[
                ["Mean decode accuracy", "Mean activity score"]
            ].to_numpy()
            fronts.append(front_data)
            present_labels.append(label)

        if len(fronts) < 2:
            continue

        results = compare_pareto_fronts(fronts, maximize=True)

        for i, label in enumerate(present_labels):
            hv_data["Trial"].append(trial)
            hv_data["Label"].append(label)
            hv_data["Normalized Hypervolume"].append(results.fronts[i].normalized_hv)

    hv_df = pd.DataFrame(hv_data)
    if hv_df.empty:
        print(f"  Note: skipped {output_path.name}: no trial has two of the config's labels.")
        return

    right_mm, bottom_mm, top_mm = HV_BAR_MARGINS_MM
    height_mm = bottom_mm + top_mm + HV_BAR_ROW_MM * len(labels)
    fig, ax = slot_figure(
        FIGURE_WIDTHS["half_page"] * MM_PER_INCH,
        height_mm,
        # The left margin is a placeholder until _fit_left_margin.
        margins_mm=(right_mm, right_mm, bottom_mm, top_mm),
    )

    # Horizontal bars; seaborn puts the first label of `order` at the top.
    sns.barplot(
        data=hv_df,
        x="Normalized Hypervolume",
        y="Label",
        hue="Label",
        palette=palette,
        width=0.5,
        order=labels,
        orient="h",
        legend=False,
        errorbar="se",
        ax=ax,
    )

    fig.suptitle(
        "Normalized Hypervolume by Method (Across All Trials)",
        y=1 - 1.0 / height_mm,  # 1 mm below the top edge at any height
        fontsize=plt.rcParams["axes.titlesize"],
    )
    ax.set_xlabel("Normalized Hypervolume")
    ax.set_ylabel("Method")
    if _fit_left_margin(fig, ax, HV_BAR_MIN_WIDTH_MM):
        print(
            f"  Note: widened {output_path.name} to "
            f"{fig.get_figwidth() * MM_PER_INCH:.0f} mm so its labels fit."
        )
    fit_last_xtick_label(fig, ax)
    save_panel(fig, output_path)
    plt.close(fig)


# =============================================================================
# CLI / main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from aggregated_metrics.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the comparison YAML config (same one used for compare_benchmarks.py).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    add_debug_plots_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress matplotlib's per-font DEBUG output under -vv (not our signal).
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    config = ComparisonConfig.from_yaml(Path(args.config))
    # ComparisonConfig ignores `visualization`; read the paper-panel list raw.
    with open(args.config) as f:
        tiers = PlotTiers.from_config(config.outdir, args.debug_plots, yaml.safe_load(f))

    agg_csv = config.outdir / "aggregated_metrics.csv"
    print(f"Reading aggregated_metrics.csv from {config.outdir}")
    if not agg_csv.exists():
        raise FileNotFoundError(
            f"{agg_csv} not found. Run "
            f"'python compare_benchmarks.py --config {args.config}' first to generate it."
        )
    agg_df = pd.read_csv(agg_csv)

    pareto_dir = config.outdir / "pareto_fronts"
    hypervolume_dir = config.outdir / "hypervolume"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    hypervolume_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    label_order = [method.label for method in config.methods]
    palette = generate_palette(config.methods)
    print(f"Color palette: {palette}")

    trials = sorted(agg_df["Trial"].unique())
    # Per-trial figures are debug plots, written only when tiers allows.
    skipped = 0

    # --- Hypervolume plots ---
    print("\nGenerating hypervolume plots...")

    print("  Barplot...")
    plot_hypervolume_barplot(
        agg_df, palette, label_order, hypervolume_dir / "hypervolume_barplot.svg"
    )

    for trial in trials:
        trial_path = hypervolume_dir / f"hypervolume_trial_{trial}.svg"
        if not tiers.writes_debug(trial_path):
            skipped += 1
            continue
        print(f"  Trial {trial}...")
        plot_hypervolume_single_trial(agg_df, trial, palette, label_order, trial_path)

    # --- Pareto front plots ---
    print("\nGenerating Pareto front plots...")

    for x_col, y_col in METRICS:
        metric_name = sanitize_filename(x_col)
        print(f"  {x_col}...")

        for trial in trials:
            trial_path = pareto_dir / f"{metric_name}_trial_{trial}.svg"
            if not tiers.writes_debug(trial_path):
                skipped += 1
                continue
            plot_pareto_single_trial(
                agg_df, trial, x_col, y_col, palette, label_order, trial_path,
            )

        plot_pareto_aggregated(
            agg_df, x_col, y_col, palette, label_order,
            pareto_dir / f"{metric_name}_aggregated.svg",
        )

    print(f"\nAll plots saved under {config.outdir}")
    if skipped:
        print(f"Skipped {skipped} per-trial debug plots; pass {DEBUG_PLOTS_FLAG} to write them.")
    tiers.report_missing_paper_panels()


if __name__ == "__main__":
    main()
