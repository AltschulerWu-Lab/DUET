#!/usr/bin/env python3
"""
Benchmark visualization script for DUET Pareto optimization results.

Generates publication-quality figures from benchmark results.csv files.

Usage:
    # Using results.csv directly:
    python visualize_benchmark.py --input /path/to/results.csv --output-dir /path/to/output

    # Using the same config file as run_benchmark.py:
    python visualize_benchmark.py --config /path/to/benchmark_config.yaml [--output-dir /path/to/output]

    # Either form, also writing the debug plots:
    python visualize_benchmark.py --config /path/to/benchmark_config.yaml --debug-plots

Outputs (tiers as in duet.plotting.tiers):
    Always: hypervolume/normalized_hypervolume_by_method_group.svg and
    pareto_fronts/<metric>_aggregated.svg for each metric.
    Only with --debug-plots: every per-trial figure (pareto_fronts/<metric>_trial_<k>,
    hypervolume_visualization/, guide_level_comparisons/, error_metrics/) and
    hypervolume_visualization/pareto_fronts_aggregated.svg. A per-trial file
    that the --config YAML lists under visualization.paper_panels is written
    by default too, at its usual path.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import yaml

from duet.benchmark.metrics import compare_pareto_fronts
from duet.plotting import (
    ERROR_CATEGORY_PALETTE,
    FIGURE_WIDTHS,
    METHOD_PALETTE,
    MM_PER_INCH,
    OVERLAY_VARIANT_PALETTE,
    apply_style,
    fit_last_xtick_label,
    fit_xlabel,
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
from duet.visualization import (
    plot_activity_scores,
    plot_codeword_probabilities,
    plot_dual_objectives,
)


# =============================================================================
# Constants
# =============================================================================

METHOD_GROUPS = ["DUET", "Feldman et al.", "Sivanandan et al."]

# METHOD_GROUPS plus the Maximum-activity anchor, in the order panels draw them.
# Iterate this rather than METHOD_PALETTE, which also carries MERFISH and
# synthetic-benchmark methods that never appear in an OPS panel.
PLOTTED_METHOD_GROUPS = ["Maximum activity", *METHOD_GROUPS]

METRICS = [
    ("Mean decode accuracy", "Mean activity score"),
    ("Standard deviation decode accuracy", "Mean activity score"),
    ("5th percentile decode accuracy", "Mean activity score"),
    ("10th percentile decode accuracy", "Mean activity score"),
    ("95th over 5th percentile decode accuracy", "Mean activity score"),
    ("Mean decode accuracy (<= 5th percentile)", "Mean activity score"),
    ("Mean decode accuracy (<= 10th percentile)", "Mean activity score"),
]

MAX_ACTIVITY_METHOD = "Maximum activity"

# Activity-fraction reference policies for the guide-level comparisons, as
# {filename suffix: fraction of the maximum achievable mean activity score}.
# Each fraction resolves to ONE DUET lambda per trial, shared by the
# max-activity panel and every baseline panel at that fraction, so all
# baselines in a fraction group are compared against the same DUET codebook.
# Insertion order determines the order panels are emitted and logged.
DUET_REFERENCE_FRACTIONS = {"97p5pct": 0.975, "95pct": 0.95}

# The three files plot_comparison writes per comparison, as filename suffixes.
COMPARISON_KINDS = ("decode_accuracy", "activity_score", "dual_objectives")

# Guide-level distribution stack (Figure 3 density panels).
#
# Width reserved to the LEFT of the shared decode-accuracy range for the bin
# of guides at exactly 0, as a fraction of that range. The slot is drawn only
# on panels where some plotted method has such guides, so the continuous range
# is the same width on every panel that lacks one.
ZERO_SLOT_FRACTION = 0.12
# Percentiles drawn as the interval strip under each density: (low, mid, high).
STRIP_PERCENTILES = (5, 50, 95)
# Stack heights in inches at the third-page width. Chosen 2026-09-08 from
# 100/90/80% trials against the assembled Figure 3: 90% of the original 2.8 and
# 3.0. At 80% the overlay's five-entry legend runs into the zero-bin label.
STACK_HEIGHT_PAIR = 2.52
STACK_HEIGHT_OVERLAY = 2.70

# A Sivanandan et al. method label as the OPS runner writes it, e.g.
# "Sivanandan et al. (ED=2)"; display_method_label shows it as HD.
_SIVANANDAN_ED = re.compile(r"^Sivanandan et al\. \(ED=(\d+)\)")


# =============================================================================
# Data Processing
# =============================================================================


def map_method_group(method: str) -> str:
    """Map a method string to its group."""
    if method.startswith("DUET"):
        return "DUET"
    if method.startswith("Sivanandan"):
        return "Sivanandan et al."
    if method.startswith("Feldman"):
        return "Feldman et al."
    if method.startswith("Maximum activity"):
        return "Maximum activity"
    raise ValueError(f"Unrecognized method label: {method}")


def display_method_label(method: str) -> str:
    """Method label as it should read in a figure.

    results.csv spells the DUET scalarization weight as the ASCII word "lambda"
    (``DUET (lambda=0.90)``); runs before 2026-08 spelled it "alpha". Figures
    show the symbol either way. Sivanandan et al. enforce a Hamming-distance
    floor, not an edit distance, so the runner's ``Sivanandan et al. (ED=1)``
    reads ``Sivanandan et al. (HD = 1)``, as in the Methods and the S2 caption.
    Feldman et al. do use edit distance and keep ``(ED=1)``. Applied at render
    time only — the dataframe keeps the ASCII labels, which every method
    lookup, palette key, and output filename still keys on.
    """
    method = _SIVANANDAN_ED.sub(r"Sivanandan et al. (HD = \1)", method)
    return method.replace("lambda=", "λ=").replace("alpha=", "λ=")


def load_and_process_results(results_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load results.csv and compute aggregated metrics.

    Returns:
        Tuple of (results_df, agg_results_df)
    """
    results_df = pd.read_csv(results_csv)

    # Filter out invalid solutions if the column exists
    if "Valid" in results_df.columns:
        n_invalid = (~results_df["Valid"]).sum()
        if n_invalid > 0:
            print(f"Warning: Dropping {n_invalid} invalid rows (wrong number of guides).")
            results_df = results_df[results_df["Valid"]].copy()

    # Add method group column
    results_df["Method group"] = results_df["Method"].apply(map_method_group)

    # Compute aggregated metrics per method/trial
    agg_results_df = (
        results_df.groupby(["Method", "Method group", "Trial"])
        .agg(
            **{
                "Mean decode accuracy": ("Decode accuracy", "mean"),
                "Standard deviation decode accuracy": ("Decode accuracy", "std"),
                "Mean activity score": ("Activity score", "mean"),
                "95th percentile decode accuracy": (
                    "Decode accuracy",
                    lambda x: x.quantile(0.95),
                ),
                "5th percentile decode accuracy": (
                    "Decode accuracy",
                    lambda x: x.quantile(0.05),
                ),
                "Mean decode accuracy (<= 5th percentile)": (
                    "Decode accuracy",
                    lambda x: x[x <= x.quantile(0.05)].mean(),
                ),
                "10th percentile decode accuracy": (
                    "Decode accuracy",
                    lambda x: x.quantile(0.1),
                ),
                "Mean decode accuracy (<= 10th percentile)": (
                    "Decode accuracy",
                    lambda x: x[x <= x.quantile(0.1)].mean(),
                ),
            }
        )
        .reset_index()
    )

    agg_results_df["95th over 5th percentile decode accuracy"] = (
        agg_results_df["95th percentile decode accuracy"]
        / agg_results_df["5th percentile decode accuracy"]
    )

    return results_df, agg_results_df


def compute_hypervolume_df(agg_results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hypervolume metrics for each trial and method group."""
    hv_data = defaultdict(list)

    for trial in agg_results_df["Trial"].unique():
        fronts = []
        for g in METHOD_GROUPS:
            front_data = agg_results_df.loc[
                (agg_results_df["Method group"].isin([g, "Maximum activity"]))
                & (agg_results_df["Trial"] == trial)
            ][["Mean decode accuracy", "Mean activity score"]].to_numpy()
            fronts.append(front_data)

        results = compare_pareto_fronts(fronts, maximize=True)

        for i, g in enumerate(METHOD_GROUPS):
            hv_data["Trial"].append(trial)
            hv_data["Method group"].append(g)
            hv_data["Normalized Hypervolume"].append(results.fronts[i].normalized_hv)

    return pd.DataFrame(hv_data)


def select_duet_lambda(
    agg_results_df: pd.DataFrame,
    trial: int,
    min_activity: float,
) -> str | None:
    """Select DUET lambda with highest decode accuracy above activity threshold.

    Args:
        agg_results_df: Aggregated results dataframe.
        trial: Trial number.
        min_activity: Minimum mean activity score threshold.

    Returns:
        Method string for selected DUET lambda, or None if no valid lambda found.
    """
    duet_df = agg_results_df.loc[
        (agg_results_df["Method group"] == "DUET")
        & (agg_results_df["Trial"] == trial)
        & (agg_results_df["Mean activity score"] >= min_activity)
    ]

    if duet_df.empty:
        return None

    best_idx = duet_df["Mean decode accuracy"].idxmax()
    return duet_df.loc[best_idx, "Method"]


@dataclass(frozen=True)
class ComparisonSpec:
    """One baseline-vs-DUET panel: which two methods, and under what policy."""

    baseline: str  # method label of the baseline arm
    duet: str  # method label of the DUET arm
    prefix: str  # filename prefix
    policy: str  # "matched", or a key of DUET_REFERENCE_FRACTIONS


def build_comparison_specs(
    agg_results_df: pd.DataFrame,
    trial: int,
    feldman_methods: List[str],
    sivanandan_methods: List[str],
) -> List[ComparisonSpec]:
    """Every baseline-vs-DUET comparison to render for one trial.

    Three reference policies:

    - ``matched``: DUET with activity >= the baseline's own activity. Each
      baseline gets its own DUET arm.
    - one per entry in DUET_REFERENCE_FRACTIONS: DUET with activity >= that
      fraction of the maximum achievable activity. Every panel in the group
      shares one DUET arm, so the baselines are compared against a common
      codebook rather than against a different one each.

    ``Maximum activity`` takes the fraction policies but not ``matched``:
    "DUET with activity >= max activity" can only resolve to lambda=0.00, which
    *is* the activity-only solution, so that panel would compare
    ``Maximum activity`` against a near-copy of itself.

    Specs are ordered max-activity fractions, then ``matched`` baselines, then
    each fraction's baseline group, so the filenames and log lines of the
    pre-existing groups are unchanged from earlier runs.
    """
    references = _reference_arms(agg_results_df, trial)
    if references is None:
        return []

    baselines = list(feldman_methods) + list(sivanandan_methods)

    def _activity(method: str) -> float | None:
        values = agg_results_df.loc[
            (agg_results_df["Method"] == method) & (agg_results_df["Trial"] == trial),
            "Mean activity score",
        ]
        return None if values.empty else values.values[0]

    specs: List[ComparisonSpec] = []

    # Maximum activity at each fraction. The prefixes are the literal historical
    # strings, not sanitize_filename(MAX_ACTIVITY_METHOD).
    for suffix, duet in references.items():
        specs.append(
            ComparisonSpec(MAX_ACTIVITY_METHOD, duet, f"max_activity_{suffix}", suffix)
        )

    # Activity-matched, per baseline.
    for method in baselines:
        activity = _activity(method)
        if activity is None:
            continue
        duet = select_duet_lambda(agg_results_df, trial, activity)
        if duet is None:
            print(
                f"    WARNING: No DUET lambda at or above {method}'s activity "
                f"in trial {trial}"
            )
            continue
        specs.append(
            ComparisonSpec(method, duet, sanitize_filename(method), "matched")
        )

    # Each fraction, per baseline.
    for suffix, duet in references.items():
        for method in baselines:
            if _activity(method) is None:
                continue
            specs.append(
                ComparisonSpec(
                    method,
                    duet,
                    f"{sanitize_filename(method)}_vs_duet_{suffix}",
                    suffix,
                )
            )

    return specs


def _reference_arms(agg_results_df: pd.DataFrame, trial: int) -> Dict[str, str] | None:
    """The DUET arm for each DUET_REFERENCE_FRACTIONS entry in one trial.

    Returns ``None`` (after a warning) when the trial has no maximum-activity
    row, since the fractions are defined relative to its mean activity. A
    fraction that cannot be resolved is dropped with its own warning.
    lambda=0.00 always attains the maximum activity, so a fraction <= 1 cannot
    normally fail to resolve; that guard is for truncated or filtered result
    sets that lack it.
    """
    max_act = agg_results_df.loc[
        (agg_results_df["Method"] == MAX_ACTIVITY_METHOD)
        & (agg_results_df["Trial"] == trial),
        "Mean activity score",
    ]
    if max_act.empty:
        print(f"    WARNING: No {MAX_ACTIVITY_METHOD} data for trial {trial}")
        return None
    max_act_mean = max_act.values[0]

    references: Dict[str, str] = {}
    for suffix, fraction in DUET_REFERENCE_FRACTIONS.items():
        duet = select_duet_lambda(agg_results_df, trial, fraction * max_act_mean)
        if duet is None:
            print(
                f"    WARNING: No DUET lambda above {fraction:.1%} of max activity "
                f"in trial {trial}; skipping that panel group"
            )
            continue
        references[suffix] = duet
    return references


def spec_colors(spec: ComparisonSpec) -> Dict[str, str]:
    """Method -> color for one comparison, from the shared publication palette.

    ``map_method_group`` already maps a method label onto exactly the
    METHOD_PALETTE keys, so the baseline's color needs no separate lookup table.
    """
    return {
        spec.baseline: METHOD_PALETTE[map_method_group(spec.baseline)],
        spec.duet: METHOD_PALETTE["DUET"],
    }


@dataclass(frozen=True)
class OverlaySpec:
    """One all-baselines panel: every baseline against a single DUET arm."""

    baselines: Tuple[str, ...]  # max activity, then Feldman, then Sivanandan
    duet: str  # method label of the DUET arm
    prefix: str  # filename prefix
    policy: str  # a key of DUET_REFERENCE_FRACTIONS


def build_overlay_specs(
    agg_results_df: pd.DataFrame,
    trial: int,
    feldman_methods: List[str],
    sivanandan_methods: List[str],
) -> List[OverlaySpec]:
    """One overlay per reference fraction: all baselines present in the trial
    against that fraction's DUET arm, the same arm the pairwise panels use.

    The distance-based baselines are close to indistinguishable from
    maximum-activity selection at the guide level, which the pairwise panels
    can only show one baseline at a time; the overlay shows it in one panel.
    """
    references = _reference_arms(agg_results_df, trial)
    if references is None:
        return []
    present = set(agg_results_df.loc[agg_results_df["Trial"] == trial, "Method"])
    baselines = tuple(
        m
        for m in [MAX_ACTIVITY_METHOD, *feldman_methods, *sivanandan_methods]
        if m in present
    )
    return [
        OverlaySpec(baselines, duet, f"all_baselines_vs_duet_{suffix}", suffix)
        for suffix, duet in references.items()
    ]


def overlay_colors(spec: OverlaySpec) -> Dict[str, str]:
    """Method -> color for an overlay, DUET last so it is drawn on top.

    Family hues come from METHOD_PALETTE; a member that shares a panel with
    another member of its family (Sivanandan ED=1 beside ED=2) takes its
    OVERLAY_VARIANT_PALETTE tint instead.
    """
    colors = {
        m: OVERLAY_VARIANT_PALETTE.get(m, METHOD_PALETTE[map_method_group(m)])
        for m in spec.baselines
    }
    colors[spec.duet] = METHOD_PALETTE["DUET"]
    return colors


# =============================================================================
# Plotting Functions
# =============================================================================


def _savefig_at_figsize(output_path: Path) -> None:
    """Save the current figure at exactly its figsize.

    The project mplstyle (`src/duet/plotting/duet_publication.mplstyle`) sets
    `savefig.bbox: tight` globally. With that rcParam in effect, the saved
    SVG canvas is re-expanded to include all artists — even when
    `constrained_layout=True` is used to keep content inside figsize.
    For panels that need to compose at exact width (e.g., 3-up strips),
    we need figsize-exact output.

    Passing `bbox_inches=None` to `savefig` does not help: matplotlib
    falls through to the rcParam when bbox_inches is None. The
    documented escape hatch is to set the rcParam itself to `"standard"`
    (which validates to `None`) for the duration of the savefig call.
    """
    with plt.rc_context({"savefig.bbox": "standard"}):
        plt.savefig(output_path)


def plot_hypervolume_barplot(hv_df: pd.DataFrame, output_path: Path) -> None:
    """Plot normalized hypervolume bar plot summarized across trials.

    Sized and titled to match the Pareto scatter panels so the two compose
    side-by-side at third_page width.
    """
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True
    )

    sns.barplot(
        data=hv_df,
        x="Method group",
        y="Normalized Hypervolume",
        hue="Method group",
        palette=METHOD_PALETTE,
        width=0.4,
        order=METHOD_GROUPS,
        legend=False,
        errorbar="se",
        ax=ax,
    )

    ax.set_title("Across Trials (± SE)")
    # The tick labels already name the groups, so an "Method Group" xlabel is a
    # wasted row at this width. Wrap "X et al." onto two lines instead — three
    # unwrapped labels collide at 60 mm.
    ax.set_xlabel("")
    ax.set_xticks(range(len(METHOD_GROUPS)))
    ax.set_xticklabels([g.replace(" et al.", "\net al.") for g in METHOD_GROUPS])
    ax.set_ylabel("Normalized hypervolume")
    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_pareto_hypervolume_single_trial(
    agg_results_df: pd.DataFrame,
    trial: int,
    output_path: Path,
) -> None:
    """Plot Pareto fronts with hypervolume polygons for a single trial.

    Sized to third_page to compose with the metric scatter panels. The polygon
    and the front line of a group share one color, so they share one legend
    entry — labelling both would spend two of the four available rows saying
    the same thing.
    """
    # Compute Pareto fronts
    fronts = []
    for g in METHOD_GROUPS:
        front_data = agg_results_df.loc[
            (agg_results_df["Method group"].isin([g, "Maximum activity"]))
            & (agg_results_df["Trial"] == trial)
        ][["Mean decode accuracy", "Mean activity score"]].to_numpy()
        fronts.append(front_data)

    results = compare_pareto_fronts(fronts, maximize=True)

    # Create plot
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True
    )

    # Plot all underlying points
    df_t = agg_results_df.loc[
        (agg_results_df["Trial"] == trial)
    ]
    sns.scatterplot(
        data=df_t,
        x="Mean decode accuracy",
        y="Mean activity score",
        hue="Method group",
        palette=METHOD_PALETTE,
        s=6,
        alpha=0.3,
        ax=ax,
        legend=False,
    )

    # Draw hypervolume polygons
    ref_point = results.reference_point
    for i, group in enumerate(METHOD_GROUPS):
        front = results.fronts[i].pareto_front
        front = front[front[:, 0].argsort()]

        # Create polygon vertices
        polygon_vertices = []
        polygon_vertices.append([ref_point[0], front[0, 1]])

        for j in range(len(front) - 1):
            polygon_vertices.append(front[j])
            polygon_vertices.append([front[j, 0], front[j + 1, 1]])

        polygon_vertices.append(front[-1])
        polygon_vertices.append([front[-1, 0], ref_point[1]])
        polygon_vertices.append(ref_point)

        # Unlabelled: the front line below carries this group's legend entry.
        polygon = patches.Polygon(
            np.array(polygon_vertices),
            closed=True,
            alpha=0.25,
            facecolor=METHOD_PALETTE[group],
        )
        ax.add_patch(polygon)

        # Plot Pareto front line
        ax.plot(
            front[:, 0],
            front[:, 1],
            marker="o",
            linestyle="--",
            color=METHOD_PALETTE[group],
            label=group,
            ms=3,
            mew=0.8,
            mfc="none",
        )

    # Plot reference point
    ax.plot(
        ref_point[0],
        ref_point[1],
        "D",
        markerfacecolor="none",  # Transparent interior
        markeredgecolor="#404040",
        ms=3,
        mew=0.8,
        label="Reference point",
    )

    ax.set_title(f"Trial {trial}")
    ax.set_xlabel(metric_axis_label("Mean decode accuracy"))
    ax.set_ylabel(metric_axis_label("Mean activity score"))
    ax.legend(title="Method", loc="best")
    _savefig_at_figsize(output_path)
    plt.close(fig)


def _compute_se(x: pd.Series) -> float:
    """Compute standard error, returning 0 for single observations."""
    n = len(x)
    if n <= 1:
        return 0.0
    return x.std() / np.sqrt(n)


def plot_pareto_hypervolume_aggregated(
    agg_results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot aggregated Pareto fronts with SE bars across trials."""
    # Compute mean and SE for each method across trials
    method_stats = (
        agg_results_df.groupby("Method")
        .agg(
            **{
                "Mean decode accuracy mean": ("Mean decode accuracy", "mean"),
                "Mean decode accuracy se": ("Mean decode accuracy", _compute_se),
                "Mean activity score mean": ("Mean activity score", "mean"),
                "Mean activity score se": ("Mean activity score", _compute_se),
                "Method group": ("Method group", "first"),
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True
    )

    # Maximum-activity anchor point — each non-MA group's connector line starts here.
    ma_df = method_stats[method_stats["Method group"] == "Maximum activity"]
    ma_x = ma_df["Mean decode accuracy mean"].iloc[0] if not ma_df.empty else None
    ma_y = ma_df["Mean activity score mean"].iloc[0] if not ma_df.empty else None

    # Plot points with error bars for each method group
    for group in PLOTTED_METHOD_GROUPS:
        group_df = method_stats[method_stats["Method group"] == group]

        if group != "Maximum activity" and ma_x is not None and not group_df.empty:
            sorted_df = group_df.sort_values("Mean activity score mean", ascending=False)
            line_x = [ma_x, *sorted_df["Mean decode accuracy mean"].tolist()]
            line_y = [ma_y, *sorted_df["Mean activity score mean"].tolist()]
            ax.plot(
                line_x,
                line_y,
                color=METHOD_PALETTE[group],
                linewidth=0.8,
                alpha=0.4,
                zorder=1,
            )

        ax.errorbar(
            group_df["Mean decode accuracy mean"],
            group_df["Mean activity score mean"],
            xerr=group_df["Mean decode accuracy se"],
            yerr=group_df["Mean activity score se"],
            fmt="o",
            color=METHOD_PALETTE[group],
            label=group,
            alpha=0.7,
            markersize=1.5,
            capsize=0,
        )

    ax.set_title("Aggregated Across Trials (± SE)")
    ax.set_xlabel(metric_axis_label("Mean decode accuracy"))
    ax.set_ylabel(metric_axis_label("Mean activity score"))
    ax.legend(title="Method Group", loc="best")
    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_metric_scatterplot_single_trial(
    agg_results_df: pd.DataFrame,
    trial: int,
    x_col: str,
    y_col: str,
    output_path: Path,
) -> None:
    """Plot scatterplot for a single metric and trial.

    Mirrors ``plot_metric_scatterplot_aggregated``'s Pareto styling — a
    Maximum-activity-anchored connector line plus small markers — so
    single-trial panels compose seamlessly alongside the aggregated frontier
    panels. A single trial carries no across-trial spread, so error bars are
    omitted (markers only).
    """
    df_t = agg_results_df.loc[agg_results_df["Trial"] == trial]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True)

    # Maximum-activity anchor point — each non-MA group's connector line starts here.
    ma_df = df_t[df_t["Method group"] == "Maximum activity"]
    ma_x = ma_df[x_col].iloc[0] if not ma_df.empty else None
    ma_y = ma_df[y_col].iloc[0] if not ma_df.empty else None

    for group in PLOTTED_METHOD_GROUPS:
        group_df = df_t[df_t["Method group"] == group]
        if group_df.empty:
            continue

        if group != "Maximum activity" and ma_x is not None:
            sorted_df = group_df.sort_values(y_col, ascending=False)
            line_x = [ma_x, *sorted_df[x_col].tolist()]
            line_y = [ma_y, *sorted_df[y_col].tolist()]
            ax.plot(
                line_x,
                line_y,
                color=METHOD_PALETTE[group],
                linewidth=0.8,
                alpha=0.4,
                zorder=1,
            )

        ax.errorbar(
            group_df[x_col],
            group_df[y_col],
            fmt="o",
            color=METHOD_PALETTE[group],
            label=group,
            alpha=0.7,
            markersize=1.5,
            capsize=0,
        )

    ax.set_title(f"Trial {trial}")
    ax.set_xlabel(metric_axis_label(x_col))
    ax.set_ylabel(metric_axis_label(y_col))
    ax.legend(title="Method", loc="best")
    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_metric_scatterplot_aggregated(
    agg_results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
) -> plt.Axes:
    """Plot aggregated scatterplot with SE bars for a metric.

    A slot-sized third-page panel (Fig 3d-f) with fixed margins, so the three
    metric panels line up when composed in a row. The bottom margin is 12 mm
    because some x labels wrap to two lines, and all panels in the row share it.
    When the last x tick label would run past the 2 mm right margin, the x range
    is widened to fit it (``fit_last_xtick_label``) rather than the margin; when
    a two-line x label would run past the bottom (matplotlib >= 3.11 draws it
    taller), its gap to the tick labels narrows instead (``fit_xlabel``).

    With a single trial (the genome-wide Fig 3d-f) there is no spread to show,
    so the points are drawn larger with no error bars and the legend uses solid
    squares; the points stay circles. A group with nothing to plot (no valid
    codebook, or no finite value of the metric, like the 95th/5th ratio when
    the 5th percentile is 0) keeps its legend row, with "N/A" for a marker.

    Returns:
        The axes (the figure is already saved and closed), so tests can inspect
        the legend.
    """
    method_stats = (
        agg_results_df.groupby("Method")
        .agg(
            **{
                f"{x_col} mean": (x_col, "mean"),
                f"{x_col} se": (x_col, _compute_se),
                f"{y_col} mean": (y_col, "mean"),
                f"{y_col} se": (y_col, _compute_se),
                "Method group": ("Method group", "first"),
            }
        )
        .reset_index()
    )

    fig, ax = slot_figure(
        FIGURE_WIDTHS["third_page"] * MM_PER_INCH,
        2.0 * MM_PER_INCH,
        margins_mm=(16.0, 2.0, 12.0, 2.0),
    )

    # Maximum-activity anchor point — each non-MA group's connector line starts here.
    ma_df = method_stats[method_stats["Method group"] == "Maximum activity"]
    ma_x = ma_df[f"{x_col} mean"].iloc[0] if not ma_df.empty else None
    ma_y = ma_df[f"{y_col} mean"].iloc[0] if not ma_df.empty else None

    single_trial = agg_results_df["Trial"].nunique() == 1
    # Group -> the artist of its points, for the groups that have any.
    drawn = {}
    for group in PLOTTED_METHOD_GROUPS:
        group_df = method_stats[method_stats["Method group"] == group]
        finite = (
            np.isfinite(group_df[f"{x_col} mean"])
            & np.isfinite(group_df[f"{y_col} mean"])
        )
        if not finite.any():
            continue  # an "N/A" legend row

        if group != "Maximum activity" and ma_x is not None:
            sorted_df = group_df.sort_values(f"{y_col} mean", ascending=False)
            line_x = [ma_x, *sorted_df[f"{x_col} mean"].tolist()]
            line_y = [ma_y, *sorted_df[f"{y_col} mean"].tolist()]
            ax.plot(
                line_x,
                line_y,
                color=METHOD_PALETTE[group],
                linewidth=0.8,
                alpha=0.4,
                zorder=1,
            )

        if single_trial:
            (drawn[group],) = ax.plot(
                group_df[f"{x_col} mean"],
                group_df[f"{y_col} mean"],
                linestyle="none",
                marker="o",
                markersize=SINGLE_TRIAL_MARKERSIZE,
                color=METHOD_PALETTE[group],
                label=group,
                alpha=0.7,
                markeredgewidth=0,
            )
        else:
            drawn[group] = ax.errorbar(
                group_df[f"{x_col} mean"],
                group_df[f"{y_col} mean"],
                xerr=group_df[f"{x_col} se"],
                yerr=group_df[f"{y_col} se"],
                fmt="o",
                color=METHOD_PALETTE[group],
                label=group,
                alpha=0.7,
                markersize=1.5,
                capsize=0,
            )

    if single_trial:
        handles, labels = square_legend_handles(
            [(g, METHOD_PALETTE[g] if g in drawn else None) for g in PLOTTED_METHOD_GROUPS],
            markersize=SINGLE_TRIAL_MARKERSIZE,
        )
    else:
        handles = [drawn[g] if g in drawn else NotAvailable() for g in PLOTTED_METHOD_GROUPS]
        labels = list(PLOTTED_METHOD_GROUPS)

    ax.set_xlabel(metric_axis_label(x_col))
    ax.set_ylabel(metric_axis_label(y_col))
    ax.legend(
        handles, labels, title="Method Group", loc="best", handler_map=LEGEND_HANDLER_MAP
    )
    fit_xlabel(fig, ax)
    fit_last_xtick_label(fig, ax)
    save_panel(fig, output_path)
    plt.close(fig)
    return ax


def plot_comparison(
    results_df: pd.DataFrame,
    method_to_hue: Dict[str, str],
    trial: int,
    output_dir: Path,
    name_prefix: str,
    kinds: Sequence[str] = COMPARISON_KINDS,
) -> None:
    """Generate comparison plots between methods.

    Creates up to three plots, one per entry of ``kinds``:
    - Decode accuracy histogram (``decode_accuracy``)
    - Activity score histogram (``activity_score``)
    - 2D scatter with KDE (``dual_objectives``)

    Args:
        results_df: Raw results dataframe.
        method_to_hue: Dict mapping method strings to colors.
        trial: Trial number.
        output_dir: Directory to save plots.
        name_prefix: Prefix for output filenames.
        kinds: Which of COMPARISON_KINDS to write.
    """
    df = results_df.loc[results_df["Trial"] == trial]

    # Decode accuracy histogram
    if "decode_accuracy" in kinds:
        fig, ax = plot_codeword_probabilities(
            [
                (display_method_label(m), df.loc[df["Method"] == m, "Decode accuracy"].values, hue)
                for m, hue in method_to_hue.items()
            ],
            print_summary=False,
            figsize=(FIGURE_WIDTHS["third_page"], 1.5),
        )
        fig.suptitle(
            f'Trial {trial}: '
            f'{", ".join(display_method_label(m) for m in method_to_hue)}'
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{name_prefix}_trial_{trial}_decode_accuracy.svg", bbox_inches="tight")
        plt.close(fig)

    # Activity score histogram
    if "activity_score" in kinds:
        fig, ax = plot_activity_scores(
            [
                (display_method_label(m), df.loc[df["Method"] == m, "Activity score"].values, hue)
                for m, hue in method_to_hue.items()
            ],
            print_summary=False,
            figsize=(FIGURE_WIDTHS["third_page"], 1.5),
        )
        fig.suptitle(
            f'Trial {trial}: '
            f'{", ".join(display_method_label(m) for m in method_to_hue)}'
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{name_prefix}_trial_{trial}_activity_score.svg", bbox_inches="tight")
        plt.close(fig)

    # 2D scatter with KDE
    if "dual_objectives" in kinds:
        fig, ax = plot_dual_objectives(
            [
                (
                    display_method_label(m),
                    df.loc[df["Method"] == m, "Decode accuracy"].values,
                    df.loc[df["Method"] == m, "Activity score"].values,
                    hue,
                )
                for m, hue in method_to_hue.items()
            ],
            figsize=(FIGURE_WIDTHS["third_page"], 1.75),
        )
        fig.suptitle(
            f'Trial {trial}: '
            f'{", ".join(display_method_label(m) for m in method_to_hue)}'
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{name_prefix}_trial_{trial}_dual_objectives.svg", bbox_inches="tight")
        plt.close(fig)


def plot_jointplot(
    results_df: pd.DataFrame,
    method_to_color: Dict[str, str],
    trial: int,
    output_path: Path,
) -> sns.JointGrid:
    """Generate jointplot with scatter, KDE contours, and marginal histograms.

    Layers, bottom to top: each method's points in ``method_to_color`` order,
    then each method's KDE contour in the same order. ``spec_colors`` lists the
    baseline first and DUET last, so DUET's points and contour land on top.
    Points are small and faint and the contours few and opaque, so the two do
    not compete: the per-guide spread stays visible as a texture while the
    contours carry the density comparison. Chosen 2026-09-08 over white halos,
    tinted or hollow points, a single 50% contour, and filled KDEs.

    Args:
        results_df: Raw results dataframe.
        method_to_color: Dict mapping method strings to colors; also the draw
            order, first key at the bottom.
        trial: Trial number.
        output_path: Path to save the figure.

    Returns:
        The seaborn JointGrid (its figure is already saved and closed), so
        callers and tests can inspect the axes and legend.
    """
    df = results_df.loc[
        (results_df["Trial"] == trial)
        & (results_df["Method"].isin(list(method_to_color.keys())))
    ]

    # Create JointGrid
    g = sns.JointGrid(
        data=df,
        x="Decode accuracy",
        y="Activity score",
        height=FIGURE_WIDTHS["third_page"],
        ratio=2,
        space=0.2,
    )

    # Points, one layer per method in method_to_color order, pinned with
    # explicit zorders. A single hue-keyed scatter draws in row order, and DUET
    # rows come first in results.csv, so the baseline used to bury the DUET
    # cloud. Small, faint markers keep the per-guide spread (and the guides at
    # accuracy 0) visible as a texture without competing with the contours.
    n_methods = len(method_to_color)
    for z, (method, color) in enumerate(method_to_color.items(), start=1):
        sns.scatterplot(
            data=df[df["Method"] == method],
            x="Decode accuracy",
            y="Activity score",
            color=color,
            s=8,
            alpha=0.35,
            edgecolor="white",
            linewidth=0.5,
            ax=g.ax_joint,
            zorder=z,
            legend=False,
        )

    # KDE contours above every point layer, same method order. Three levels at
    # full opacity: at third_page width five nested rings merge into a blob,
    # and the 0.7 alpha they used to carry let the points show through them.
    for z, (method, color) in enumerate(method_to_color.items(), start=n_methods + 1):
        sns.kdeplot(
            data=df[df["Method"] == method],
            x="Decode accuracy",
            y="Activity score",
            ax=g.ax_joint,
            levels=3,
            color=color,
            linewidths=1.5,
            alpha=1.0,
            zorder=z,
        )

    # Add marginal histograms with KDE
    for method, color in method_to_color.items():
        method_data = df[df["Method"] == method]

        # X marginal
        sns.histplot(
            data=method_data,
            x="Decode accuracy",
            ax=g.ax_marg_x,
            color=color,
            alpha=0.35,
            edgecolor=None,
            kde=True,
            stat="density",
            line_kws={"linewidth": 1.6, "alpha": 1.0},
            kde_kws={"cut": 0},
        )

        # Y marginal
        sns.histplot(
            data=method_data,
            y="Activity score",
            ax=g.ax_marg_y,
            color=color,
            alpha=0.35,
            edgecolor=None,
            kde=True,
            stat="density",
            line_kws={"linewidth": 1.6, "alpha": 1.0},
            kde_kws={"cut": 0},
        )

    # Legend: one filled square per method, the look of the shipped Figure 3b
    # and S4d-f panels, rather than the scatter marker seaborn adds by default or
    # a line swatch. The square reads as the method's colour without mimicking
    # either a data point or the contour lines. handlelength == handleheight
    # (in font-size units) makes the patch square instead of matplotlib's
    # default 2:0.7 bar. Colors come from method_to_color (the same source the
    # scatter and KDE use). Entries follow the order methods first appear in the
    # data (DUET before the baselines in results.csv), which is what the
    # hue-keyed scatter legend used to produce, so regenerated panels keep their
    # legend order; the per-method scatter layers above pass legend=False, so
    # seaborn registers no handles to read that order from.
    method_labels = [m for m in dict.fromkeys(df["Method"]) if m in method_to_color]
    swatches = [
        patches.Patch(facecolor=method_to_color[label], edgecolor="none")
        for label in method_labels
    ]
    g.ax_joint.legend(
        swatches, [display_method_label(label) for label in method_labels],
        loc="upper left", fontsize="small", frameon=True, framealpha=0.6,
        edgecolor="none", title=None, borderpad=0.3, handletextpad=0.4,
        labelspacing=0.3, handlelength=1.0, handleheight=1.0,
    )

    g.set_axis_labels("Decode accuracy", "Activity score")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return g


# -----------------------------------------------------------------------------
# Guide-level distribution stack
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisRanges:
    """Display ranges shared by every distribution stack in one trial."""

    activity: Tuple[float, float]
    accuracy: Tuple[float, float]


@dataclass(frozen=True)
class DistributionStackInfo:
    """What one distribution stack drew, for callers and tests."""

    zero_fractions: Dict[str, float]  # method -> fraction of guides at accuracy 0
    zero_bin: bool  # whether the broken-axis zero slot was drawn
    accuracy_xlim: Tuple[float, float]  # final x-limits of the accuracy panel
    filled: Tuple[str, ...]  # methods drawn with a filled density


def shared_axis_ranges(
    results_df: pd.DataFrame, trial: int, methods: List[str]
) -> AxisRanges:
    """Axis ranges for the distribution stacks of one trial.

    Computed over every method that appears in any comparison of the trial, so
    the DUET curve (the same codebook in each panel) is drawn at the same scale
    everywhere and the panels read as a strip.

    Activity is truncated at the pooled 99.5th percentile: its right tail is a
    handful (about 0.5%) of very high-scoring guides that the methods largely
    share, identically in some trials, so the truncation is close to symmetric
    across methods; the jointplots keep the full range, which can run to 10 for
    a single guide. Decode accuracy starts at the lowest
    NON-ZERO value: guides at exactly 0 (duplicate codewords) are a point mass
    that the density cannot represent, and they get their own bin instead
    (see ``plot_distribution_stack``). Bounds are rounded outward to 0.1
    (activity) and 0.05 (accuracy).
    """
    df = results_df.loc[
        (results_df["Trial"] == trial) & (results_df["Method"].isin(methods))
    ]
    activity = df["Activity score"]
    accuracy = df["Decode accuracy"]
    nonzero = accuracy[accuracy > 0]
    return AxisRanges(
        activity=(
            float(np.floor(activity.min() * 10) / 10),
            float(np.ceil(activity.quantile(0.995) * 10) / 10),
        ),
        accuracy=(
            float(np.floor(nonzero.min() * 20) / 20),
            float(min(1.0, np.ceil(accuracy.max() * 20) / 20)),
        ),
    )


def continuous_density(
    x: np.ndarray, grid: np.ndarray, split_zeros: bool = True
) -> Tuple[np.ndarray, float]:
    """Gaussian kernel density of ``x`` on ``grid``, and the bandwidth used.

    With ``split_zeros`` (the decode-accuracy case), guides at exactly 0 are a
    point mass: they are removed before the fit, so they neither inflate the
    bandwidth (Scott's rule scales with the standard deviation, which a pile
    of zeros far from the mode blows up) nor smear into a spurious bump. The
    curve is then scaled by the non-zero fraction so that curve area plus the
    zero bin's area is one.
    """
    x = np.asarray(x, dtype=float)
    fitted = x[x > 0] if split_zeros else x
    fraction = fitted.size / x.size
    kde = gaussian_kde(fitted)
    bandwidth = float(kde.factor * fitted.std(ddof=1))
    return fraction * kde(grid), bandwidth


def zero_bin_bar(zero_fraction: float, bandwidth: float) -> Tuple[float, float]:
    """(height, width) of the bar for the guides at accuracy 0.

    A point mass has no density, so the bar is a histogram bin on the same
    density axis: its area equals the fraction of guides it holds and its width
    is the kernel bandwidth, so every mark on the axis is at one resolution.
    """
    return zero_fraction / bandwidth, bandwidth


def _draw_density_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    method_to_color: Dict[str, str],
    column: str,
    lims: Tuple[float, float],
    split_zeros: bool,
    filled: Tuple[str, ...],
    fill_alpha: float,
) -> Tuple[float, Dict[str, float]]:
    """One row of the stack: densities plus the percentile strip.

    Methods in ``filled`` get a filled density; the rest are outlines. Returns
    the density-axis ceiling and each method's bandwidth (the zero bin needs
    the latter).
    """
    ymax = 0.0
    bandwidths: Dict[str, float] = {}
    for method, color in method_to_color.items():
        x = df.loc[df["Method"] == method, column].to_numpy()
        fitted = x[x > 0] if split_zeros else x
        grid = np.linspace(
            max(lims[0], fitted.min()), min(lims[1], fitted.max()), 512
        )
        y, bandwidths[method] = continuous_density(x, grid, split_zeros)
        if method in filled:
            ax.fill_between(grid, 0, y, color=color, alpha=fill_alpha, linewidth=0)
        ax.plot(grid, y, color=color, linewidth=1.2)
        ymax = max(ymax, float(y.max()))
    ymax *= 1.05

    # Percentile strips below the zero line, one per method, on ALL guides.
    # Two-method panels put the rows at 11% and 22% of the axis height and
    # reserve 28%; overlays pack their rows at 7.5% plus one row of margin.
    n_methods = len(method_to_color)
    step = 0.11 if n_methods <= 2 else 0.075
    reserve = 0.28 if n_methods <= 2 else step * (n_methods + 1.2)
    for i, (method, color) in enumerate(method_to_color.items()):
        x = df.loc[df["Method"] == method, column].to_numpy()
        lo, mid, hi = np.percentile(x, STRIP_PERCENTILES)
        y = -ymax * step * (i + 1)
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, solid_capstyle="butt")
        ax.plot([mid], [y], marker="o", markersize=3.0, color=color,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.axhline(0, color="0.6", linewidth=0.5)
    ax.set_ylim(-ymax * reserve, ymax)
    ax.set_xlim(*lims)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Density")
    ax.set_xlabel(column)
    return ymax, bandwidths


def _draw_zero_bin(
    ax: plt.Axes,
    method_to_color: Dict[str, str],
    zero_fractions: Dict[str, float],
    bandwidths: Dict[str, float],
    lims: Tuple[float, float],
    ymax: float,
) -> Tuple[float, float]:
    """Add the broken-axis slot for guides at accuracy 0 to a density panel.

    The continuous range ``lims`` is kept intact; a slot is reserved to its
    left, ticked "0" and separated by a double slash on the bottom spine. Each
    method with zero-accuracy guides gets an area-true bar (``zero_bin_bar``)
    with its fraction printed above. Returns the panel's new x-limits.
    """
    lo, hi = lims
    slot = ZERO_SLOT_FRACTION * (hi - lo)
    left = lo - slot
    center = left + 0.45 * slot
    break_x = lo - 0.18 * slot

    # Keep the ticks matplotlib chose for the continuous range, prepend "0".
    # A tick within half a slot of the range's left edge would sit against the
    # "0" tick and the break marks (a range starting at 0.4 gets one at 0.4),
    # so it is dropped; the next tick in is always clear of the slot.
    ticks = [t for t in ax.get_xticks() if lo + 0.5 * slot <= t <= hi]
    ax.set_xlim(left, hi)
    ax.set_xticks([center, *ticks])
    ax.set_xticklabels(["0", *(f"{t:g}" for t in ticks)])

    # Redraw the bottom spine as two segments with a slash pair between them.
    ax.spines["bottom"].set_visible(False)
    spine_kw = dict(
        color=plt.rcParams["axes.edgecolor"],
        linewidth=plt.rcParams["axes.linewidth"],
        clip_on=False,
        solid_capstyle="butt",
        transform=blended_transform_factory(ax.transData, ax.transAxes),
    )
    gap, slash_half, slash_offset = 0.10 * slot, 0.067 * slot, 0.133 * slot
    ax.plot([left, break_x - gap], [0, 0], **spine_kw)
    ax.plot([break_x + gap + slash_offset, hi], [0, 0], **spine_kw)
    for offset in (0.0, slash_offset):
        ax.plot(
            [break_x - slash_half + offset, break_x + slash_half + offset],
            [-0.045, 0.045],
            **spine_kw,
        )

    # One bar per method with zero-mass, laid side by side and centred on the
    # slot as a group.
    bars = [
        (method, *zero_bin_bar(zero_fractions[method], bandwidths[method]))
        for method in method_to_color
        if zero_fractions[method] > 0
    ]
    x = center - sum(width for _, _, width in bars) / 2
    for method, height, width in bars:
        ax.bar([x + width / 2], [height], width=width,
               color=method_to_color[method], linewidth=0)
        x += width

    # Labels. A percent label is far wider than its bar, so centring one on each
    # bar collides as soon as there are two. With one bar the label sits on it in
    # grey; with several they stack as a column above the TALLEST bar, each in
    # its bar's colour, which is the only cue tying a label to its bar.
    tallest = max(height for _, height, _ in bars)
    line = ymax * 0.09
    for j, (method, height, _) in enumerate(bars):
        single = len(bars) == 1
        ax.text(
            center,
            (height if single else tallest + j * line) + ymax * 0.03,
            f"{zero_fractions[method]:.1%}", ha="center", va="bottom", fontsize=6,
            color="0.3" if single else method_to_color[method],
        )
    return left, hi


def plot_distribution_stack(
    results_df: pd.DataFrame,
    method_to_color: Dict[str, str],
    trial: int,
    output_path: Path,
    ranges: AxisRanges,
    emphasize: str | None = None,
) -> DistributionStackInfo:
    """The Figure 3 per-guide distribution stack for one comparison.

    A third-page-wide 2x1 panel: activity-score density on top, decode-accuracy
    density below, one curve per method with a 5th-95th percentile strip
    (median dot) beneath. Guides at decode accuracy exactly 0 are drawn as an
    area-true bin in a broken-axis slot, only when some plotted method has any.
    The legend sits in the accuracy panel, whose upper-left corner is empty;
    the activity peak collides with every corner of its own panel.

    Args:
        results_df: Raw per-guide results.
        method_to_color: Method label -> colour, baselines first, DUET last.
        trial: Trial number.
        output_path: SVG to write, at exactly the third-page width.
        ranges: From ``shared_axis_ranges``, shared by every stack in the trial.
        emphasize: If given, only this method's density is filled and the rest
            are outlines. The all-baselines overlay uses it for DUET, since
            five filled curves would be unreadable.
    """
    df = results_df.loc[
        (results_df["Trial"] == trial)
        & (results_df["Method"].isin(list(method_to_color)))
    ]
    filled = tuple(method_to_color) if emphasize is None else (emphasize,)
    fill_alpha = 0.22 if emphasize is None else 0.18
    overlay = len(method_to_color) > 2  # more strip rows and legend entries

    height = STACK_HEIGHT_OVERLAY if overlay else STACK_HEIGHT_PAIR
    fig, (ax_act, ax_acc) = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTHS["third_page"], height), constrained_layout=True
    )
    _draw_density_panel(
        ax_act, df, method_to_color, "Activity score", ranges.activity,
        split_zeros=False, filled=filled, fill_alpha=fill_alpha,
    )
    ymax, bandwidths = _draw_density_panel(
        ax_acc, df, method_to_color, "Decode accuracy", ranges.accuracy,
        split_zeros=True, filled=filled, fill_alpha=fill_alpha,
    )

    zero_fractions = {
        method: float((df.loc[df["Method"] == method, "Decode accuracy"] == 0).mean())
        for method in method_to_color
    }
    zero_bin = any(fraction > 0 for fraction in zero_fractions.values())
    accuracy_xlim = ranges.accuracy
    if zero_bin:
        accuracy_xlim = _draw_zero_bin(
            ax_acc, method_to_color, zero_fractions, bandwidths, ranges.accuracy, ymax
        )

    handles = [
        Line2D([0], [0], color=color, linewidth=2.5 if not overlay else 2.0)
        for color in method_to_color.values()
    ]
    ax_acc.legend(
        handles, [display_method_label(m) for m in method_to_color],
        loc="upper left", fontsize=6 if overlay else "small", frameon=False,
        handlelength=1.6 if overlay else 1.2, borderaxespad=0.1,
        labelspacing=0.25 if overlay else 0.3, handletextpad=0.5,
    )

    _savefig_at_figsize(output_path)
    plt.close(fig)
    return DistributionStackInfo(zero_fractions, zero_bin, accuracy_xlim, filled)


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
    )


# =============================================================================
# Error Correction Metrics Plotting Functions
# =============================================================================


def plot_error_metrics_pie_comparison(
    results_df: pd.DataFrame,
    baseline_method: str,
    duet_method: str,
    trial: int,
    output_path: Path,
) -> None:
    """Plot side-by-side pie charts comparing error correction metrics.

    Shows three categories for each method:
    - No error: Samples received without channel errors
    - Corrected: Samples with errors that were successfully corrected
    - Failed: Samples with errors that failed to decode correctly

    Args:
        results_df: Raw results dataframe with No_error, Corrected, Failed columns.
        baseline_method: Name of baseline method (e.g., "Feldman et al. (ED=2)").
        duet_method: Name of matched DUET method (e.g., "DUET (lambda=0.50)").
        trial: Trial number.
        output_path: Path to save the figure.
    """
    df = results_df.loc[results_df["Trial"] == trial]

    # Compute codebook-level rates (mean across codewords)
    baseline_data = df.loc[df["Method"] == baseline_method]
    duet_data = df.loc[df["Method"] == duet_method]

    if baseline_data.empty or duet_data.empty:
        print(f"Warning: Missing data for pie chart comparison. "
              f"Baseline ({baseline_method}): {len(baseline_data)}, "
              f"DUET ({duet_method}): {len(duet_data)}")
        return

    baseline_rates = [
        baseline_data["No_error"].mean(),
        baseline_data["Corrected"].mean(),
        baseline_data["Failed"].mean(),
    ]
    duet_rates = [
        duet_data["No_error"].mean(),
        duet_data["Corrected"].mean(),
        duet_data["Failed"].mean(),
    ]

    labels = ["No error", "Corrected", "Failed"]
    colors = [ERROR_CATEGORY_PALETTE[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True)

    # Baseline pie
    axes[0].pie(
        baseline_rates,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    axes[0].set_title(f"{display_method_label(baseline_method)}\n(Trial {trial})")

    # DUET pie
    axes[1].pie(
        duet_rates,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    axes[1].set_title(f"{display_method_label(duet_method)}\n(Trial {trial})")

    fig.suptitle("Error Correction Breakdown", fontweight="bold")
    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_error_metrics_bar_comparison(
    results_df: pd.DataFrame,
    method_pairs: List[Tuple[str, str]],
    trial: int,
    output_path: Path,
) -> None:
    """Plot bar chart comparing % Corrected across methods.

    Each method contributes at most one bar. A fraction-anchored policy pairs
    every baseline with the *same* DUET arm, so without this the DUET bar would
    be redrawn once per baseline; two baselines resolving to one DUET lambda
    under the matched policy would do the same.

    Args:
        results_df: Raw results dataframe with Corrected column.
        method_pairs: List of (baseline_method, duet_method) tuples.
        trial: Trial number.
        output_path: Path to save the figure.
    """
    df = results_df.loc[results_df["Trial"] == trial]

    # Collect data for bar chart
    methods = []
    corrected_rates = []
    colors = []

    def _add(method: str, color: str) -> None:
        if method in methods:
            return
        data = df.loc[df["Method"] == method]
        if data.empty:
            return
        methods.append(method)
        corrected_rates.append(data["Corrected"].mean() * 100)
        colors.append(color)

    for baseline_method, duet_method in method_pairs:
        if "Feldman" in baseline_method:
            baseline_color = METHOD_PALETTE["Feldman et al."]
        elif "Sivanandan" in baseline_method:
            baseline_color = METHOD_PALETTE["Sivanandan et al."]
        else:
            baseline_color = METHOD_PALETTE["Maximum activity"]

        _add(baseline_method, baseline_color)
        _add(duet_method, METHOD_PALETTE["DUET"])

    if not methods:
        print("Warning: No data for error metrics bar chart")
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True)

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, corrected_rates, color=colors, edgecolor="white", linewidth=1)

    # Add value labels on bars
    for bar, rate in zip(bars, corrected_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [display_method_label(m) for m in methods], rotation=45, ha="right"
    )
    ax.set_ylabel("% Corrected")
    ax.set_title(f"Error Correction Rate Comparison (Trial {trial})")
    ax.set_ylim(0, max(corrected_rates) * 1.15 if corrected_rates else 100)

    _savefig_at_figsize(output_path)
    plt.close(fig)


def plot_error_metrics_delta_histogram(
    results_df: pd.DataFrame,
    baseline_method: str,
    duet_method: str,
    trial: int,
    output_path: Path,
) -> None:
    """Plot histogram of codeword-level delta in % Corrected (DUET - Baseline).

    Positive values indicate DUET improvement over baseline.

    Args:
        results_df: Raw results dataframe with Corrected column.
        baseline_method: Name of baseline method.
        duet_method: Name of matched DUET method.
        trial: Trial number.
        output_path: Path to save the figure.
    """
    df = results_df.loc[results_df["Trial"] == trial]

    baseline_data = df.loc[df["Method"] == baseline_method].sort_values("Index")
    duet_data = df.loc[df["Method"] == duet_method].sort_values("Index")

    if len(baseline_data) != len(duet_data):
        print(f"Warning: Mismatched codeword counts for delta histogram. "
              f"Baseline: {len(baseline_data)}, DUET: {len(duet_data)}")
        return

    if baseline_data.empty:
        print(f"Warning: No data for delta histogram")
        return

    # Compute delta (DUET - Baseline) as percentage points
    delta = (duet_data["Corrected"].values - baseline_data["Corrected"].values) * 100

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["third_page"], 2.0), constrained_layout=True)

    # Histogram with KDE
    sns.histplot(
        delta,
        kde=True,
        color=METHOD_PALETTE["DUET"],
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # Add vertical line at 0
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="No change")

    # Add mean line
    mean_delta = delta.mean()
    ax.axvline(
        x=mean_delta,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_delta:+.2f}pp",
    )

    ax.set_xlabel("Delta % Corrected (DUET - Baseline) [percentage points]")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Codeword-Level Error Correction Improvement\n"
        f"DUET ({display_method_label(duet_method)}) vs "
        f"{display_method_label(baseline_method)} (Trial {trial})"
    )
    ax.legend()

    # Add summary statistics as text
    positive_pct = (delta > 0).sum() / len(delta) * 100
    stats_text = (
        f"Codewords improved: {positive_pct:.1f}%\n"
        f"Mean delta: {mean_delta:+.2f}pp\n"
        f"Std: {delta.std():.2f}pp"
    )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    _savefig_at_figsize(output_path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def _log_zero_bin(info: DistributionStackInfo) -> None:
    """Note in the run log which methods put guides into the zero bin."""
    if info.zero_bin:
        zeros = ", ".join(
            f"{m}: {f:.1%}" for m, f in info.zero_fractions.items() if f > 0
        )
        print(f"      zero-accuracy guides drawn in the zero bin ({zeros})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark visualizations from results.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to results.csv generated by run_benchmark.py",
    )
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to benchmark config YAML file (will use outdir/results.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Directory to save output figures (defaults to outdir from config if using --config)",
    )
    add_debug_plots_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input path and output directory
    if args.config:
        # Load config to get outdir
        config_path = Path(args.config)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        outdir = Path(config["outdir"])
        results_csv = outdir / "results.csv"

        # Default output_dir to outdir if not specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = outdir
    else:
        # Using --input directly
        results_csv = Path(args.input)
        config = None  # so no visualization.paper_panels

        if not args.output_dir:
            raise ValueError("--output-dir is required when using --input")
        output_dir = Path(args.output_dir)

    # Apply shared publication stylesheet
    apply_style()

    # The hypervolume bar plot and the aggregated Pareto fronts are always
    # written. Every other figure is a debug plot: written with --debug-plots,
    # or when the config lists it under visualization.paper_panels.
    tiers = PlotTiers.from_config(output_dir, args.debug_plots, config)
    n_skipped = 0

    def writes_debug(path: Path) -> bool:
        """Whether this run writes the debug plot at ``path``.

        Makes the plot's folder when it does, so a default run leaves no empty
        debug folders behind.
        """
        nonlocal n_skipped
        if not tiers.writes_debug(path):
            n_skipped += 1
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return True

    # Create output directories. The debug-only ones are made by writes_debug.
    hypervolume_dir = output_dir / "hypervolume"
    hv_viz_dir = output_dir / "hypervolume_visualization"
    pareto_dir = output_dir / "pareto_fronts"
    comparison_dir = output_dir / "guide_level_comparisons"
    error_metrics_dir = output_dir / "error_metrics"

    for d in [hypervolume_dir, pareto_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load and process data
    print("Loading results...")
    results_df, agg_results_df = load_and_process_results(results_csv)
    trials = sorted(agg_results_df["Trial"].unique())
    print(f"Found {len(trials)} trials")

    # =========================================================================
    # 1. Normalized Hypervolume Bar Plot
    # =========================================================================
    print("\nGenerating hypervolume bar plot...")
    hv_df = compute_hypervolume_df(agg_results_df)
    plot_hypervolume_barplot(
        hv_df, hypervolume_dir / "normalized_hypervolume_by_method_group.svg"
    )

    # =========================================================================
    # 2. Pareto Front Hypervolume Visualization
    # =========================================================================
    print("\nGenerating hypervolume visualizations...")
    for trial in trials:
        path = hv_viz_dir / f"pareto_fronts_trial_{trial}.svg"
        if writes_debug(path):
            print(f"  Trial {trial}...")
            plot_pareto_hypervolume_single_trial(agg_results_df, trial, path)

    # A debug plot even though it is aggregated: it shows the same points as
    # pareto_fronts/Mean_decode_accuracy_aggregated.svg.
    path = hv_viz_dir / "pareto_fronts_aggregated.svg"
    if writes_debug(path):
        print("  Aggregated...")
        plot_pareto_hypervolume_aggregated(agg_results_df, path)

    # =========================================================================
    # 3. Metric Scatterplots (Pareto Fronts)
    # =========================================================================
    print("\nGenerating metric scatterplots...")
    for x_col, y_col in METRICS:
        metric_name = sanitize_filename(x_col)
        print(f"  {x_col}...")

        # Per-trial plots
        for trial in trials:
            path = pareto_dir / f"{metric_name}_trial_{trial}.svg"
            if writes_debug(path):
                plot_metric_scatterplot_single_trial(
                    agg_results_df, trial, x_col, y_col, path
                )

        # Aggregated plot
        plot_metric_scatterplot_aggregated(
            agg_results_df,
            x_col,
            y_col,
            pareto_dir / f"{metric_name}_aggregated.svg",
        )

    # =========================================================================
    # 4. Guide-Level Comparisons
    # =========================================================================
    print("\nGenerating guide-level comparisons...")

    # Get baseline methods present in data
    all_methods = set(agg_results_df["Method"].unique())
    feldman_methods = sorted([m for m in all_methods if m.startswith("Feldman")])
    sivanandan_methods = sorted([m for m in all_methods if m.startswith("Sivanandan")])

    # Built once per trial and reused by the error-metrics section below, so the
    # two sections cannot disagree about which DUET point a baseline is compared
    # against.
    specs_by_trial: Dict[int, List[ComparisonSpec]] = {}

    for trial in trials:
        print(f"  Trial {trial}...")
        specs = build_comparison_specs(
            agg_results_df, trial, feldman_methods, sivanandan_methods
        )
        specs_by_trial[trial] = specs
        if not specs:
            continue

        # One set of axis ranges for every distribution stack in the trial, so
        # the panels can be composed as a strip. Taken over every comparison,
        # written or not, so a stack's axes do not depend on --debug-plots.
        trial_methods = sorted({m for spec in specs for m in (spec.baseline, spec.duet)})
        ranges = shared_axis_ranges(results_df, trial, trial_methods)

        for spec in specs:
            print(f"    {spec.prefix}: {spec.baseline} vs {spec.duet}")
            method_colors = spec_colors(spec)
            stem = f"{spec.prefix}_trial_{trial}"
            kinds = [
                k for k in COMPARISON_KINDS
                if writes_debug(comparison_dir / f"{stem}_{k}.svg")
            ]
            if kinds:
                plot_comparison(
                    results_df,
                    method_colors,
                    trial,
                    comparison_dir,
                    spec.prefix,
                    kinds,
                )
            path = comparison_dir / f"{stem}_jointplot.svg"
            if writes_debug(path):
                plot_jointplot(results_df, method_colors, trial, path)
            path = comparison_dir / f"{stem}_distributions.svg"
            if writes_debug(path):
                info = plot_distribution_stack(
                    results_df, method_colors, trial, path, ranges
                )
                _log_zero_bin(info)

        # Every baseline on one stack, per reference fraction.
        for overlay in build_overlay_specs(
            agg_results_df, trial, feldman_methods, sivanandan_methods
        ):
            print(f"    {overlay.prefix}: {', '.join(overlay.baselines)} vs {overlay.duet}")
            path = comparison_dir / f"{overlay.prefix}_trial_{trial}_distributions.svg"
            if writes_debug(path):
                info = plot_distribution_stack(
                    results_df,
                    overlay_colors(overlay),
                    trial,
                    path,
                    ranges,
                    emphasize=overlay.duet,
                )
                _log_zero_bin(info)

    # =========================================================================
    # 5. Error Correction Metrics Visualizations
    # =========================================================================
    # Check if error metrics columns exist
    has_error_metrics = all(col in results_df.columns for col in ["No_error", "Corrected", "Failed"])

    if has_error_metrics:
        print("\nGenerating error correction metrics visualizations...")

        for trial in trials:
            print(f"  Trial {trial}...")

            specs = specs_by_trial.get(trial, [])
            if not specs:
                print(f"    WARNING: No comparisons available for trial {trial}")
                continue

            for spec in specs:
                path = error_metrics_dir / f"{spec.prefix}_trial_{trial}_pie.svg"
                if writes_debug(path):
                    plot_error_metrics_pie_comparison(
                        results_df, spec.baseline, spec.duet, trial, path
                    )
                path = error_metrics_dir / f"{spec.prefix}_trial_{trial}_delta_hist.svg"
                if writes_debug(path):
                    plot_error_metrics_delta_histogram(
                        results_df, spec.baseline, spec.duet, trial, path
                    )

            # One bar chart per reference policy. Mixing policies into a single
            # chart would show each baseline once per policy.
            #
            # The unsuffixed chart keeps its historical contents: the
            # max-activity 97.5% pair followed by the activity-matched
            # baselines. Iterating `specs` preserves that order, since
            # build_comparison_specs emits the max-activity fractions before the
            # matched group.
            legacy_pairs = [
                (spec.baseline, spec.duet)
                for spec in specs
                if spec.policy == "matched"
                or (spec.policy == "97p5pct" and spec.baseline == MAX_ACTIVITY_METHOD)
            ]
            path = error_metrics_dir / f"corrected_rate_comparison_trial_{trial}.svg"
            if legacy_pairs and writes_debug(path):
                plot_error_metrics_bar_comparison(
                    results_df, legacy_pairs, trial, path
                )

            for suffix in DUET_REFERENCE_FRACTIONS:
                pairs = [
                    (spec.baseline, spec.duet)
                    for spec in specs
                    if spec.policy == suffix
                ]
                path = (
                    error_metrics_dir
                    / f"corrected_rate_comparison_duet_{suffix}_trial_{trial}.svg"
                )
                if pairs and writes_debug(path):
                    plot_error_metrics_bar_comparison(results_df, pairs, trial, path)
    else:
        print("\nSkipping error correction metrics visualizations (columns not found in results.csv)")
        print("  Required columns: No_error, Corrected, Failed")

    if n_skipped:
        print(f"\nSkipped {n_skipped} debug plots; pass {DEBUG_PLOTS_FLAG} to write them.")
    tiers.report_missing_paper_panels()
    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()