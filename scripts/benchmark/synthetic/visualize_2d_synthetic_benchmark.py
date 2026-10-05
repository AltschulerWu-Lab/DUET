#!/usr/bin/env python3
"""
Visualize 2-D synthetic benchmark output.

Reads results.parquet and hv_summary.parquet from a benchmark run and
renders the following artifacts. Items 3 and 4 are debug plots
(duet.plotting.tiers), written only with --debug-plots; the rest are written
on every run. aggregate_hvr.svg is the Supp Fig S1 panel.

1. recovered_fronts.svg — small multiples, one panel per (noise_channel,
   error_rate). Per (method, lambda) mean ± SE markers across trials, with
   a thin connector through means sorted by mean decode accuracy. DUET,
   Greedy-NLL-MO, and Greedy-Hamming-MO; exhaustive is excluded from this
   view because per-trial pools are different problems and their Pareto
   fronts cannot be aggregated meaningfully.

2. recovered_fronts_per_trial.svg — same panel layout, all four methods
   (DUET, Greedy-NLL-MO, Greedy-Hamming-MO, Exhaustive). Per (method,
   trial) Pareto front rendered as a thin line connecting non-dominated
   points sorted by decode accuracy, every trial overlaid in one file (a
   diagnostic, despite the name: the only default view that draws the
   exhaustive front). Method = color, trial = no visual
   encoding (overlap density carries the trial-to-trial spread). Straight
   lines rather than step plots — overlaying staircases of different
   densities produces a misleading visual where the coarser front's
   flat-tops appear to dominate the finer front between its kinks.

3. raw_scatter_per_trial/trial_NN.svg (debug) — one figure per trial,
   same panel layout, only that trial's raw points (one per lambda for
   DUET, Greedy-NLL-MO and Greedy-Hamming-MO, duplicates dropped, plus the
   exhaustive front as a line per panel).

4. pareto_per_trial/trial_NN.svg (debug) — companion to artifact 3. Same per-
   trial panel layout, but each method's points are filtered to its
   recovered Pareto front (markers only; no connecting line, since
   intermediate trade-offs between discrete codebooks aren't actually
   achievable). Exhaustive remains drawn as a line — in the small
   regime it represents the true Pareto front and is the ceiling
   reference. Useful when the raw scatter is cluttered by dominated
   points near the front.

5. aggregate_hv.svg — 4×3 grid of normalized-HV box plots (one panel per
   noise channel × error rate). Methods on the x-axis: Optimal (exhaustive,
   small regime only), DUET, NLL, Hamming. Mirrors the 1-D synthetic
   benchmark's grid layout. No y=1 reference line — normalized_hv's
   ceiling (utopia bounding box) is unreachable by construction.

6. aggregate_hvr.svg (small regime only) — 4×3 grid of HVR box plots,
   parallel to aggregate_hv.svg. HVR = HV(method) / HV(exhaustive).
   Exhaustive is omitted from the x-axis (its HVR is 1.0 by
   construction, redundant with the y=1 reference line); a dashed
   y=1 line is drawn per panel as the achievable ceiling. Skipped in
   the large regime since HVR requires a known true Pareto front.

7. aggregate_igd.svg (small regime only) — bar plot of IGD against the
   exhaustive front per (method, noise, error) with 95% CI across
   trials. DUET, Greedy-NLL-MO, and Greedy-Hamming-MO; lower is better.
   Skipped in the large regime since IGD requires a known true Pareto
   front.

Usage:
    python -m scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark \
        --indir results/benchmark/2d_synthetic/small [--debug-plots]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from duet.benchmark.metrics import compute_igd
from duet.plotting import (
    FIGURE_WIDTHS,
    METHOD_MARKERS,
    METHOD_PALETTE,
    OKABE_ITO,
    apply_style,
)
from duet.plotting.tiers import add_debug_plots_arg


METHOD_LABEL = {
    "duet": "DUET",
    "greedy_nll_mo": "Greedy-NLL-MO",
    "greedy_hamming_mo": "Greedy-Hamming-MO",
    "exhaustive": "Exhaustive",
}

# Methods that have a lambda sweep and so can be aggregated by lambda
# across trials. Exhaustive does not have a lambda; per-trial pools are
# different problems, so exhaustive Pareto fronts cannot be aggregated.
# This is also the canonical "non-exhaustive method set" — the two
# inline tuples in plot_per_trial_raw_scatters and plot_aggregate_igd
# both iterate AGGREGATABLE_METHODS so adding a future method only
# requires extending this constant.
AGGREGATABLE_METHODS = ("duet", "greedy_nll_mo", "greedy_hamming_mo")

# Box-plot grid display conventions. Order: ceiling first (exhaustive →
# Optimal), then DUET, then the two greedy baselines from most-informed
# (NLL, noise-aware distance) to least-informed (Hamming, noise-agnostic).
HV_METHOD_ORDER = ["exhaustive", "duet", "greedy_nll_mo", "greedy_hamming_mo"]
HV_METHOD_DISPLAY = {
    "exhaustive": "Optimal",
    "duet": "DUET",
    # Newline keeps tick labels narrow so they don't overlap at the 4×3
    # grid's per-panel x-axis width.
    "greedy_nll_mo": "Greedy\nNLL",
    "greedy_hamming_mo": "Greedy\nHamming",
}
HV_BOX_PALETTE = {
    "Optimal": OKABE_ITO["bluish_green"],
    "DUET": METHOD_PALETTE["DUET"],
    "Greedy\nNLL": METHOD_PALETTE["Greedy-NLL-MO"],
    "Greedy\nHamming": METHOD_PALETTE["Greedy-Hamming-MO"],
}
NOISE_CHANNEL_LABELS = {
    "symmetric": "Symmetric",
    "position_varying": "Position-varying",
    "asymmetric": "Asymmetric",
    "position_varying_asymmetric": "Position-varying, asymmetric",
}


def _color(method_key: str) -> str:
    return METHOD_PALETTE[METHOD_LABEL[method_key]]


def _marker(method_key: str) -> str:
    """Marker shape for a method, keyed by the snake_case method id."""
    return METHOD_MARKERS[METHOD_LABEL[method_key]]


def _compute_se(x: pd.Series) -> float:
    """Standard error of the mean. Matches scripts/benchmark/visualize_comparison.py:_compute_se."""
    n = x.notna().sum()
    if n < 2:
        return 0.0
    return float(x.std() / np.sqrt(n))


def _panel_grid(results_df: pd.DataFrame):
    """Return (noise_levels, error_rates, nrows, ncols) for the small-multiples grid.

    `noise_levels` is ordered by `NOISE_CHANNEL_LABELS` (symmetric → position_varying →
    asymmetric → position_varying_asymmetric) — the conceptual progression from simplest
    noise to most-structured. Default alphabetical sort would put `asymmetric`
    first, which reads less naturally for a reader scanning the grid.
    Unknown noise channels (defensive — shouldn't happen in practice) sort
    after the known ones.
    """
    noise_keys = list(NOISE_CHANNEL_LABELS.keys())
    noise_levels = sorted(
        results_df["noise_channel"].unique(),
        key=lambda x: noise_keys.index(x) if x in noise_keys else len(noise_keys),
    )
    rates = sorted(results_df["error_rate"].unique())
    return noise_levels, rates, len(noise_levels), len(rates)


def _label_panel(ax, *, i, j, nrows, noise, rate):
    """Common per-panel labeling for the small-multiples figures."""
    if i == 0:
        ax.set_title(f"err={rate}")
    if j == 0:
        ax.set_ylabel(f"{noise}\nmean score")
    if i == nrows - 1:
        ax.set_xlabel("decode accuracy")


def _pareto_front_xy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Filter (decode, score) points to the non-dominated subset and sort by decode ascending.

    Returns (xs, ys) suitable for either ax.plot or ax.step, or None for empty input.
    Maximize-maximize convention: pymoo assumes minimization, so points are negated.
    """
    if len(points) == 0:
        return None
    keep = find_non_dominated(-points)
    front = points[keep]
    order = np.argsort(front[:, 0])
    return front[order, 0], front[order, 1]


# ---------------------------------------------------------------------------
# Plot 1: aggregated recovered fronts (DUET + Greedy, mean ± SE)
# ---------------------------------------------------------------------------

def plot_recovered_fronts_aggregated(results_df: pd.DataFrame, outdir: Path) -> None:
    """Per (method, lambda) mean ± SE markers connected by sorted decode_mean.

    Excludes exhaustive (cannot be aggregated across distinct pools).
    """
    if results_df.empty:
        print("results_df is empty; skipping recovered_fronts.svg.")
        return

    noise_levels, rates, nrows, ncols = _panel_grid(results_df)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIGURE_WIDTHS["full_page"], 2.0 * nrows),
        squeeze=False,
        sharex=True, sharey=True,
    )

    agg_methods = [m for m in AGGREGATABLE_METHODS if m in results_df["method"].unique()]

    for i, noise in enumerate(noise_levels):
        for j, rate in enumerate(rates):
            ax = axes[i, j]
            cell = results_df[
                (results_df["noise_channel"] == noise) &
                (np.isclose(results_df["error_rate"], rate))
            ]
            for method_key in agg_methods:
                method_pts = cell[cell["method"] == method_key]
                if method_pts.empty:
                    continue
                stats = (
                    method_pts.groupby("lambda")
                    .agg(
                        x_mean=("decode_accuracy", "mean"),
                        x_se=("decode_accuracy", _compute_se),
                        y_mean=("mean_score", "mean"),
                        y_se=("mean_score", _compute_se),
                    )
                    .reset_index()
                    .sort_values("x_mean")
                )
                color = _color(method_key)
                ax.plot(
                    stats["x_mean"], stats["y_mean"],
                    color=color, linewidth=0.8, alpha=0.4, zorder=1,
                )
                ax.errorbar(
                    stats["x_mean"], stats["y_mean"],
                    xerr=stats["x_se"], yerr=stats["y_se"],
                    fmt=_marker(method_key), color=color,
                    label=METHOD_LABEL[method_key],
                    markersize=2.5, capsize=0, alpha=0.85, zorder=2,
                )
            _label_panel(ax, i=i, j=j, nrows=nrows, noise=noise, rate=rate)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.tight_layout(rect=[0, 0, 0.95, 1.0])
    fig.savefig(outdir / "recovered_fronts.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: per-trial staircases overlay (all three methods)
# ---------------------------------------------------------------------------

def plot_recovered_fronts_per_trial(results_df: pd.DataFrame, outdir: Path) -> None:
    """Per (method, trial) Pareto front as a thin line.

    Method encoded by color (METHOD_PALETTE); trial not visually encoded —
    overlap density carries the trial-to-trial variability signal.
    Straight lines rather than step plots: overlaying staircases of
    different densities produces a misleading visual where the coarser
    method's flat-tops appear to dominate the finer one's kinked region,
    even though no point of one method actually dominates any point of
    the other.
    """
    if results_df.empty:
        print("results_df is empty; skipping recovered_fronts_per_trial.svg.")
        return

    noise_levels, rates, nrows, ncols = _panel_grid(results_df)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIGURE_WIDTHS["full_page"], 2.0 * nrows),
        squeeze=False,
        sharex=True, sharey=True,
    )

    methods_present = [m for m in METHOD_LABEL if m in results_df["method"].unique()]

    for i, noise in enumerate(noise_levels):
        for j, rate in enumerate(rates):
            ax = axes[i, j]
            cell = results_df[
                (results_df["noise_channel"] == noise) &
                (np.isclose(results_df["error_rate"], rate))
            ]
            for method_key in methods_present:
                method_pts = cell[cell["method"] == method_key]
                if method_pts.empty:
                    continue
                color = _color(method_key)
                for trial, trial_pts in method_pts.groupby("trial"):
                    points = trial_pts[["decode_accuracy", "mean_score"]].to_numpy()
                    front = _pareto_front_xy(points)
                    if front is None:
                        continue
                    xs, ys = front
                    ax.plot(
                        xs, ys,
                        color=color, linewidth=0.6, alpha=0.4,
                    )
            _label_panel(ax, i=i, j=j, nrows=nrows, noise=noise, rate=rate)

    # Method-level legend (one handle per method, full opacity for legibility).
    legend_handles = [
        plt.Line2D([0], [0], color=_color(m), linewidth=1.2, label=METHOD_LABEL[m])
        for m in methods_present
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.tight_layout(rect=[0, 0, 0.95, 1.0])
    fig.savefig(outdir / "recovered_fronts_per_trial.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: per-trial raw scatters (one SVG per trial)
# ---------------------------------------------------------------------------

def plot_per_trial_raw_scatters(results_df: pd.DataFrame, outdir: Path) -> None:
    """One SVG per trial — same panel layout, only that trial's raw points."""
    if results_df.empty:
        print("results_df is empty; skipping raw_scatter_per_trial/.")
        return

    subdir = outdir / "raw_scatter_per_trial"
    subdir.mkdir(parents=True, exist_ok=True)

    noise_levels, rates, nrows, ncols = _panel_grid(results_df)
    trials = sorted(results_df["trial"].unique())

    for trial in trials:
        trial_df = results_df[results_df["trial"] == trial]
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(FIGURE_WIDTHS["full_page"], 2.0 * nrows),
            squeeze=False,
            sharex=True, sharey=True,
        )
        for i, noise in enumerate(noise_levels):
            for j, rate in enumerate(rates):
                ax = axes[i, j]
                cell = trial_df[
                    (trial_df["noise_channel"] == noise) &
                    (np.isclose(trial_df["error_rate"], rate))
                ]
                for method_key in AGGREGATABLE_METHODS:
                    pts = cell[cell["method"] == method_key]
                    if len(pts):
                        # Deduplicate within method so identical lambda-runs
                        # don't stack at the same (decode, score) and read as
                        # a single near-opaque marker.
                        unique_pts = pts.drop_duplicates(
                            subset=["decode_accuracy", "mean_score"]
                        )
                        ax.scatter(
                            unique_pts["decode_accuracy"], unique_pts["mean_score"],
                            s=14, alpha=0.7,
                            color=_color(method_key),
                            marker=_marker(method_key),
                            edgecolors="none",
                            label=METHOD_LABEL[method_key],
                        )
                exh = cell[cell["method"] == "exhaustive"]
                if len(exh):
                    points = exh[["decode_accuracy", "mean_score"]].to_numpy()
                    front = _pareto_front_xy(points)
                    if front is not None:
                        xs, ys = front
                        ax.plot(
                            xs, ys,
                            color=_color("exhaustive"),
                            linewidth=1.0,
                            label=METHOD_LABEL["exhaustive"],
                        )
                _label_panel(ax, i=i, j=j, nrows=nrows, noise=noise, rate=rate)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=True)
        fig.suptitle(f"trial {int(trial):02d}", y=0.995)
        fig.tight_layout(rect=[0, 0, 0.95, 0.99])
        fig.savefig(subdir / f"trial_{int(trial):02d}.svg")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3b: per-trial Pareto fronts (one SVG per trial, filtered to fronts)
# ---------------------------------------------------------------------------

def plot_per_trial_pareto_fronts(results_df: pd.DataFrame, outdir: Path) -> None:
    """One SVG per trial — same panel layout as raw_scatter_per_trial, but
    each method's points filtered to its recovered Pareto front.

    Companion to plot_per_trial_raw_scatters: the raw-scatter view shows
    every lambda landing for every method (useful for seeing lambda-sweep
    density and dominated-point clusters near the front). This view
    collapses each method down to its non-dominated subset, making
    cross-method front comparisons easier when the raw scatter is
    cluttered.

    Markers only (no connecting lines) for DUET / Greedy-NLL-MO /
    Greedy-Hamming-MO — connectors between non-dominated points imply
    that intermediate trade-offs along the segment are achievable, which
    is not true for discrete codebooks. Exhaustive remains drawn as a
    line because in the small regime it represents the true Pareto
    front, not a method's recovered front, and the connecting line is
    the established visual convention for "ceiling reference" elsewhere
    in the suite.
    """
    if results_df.empty:
        print("results_df is empty; skipping pareto_per_trial/.")
        return

    subdir = outdir / "pareto_per_trial"
    subdir.mkdir(parents=True, exist_ok=True)

    noise_levels, rates, nrows, ncols = _panel_grid(results_df)
    trials = sorted(results_df["trial"].unique())

    for trial in trials:
        trial_df = results_df[results_df["trial"] == trial]
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(FIGURE_WIDTHS["full_page"], 2.0 * nrows),
            squeeze=False,
            sharex=True, sharey=True,
        )
        for i, noise in enumerate(noise_levels):
            for j, rate in enumerate(rates):
                ax = axes[i, j]
                cell = trial_df[
                    (trial_df["noise_channel"] == noise) &
                    (np.isclose(trial_df["error_rate"], rate))
                ]
                for method_key in AGGREGATABLE_METHODS:
                    pts = cell[cell["method"] == method_key]
                    if len(pts) == 0:
                        continue
                    points = pts[["decode_accuracy", "mean_score"]].to_numpy()
                    front = _pareto_front_xy(points)
                    if front is None:
                        continue
                    xs, ys = front
                    ax.scatter(
                        xs, ys,
                        s=14, alpha=0.85,
                        color=_color(method_key),
                        marker=_marker(method_key),
                        edgecolors="none",
                        label=METHOD_LABEL[method_key],
                    )
                exh = cell[cell["method"] == "exhaustive"]
                if len(exh):
                    points = exh[["decode_accuracy", "mean_score"]].to_numpy()
                    front = _pareto_front_xy(points)
                    if front is not None:
                        xs, ys = front
                        ax.plot(
                            xs, ys,
                            color=_color("exhaustive"),
                            linewidth=1.0,
                            label=METHOD_LABEL["exhaustive"],
                        )
                _label_panel(ax, i=i, j=j, nrows=nrows, noise=noise, rate=rate)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=True)
        fig.suptitle(f"trial {int(trial):02d}", y=0.995)
        fig.tight_layout(rect=[0, 0, 0.95, 0.99])
        fig.savefig(subdir / f"trial_{int(trial):02d}.svg")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: aggregate HV bar plot (unchanged)
# ---------------------------------------------------------------------------

def plot_aggregate_hv(hv_df: pd.DataFrame, outdir: Path) -> None:
    """4×3 grid of normalized-HV box plots — one panel per (noise, error_rate).

    Mirrors the 1-D synthetic benchmark layout (boxplots per method per cell,
    bold noise-model row labels, error-rate column titles, sharey=False so
    each cell scales independently). Includes Exhaustive ("Optimal") as the
    ceiling reference; falls back to two-method panels in the large regime
    where exhaustive rows are absent.
    """
    if hv_df.empty:
        print("hv_df is empty; skipping aggregate_hv.svg.")
        return

    df = hv_df.copy()
    methods_present = [m for m in HV_METHOD_ORDER if m in df["method"].unique()]
    df = df[df["method"].isin(methods_present)].copy()
    df["method_label"] = df["method"].map(HV_METHOD_DISPLAY)
    method_order_display = [HV_METHOD_DISPLAY[m] for m in methods_present]

    noise_keys = list(NOISE_CHANNEL_LABELS.keys())
    noise_channels = sorted(
        df["noise_channel"].unique(),
        key=lambda x: noise_keys.index(x) if x in noise_keys else len(noise_keys),
    )
    error_rates = sorted(df["error_rate"].unique())

    n_rows = len(noise_channels)
    n_cols = len(error_rates)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGURE_WIDTHS["full_page"], 6.5),
        squeeze=False,
        sharey=False,
    )

    for i, noise in enumerate(noise_channels):
        for j, rate in enumerate(error_rates):
            ax = axes[i, j]
            cell = df[
                (df["noise_channel"] == noise)
                & (np.isclose(df["error_rate"], rate))
            ].copy()
            if cell.empty:
                ax.set_visible(False)
                continue
            cell["method_label"] = pd.Categorical(
                cell["method_label"],
                categories=method_order_display,
                ordered=True,
            )

            sns.boxplot(
                data=cell,
                x="method_label",
                y="normalized_hv",
                hue="method_label",
                order=method_order_display,
                hue_order=method_order_display,
                palette=HV_BOX_PALETTE,
                legend=False,
                dodge=False,
                width=0.3,
                flierprops={"marker": ".", "markersize": 4},
                ax=ax,
            )

            if i == 0:
                ax.set_title(f"Error rate = {rate * 100:.0f}%")

            if j == 0:
                ax.set_ylabel("Normalized\nhypervolume")
                ax.annotate(
                    NOISE_CHANNEL_LABELS.get(noise, noise),
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 30, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    fontweight="bold",
                    ha="right",
                    va="center",
                    rotation=90,
                )
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            if i == n_rows - 1:
                ax.set_xticks(range(len(method_order_display)))
                ax.set_xticklabels(method_order_display)
            else:
                ax.set_xticklabels([])

    plt.tight_layout(rect=[0.04, 0.04, 1.0, 1.0])
    frame_size = plt.rcParams["figure.titlesize"]
    fig.text(
        0.5, 0.01, "Method",
        ha="center", fontsize=frame_size, fontweight="bold",
    )
    fig.text(
        0.01, 0.5, "Noise Channel",
        ha="center", va="center",
        fontsize=frame_size, fontweight="bold", rotation="vertical",
    )
    fig.savefig(outdir / "aggregate_hv.svg")
    plt.close(fig)


def plot_aggregate_hvr(hv_df: pd.DataFrame, outdir: Path) -> None:
    """4×3 grid of HVR box plots — one panel per (noise, error_rate).

    Small-regime only. HVR = HV(method) / HV(exhaustive); 1.0 means the
    method recovered the full true-PF hypervolume. A dashed y=1 reference
    line is drawn per panel so readers can read off "what fraction of
    achievable HV did this method capture?". The line uses the same
    vermillion dashed style as Fig. 2B's rho = 1 reference, and is labelled
    "Optimal" once per row — inside the last panel, just above the line —
    so the 4x3 grid carries four labels rather than twelve while keeping
    the panel width unchanged. Skipped entirely when no cell has a finite
    hvr (i.e., the input data is all large-regime).

    Exhaustive is omitted from the x-axis: its HVR is 1.0 by construction,
    making its box redundant with the y=1 reference line.
    """
    if hv_df.empty:
        print("hv_df is empty; skipping aggregate_hvr.svg.")
        return

    df = hv_df.copy()
    # Backward compat: hv_summary.parquet files written before 2026-05-11
    # do not have an `hvr` column. Without this guard, rendering such an
    # older run would raise KeyError. The notna() check additionally skips runs that
    # have the column but are pure-large-regime (every value NaN).
    if "hvr" not in df.columns or df["hvr"].notna().sum() == 0:
        print("No finite hvr values (large regime only); skipping aggregate_hvr.svg.")
        return

    df = df[df["hvr"].notna()].copy()
    # Drop exhaustive: its HVR is 1.0 by construction, so its box would
    # collapse to a single point on the y=1 reference line and add no
    # information beyond what the line already conveys. Keep it in
    # aggregate_hv.svg (which uses normalized_hv, not HVR — exhaustive
    # there does carry information).
    methods_present = [
        m for m in HV_METHOD_ORDER
        if m in df["method"].unique() and m != "exhaustive"
    ]
    df = df[df["method"].isin(methods_present)].copy()
    df["method_label"] = df["method"].map(HV_METHOD_DISPLAY)
    method_order_display = [HV_METHOD_DISPLAY[m] for m in methods_present]

    noise_keys = list(NOISE_CHANNEL_LABELS.keys())
    noise_channels = sorted(
        df["noise_channel"].unique(),
        key=lambda x: noise_keys.index(x) if x in noise_keys else len(noise_keys),
    )
    error_rates = sorted(df["error_rate"].unique())

    n_rows = len(noise_channels)
    n_cols = len(error_rates)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGURE_WIDTHS["full_page"], 6.5),
        squeeze=False,
        sharey=False,
    )

    for i, noise in enumerate(noise_channels):
        for j, rate in enumerate(error_rates):
            ax = axes[i, j]
            cell = df[
                (df["noise_channel"] == noise)
                & (np.isclose(df["error_rate"], rate))
            ].copy()
            if cell.empty:
                ax.set_visible(False)
                continue
            cell["method_label"] = pd.Categorical(
                cell["method_label"],
                categories=method_order_display,
                ordered=True,
            )

            # y=1 reference (true-PF ceiling) — drawn behind the boxes, in
            # the same style as Fig. 2B's "Optimal" line.
            ax.axhline(
                1.0, ls="--", lw=1, color=OKABE_ITO["vermillion"], zorder=0,
            )

            sns.boxplot(
                data=cell,
                x="method_label",
                y="hvr",
                hue="method_label",
                order=method_order_display,
                hue_order=method_order_display,
                palette=HV_BOX_PALETTE,
                legend=False,
                dodge=False,
                width=0.3,
                flierprops={"marker": ".", "markersize": 4},
                ax=ax,
            )

            # Headroom above the ceiling so whiskers that reach 1.0 and the
            # row's "Optimal" label clear the line; no tick is placed above
            # 1.0 because HVR cannot exceed it.
            y_min, _ = ax.get_ylim()
            ax.set_ylim(y_min, 1.0 + 0.12 * (1.0 - y_min))
            ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0 + 1e-9])

            # Label the ceiling once per row, in the last column, riding the
            # right end of the line (Fig. 2B convention, one label per row).
            if j == n_cols - 1:
                ax.annotate(
                    "Optimal",
                    xy=(1.0, 1.0),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 1.5),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    color=OKABE_ITO["vermillion"],
                    fontsize=8,
                )

            if i == 0:
                ax.set_title(f"Error rate = {rate * 100:.0f}%")

            if j == 0:
                ax.set_ylabel("Hypervolume ratio")
                ax.annotate(
                    NOISE_CHANNEL_LABELS.get(noise, noise),
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 30, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    fontweight="bold",
                    ha="right",
                    va="center",
                    rotation=90,
                )
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            if i == n_rows - 1:
                ax.set_xticks(range(len(method_order_display)))
                ax.set_xticklabels(method_order_display)
            else:
                ax.set_xticklabels([])

    plt.tight_layout(rect=[0.04, 0.04, 1.0, 1.0])
    frame_size = plt.rcParams["figure.titlesize"]
    fig.text(
        0.5, 0.01, "Method",
        ha="center", fontsize=frame_size, fontweight="bold",
    )
    fig.text(
        0.01, 0.5, "Noise Channel",
        ha="center", va="center",
        fontsize=frame_size, fontweight="bold", rotation="vertical",
    )
    fig.savefig(outdir / "aggregate_hvr.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: aggregate IGD bar plot (small regime only)
# ---------------------------------------------------------------------------

def plot_aggregate_igd(results_df: pd.DataFrame, outdir: Path) -> None:
    """IGD against the exhaustive front, per (method, noise, error_rate).

    Computed only when exhaustive rows exist (small regime). For each
    (trial, noise, error) cell, the recovered front of each non-exhaustive
    method is filtered to its non-dominated subset and IGD is computed
    against that cell's exhaustive front. Bars are then aggregated across
    trials with 95% CI error bars, mirroring plot_aggregate_hv.
    """
    exh_df = results_df[results_df["method"] == "exhaustive"]
    if exh_df.empty:
        print("No exhaustive front (large regime); skipping aggregate_igd.svg.")
        return

    rows = []
    group_cols = ["trial", "noise_channel", "error_rate"]
    for (trial, noise, error), group in results_df.groupby(group_cols):
        true_pts = group[group["method"] == "exhaustive"][
            ["decode_accuracy", "mean_score"]
        ].to_numpy()
        if len(true_pts) == 0:
            continue
        for method_key in AGGREGATABLE_METHODS:
            method_df = group[group["method"] == method_key]
            if method_df.empty:
                continue
            method_pts = method_df[["decode_accuracy", "mean_score"]].to_numpy()
            front = _pareto_front_xy(method_pts)
            if front is None:
                continue
            xs, ys = front
            approx_pts = np.column_stack([xs, ys])
            igd_val = compute_igd(approx_pts, true_pts, maximize=True)
            rows.append({
                "trial": trial,
                "noise_channel": noise,
                "error_rate": error,
                "method": method_key,
                "igd": igd_val,
            })

    if not rows:
        print("No aggregatable method points found; skipping aggregate_igd.svg.")
        return

    plot_df = pd.DataFrame(rows)
    plot_df["method_label"] = plot_df["method"].map(METHOD_LABEL)
    plot_df["cell"] = plot_df.apply(
        lambda r: f"{r['noise_channel']}\nerr={r['error_rate']}", axis=1,
    )

    palette = {METHOD_LABEL[m]: _color(m) for m in AGGREGATABLE_METHODS}

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["full_page"], 3.0))
    sns.barplot(
        data=plot_df,
        x="cell", y="igd", hue="method_label",
        palette=palette,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("IGD (vs exhaustive front)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outdir / "aggregate_igd.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2-D synthetic benchmark output.",
    )
    parser.add_argument(
        "--indir", required=True,
        help="Run directory containing results.parquet and hv_summary.parquet.",
    )
    add_debug_plots_arg(parser)
    args = parser.parse_args()

    indir = Path(args.indir)
    results_df = pd.read_parquet(indir / "results.parquet")
    hv_df = pd.read_parquet(indir / "hv_summary.parquet")

    apply_style()
    plot_recovered_fronts_aggregated(results_df, indir)
    plot_recovered_fronts_per_trial(results_df, indir)
    if args.debug_plots:
        plot_per_trial_raw_scatters(results_df, indir)
        plot_per_trial_pareto_fronts(results_df, indir)
    plot_aggregate_hv(hv_df, indir)
    plot_aggregate_hvr(hv_df, indir)
    plot_aggregate_igd(results_df, indir)
    # Every plot function that draws nothing prints a "skipping" line.
    print(
        f"Wrote recovered_fronts.svg, recovered_fronts_per_trial.svg, "
        f"aggregate_hv.svg, aggregate_hvr.svg and aggregate_igd.svg to {indir}, "
        f"except any named in a 'skipping' line above"
    )
    if args.debug_plots:
        print(f"Wrote the per-trial debug plots to {indir / 'raw_scatter_per_trial'}/ "
              f"and {indir / 'pareto_per_trial'}/")
    else:
        n_skipped = 2 * results_df["trial"].nunique()
        print(f"Skipped {n_skipped} per-trial debug plots (raw_scatter_per_trial/, "
              f"pareto_per_trial/); pass --debug-plots to write them.")


if __name__ == "__main__":
    main()
