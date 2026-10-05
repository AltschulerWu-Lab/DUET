#!/usr/bin/env python3
"""Sweep-level figures for the OPS rounds x epsilon robustness experiment (Fig 3c).

Reads each grid point's results.csv plus the config.yaml the runner snapshots
beside it, computes one shared DUET reference arm per (point, trial) at >= 97.5%
of maximum activity, and renders epsilon x rounds heatmaps of the delta against
each baseline. Fig 3c is two of them (FIG3C_PANELS below).

Analysis functions are imported from scripts/benchmark/visualize_benchmark.py
rather than reimplemented, so this sweep and the per-point panels cannot drift
apart in aggregation semantics or reference policy.

Migrated copy of scripts/benchmark/archive/08-20-2026/robustness_sweep/visualize_sweep.py.
Changes from the as-run script:
  * Paths. The point list and results directory come from this folder's configs
    (`<stem>.yaml`, or `<stem>.<variant>.yaml` with --variant / VARIANT): each
    config's `outdir`, resolved against this folder. The figures go to
    `<results dir>/sweep_figures/` as before.
  * The built-in check. The as-run `check_overlap` compared the rounds=10 cells
    with the 08-18 runs to 1e-9; it never passed (the 31-value lambda grid
    changes the union-indexed ground-truth draws). It is replaced by
    `check_reference`, against experiments/ops_crispri_symmetric (Fig 3b; same
    pool, channel and seeds at rounds 10, epsilon 0.1): Tier A exact, Tier B
    statistical. See that function and README.md.
  * The closing print names the two Fig 3c panels instead of the
    mean-decode-accuracy heatmaps.
  * A point whose snapshotted config differs from the current config is flagged.
  * Output tiers (duet.plotting.tiers). By default only the two Fig 3c panels
    and the six mean-decode-accuracy heatmaps (DIAGNOSTIC_METRICS) are drawn;
    the other 52 heatmaps are debug plots, drawn with --debug-plots. The as-run
    script drew all 60. sweep_summary.csv holds every cell either way.
No lambda value is hard-coded here. The 97.5% arm comes from `select_duet_lambda`
(first idxmax over label-sorted rows), unchanged; under the flipped convention a
tie would resolve to the other end of the grid, but the as-run picks had none.

Usage:
    python visualize_sweep.py                  # the 18 <stem>.yaml points
    python visualize_sweep.py --variant smoke  # the <stem>.smoke.yaml points present
    python visualize_sweep.py --debug-plots    # also the other 52 heatmaps
    python visualize_sweep.py --results-base DIR --figures-dir DIR --no-reference-check
                                               # re-render another copy of the sweep
`--variant` defaults to $VARIANT. Exits 1, after writing the figures, if the
Tier A reference check finds a mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import Normalize, TwoSlopeNorm

from duet.plotting import (
    FIGURE_WIDTHS,
    MM_PER_INCH,
    PANEL_GAP_MM,
    apply_style,
    delta_diverging_cmap,
    gain_sequential_cmap,
    reduction_sequential_cmap,
    save_panel,
    slot_figure,
)
from duet.plotting.tiers import add_debug_plots_arg

apply_style()

SWEEP_DIR = Path(__file__).resolve().parent
# experiments/<name>/ -> the repo root is two levels up.
REPO_ROOT = SWEEP_DIR.parents[1]
BENCHMARK_SCRIPTS = REPO_ROOT / "scripts" / "benchmark"
REFERENCE_EXPERIMENT = REPO_ROOT / "experiments" / "ops_crispri_symmetric"

sys.path.insert(0, str(BENCHMARK_SCRIPTS))
from visualize_benchmark import (  # noqa: E402
    DUET_REFERENCE_FRACTIONS,
    MAX_ACTIVITY_METHOD,
    STACK_HEIGHT_OVERLAY,
    load_and_process_results,
    select_duet_lambda,
)

SEQ_ROUNDS = [8, 9, 10, 11, 12, 13]
EPSILONS = [0.03, 0.05, 0.10]
STEMS = [f"sr{r:02d}_eps{e:.2f}" for r in SEQ_ROUNDS for e in EPSILONS]

_STEM_RE = re.compile(r"^sr(\d{2})_eps(\d\.\d{2})$")


def parse_stem(stem: str) -> tuple[int, float]:
    """Recover (rounds, epsilon) from a stem such as 'sr08_eps0.03'."""
    m = _STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"Malformed sweep stem: {stem!r}")
    return int(m.group(1)), float(m.group(2))


def config_path(stem: str, variant: str = "") -> Path:
    """This folder's config for `stem`: `<stem>.yaml` or `<stem>.<variant>.yaml`."""
    return SWEEP_DIR / f"{stem}{'.' + variant if variant else ''}.yaml"


def sweep_points(variant: str = "") -> dict[str, Path]:
    """stem -> results directory, read from the configs beside this script.

    Without a variant all 18 committed configs must exist. With one (e.g.
    `smoke`) the points are whichever `<stem>.<variant>.yaml` files exist.
    Each `outdir` is relative to this folder (run.sh cd's here to run it).
    """
    points: dict[str, Path] = {}
    for stem in STEMS:
        cfg = config_path(stem, variant)
        if not cfg.is_file():
            if not variant:
                raise SystemExit(f"Missing committed config {cfg}; run make_configs.py --check")
            continue
        outdir = yaml.safe_load(cfg.read_text())["outdir"]
        points[stem] = Path(os.path.normpath(SWEEP_DIR / outdir))
    if not points:
        raise SystemExit(f"No {variant or 'real'} configs found in {SWEEP_DIR}")
    return points


def results_base_of(points: dict[str, Path]) -> Path:
    """The one directory every point's outdir sits in (`<base>/<stem>`)."""
    bases = {p.parent for p in points.values()}
    wrong = [s for s, p in points.items() if p.name != s]
    if len(bases) != 1 or wrong:
        raise SystemExit(
            f"Point outdirs must be <one base>/<stem>; got bases {sorted(map(str, bases))}"
            + (f", misnamed {wrong}" if wrong else "")
        )
    return bases.pop()


def point_params(results_dir: Path) -> tuple[int, float]:
    """Recover (rounds, epsilon) from the runner's snapshotted config.yaml.

    Reading the snapshot rather than the stem is what makes provenance
    recorded rather than reconstructed: the snapshot is the config the run
    actually used.
    """
    cfg = yaml.safe_load((results_dir / "config.yaml").read_text())
    return (
        int(cfg["candidate_pool"]["seq_rounds"]),
        float(cfg["evaluator"]["noise_channel"]["epsilon"]),
    )


def augment_agg(results_df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Add the three dispersion metrics `load_and_process_results` omits."""
    extra = (
        results_df.groupby(["Method", "Trial"])["Decode accuracy"]
        .agg(
            **{
                "Median absolute deviation decode accuracy": lambda x: (
                    x - x.median()
                ).abs().median(),
                "Mean absolute deviation decode accuracy": lambda x: (
                    x - x.mean()
                ).abs().mean(),
            }
        )
        .reset_index()
    )
    agg = agg.merge(extra, on=["Method", "Trial"], how="left")
    agg["90% percentile interval length decode accuracy"] = (
        agg["95th percentile decode accuracy"] - agg["5th percentile decode accuracy"]
    )
    return agg


def expected_baselines(cfg: dict) -> set[str]:
    """Every non-DUET method label this config's point is committed to.

    Always includes Maximum activity, plus one label per configured
    Sivanandan/Feldman edit distance -- regardless of whether that baseline
    ends up with any valid codebook. Feldman ED=2 and Sivanandan ED=3 have
    historically produced zero valid rows on this candidate pool at some
    points; a baseline that resolves to nothing must still be named as
    expected, never inferred from which methods happened to survive the Valid
    filter.
    """
    labels = {MAX_ACTIVITY_METHOD}
    for ed in cfg.get("sivanandan", {}).get("edit_distances", []) or []:
        labels.add(f"Sivanandan et al. (ED={ed})")
    for ed in cfg.get("feldman", {}).get("edit_distances", []) or []:
        labels.add(f"Feldman et al. (ED={ed})")
    return labels


def load_point(results_dir: Path, stem: str | None = None) -> pd.DataFrame | None:
    """Load and aggregate one grid point. Returns None (loudly) if absent.

    If stem is provided, use it instead of parsing from the directory name.
    This is useful for reference directories that aren't named with the stem convention.
    """
    if stem is None:
        stem = results_dir.name
    results_csv = results_dir / "results.csv"
    config_yaml = results_dir / "config.yaml"
    if not results_csv.exists() or not config_yaml.exists():
        print(f"SKIP {stem}: missing results.csv or config.yaml in {results_dir}")
        return None

    stem_rounds, stem_eps = parse_stem(stem)
    cfg_rounds, cfg_eps = point_params(results_dir)
    if (stem_rounds, stem_eps) != (cfg_rounds, cfg_eps):
        raise ValueError(
            f"{stem}: stem disagrees with the snapshotted config "
            f"(stem says rounds={stem_rounds}, eps={stem_eps}; "
            f"config says rounds={cfg_rounds}, eps={cfg_eps})"
        )

    results_df, agg = load_and_process_results(results_csv)
    agg = augment_agg(results_df, agg)
    agg["stem"] = stem
    agg["rounds"] = stem_rounds
    agg["epsilon"] = stem_eps

    # A baseline every row of which failed the Valid filter produces no rows
    # at all, so it would otherwise vanish from every downstream table without
    # a trace. Name it explicitly, from the config's commitment rather than
    # from what survived.
    cfg = yaml.safe_load(config_yaml.read_text())
    present_methods = set(agg["Method"].unique())
    for baseline in sorted(expected_baselines(cfg)):
        if baseline not in present_methods:
            print(f"SKIP {stem}: baseline {baseline} has no valid rows")

    return agg


def load_sweep(results_base: Path, stems: list[str]) -> dict[str, pd.DataFrame]:
    """Load every grid point present. Missing points are skipped, loudly."""
    loaded: dict[str, pd.DataFrame] = {}
    for stem in stems:
        agg = load_point(Path(results_base) / stem)
        if agg is not None:
            loaded[stem] = agg
    print(f"Loaded {len(loaded)}/{len(stems)} grid points")
    return loaded


def warn_stale_points(points: dict[str, Path], variant: str, loaded) -> None:
    """Flag a loaded point whose snapshotted config differs from this folder's config.

    The runner copies the config verbatim to `<outdir>/config.yaml`, so a byte
    difference means those results came from another version of the config
    (for example a point left over from an earlier run).
    """
    for stem in loaded:
        snap = points[stem] / "config.yaml"
        if snap.read_bytes() != config_path(stem, variant).read_bytes():
            print(
                f"WARN {stem}: {snap} differs from {config_path(stem, variant).name}; "
                "these results were produced by a different config -- re-run the point"
            )


# =============================================================================
# Delta computation
# =============================================================================

# Fraction of the maximum achievable mean activity that the shared DUET
# reference arm must attain. Imported (not duplicated) from
# DUET_REFERENCE_FRACTIONS["97p5pct"] in visualize_benchmark.py, so the sweep
# cells and the per-point panels answer the same question and cannot drift.
ACTIVITY_FRACTION = DUET_REFERENCE_FRACTIONS["97p5pct"]


@dataclass(frozen=True)
class MetricSpec:
    """One metric and how its delta is expressed.

    kind:
      'absolute'          -- duet - baseline (reported in percentage points)
      'relative'          -- (duet - baseline) / baseline
      'relative_over_one' -- (duet - 1 - (baseline - 1)) / (baseline - 1), for
                             ratio metrics whose floor is 1

    direction:
      'max' -- higher is better, so DUET should show a positive delta (a gain)
      'min' -- lower is better, so DUET should show a negative delta (a
               reduction)

    Direction selects the panel's colour hue, following the published Figure 3C
    convention. It is a property of the METRIC, not of the observed data: keying
    the hue off the data's sign would let a panel flip colour between runs when
    a single cell crosses zero, and the figure set would stop being consistent.
    """

    name: str
    kind: str
    direction: str


SWEEP_METRICS: list[MetricSpec] = [
    MetricSpec("Mean decode accuracy", "absolute", "max"),
    MetricSpec("Mean decode accuracy (<= 5th percentile)", "absolute", "max"),
    MetricSpec("Mean decode accuracy (<= 10th percentile)", "absolute", "max"),
    MetricSpec("5th percentile decode accuracy", "absolute", "max"),
    MetricSpec("10th percentile decode accuracy", "absolute", "max"),
    MetricSpec("Standard deviation decode accuracy", "relative", "min"),
    MetricSpec("90% percentile interval length decode accuracy", "relative", "min"),
    MetricSpec("Median absolute deviation decode accuracy", "relative", "min"),
    MetricSpec("Mean absolute deviation decode accuracy", "relative", "min"),
    MetricSpec("95th over 5th percentile decode accuracy", "relative_over_one", "min"),
]

# The two Fig 3c panels (as the Fig 3c caption describes them: absolute gain in
# mean bottom-decile decoding accuracy, top; relative reduction in the
# inter-guide standard deviation, bottom; both against maximum activity).
FIG3C_PANELS = [
    ("top", MAX_ACTIVITY_METHOD, "Mean decode accuracy (<= 10th percentile)"),
    ("bottom", MAX_ACTIVITY_METHOD, "Standard deviation decode accuracy"),
]

# Diagnostics: these metrics' heatmaps are drawn against every baseline by
# default. Mean decode accuracy is the headline objective, and its six panels
# show where a baseline has no valid codebook (blank cells). Every other
# (baseline, metric) heatmap outside FIG3C_PANELS is a debug plot.
DIAGNOSTIC_METRICS = ("Mean decode accuracy",)


def heatmap_name(baseline: str, metric: str) -> str:
    """Output filename of one (baseline, metric) heatmap, as run."""
    slug_b = re.sub(r"[^A-Za-z0-9]+", "_", baseline).strip("_").lower()
    slug_m = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_").lower()
    return f"heatmap_{slug_b}_{slug_m}.svg"


def heatmap_is_default(baseline: str, metric: str) -> bool:
    """Whether a heatmap is drawn without --debug-plots (a Fig 3c panel or a diagnostic)."""
    return metric in DIAGNOSTIC_METRICS or any(
        (baseline, metric) == (b, m) for _, b, m in FIG3C_PANELS
    )


def _delta(kind: str, duet: float, baseline: float) -> float:
    if kind == "absolute":
        return duet - baseline
    if kind == "relative":
        return (duet - baseline) / baseline
    if kind == "relative_over_one":
        return ((duet - 1.0) - (baseline - 1.0)) / (baseline - 1.0)
    raise ValueError(f"Unknown metric kind: {kind!r}")


# A selected arm should sit just above the activity floor. A large overshoot
# means no lambda samples the band right above the floor, so the policy was
# forced onto a higher-activity arm and DUET's gain is understated. The 08-20
# run overshot by up to 0.02 at 10 of 18 points and nothing said so.
OVERSHOOT_WARN = 0.005


def compute_point_deltas(
    agg: pd.DataFrame,
    fraction: float = ACTIVITY_FRACTION,
    overshoot_warn: float = OVERSHOOT_WARN,
) -> pd.DataFrame:
    """Delta of the shared DUET reference arm against every baseline, per trial."""
    stem = agg["stem"].iloc[0]
    rows = []
    for trial in sorted(agg["Trial"].unique()):
        trial_df = agg[agg["Trial"] == trial]

        max_act = trial_df.loc[trial_df["Method"] == MAX_ACTIVITY_METHOD, "Mean activity score"]
        if max_act.empty:
            print(f"SKIP {stem} trial {trial}: no {MAX_ACTIVITY_METHOD} row")
            continue

        duet_method = select_duet_lambda(trial_df, trial, fraction * max_act.values[0])
        if duet_method is None:
            print(
                f"SKIP {stem} trial {trial}: no DUET lambda reaches "
                f"{fraction:.1%} of max activity"
            )
            continue

        duet_row = trial_df[trial_df["Method"] == duet_method].iloc[0]

        floor = fraction * max_act.values[0]
        overshoot = float(duet_row["Mean activity score"]) - floor
        if overshoot > overshoot_warn:
            print(
                f"WARN {stem} trial {trial}: selected arm {duet_method} overshoots "
                f"the {fraction:.1%} activity floor by {overshoot:.4f} "
                f"(activity {duet_row['Mean activity score']:.4f} vs floor "
                f"{floor:.4f}) -- the lambda grid is too coarse just above the "
                f"floor here, so this cell understates DUET's gain"
            )
        baselines = trial_df[~trial_df["Method"].str.startswith("DUET")]["Method"]

        for baseline in baselines:
            base_row = trial_df[trial_df["Method"] == baseline].iloc[0]
            for spec in SWEEP_METRICS:
                if spec.name not in trial_df.columns:
                    print(f"SKIP {stem}: metric column absent: {spec.name}")
                    continue
                rows.append(
                    {
                        "stem": stem,
                        "rounds": int(duet_row["rounds"]),
                        "epsilon": float(duet_row["epsilon"]),
                        "trial": int(trial),
                        "baseline": baseline,
                        "metric": spec.name,
                        "kind": spec.kind,
                        "direction": spec.direction,
                        "duet_lambda": duet_method,
                        "duet_value": duet_row[spec.name],
                        "baseline_value": base_row[spec.name],
                        "delta": _delta(
                            spec.kind, duet_row[spec.name], base_row[spec.name]
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "stem", "rounds", "epsilon", "trial", "baseline", "metric",
            "kind", "direction", "duet_lambda", "duet_value", "baseline_value",
            "delta",
        ],
    )


def compute_sweep_deltas(
    loaded: dict[str, pd.DataFrame], fraction: float = ACTIVITY_FRACTION
) -> pd.DataFrame:
    """Concatenate per-point deltas across the whole grid."""
    frames = [compute_point_deltas(agg, fraction) for agg in loaded.values()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-trial deltas to mean / sd / se / n per cell."""
    if deltas.empty:
        return pd.DataFrame()
    keys = ["stem", "rounds", "epsilon", "baseline", "metric", "kind", "direction"]
    summary = (
        deltas.groupby(keys)["delta"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    lambdas = (
        deltas.groupby(keys)["duet_lambda"]
        .agg(lambda s: ";".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"duet_lambda": "lambdas"})
    )
    return summary.merge(lambdas, on=keys, how="left")


# =============================================================================
# Rendering
# =============================================================================

# Heatmap panels are third-page slot panels with fixed millimetre margins
# (duet.plotting.slot_figure), so the two Fig 3c panels drop into the right-hand
# third of Fig 3 unscaled, with their cells and colour bars lined up. They used
# to be drawn half-page wide and saved with a tight box (89-94 mm), so they had
# to be scaled to about 0.6 at assembly, shrinking 7 pt text to about 4 pt.
HEATMAP_WIDTH_MM = FIGURE_WIDTHS["third_page"] * MM_PER_INCH
# Two stacked panels and the gap between them span the height of Fig 3b right
# (the all-baselines distribution stack), when the title takes two lines.
HEATMAP_HEIGHT_MM = (STACK_HEIGHT_OVERLAY * MM_PER_INCH - PANEL_GAP_MM) / 2
HEATMAP_LEFT_MM = 12.0     # the "Error rate" label and "10%" ticks: 11.2 mm in Arial, 11.7 in DejaVu Sans
HEATMAP_BOTTOM_MM = 9.5    # the round tick labels and the "Sequencing rounds" label
HEATMAP_RIGHT_MM = 11.5    # gap, colour bar and its tick labels (up to "-80")
HEATMAP_CBAR_GAP_MM = 1.5  # between the cells and the colour bar
HEATMAP_CBAR_WIDTH_MM = 2.0
# The colour bar stops this far below the top of the cells; its unit ("pp" or
# "%") sits in the space. A rotated "Difference (percentage points)" would be
# longer than the bar.
HEATMAP_CBAR_UNIT_MM = 3.5
# Titles wrap at this many characters, so the longest line is about 43 mm at
# 9 pt, inside the ~53 mm the slot leaves.
HEATMAP_TITLE_WRAP = 30
HEATMAP_TITLE_PAD_MM = 1.0
# One 9 pt title line (matplotlib's 1.2 line spacing), in mm.
_TITLE_LINE_MM = 9 * 1.2 / 72 * MM_PER_INCH
# Cell labels at the tick size: a four-character label such as "+7.4" is
# about 5.0 mm, in a ~5.8 mm cell (see cell_label).
HEATMAP_ANNOT_FONTSIZE = 7


def _cells(summary: pd.DataFrame, baseline: str, metric: str) -> pd.DataFrame:
    return summary[(summary["baseline"] == baseline) & (summary["metric"] == metric)]


def deltas_to_matrix(
    summary: pd.DataFrame, baseline: str, metric: str, value: str = "mean"
) -> pd.DataFrame:
    """Pivot one (baseline, metric) slice to epsilon x rounds.

    Rows are epsilon descending (higher error on top) and columns are rounds
    ascending (fewer rounds on the left), matching the reading order of the
    panel: the top-left corner is the hardest channel.
    """
    cells = _cells(summary, baseline, metric)
    matrix = cells.pivot(index="epsilon", columns="rounds", values=value)
    return matrix.sort_index(ascending=False).sort_index(axis=1)


def cell_label(value: float) -> str:
    """One cell's label: the signed value to at most one decimal place.

    '+45', '-64', '+7.4', '+9.0', '+0.4'. That is two significant figures from
    1 to 99, as in the published Figure 3C panels, and never more than four
    characters, so a label fits its cell (about 5.8 mm at the third-page width;
    '+0.38' at 7 pt is 6.3 mm). There is no percent sign: the unit is on the
    colour bar ("pp" for an absolute difference, "%" for a relative one).
    """
    if pd.isna(value):
        return ""
    return f"{value:+.0f}" if abs(round(value, 1)) >= 10 else f"{value:+.1f}"


def annotation_matrix(summary: pd.DataFrame, baseline: str, metric: str) -> pd.DataFrame:
    """Per-cell labels (``cell_label``) for one (baseline, metric) heatmap.

    No standard error: '+45.3+/-1.29' overran its neighbour even at the old
    half-page width. Per-cell sd, se and n are carried in sweep_summary.csv for
    the caption instead.
    """
    mean = deltas_to_matrix(summary, baseline, metric, "mean") * 100
    return mean.apply(lambda col: col.map(cell_label))


def _metric_phrase(metric: str) -> str:
    """A metric column name as it reads inside a title.

    '<=' becomes the '≤' glyph, a statistic of decode accuracy other than its
    mean takes 'of' ('standard deviation of decode accuracy'), and the first
    letter is lower-cased.
    """
    phrase = metric.replace("<=", "≤")
    suffix = " decode accuracy"
    if phrase.endswith(suffix) and not phrase.startswith("Mean decode accuracy"):
        phrase = phrase[: -len(suffix)] + " of" + suffix
    return phrase[0].lower() + phrase[1:]


def panel_title(metric: str, kind: str, direction: str) -> str:
    """Panel title, wrapped to fit a third-page panel.

    e.g. 'Absolute gain in mean decode\naccuracy (≤ 10th percentile)' (Fig 3c
    top) and 'Relative reduction in standard\ndeviation of decode accuracy'
    (Fig 3c bottom), as in the Fig 3c caption. The unit is on the colour bar,
    not in the title. The metric is the subject; which baseline it is measured
    against belongs in the figure caption, not in every panel title.
    """
    scale = "Absolute" if kind == "absolute" else "Relative"
    move = "gain" if direction == "max" else "reduction"
    return textwrap.fill(f"{scale} {move} in {_metric_phrase(metric)}", HEATMAP_TITLE_WRAP)


def panel_violations(cells: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Cells whose delta runs opposite to the metric's direction.

    For a 'max' metric that means DUET lost; for 'min' it means DUET was more
    variable. Either is a real result and must stay visible in the figure.
    """
    if direction == "max":
        return cells[cells["mean"] < 0]
    if direction == "min":
        return cells[cells["mean"] > 0]
    raise ValueError(f"Unknown metric direction: {direction!r}")


def panel_scale(cells: pd.DataFrame, direction: str):
    """Colormap and normalization for one panel.

    Sign-clean panels get the single-hue map for their direction, clamped at
    zero so colour encodes magnitude away from "no difference" -- the Figure 3C
    convention. A panel containing a wrong-sign cell gets the diverging map and
    is NOT clamped: clamping would paint a genuine loss the same colour as a
    true zero, hiding it.
    """
    values = cells["mean"].to_numpy(dtype=float) * 100
    limit = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    if limit == 0.0:
        limit = 1.0

    if len(panel_violations(cells, direction)):
        return delta_diverging_cmap(), TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    if direction == "max":
        return gain_sequential_cmap(), Normalize(vmin=0.0, vmax=limit)
    return reduction_sequential_cmap(), Normalize(vmin=-limit, vmax=0.0)


def colorbar_unit(kind: str) -> str:
    """Colour-bar unit for a metric kind: pp (percentage points) for an
    absolute difference, % for the ratios."""
    if kind == "absolute":
        return "pp"
    if kind in ("relative", "relative_over_one"):
        return "%"
    raise ValueError(f"Unknown metric kind: {kind!r}")


def plot_delta_heatmap(
    summary: pd.DataFrame, baseline: str, metric: str, out_path: Path
) -> bool:
    """Render one epsilon x rounds heatmap of the delta against `baseline`.

    Returns True if a figure was written, False if the slice was empty and
    the render was skipped. The caller uses this to count renders rather
    than inferring success from whether the output path is new -- an SVG
    already exists on every re-render, so "did the file not exist before"
    undercounts.
    """
    cells = _cells(summary, baseline, metric)
    if cells.empty:
        print(f"SKIP heatmap {baseline} / {metric}: no cells in the summary")
        return False

    matrix = deltas_to_matrix(summary, baseline, metric) * 100
    annot = annotation_matrix(summary, baseline, metric)
    kind = cells["kind"].iloc[0]
    direction = cells["direction"].iloc[0]

    violations = panel_violations(cells, direction)
    for _, row in violations.iterrows():
        print(
            f"NOTE {baseline} / {metric}: {row['stem']} runs opposite to the "
            f"metric direction ({row['mean'] * 100:+.3g}% for a '{direction}' "
            f"metric, SE {row['se'] * 100:.3g}) -- panel rendered diverging, "
            f"not clamped, so the cell stays visible"
        )
    cmap, norm = panel_scale(cells, direction)

    title = panel_title(metric, kind, direction)
    fig, ax, cax = heatmap_slot_figure(title.count("\n") + 1)
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": HEATMAP_ANNOT_FONTSIZE},
        cmap=cmap,
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_ax=cax,
    )
    ax.set_xlabel("Sequencing rounds")
    ax.set_ylabel("Error rate")
    ax.set_yticklabels([f"{float(t.get_text()) * 100:.0f}%" for t in ax.get_yticklabels()],
                       rotation=0)
    ax.set_title(title, pad=HEATMAP_TITLE_PAD_MM * 72 / MM_PER_INCH)
    cax.set_title(colorbar_unit(kind), fontsize=plt.rcParams["ytick.labelsize"], pad=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_panel(fig, out_path)
    plt.close(fig)
    return True


def heatmap_slot_figure(title_lines: int):
    """``(fig, ax, cax)`` for one heatmap panel: third-page wide, fixed margins.

    ``ax`` holds the cells and ``cax`` the colour bar, at fixed millimetre
    positions, so every heatmap's cells and colour bar sit in the same place.
    The top margin holds ``title_lines`` lines of title. With two lines (both
    Fig 3c panels) the panel is ``HEATMAP_HEIGHT_MM`` tall; a longer title (some
    debug heatmaps) makes the panel taller, not the cells shorter.
    """
    cells_mm = HEATMAP_HEIGHT_MM - HEATMAP_BOTTOM_MM - _heatmap_top_mm(2)
    top_mm = _heatmap_top_mm(title_lines)
    height_mm = HEATMAP_BOTTOM_MM + cells_mm + top_mm
    fig, ax = slot_figure(
        HEATMAP_WIDTH_MM,
        height_mm,
        margins_mm=(HEATMAP_LEFT_MM, HEATMAP_RIGHT_MM, HEATMAP_BOTTOM_MM, top_mm),
    )
    cax_left = HEATMAP_WIDTH_MM - HEATMAP_RIGHT_MM + HEATMAP_CBAR_GAP_MM
    cax = fig.add_axes([
        cax_left / HEATMAP_WIDTH_MM,
        HEATMAP_BOTTOM_MM / height_mm,
        HEATMAP_CBAR_WIDTH_MM / HEATMAP_WIDTH_MM,
        (cells_mm - HEATMAP_CBAR_UNIT_MM) / height_mm,
    ])
    return fig, ax, cax


def _heatmap_top_mm(title_lines: int) -> float:
    """Top margin for a title of ``title_lines`` lines, with 0.5 mm to spare."""
    return HEATMAP_TITLE_PAD_MM + title_lines * _TITLE_LINE_MM + 0.5


def render_heatmaps(
    summary: pd.DataFrame, figures_dir: Path, debug_plots: bool = False
) -> tuple[int, int]:
    """Draw every (baseline, metric) heatmap this run writes.

    Without `debug_plots` only the Fig 3c panels and DIAGNOSTIC_METRICS are
    drawn. Returns (rendered, skipped as debug plots). Each panel's colour
    scale comes from its own cells, so skipping a panel changes no other.
    """
    rendered = skipped = 0
    for baseline in sorted(summary["baseline"].unique()):
        for spec in SWEEP_METRICS:
            if not (debug_plots or heatmap_is_default(baseline, spec.name)):
                skipped += 1
                continue
            out = figures_dir / heatmap_name(baseline, spec.name)
            if plot_delta_heatmap(summary, baseline, spec.name, out):
                rendered += 1
    return rendered, skipped


# =============================================================================
# Summary
# =============================================================================


def write_summary(summary: pd.DataFrame, out_path: Path) -> None:
    """Write one row per (point, baseline, metric) so caption numbers are traceable."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "stem", "rounds", "epsilon", "baseline", "metric", "kind", "direction",
        "mean", "sd", "se", "n", "lambdas",
    ]
    summary[columns].sort_values(["stem", "baseline", "metric"]).to_csv(
        out_path, index=False
    )
    print(f"Wrote {len(summary)} summary rows to {out_path}")


def sweep_expected_baselines(results_base: Path, stems) -> set[str]:
    """Union of expected baselines across every grid point's config.yaml.

    Reads the config each point actually snapshotted rather than inspecting
    which methods survived into the summary -- a point missing entirely
    contributes nothing here (it is already named by the "missing results.csv
    or config.yaml" SKIP line), but a baseline configured at every point yet
    valid at none of them still counts.
    """
    labels: set[str] = set()
    for stem in stems:
        config_yaml = Path(results_base) / stem / "config.yaml"
        if not config_yaml.exists():
            continue
        labels |= expected_baselines(yaml.safe_load(config_yaml.read_text()))
    return labels


# =============================================================================
# Reference check against experiments/ops_crispri_symmetric (Fig 3b)
# =============================================================================
#
# Fig 3b runs this sweep's rounds-10 design: same pool (candidate_pool block,
# seed 42, 5 trials), same optimizer settings, and at epsilon 0.1 the same
# channel, PEP (seed 43, 5000 samples) and evaluator (seed 42, 2000 samples).
# Its 18 lambdas are a subset of the sweep's 31.
#
# Tier A, exact (a mismatch fails the run). Per trial, at every sweep point
# whose pool matches the reference's:
#   * trial_NN/guides.csv, byte for byte (the pool depends only on seed and
#     candidate_pool, not on epsilon);
#   * every baseline's rows (Index, Group, Sequence, Activity score, Valid), in
#     order, from the raw results.csv (invalid rows included). Maximum activity,
#     Sivanandan and Feldman are deterministic given the pool, and Feldman's
#     child process pins PYTHONHASHSEED;
#   * DUET at lambda = 0: its decode weight is exactly 0, so the PEP (its only
#     epsilon-dependent input) cannot influence it;
#   * at the reference's epsilon only, DUET at every lambda both grids share.
#     Same pool, random init and per-lambda seed (trial seed + int(lambda*1000)),
#     and the same seeded, integer-valued Hamming PEP (its fingerprint holds the
#     library, channel, metric, rule, samples, num_cpus and seed, not the path,
#     so the sweep rebuilds it in its own cache_dir with identical counts). The
#     lambda jobs are independent, so the extra 13 values and the worker count
#     change nothing.
# Tier B, statistical (report only). Decode accuracy cannot match exactly: the
# ground truth draws one stream per codeword, indexed by its position in the
# sorted union of every evaluated codebook, and the 13 extra DUET codebooks
# shift almost every position. So at the reference's epsilon the sweep's cell
# means are compared with the reference's, recomputed by this script, as z-scores
# against the Monte Carlo noise of two independent evaluations of the same
# codebooks (parametric bootstrap; see `mc_sd_of_cell_difference`).

# Config blocks that must match for each comparison (dotted path, keys ignored).
POOL_KEYS = [(("seed",), ()), (("trials",), ()), (("candidate_pool",), ())]
DUET_KEYS = [
    (("duet", "use_mmap"), ()),
    (("duet", "initialization"), ()),
    (("duet", "optimizer"), ("lambda", "num_cpus")),  # num_cpus: speed only
]
PEP_KEYS = [(("duet", "pep"), ("num_cpus", "scratch_dir"))]  # counts do not depend on num_cpus
EVAL_KEYS = [(("evaluator",), ("num_cpus", "scratch_dir", "device", "mem_budget_gb"))]

COMPARE_COLUMNS = ["Index", "Group", "Sequence", "Activity score", "Valid"]
TIER_B_Z = 4.0          # |z| above this is flagged in the Tier B report
MC_REPLICATES = 200     # parametric-bootstrap replicates per (trial, codebook)


def _get(cfg: dict, path: tuple[str, ...]):
    for key in path:
        cfg = cfg.get(key) if isinstance(cfg, dict) else None
    return cfg


def _config_differences(a: dict, b: dict, specs) -> list[str]:
    diffs = []
    for path, ignored in specs:
        va, vb = _get(a, path), _get(b, path)
        if isinstance(va, dict) and isinstance(vb, dict):
            va = {k: v for k, v in va.items() if k not in ignored}
            vb = {k: v for k, v in vb.items() if k not in ignored}
        if va != vb:
            diffs.append(f"{'.'.join(path)} (sweep {va!r} vs reference {vb!r})")
    return diffs


def _epsilon(cfg: dict) -> float:
    return float(cfg["evaluator"]["noise_channel"]["epsilon"])


def _duet_label(lam: float) -> str:
    return f"DUET (lambda={lam:.2f})"  # the runner's label (ops_benchmark/runner.py)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_rows(sweep_raw, ref_raw, method: str, trial: int) -> str | None:
    """None if `method`'s rows in `trial` are identical in both results.csv."""
    a = sweep_raw.loc[(sweep_raw["Method"] == method) & (sweep_raw["Trial"] == trial), COMPARE_COLUMNS]
    b = ref_raw.loc[(ref_raw["Method"] == method) & (ref_raw["Trial"] == trial), COMPARE_COLUMNS]
    if a.empty or b.empty:
        return f"rows missing ({len(a)} in sweep, {len(b)} in reference)"
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if a.equals(b):
        return None
    same_set = set(a["Index"]) == set(b["Index"])
    n_diff = int((a["Index"].to_numpy() != b["Index"].to_numpy()).sum()) if len(a) == len(b) else None
    return (
        f"{len(a)} vs {len(b)} rows; index set {'equal' if same_set else 'differs'}"
        + (f"; {n_diff} positions differ" if n_diff is not None else "")
        + f"; mean activity {a['Activity score'].mean():.6f} vs {b['Activity score'].mean():.6f}"
        + f"; valid {a['Valid'].iloc[0]} vs {b['Valid'].iloc[0]}"
    )


def reference_dir_for(variant: str) -> tuple[Path | None, str]:
    """The Fig 3b outdir for this variant, from its config (not hard-coded)."""
    cfg = REFERENCE_EXPERIMENT / f"config{'.' + variant if variant else ''}.yaml"
    if not cfg.is_file():
        return None, f"reference config {cfg} not found"
    outdir = yaml.safe_load(cfg.read_text())["outdir"]
    return Path(os.path.normpath(REFERENCE_EXPERIMENT / outdir)), f"outdir of {cfg}"


def _metric_values(acc: np.ndarray) -> dict[str, np.ndarray]:
    """SWEEP_METRICS for each row of `acc` (replicates x guides).

    The same definitions as `load_and_process_results` plus `augment_agg`
    (pandas' default linear quantiles, sample standard deviation).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        q05, q10, q95 = (np.quantile(acc, q, axis=1) for q in (0.05, 0.1, 0.95))
        mean, med = acc.mean(axis=1), np.median(acc, axis=1)

        def tail_mean(q):
            m = acc <= q[:, None]
            return (acc * m).sum(axis=1) / m.sum(axis=1)

        return {
            "Mean decode accuracy": mean,
            "Mean decode accuracy (<= 5th percentile)": tail_mean(q05),
            "Mean decode accuracy (<= 10th percentile)": tail_mean(q10),
            "5th percentile decode accuracy": q05,
            "10th percentile decode accuracy": q10,
            "Standard deviation decode accuracy": acc.std(axis=1, ddof=1),
            "90% percentile interval length decode accuracy": q95 - q05,
            "Median absolute deviation decode accuracy": np.median(np.abs(acc - med[:, None]), axis=1),
            "Mean absolute deviation decode accuracy": np.abs(acc - mean[:, None]).mean(axis=1),
            "95th over 5th percentile decode accuracy": q95 / q05,
        }


def mc_sd_of_cell_difference(
    ref_valid: pd.DataFrame, ref_deltas: pd.DataFrame, num_samples: int,
    replicates: int = MC_REPLICATES, seed: int = 0,
) -> pd.DataFrame:
    """Monte Carlo SD of (cell mean from one evaluation) - (from another), per cell.

    Given the codebooks, each guide's decode accuracy is Binomial(num_samples, p)
    / num_samples, independently across guides (one stream per guide). Plugging
    in the reference's per-guide accuracies and redrawing gives, per trial, the
    SD of that trial's delta for the reference's own arm and baseline. A cell
    mean over n trials then has SD sqrt(sum of variances) / n, and the
    difference of two independent evaluations sqrt(2) times that. Ignoring the
    shared reads of a guide present in both codebooks overstates the SD
    (conservative).
    """
    rng = np.random.default_rng(seed)
    kinds = {s.name: s.kind for s in SWEEP_METRICS}
    draws: dict[tuple[int, str], dict[str, np.ndarray]] = {}

    def metrics_for(trial: int, method: str) -> dict[str, np.ndarray]:
        if (trial, method) not in draws:
            p = ref_valid.loc[
                (ref_valid["Trial"] == trial) & (ref_valid["Method"] == method), "Decode accuracy"
            ].to_numpy(dtype=float)
            acc = rng.binomial(num_samples, np.clip(p, 0, 1), size=(replicates, len(p))) / num_samples
            draws[(trial, method)] = _metric_values(acc)
        return draws[(trial, method)]

    var = {}
    for (trial, baseline, arm), _ in ref_deltas.groupby(["trial", "baseline", "duet_lambda"]):
        duet_m, base_m = metrics_for(trial, arm), metrics_for(trial, baseline)
        for metric, kind in kinds.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                d = _delta(kind, duet_m[metric], base_m[metric])
            d = d[np.isfinite(d)]
            v = float(np.var(d, ddof=1)) if len(d) > 1 else np.nan
            var.setdefault((baseline, metric), []).append(v)
    rows = [
        {"baseline": b, "metric": m, "sd_diff": np.sqrt(2.0 * np.sum(v)) / len(v)}
        for (b, m), v in var.items()
    ]
    return pd.DataFrame(rows)


def tier_b_report(stem: str, sweep_deltas: pd.DataFrame, summary: pd.DataFrame,
                  ref_dir: Path, ref_cfg: dict) -> None:
    """Print the statistical comparison of one point's cells with the reference."""
    print(f"\nTier B (statistical, report only): {stem} vs {ref_dir}")
    ref_agg = load_point(ref_dir, stem=stem)
    if ref_agg is None:
        print("  SKIPPED: the reference could not be loaded")
        return
    ref_deltas = compute_point_deltas(ref_agg)
    ref_summary = summarize_deltas(ref_deltas)
    if ref_summary.empty:
        print("  SKIPPED: no reference cells")
        return

    arms_s = sweep_deltas[sweep_deltas["stem"] == stem].groupby("trial")["duet_lambda"].first()
    arms_r = ref_deltas.groupby("trial")["duet_lambda"].first()
    for trial in sorted(set(arms_s.index) | set(arms_r.index)):
        a, r = arms_s.get(trial, "none"), arms_r.get(trial, "none")
        print(f"  trial {trial}: 97.5% arm {a} (sweep) vs {r} (reference)"
              + ("" if a == r else "  <-- DIFFERENT ARM: cells in this trial are not like for like"))

    ref_valid = pd.read_csv(ref_dir / "results.csv")
    ref_valid = ref_valid[ref_valid["Valid"]]
    sd = mc_sd_of_cell_difference(ref_valid, ref_deltas, int(ref_cfg["evaluator"]["num_samples"]))
    merged = (
        summary[summary["stem"] == stem]
        .merge(ref_summary, on=["stem", "baseline", "metric"], suffixes=("_sweep", "_ref"))
        .merge(sd, on=["baseline", "metric"], how="left")
    )
    if merged.empty:
        print("  no comparable cells")
        return
    merged["diff"] = merged["mean_sweep"] - merged["mean_ref"]
    merged["z"] = merged["diff"] / merged["sd_diff"]
    flagged = merged[merged["z"].abs() > TIER_B_Z]
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.6f}".format):
        print(merged[["baseline", "metric", "mean_sweep", "mean_ref", "diff", "sd_diff", "z"]]
              .sort_values(["baseline", "metric"]).to_string(index=False))
    zmax = merged["z"].abs().max()
    print(f"  {len(merged)} cells compared; max |z| = {zmax:.2f}; "
          f"{len(flagged)} with |z| > {TIER_B_Z:g}; max |diff| = {merged['diff'].abs().max():.6f}")
    for _, row in flagged.iterrows():
        print(f"  NOTE Tier B: {row['baseline']} / {row['metric']}: diff {row['diff']:+.6f}, "
              f"z {row['z']:+.2f}")


def check_reference(
    points: dict[str, Path], loaded: dict[str, pd.DataFrame], deltas: pd.DataFrame,
    summary: pd.DataFrame, ref_dir: Path | None, ref_source: str,
) -> tuple[str, list[str]]:
    """Tier A and Tier B checks against the Fig 3b outputs.

    Returns (status, Tier A problems), status one of PASSED, FAILED, SKIPPED. A
    missing reference, or no loaded sweep point with the reference's number of
    rounds, is SKIPPED and is not a failure.
    """
    print("\n" + "=" * 78 + "\nReference check vs experiments/ops_crispri_symmetric\n" + "=" * 78)
    if ref_dir is None:
        print(f"SKIPPED: {ref_source}. Run experiments/ops_crispri_symmetric first to enable this check.")
        return "SKIPPED", []
    if not ((ref_dir / "results.csv").is_file() and (ref_dir / "config.yaml").is_file()):
        print(f"SKIPPED: no results.csv and config.yaml in {ref_dir} ({ref_source}). "
              "Run experiments/ops_crispri_symmetric first to enable this check.")
        return "SKIPPED", []
    ref_cfg = yaml.safe_load((ref_dir / "config.yaml").read_text())
    ref_rounds, ref_eps = int(ref_cfg["candidate_pool"]["seq_rounds"]), _epsilon(ref_cfg)
    print(f"Reference: {ref_dir} ({ref_source}); rounds {ref_rounds}, epsilon {ref_eps:g}")
    candidates = [
        s for s in loaded
        if int(yaml.safe_load((points[s] / "config.yaml").read_text())["candidate_pool"]["seq_rounds"]) == ref_rounds
    ]
    if not candidates:
        print(f"SKIPPED: no loaded sweep point has {ref_rounds} rounds")
        return "SKIPPED", []

    problems: list[str] = []
    ref_raw = pd.read_csv(ref_dir / "results.csv")
    ref_lams = set(ref_cfg["duet"]["optimizer"]["lambda"])
    counts = {"guides.csv": 0, "baseline rows": 0, "DUET rows": 0}
    tier_b_points = []
    for stem in candidates:
        cfg = yaml.safe_load((points[stem] / "config.yaml").read_text())
        diffs = _config_differences(cfg, ref_cfg, POOL_KEYS)
        if diffs:
            problems.append(f"{stem}: pool not comparable with the reference: {'; '.join(diffs)}")
            continue
        duet_diffs = _config_differences(cfg, ref_cfg, DUET_KEYS)
        same_eps = _epsilon(cfg) == ref_eps
        pep_diffs = _config_differences(cfg, ref_cfg, PEP_KEYS) if same_eps else []
        if duet_diffs or pep_diffs:
            problems.append(f"{stem}: DUET not comparable with the reference: {'; '.join(duet_diffs + pep_diffs)}")
        shared = sorted(ref_lams & set(cfg["duet"]["optimizer"]["lambda"]))
        if duet_diffs:
            arms = []
        elif same_eps and not pep_diffs:
            arms = shared                                  # same PEP: every shared arm
        else:
            arms = [lam for lam in shared if lam == 0.0]  # PEP-independent arm only
        sweep_raw = pd.read_csv(points[stem] / "results.csv")
        trials = sorted(set(sweep_raw["Trial"]) & set(ref_raw["Trial"]))
        methods = set(sweep_raw["Method"]) | set(ref_raw["Method"])
        baselines = sorted(m for m in methods if not m.startswith("DUET"))
        for trial in trials:
            g_s = points[stem] / f"trial_{trial:02d}" / "guides.csv"
            g_r = ref_dir / f"trial_{trial:02d}" / "guides.csv"
            if not (g_s.is_file() and g_r.is_file()) or _sha256(g_s) != _sha256(g_r):
                problems.append(f"{stem} trial {trial}: guides.csv differs from {g_r}")
            else:
                counts["guides.csv"] += 1
            for method in baselines:
                why = _compare_rows(sweep_raw, ref_raw, method, trial)
                if why:
                    problems.append(f"{stem} trial {trial}: {method}: {why}")
                else:
                    counts["baseline rows"] += 1
            for lam in arms:
                why = _compare_rows(sweep_raw, ref_raw, _duet_label(lam), trial)
                if why:
                    problems.append(f"{stem} trial {trial}: {_duet_label(lam)}: {why}")
                else:
                    counts["DUET rows"] += 1
        scope = (f"all {len(arms)} shared DUET arms" if len(arms) > 1 else
                 "DUET lambda=0 only (epsilon-independent)" if arms else "no DUET arm")
        print(f"Tier A {stem}: {len(trials)} trial(s); guides.csv; {len(baselines)} baselines; {scope}"
              + ("" if same_eps else f" (epsilon {_epsilon(cfg):g} != {ref_eps:g}: no equivalent "
                                     "reference, so shared arms other than lambda=0 and the "
                                     "decode-accuracy deltas are not compared)"))
        if not trials:
            problems.append(f"{stem}: no trial in common with the reference")
        if same_eps:
            if _config_differences(cfg, ref_cfg, EVAL_KEYS):
                print(f"Tier B {stem}: SKIPPED, evaluator differs: "
                      + "; ".join(_config_differences(cfg, ref_cfg, EVAL_KEYS)))
            else:
                tier_b_points.append(stem)

    n_compared = sum(counts.values())
    if n_compared == 0:
        problems.append("Tier A compared nothing although the reference and a comparable point exist")
    print(f"Tier A compared {counts['guides.csv']} guides.csv, {counts['baseline rows']} baseline "
          f"and {counts['DUET rows']} DUET (trial, method) row sets identical; "
          f"{len(problems)} problem(s).")

    for stem in tier_b_points:
        tier_b_report(stem, deltas, summary, ref_dir, ref_cfg)
    return ("FAILED" if problems else "PASSED"), problems


# =============================================================================
# Entry point
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fig 3c sweep heatmaps and reference check.")
    parser.add_argument("--variant", default=os.environ.get("VARIANT", ""),
                        help="config variant: '' (the 18 committed configs) or e.g. 'smoke' "
                             "(default: $VARIANT)")
    parser.add_argument("--results-base", type=Path,
                        help="read <DIR>/<stem>/ instead of the configs' outdirs")
    parser.add_argument("--figures-dir", type=Path,
                        help="write here instead of <results base>/sweep_figures")
    parser.add_argument("--reference-dir", type=Path,
                        help="Fig 3b outputs to check against (default: the outdir of "
                             "experiments/ops_crispri_symmetric/config[.<variant>].yaml)")
    parser.add_argument("--no-reference-check", action="store_true")
    add_debug_plots_arg(parser)
    args = parser.parse_args(argv)

    points = sweep_points(args.variant)
    stems = list(points)
    if args.results_base is not None:
        points = {s: args.results_base.resolve() / s for s in stems}
    results_base = results_base_of(points)
    figures_dir = args.figures_dir or results_base / "sweep_figures"
    print(f"Variant: {args.variant or '(real)'}; {len(stems)} configured points; results in {results_base}")

    loaded = load_sweep(results_base, stems)
    if not loaded:
        raise SystemExit(f"No grid points found under {results_base}; nothing to plot.")
    if args.results_base is None:
        warn_stale_points(points, args.variant, loaded)

    deltas = compute_sweep_deltas(loaded)
    if deltas.empty:
        raise SystemExit("No deltas computed; see the SKIP lines above.")
    summary = summarize_deltas(deltas)

    figures_dir.mkdir(parents=True, exist_ok=True)
    write_summary(summary, figures_dir / "sweep_summary.csv")

    rendered, skipped = render_heatmaps(summary, figures_dir, args.debug_plots)
    print(f"Rendered {rendered} heatmaps to {figures_dir}/; skipped {skipped} debug "
          f"heatmaps" + ("" if args.debug_plots else " (pass --debug-plots to draw them)"))

    # Skip accounting: covered cells plus skipped cells must equal the full
    # grid. A shortfall is legitimate (a missing point, a baseline with no
    # valid codebook) but must be stated, never inferred from a figure count.
    # The baseline count comes from what the configs commit to producing, not
    # from which baselines happened to survive the Valid filter -- otherwise a
    # baseline that is invalid at every point shrinks the denominator to match
    # the numerator and the run reports full coverage while omitting it
    # entirely.
    n_baselines = len(sweep_expected_baselines(results_base, loaded.keys()))
    expected = len(stems) * n_baselines * len(SWEEP_METRICS)
    covered = len(summary)
    print(
        f"Cell accounting: {covered}/{expected} covered "
        f"({len(stems)} points x {n_baselines} baselines x {len(SWEEP_METRICS)} metrics); "
        f"{expected - covered} skipped -- see the SKIP lines above."
    )
    for position, baseline, metric in FIG3C_PANELS:
        path = figures_dir / heatmap_name(baseline, metric)
        print(f"Fig 3c {position} panel: {path}" + ("" if path.is_file() else "  (NOT RENDERED)"))

    if args.no_reference_check:
        print("Reference check disabled (--no-reference-check).")
        return
    if args.reference_dir is not None:
        ref_dir, ref_source = args.reference_dir.resolve(), "--reference-dir"
    else:
        ref_dir, ref_source = reference_dir_for(args.variant)
    status, problems = check_reference(points, loaded, deltas, summary, ref_dir, ref_source)
    if problems:
        print("\nREFERENCE CHECK FAILED (Tier A, exact):")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print(f"\nReference check: Tier A {status}" + ("; Tier B is report-only." if status == "PASSED" else "."))


if __name__ == "__main__":
    main()
