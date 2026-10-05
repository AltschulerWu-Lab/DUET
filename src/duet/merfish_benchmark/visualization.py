# src/duet/merfish_benchmark/visualization.py
"""MERFISH-specific plotting (Pareto front, crowding bar chart, barplots, histogram).

DataFrame-driven so the same plot code can run in-line under
``run_merfish_benchmark`` (legacy path, removed in this PR) or from the
top-level ``scripts/benchmark/visualize_merfish.py`` script after the
analysis has persisted ``results.csv`` and ``metrics.csv``.

Style note: ``apply_style()`` MUST be called once by the caller before
invoking any function here. The functions do not call ``apply_style()``
themselves so callers can keep style configuration in one place
(matching ``scripts/benchmark/visualize_benchmark.py``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from duet.plotting import (
    DEFAULT_MARGINS_MM,
    FIGURE_WIDTHS,
    METHOD_PALETTE,
    MM_PER_INCH,
    OKABE_ITO,
    UNKNOWN_METHOD_COLOR,
    fit_last_xtick_label,
    save_panel,
    sequential_shades,
    slot_figure,
)
from duet.visualization import plot_codeword_probabilities


# Columns excluded from per-metric bar plots: non-metric metadata (Method,
# lambda), the four crowding columns (already rendered by plot_crowding_bar_chart
# / the Pareto front), and the two per-axis SE columns (error estimates, not
# displayable summary metrics).
_BARPLOT_EXCLUDED = {
    "Method",
    "lambda",
    "mean_identified_fraction",
    "std_identified_fraction",
    "mean_conflict_fraction",
    "std_conflict_fraction",
    "se_identified_fraction",
    "se_decode_accuracy",
}

# A per-codeword identified fraction is approximately a binomial proportion over
# the gene's realized transcript count N (results.csv 'n_transcripts') -- strictly
# a per-trial mean of ratios, so its sampling standard error ~ sqrt(p*(1-p)/N) <=
# sqrt(0.25/N) is a close but slightly optimistic bound. A gene seen only 1-2 times
# gives a quantized, near-meaningless value (a hard 0 if its lone transcript
# conflicted). The jointplot drops genes below this count so every plotted
# per-gene value carries a bounded error. N >= 25 bounds the worst-case s.e. at
# sqrt(0.25/25) = 0.10, which clears the small-sample tail (in the 1000-gene
# panel every codeword below identified fraction 0.6 had N <= 7). This is the
# total-appearances analogue of Boström's detectable-gene handling
# (round(reads*prop) >= 1 per cell), summed across all trials.
MIN_TRANSCRIPTS_FOR_JOINTPLOT = 25

# Crowding-Pareto marker area (pt^2). Enlarged from the original 8 so each point's
# per-Hamming-weight marker shape and method color read clearly at the third_page
# panel size. The smaller size used to leave room for the per-point SE error bars,
# but those bars are no longer drawn (they were tiny -- smaller than the marker --
# and added clutter without conveying much), so the marker can grow.
_PARETO_MARKER_SIZE = 20

# Connector line width for the crowding-Pareto sweeps (the DUET frontier line and
# the per-family Hamming-weight mini-Pareto connectors). Widened from 0.8 together
# with the larger markers so the connectors stay visually proportionate.
_PARETO_LINE_WIDTH = 1.5

# Horizontal method bar charts (plot_metrics_barplots, plot_crowding_bar_chart)
# put one method per row with its full name on the y axis, so the names read
# horizontally however many methods there are. The panel is half_page wide: the
# longest baseline names take ~42 mm at the 7 pt tick size, which would leave
# ~10 mm for the bars in a third_page slot. The height grows with the rows; the
# width grows only for names past ~50 characters (``_fit_hbar_names``).
_HBAR_ROW_MM = 4.0
_HBAR_BAR_HEIGHT = 0.6     # bar thickness as a fraction of a row
_HBAR_EDGE_MM = 1.0        # gap between the method names and the figure's left edge
_HBAR_MIN_AXES_MM = 20.0   # narrowest bars; longer names widen the figure instead
_HBAR_TITLE_MM = 6.5       # top margin: the title row
_HBAR_LEGEND_ROW_MM = 4.0  # added under the title for a legend above the bars
_HBAR_TICKS_MM = 6.0       # bottom margin: value tick labels, no axis label
# Right margin: half a six-character value tick label ("0.0125", ~3.8 mm at
# 7 pt). The value axis autoscales, so its last tick can sit on the axes edge.
_HBAR_RIGHT_MM = 4.0


def metric_barplot_stem(metric: str) -> str:
    """Filename stem of ``metric``'s bar plot (``plot_metrics_barplots``)."""
    safe = metric.lower().replace(" ", "_").replace("/", "_over_")
    safe = safe.replace("(", "").replace(")", "").replace("<=", "le")
    safe = safe.replace("%", "pct")
    return safe


def _method_color(method: str) -> str:
    """Look up a METHOD_PALETTE color.

    DUET sweep rows are labeled like ``DUET (lambda=0.50)`` by the runner.
    Strip the lambda suffix so all DUET points share the canonical DUET
    palette entry. Falls back to ``UNKNOWN_METHOD_COLOR`` for genuinely
    unknown methods.
    """
    if method.startswith("DUET"):
        return METHOD_PALETTE["DUET"]
    return METHOD_PALETTE.get(method, UNKNOWN_METHOD_COLOR)


def _ordered_method_palette(methods: Iterable[str]) -> dict[str, str]:
    """Build a per-method color dict, preserving caller-provided iteration order.

    Known methods (DUET sweep rows + ``METHOD_PALETTE`` keys) keep their curated
    color. Genuinely unknown methods are each assigned a DISTINCT Okabe-Ito hue
    not already used by a known method in this set, so multiple unrecognized
    baselines no longer collapse to a single ``UNKNOWN_METHOD_COLOR`` grey (the
    scalar fallback in ``_method_color``). Assignment is deterministic by first
    appearance; if unknowns outnumber the spare hues it cycles.
    """
    result: dict[str, str] = {}
    used: set[str] = set()
    unknown: list[str] = []
    for m in methods:
        if m in result or m in unknown:
            continue
        if m.startswith("DUET"):
            color = METHOD_PALETTE["DUET"]
        elif m in METHOD_PALETTE:
            color = METHOD_PALETTE[m]
        else:
            unknown.append(m)
            continue
        result[m] = color
        used.add(color)
    spare = [h for h in OKABE_ITO.values() if h not in used] or list(OKABE_ITO.values())
    for i, m in enumerate(unknown):
        result[m] = spare[i % len(spare)]
    return result


# HW color scheme: a single-hue *sequential* ramp (light -> dark = low -> high
# Hamming weight), built from OKABE_ITO["reddish_purple"] via sequential_shades.
# HW is ordinal, so a sequential ramp reads more naturally than categorical
# hues; purple is also distinct from every METHOD_PALETTE hue (DUET=blue,
# baselines=bluish_green/orange), so a HW segment cannot be mistaken for a
# method color when Figure-4 panels are composited. Shared by the stacked
# Hamming-weight distribution, the decode-accuracy-by-HW histogram, and the
# expression-by-HW panel so all HW-grouped panels use one scheme.
_HAMMING_WEIGHT_BASE = OKABE_ITO["reddish_purple"]


def _hamming_weight_colors(results_df: pd.DataFrame) -> dict[int, str]:
    """Map each Hamming weight present in ``results_df`` to a ramp color.

    Colors are a purple ``sequential_shades`` ramp assigned in ascending-HW
    order (lightest = lowest HW, darkest = highest HW) -- the same assignment
    every HW-grouped panel uses, so they share one scheme. Returns ``{}`` for
    an empty frame.
    """
    hws = sorted(results_df["Sequence"].astype(str).str.count("1").unique())
    if not hws:
        return {}
    shades = sequential_shades(_HAMMING_WEIGHT_BASE, len(hws))
    return {hw: shades[i] for i, hw in enumerate(hws)}


def _savefig_at_figsize(svg_path: Path, png_path: Path, *, png_dpi: int = 150) -> None:
    """Save the current figure to svg_path and png_path with canvas pinned to figsize.

    The shared stylesheet sets `savefig.bbox: tight`, which re-expands the
    saved canvas to fit all artists even when constrained_layout keeps
    content inside figsize. Flipping the rcParam to `"standard"` (validates
    to None) for the savefig call is matplotlib's documented escape hatch.

    Mirrors `_savefig_at_figsize` in `scripts/benchmark/visualize_comparison.py`
    but accepts two paths so MERFISH's SVG + PNG companion both render at
    the intended figsize.
    """
    with plt.rc_context({"savefig.bbox": "standard"}):
        plt.savefig(svg_path)
        plt.savefig(png_path, dpi=png_dpi)


def _lambda_to_symbol(text: str) -> str:
    """Render the ``lambda=`` token as ``λ=`` for figure text (tick labels,
    legends, titles) only -- never the underlying ``Method`` data, which the
    selection and sort helpers still match on the literal ``DUET (lambda=...)``.
    """
    return text.replace("lambda=", "λ=")


# Display spellings of baseline names, as the manuscript writes them. The
# configs, results.csv and METHOD_PALETTE keep the ASCII names, so existing
# results still plot; only figure text changes.
_DISPLAY_NAME_REPLACEMENTS = (
    ("Bostrom et al.", "Boström et al."),
    ("Zhang et al. codebook #", "Zhang et al. codebook "),
)


def _abbrev_legend_labels(labels: Iterable[str]) -> list[str]:
    """Compact the verbose tokens in method names for legend display only.

    ``Hamming weight N`` -> ``HWN`` and ``lambda=`` -> ``λ=`` so the legend
    stays narrow enough to sit inside a Nature-column panel without clipping,
    and the baseline names take their display spellings
    (``_DISPLAY_NAME_REPLACEMENTS``). The underlying data labels (the
    ``Method`` column) are untouched. Matches the abbreviations already applied
    in the decode-vs-identified jointplot.
    """
    out = []
    for lbl in labels:
        lbl = lbl.replace("Hamming weight ", "HW")
        for old, new in _DISPLAY_NAME_REPLACEMENTS:
            lbl = lbl.replace(old, new)
        out.append(_lambda_to_symbol(lbl))
    return out


def _boxed_legend(ax, handles=None, labels=None, **kwargs):
    """Draw a legend with a hairline frame + semi-opaque white fill.

    The shared stylesheet sets ``legend.frameon: False``, so a legend placed
    inside the axes floats borderless and its swatches read as stray data
    points. For the marker-based crowding panels we want the space savings of
    an in-axes legend without that ambiguity, so this override reinstates a
    thin grey border over a near-opaque white background: the border demarcates
    the legend and the fill masks any markers behind it. Any keyword (``loc``,
    ``title``, ...) overrides the defaults.
    """
    style = dict(
        fontsize="small", frameon=True, framealpha=0.9, facecolor="white",
        edgecolor="0.7", handletextpad=0.4, labelspacing=0.3, borderpad=0.4,
    )
    style.update(kwargs)
    leg = ax.legend(handles, labels, **style) if handles is not None else ax.legend(**style)
    leg.get_frame().set_linewidth(0.6)
    return leg


def _hbar_slot_figure(n_rows: int, *, top_mm: float, bottom_mm: float):
    """``(fig, ax)`` for a horizontal method bar chart: a half_page slot panel
    with one ``_HBAR_ROW_MM`` row per method.

    The left margin is provisional; ``_fit_hbar_names`` sizes it to the method
    names once they are drawn.
    """
    return slot_figure(
        FIGURE_WIDTHS["half_page"] * MM_PER_INCH,
        top_mm + n_rows * _HBAR_ROW_MM + bottom_mm,
        margins_mm=(DEFAULT_MARGINS_MM[0], _HBAR_RIGHT_MM, bottom_mm, top_mm),
    )


def _fit_hbar_names(fig, ax, chart: str) -> float:
    """Size the left margin to the method names drawn as y tick labels.

    Names are never shortened (they are ``METHOD_PALETTE`` keys), so the widest
    one sets the margin, plus ``_HBAR_EDGE_MM``, and the bars take the rest of
    the width. Names so long (~50 characters) that the bars would get less than
    ``_HBAR_MIN_AXES_MM`` widen the figure instead, just enough, keeping the
    other margins; a note names ``chart`` and its new width. The panel then
    leaves its slot, which a diagnostic can afford. Returns the x position, in
    axes coordinates, where the names start: a title placed there
    (``loc="left"``) spans the whole panel width instead of the narrower axes.
    """
    renderer = fig.canvas.get_renderer()
    axes_x0 = ax.get_window_extent(renderer).x0
    names_x0 = min((t.get_window_extent(renderer).x0 for t in ax.get_yticklabels()),
                   default=axes_x0)
    left_mm = (axes_x0 - names_x0) / fig.dpi * MM_PER_INCH + _HBAR_EDGE_MM
    fig_w_mm = fig.get_figwidth() * MM_PER_INCH
    box = ax.get_position()
    axes_w_mm = box.x1 * fig_w_mm - left_mm
    if axes_w_mm < _HBAR_MIN_AXES_MM:
        fig_w_mm += _HBAR_MIN_AXES_MM - axes_w_mm
        axes_w_mm = _HBAR_MIN_AXES_MM
        fig.set_figwidth(fig_w_mm / MM_PER_INCH)
        print(f"Note: widened {chart} to {fig_w_mm:.1f} mm to fit its method names")
    ax.set_position([left_mm / fig_w_mm, box.y0, axes_w_mm / fig_w_mm, box.height])
    return -(left_mm - _HBAR_EDGE_MM) / axes_w_mm


def summary_metric_columns(metrics_df: pd.DataFrame) -> list[str]:
    """The metrics ``plot_metrics_barplots`` can draw, in column order.

    These are the eight summary-stat columns from
    ``duet.benchmark.metrics.compute_metrics``. ``lambda``, the four crowding
    columns, and the two per-axis SE columns are excluded (see
    ``_BARPLOT_EXCLUDED``).
    """
    return [c for c in metrics_df.columns if c not in _BARPLOT_EXCLUDED]


def plot_metrics_barplots(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    metrics: Iterable[str] | None = None,
) -> None:
    """Emit one horizontal bar plot per summary-stat metric in `metrics_df`.

    ``metrics`` picks which of ``summary_metric_columns`` to draw; None draws
    all of them. Each plot has one bar per method, top to bottom in
    ``metrics_df`` row order, with the method name on the y axis
    (``_hbar_slot_figure``). The title names the metric, so the value axis
    carries no label.
    """
    metric_cols = summary_metric_columns(metrics_df)
    if metrics is not None:
        wanted = set(metrics)
        unknown = wanted.difference(metric_cols)
        if unknown:
            raise ValueError(f"not summary-stat columns of metrics_df: {sorted(unknown)}")
        metric_cols = [c for c in metric_cols if c in wanted]
    palette = _ordered_method_palette(metrics_df["Method"].tolist())
    n_methods = metrics_df["Method"].nunique()

    for metric in metric_cols:
        fig, ax = _hbar_slot_figure(n_methods, top_mm=_HBAR_TITLE_MM, bottom_mm=_HBAR_TICKS_MM)
        sns.barplot(
            data=metrics_df,
            x=metric, y="Method",
            hue="Method", palette=palette,
            width=_HBAR_BAR_HEIGHT,
            ax=ax, legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Capture the seaborn-assigned category labels before pinning ticks
        # (set_yticks resets the formatter, which would wipe the text otherwise),
        # so the lambda->λ relabel lands without a FixedLocator warning.
        ylabels = [_lambda_to_symbol(t.get_text()) for t in ax.get_yticklabels()]
        ax.set_yticks(ax.get_yticks(), ylabels)
        stem = metric_barplot_stem(metric)
        ax.set_title(metric, loc="left", x=_fit_hbar_names(fig, ax, stem))
        # The value axis is autoscaled to the bars, so its last tick label can
        # land on the right edge.
        fit_last_xtick_label(fig, ax)

        save_panel(fig, output_dir / f"{stem}.svg", output_dir / f"{stem}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(metric_cols)} metric barplots to {output_dir}")


def _non_duet_methods(metrics_df: pd.DataFrame) -> list[str]:
    """Baseline (non-DUET) method labels, in ``metrics_df`` row order."""
    if "lambda" in metrics_df.columns:
        duet = set(metrics_df.loc[metrics_df["lambda"].notna(), "Method"])
    else:
        duet = set()
    return [m for m in metrics_df["Method"].tolist() if m not in duet]


def _largest_lambda_method(metrics_df: pd.DataFrame) -> str:
    """The DUET method at the largest lambda (lambda=1, the pure-decode sweep point)."""
    duet_mask = metrics_df["lambda"].notna() if "lambda" in metrics_df.columns else pd.Series(False, index=metrics_df.index)
    duet_rows = metrics_df.loc[duet_mask]
    if duet_rows.empty:
        raise ValueError("metrics_df has no DUET rows (no non-null 'lambda' column)")
    return str(duet_rows.loc[duet_rows["lambda"].idxmax(), "Method"])


def _render_decode_accuracy_histogram(
    results_df: pd.DataFrame,
    methods: list[str],
    output_dir: Path,
    stem: str,
    *,
    title: str = "Per-codeword decode accuracy distribution",
) -> None:
    """Render one decode-accuracy histogram over ``methods`` (first entry the
    DUET codebook, the rest baselines) to ``output_dir/<stem>.{svg,png}``.

    Factored out of ``plot_decode_accuracy_histogram`` so the canonical panel
    and the per-lambda sweep share one recipe; ``title`` lets the sweep label
    each panel with its lambda while the canonical panel keeps its fixed title.
    """
    groups = []
    for method in methods:
        accs = results_df.loc[results_df["Method"] == method, "Decode accuracy"].to_numpy()
        if accs.size == 0:
            continue
        groups.append((_lambda_to_symbol(method), accs, _method_color(method)))

    fig, ax = plot_codeword_probabilities(
        groups,
        bins=50,
        kde=True,
        print_summary=True,
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
    )
    # plot_codeword_probabilities builds its own figure with no
    # constrained_layout kwarg; switch the engine post-hoc so the
    # canvas behaves like the other OPS-style panels.
    fig.set_layout_engine("constrained")

    ax.set_xlabel("Decode accuracy")
    ax.set_title(title)

    _savefig_at_figsize(
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.png",
    )
    plt.close(fig)

    print(f"Saved decode accuracy histogram ({stem}) to {output_dir}")


def plot_decode_accuracy_histogram(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Histogram of per-codeword decode accuracies.

    DUET is represented by the row in ``metrics_df`` with the largest
    ``lambda`` value (the lambda=1 sweep point, i.e., pure decode accuracy).
    All non-DUET methods in ``results_df`` are layered on top.
    """
    _render_decode_accuracy_histogram(
        results_df,
        [_largest_lambda_method(metrics_df), *_non_duet_methods(metrics_df)],
        output_dir,
        "decode_accuracy_histogram",
    )


def _included_sweep_panels(
    duet_rows: pd.DataFrame,
    sweep_dir: Path,
    stem_prefix: str,
    include: Callable[[Path], bool] | None,
) -> tuple[list[tuple[pd.Series, str]], int]:
    """The ``(row, stem)`` pairs of a per-lambda sweep that ``include`` keeps,
    and how many panels it drops.

    Each panel's stem is ``<stem_prefix>_lambda{λ:.2f}``; ``include`` is called
    with its SVG path, and None keeps every panel. ``sweep_dir`` is created only
    when a panel is kept, so a sweep that writes nothing leaves no empty folder.
    """
    kept = []
    for _, row in duet_rows.iterrows():
        stem = f"{stem_prefix}_lambda{row['lambda']:.2f}"
        if include is None or include(sweep_dir / f"{stem}.svg"):
            kept.append((row, stem))
    if kept:
        sweep_dir.mkdir(parents=True, exist_ok=True)
    return kept, len(duet_rows) - len(kept)


def sweep_decode_accuracy_histogram(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
    include: Callable[[Path], bool] | None = None,
) -> int:
    """Emit one decode-accuracy histogram per DUET lambda into a
    ``decode_hist_sweep/`` sub-directory (each layered over the same baselines).

    The canonical ``decode_accuracy_histogram`` shows only the largest-lambda
    (decode-only) DUET point; this sweep renders every lambda so a reader can
    watch the decode-accuracy distribution degrade as decreasing lambda trades
    accuracy for reduced crowding. Filed in a sub-directory so the per-lambda panels don't flood the
    main figures directory. ``include`` picks panels by SVG path (see
    ``_included_sweep_panels``). Returns the number of panels it skipped.
    """
    if "lambda" not in metrics_df.columns:
        print("Skipping decode-accuracy-histogram sweep: no 'lambda' column")
        return 0
    duet_rows = metrics_df.loc[metrics_df["lambda"].notna()].sort_values("lambda")
    if duet_rows.empty:
        print("Skipping decode-accuracy-histogram sweep: no DUET rows")
        return 0
    non_duet = _non_duet_methods(metrics_df)
    sweep_dir = output_dir / "decode_hist_sweep"
    kept, skipped = _included_sweep_panels(
        duet_rows, sweep_dir, "decode_accuracy_histogram", include
    )
    for row, stem in kept:
        duet_method = str(row["Method"])
        _render_decode_accuracy_histogram(
            results_df,
            [duet_method, *non_duet],
            sweep_dir,
            stem,
            title=f"Per-codeword decode accuracy\n{_lambda_to_symbol(duet_method)}",
        )
    if kept:
        print(f"Saved {len(kept)} decode-accuracy-histogram sweep panels to {sweep_dir}")
    return skipped


def _annotate_duet_lambdas(ax, xs, ys, lambdas) -> None:
    """Label each DUET point (xs, ys) with ``λ=…`` via de-collided leader lines.

    Each label starts at a small up-and-left offset from its point (into the
    typically-empty upper-left region of a decode-vs-crowding Pareto front); a
    greedy vertical-spread pass then enforces a minimum gap so labels at the
    nearly-coincident decode-knee cluster don't overlap. The de-collided
    positions are converted to per-point ``offset points`` (relative to each
    data point) so they survive any post-draw layout shift — robust where
    absolute-pixel placement is not. Each label keeps a thin leader line back to
    its point; ``annotation_clip=False`` + the figure's bbox=tight save let
    callouts sit in the margin instead of being clipped.
    """
    n = len(xs)
    if n == 0:
        return
    fig = ax.figure
    fig.canvas.draw()  # realize transforms against current data limits
    pts_disp = ax.transData.transform(np.column_stack([xs, ys]))
    labels_disp = pts_disp + np.array([-10.0, 3.0])  # initial up-left offset (px)

    # Greedy bottom-to-top spread: push each label up to keep a minimum gap so
    # the clustered-lambda labels stack instead of overprinting.
    min_gap = 11.0  # px — ~the 6 pt label height with breathing room
    order = np.argsort(labels_disp[:, 1])
    for k in range(1, n):
        below, cur = order[k - 1], order[k]
        if labels_disp[cur, 1] - labels_disp[below, 1] < min_gap:
            labels_disp[cur, 1] = labels_disp[below, 1] + min_gap

    # Express each label as an offset (in points) from its own data point, so
    # the placement is robust to the axes moving on the final (savefig) draw.
    offsets_pt = (labels_disp - pts_disp) * 72.0 / fig.dpi
    for i in range(n):
        ax.annotate(
            f"λ={lambdas[i]:g}",
            xy=(xs[i], ys[i]), xycoords="data",
            xytext=tuple(offsets_pt[i]), textcoords="offset points",
            fontsize=6, ha="right", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55",
                            shrinkA=1.0, shrinkB=2.0),
            annotation_clip=False,
            zorder=5,
        )


def _place_legend_clear_of_data(ax, legend, *, n_grid: int = 25, pad_mm: float = 1.0) -> None:
    """Move an in-axes ``legend`` to the spot that hides the fewest points.

    ``loc="best"`` tries only nine fixed spots. On the crowding fronts the
    four-row legend fits none of them cleanly, so it settled where it hid one
    marker: Boström HW4 on Fig 4b. This scans a ``n_grid`` x ``n_grid`` grid of
    positions inside the axes, ``pad_mm`` from the spines, and keeps the one
    that overlaps the fewest markers, then the least connector line (sampled
    every 0.25 mm), then the one nearest an axes corner. The legend frame is
    near-opaque, so a marker under it is lost, while a short stretch of
    connector under it still reads.

    Call it after the title and labels are set: it draws the figure once so the
    layout engine fixes the axes box first.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    px_per_mm = fig.dpi / MM_PER_INCH
    pad = pad_mm * px_per_mm
    w, h = legend_box.width, legend_box.height
    if w + 2 * pad > axes_box.width or h + 2 * pad > axes_box.height:
        return  # no room to move it; leave matplotlib's placement

    # Markers as (x, y, radius) in display units.
    markers = []
    for coll in ax.collections:
        offsets = np.asarray(coll.get_offsets())
        if not len(offsets):
            continue
        xy = coll.get_offset_transform().transform(offsets)
        sizes = coll.get_sizes()
        radius = np.sqrt(sizes.max() if len(sizes) else _PARETO_MARKER_SIZE) / 2 * fig.dpi / 72
        markers.extend((x, y, radius) for x, y in xy)
    markers = np.asarray(markers).reshape(-1, 3)

    line_points = [np.empty((0, 2))]
    for line in ax.lines:
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / (0.25 * px_per_mm)) + 1)
            t = np.linspace(0.0, 1.0, n)[:, None]
            line_points.append(np.hstack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]))
    line_points = np.vstack(line_points)

    corners = np.array([
        [axes_box.x0, axes_box.y0], [axes_box.x1, axes_box.y0],
        [axes_box.x0, axes_box.y1], [axes_box.x1, axes_box.y1],
    ])
    best = None
    for x in np.linspace(axes_box.x0 + pad, axes_box.x1 - pad - w, n_grid):
        for y in np.linspace(axes_box.y0 + pad, axes_box.y1 - pad - h, n_grid):
            mx, my, mr = markers.T
            n_markers = int(np.sum(
                (mx + mr >= x) & (mx - mr <= x + w) & (my + mr >= y) & (my - mr <= y + h)
            ))
            lx, ly = line_points.T
            n_line = int(np.sum((lx >= x) & (lx <= x + w) & (ly >= y) & (ly <= y + h)))
            box_corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            corner_dist = float(np.min(np.linalg.norm(box_corners - corners, axis=1)))
            score = (n_markers, n_line, corner_dist)
            if best is None or score < best[0]:
                best = (score, x, y)
    _, x, y = best
    legend.set_loc(((x - axes_box.x0) / axes_box.width, (y - axes_box.y0) / axes_box.height))


# Baseline method families whose Hamming-weight variants form a 2-point
# mini-Pareto on the crowding front: both variants share the family color
# (see METHOD_PALETTE) and are joined by a connector, so Hamming weight reads
# as the family's only lever. Methods outside these prefixes (e.g. a published
# Zhang codebook) render as standalone markers.
_BASELINE_FAMILY_PREFIXES = ("Bostrom et al.", "MERFISH MHD4")
# Per-Hamming-weight marker so the two same-color points in a family (and their
# legend swatches) stay distinguishable without spending a second color channel.
_BASELINE_HW_MARKER = {4: "o", 5: "s", 6: "^"}


def _baseline_family_and_hw(method: str) -> tuple[str | None, int | None]:
    """Parse a baseline label into ``(family, hamming_weight)``.

    ``family`` is the shared prefix for methods grouped into one connected
    mini-Pareto (``Bostrom et al.`` / ``MERFISH MHD4``), or None for standalone
    baselines. ``hamming_weight`` is parsed from a ``Hamming weight N`` token
    when present, else None.
    """
    hw_match = re.search(r"Hamming weight (\d+)", method)
    hw = int(hw_match.group(1)) if hw_match else None
    for prefix in _BASELINE_FAMILY_PREFIXES:
        if method.startswith(prefix):
            return prefix, hw
    return None, hw


def _draw_crowding_pareto(ax, metrics_df: pd.DataFrame, *, annotate_lambda: bool = False) -> None:
    """Draw the decode-accuracy vs identified-fraction Pareto front onto ``ax``.

    DUET rows (non-null ``lambda``) form a connected sweep (thin alpha-0.4 line
    through points sorted by mean identified fraction — the y axis — descending)
    in the canonical DUET color. Non-DUET baseline rows with non-null
    ``mean_identified_fraction`` render as markers colored from
    ``_ordered_method_palette``. Baselines belonging to a recognized method
    family (``Bostrom et al.`` / ``MERFISH MHD4``; see
    ``_baseline_family_and_hw``) share one family color and are joined by a thin
    connector through their Hamming-weight points — a 2-point mini-Pareto that
    treats Hamming weight as the family's only lever. A family with a single
    point (e.g. only HW5 present) draws just the marker, no line. Standalone
    baselines outside any family render as lone markers, as before. Within a
    family the two points are kept distinct by per-Hamming-weight marker shape.
    When ``annotate_lambda`` is set, each DUET point is labeled with its lambda.
    """
    has_lambda = "lambda" in metrics_df.columns
    duet_rows = metrics_df.loc[metrics_df["lambda"].notna()] if has_lambda else metrics_df.iloc[0:0]
    baseline_rows = metrics_df.loc[
        (~metrics_df.index.isin(duet_rows.index))
        & metrics_df["mean_identified_fraction"].notna()
    ]
    color_map = _ordered_method_palette(metrics_df["Method"].tolist())

    sorted_duet = None
    if not duet_rows.empty:
        # Order the connector by the Y axis (identified fraction), descending —
        # mirrors visualize_benchmark.py's Pareto connector (sort_values on the
        # y-column, ascending=False). It traces the frontier cleanly through the
        # clustered-lambda knee, where sorting by the X axis (decode accuracy)
        # makes the line zig-zag: e.g. lambda=1 (decode-only) has the lowest identified fraction
        # but a mid-range decode, so an x-sort dips the line down to it and back.
        sorted_duet = duet_rows.sort_values("mean_identified_fraction", ascending=False)
        duet_color = _method_color("DUET")
        # Lines and markers are drawn fully opaque (alpha=1.0). The tight cluster
        # of DUET lambdas at the decode knee stays separable via the white marker
        # edges rather than via translucency.
        ax.plot(
            sorted_duet["Mean decode accuracy"],
            sorted_duet["mean_identified_fraction"],
            color=duet_color,
            linewidth=_PARETO_LINE_WIDTH,
            alpha=1.0,
            zorder=1,
        )
        ax.scatter(
            sorted_duet["Mean decode accuracy"],
            sorted_duet["mean_identified_fraction"],
            color=duet_color,
            s=_PARETO_MARKER_SIZE,
            alpha=1.0,
            edgecolors="white",
            linewidth=0.3,
            label="DUET",
            zorder=3,
        )

    # Connect each multi-point method family (Bostrom / MERFISH MHD4) with a thin
    # family-colored line FIRST (under the markers) so the two Hamming-weight
    # points read as a 2-point mini-Pareto. Drawn before the scatters so markers
    # sit on top; single-point families contribute no line.
    family_rows: dict[str, list] = {}
    for _, row in baseline_rows.iterrows():
        family, _hw = _baseline_family_and_hw(row["Method"])
        if family is not None:
            family_rows.setdefault(family, []).append(row)
    for fam, rows in family_rows.items():
        if len(rows) < 2:
            continue
        fam_df = pd.DataFrame(rows).sort_values("Mean decode accuracy")
        ax.plot(
            fam_df["Mean decode accuracy"],
            fam_df["mean_identified_fraction"],
            color=color_map.get(rows[0]["Method"], UNKNOWN_METHOD_COLOR),
            linewidth=_PARETO_LINE_WIDTH,
            alpha=1.0,
            zorder=2,
        )

    for _, row in baseline_rows.iterrows():
        method = row["Method"]
        _family, hw = _baseline_family_and_hw(method)
        ax.scatter(
            row["Mean decode accuracy"],
            row["mean_identified_fraction"],
            color=color_map.get(method, UNKNOWN_METHOD_COLOR),
            s=_PARETO_MARKER_SIZE,
            alpha=1.0,
            edgecolors="white",
            linewidth=0.3,
            marker=_BASELINE_HW_MARKER.get(hw, "o"),
            label=method,
            zorder=4,
        )

    ax.set_xlabel("Mean decode accuracy")
    ax.set_ylabel("Mean resolved fraction")
    # Centred on the figure, not the axes: at 9 pt the title is ~52 mm, and
    # centred on the axes of a 57.33 mm panel it ran ~3 mm past the right edge
    # (Fig 4b,c). Constrained layout keeps room for it above the axes.
    ax.figure.suptitle(
        "Decode accuracy vs optical crowding", fontsize=plt.rcParams["axes.titlesize"]
    )
    # No "Method" title and tight spacing (31.6 x 12.0 mm on Zhang v2, from
    # 34.2 x 16.1): with the title, the legend covered a marker wherever it
    # sat in the 44 x 35 mm axes of a third-page panel (Boström HW4 on Fig 4b).
    # The entries are method names, so the title said nothing they do not.
    handles, labels = ax.get_legend_handles_labels()
    legend = _boxed_legend(
        ax, handles, _abbrev_legend_labels(labels),
        labelspacing=0.2, borderpad=0.3, handlelength=1.2, handletextpad=0.3,
    )
    _place_legend_clear_of_data(ax, legend)

    if annotate_lambda and sorted_duet is not None:
        _annotate_duet_lambdas(
            ax,
            sorted_duet["Mean decode accuracy"].to_numpy(),
            sorted_duet["mean_identified_fraction"].to_numpy(),
            sorted_duet["lambda"].to_numpy(),
        )


def plot_crowding_pareto_front(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Decode accuracy vs identified fraction Pareto front.

    DUET rows (non-null ``lambda``) form the sweep — connected by a thin
    alpha-0.4 line through points sorted by mean identified fraction (the y
    axis), descending. Non-DUET baselines render as markers; the Bostrom and
    MERFISH-MHD4 families share one color per family and join their two
    Hamming-weight points with a thin connector (a 2-point mini-Pareto —
    Hamming weight is the family's only lever), while standalone baselines stay
    lone markers (see ``_draw_crowding_pareto``). A companion
    ``crowding_pareto_front_lambda_labeled`` figure adds per-point lambda callouts
    (see ``plot_crowding_pareto_front_lambda_labeled``).
    """
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
        constrained_layout=True,
    )
    _draw_crowding_pareto(ax, metrics_df, annotate_lambda=False)
    _savefig_at_figsize(
        output_dir / "crowding_pareto_front.svg",
        output_dir / "crowding_pareto_front.png",
    )
    plt.close(fig)

    print(f"Saved crowding Pareto front to {output_dir}")


def plot_crowding_pareto_front_lambda_labeled(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Companion to ``plot_crowding_pareto_front`` with each DUET point labeled
    by its lambda.

    Same axes/colors, but every DUET sweep point carries an ``λ=…`` callout with
    a thin leader line. Labels are de-collided by vertical spread (see
    ``_annotate_duet_lambdas``) so the tight cluster of lambdas at the decode knee
    stays readable. Rendered on a larger canvas and saved with the stylesheet's
    bbox=tight (rather than the figsize-pinned helper) so the callouts aren't
    clipped. Baselines (NaN lambda) are not labeled. No-op when there is no DUET
    sweep.
    """
    if "lambda" not in metrics_df.columns or metrics_df["lambda"].notna().sum() == 0:
        return
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["half_page"], 3.0),
        constrained_layout=True,
    )
    _draw_crowding_pareto(ax, metrics_df, annotate_lambda=True)
    # bbox=tight (stylesheet default) so the leader-line callouts that extend
    # past the axes are captured rather than clipped.
    plt.savefig(output_dir / "crowding_pareto_front_lambda_labeled.svg")
    plt.savefig(output_dir / "crowding_pareto_front_lambda_labeled.png", dpi=150)
    plt.close(fig)

    print(f"Saved lambda-labeled crowding Pareto front to {output_dir}")


def plot_crowding_bar_chart(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Stacked identified/missed horizontal bars for every method with crowding
    metrics, one row per method, top to bottom in ``metrics_df`` row order.

    The bars fill the axes width, so the legend sits above them, in a row of
    its own between the title and the bars.
    """
    rows = metrics_df.loc[metrics_df["mean_identified_fraction"].notna()]
    methods = rows["Method"].tolist()
    id_fracs = rows["mean_identified_fraction"].to_numpy()
    conflict_fracs = rows["mean_conflict_fraction"].to_numpy()

    fig, ax = _hbar_slot_figure(
        len(methods),
        top_mm=_HBAR_TITLE_MM + _HBAR_LEGEND_ROW_MM,
        bottom_mm=DEFAULT_MARGINS_MM[2],
    )
    y = np.arange(len(methods))
    color_map = _ordered_method_palette(methods)
    identified_colors = [color_map[m] for m in methods]
    ax.barh(y, id_fracs, _HBAR_BAR_HEIGHT, color=identified_colors, label="Resolved")
    ax.barh(y, conflict_fracs, _HBAR_BAR_HEIGHT, left=id_fracs,
            color="#d32f2f", label="Missed", alpha=0.7)

    ax.set_xlabel("Fraction of transcripts")
    ax.set_yticks(y, [_lambda_to_symbol(m) for m in methods])
    ax.set_ylim(len(methods) - 0.5, -0.5)  # first method on top
    ax.set_xlim(0, 1.05)
    # The pad (pt) lifts the title over the legend row.
    ax.set_title("Optical crowding: resolved vs missed", loc="left",
                 x=_fit_hbar_names(fig, ax, "crowding_bar_chart"),
                 pad=plt.rcParams["axes.titlepad"] + _HBAR_LEGEND_ROW_MM / MM_PER_INCH * 72)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
              fontsize="small", borderaxespad=0.3, columnspacing=1.0)

    save_panel(
        fig,
        output_dir / "crowding_bar_chart.svg",
        output_dir / "crowding_bar_chart.png",
        dpi=150,
    )
    plt.close(fig)

    print(f"Saved crowding bar chart to {output_dir}")


def _method_sort_key(method: str) -> tuple[int, float, str]:
    """Sort key: DUET lambdas first (descending, so the decode-only lambda=1
    arm leads), then baselines alphabetically.

    Returns (group, -lambda, name) where group=0 for DUET and 1 for baselines.
    Baselines sort alphabetically by method name within the second group
    (i.e. 'Maximum activity' sorts before 'Zhang et al. codebook #2').
    """
    if method.startswith("DUET (lambda="):
        try:
            lambda_val = float(method.split("lambda=")[1].rstrip(")"))
        except ValueError:
            lambda_val = float("inf")
        return (0, -lambda_val, method)
    return (1, 0.0, method)


def plot_hamming_weight_distribution(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Stacked-bar Hamming-weight distribution per method.

    For each method in `results_df`, plots the fraction of selected codewords
    at each Hamming weight (number of '1' bits in `Sequence`). Methods are
    ordered with DUET lambdas ascending followed by baselines. HW segments
    are hued with Okabe-Ito colors chosen to NOT overlap METHOD_PALETTE
    (so a reader scanning a composited multi-panel figure can't misread an
    HW segment as a method color).
    """
    df = results_df.copy()
    df["HW"] = df["Sequence"].astype(str).str.count("1")
    methods = sorted(df["Method"].unique(), key=_method_sort_key)

    # Rows sum to 1 (fraction of selected codewords at each HW per method).
    ct = pd.crosstab(df["Method"], df["HW"], normalize="index").reindex(methods)

    hw_to_color = _hamming_weight_colors(df)
    colors = [hw_to_color[hw] for hw in ct.columns]

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
        constrained_layout=True,
    )
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.6,
            edgecolor="white", linewidth=0.5)

    xlabels = [_lambda_to_symbol(t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of selected codewords")
    ax.set_title("Hamming weight distribution by method")
    ax.legend(title="Hamming weight", loc="best", fontsize="small")

    _savefig_at_figsize(
        output_dir / "hamming_weight_distribution.svg",
        output_dir / "hamming_weight_distribution.png",
    )
    plt.close(fig)

    print(f"Saved Hamming weight distribution to {output_dir}")


def _select_matched_duet_lambda(
    metrics_df: pd.DataFrame, baseline_method: str
) -> str | None:
    """Pick the DUET method (label) with highest decode accuracy among rows
    whose mean_identified_fraction >= baseline's. None if no DUET rows.

    Adapted from `scripts/benchmark/visualize_benchmark.py:select_duet_lambda`
    (OPS version). The OPS function returns None when no DUET row satisfies
    the threshold; this MERFISH variant adds a fallback to the DUET row with
    highest mean_identified_fraction so the jointplot always renders, with
    a printed warning when the fallback fires.
    """
    baseline_row = metrics_df.loc[metrics_df["Method"] == baseline_method]
    if baseline_row.empty:
        return None
    threshold = float(baseline_row["mean_identified_fraction"].iloc[0])

    duet = metrics_df.loc[metrics_df["lambda"].notna()] if "lambda" in metrics_df.columns else metrics_df.iloc[0:0]
    if duet.empty or "mean_identified_fraction" not in duet.columns:
        return None

    qualifying = duet.loc[duet["mean_identified_fraction"] >= threshold]
    if not qualifying.empty:
        return str(qualifying.loc[qualifying["Mean decode accuracy"].idxmax(), "Method"])

    # Fallback: drop NaN before idxmax (pandas raises on all-NaN slices).
    duet_with_idf = duet.dropna(subset=["mean_identified_fraction"])
    if duet_with_idf.empty:
        return None
    return str(duet_with_idf.loc[duet_with_idf["mean_identified_fraction"].idxmax(), "Method"])


def plot_decode_vs_identified_jointplot(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Jointplot of per-codeword decode accuracy vs identified fraction.

    Picks the highest-mean-identified-fraction baseline (a Bostrom / MERFISH-MHD4
    / Zhang family member, depending on the panel) and the DUET lambda whose mean
    identified fraction >= that baseline's AND has the highest mean decode
    accuracy. Overlays the baseline + matched-DUET as scatter + KDE contours
    with marginal histograms (recipe from
    scripts/benchmark/visualize_benchmark.py:plot_jointplot).

    Codewords with fewer than ``MIN_TRANSCRIPTS_FOR_JOINTPLOT`` transcripts are
    dropped (small-sample noise). The identified-fraction marginal is weighted by
    transcript count (transcript-weighted; approximately inverse-variance, since
    var(IF) ~ p(1-p)/n_transcripts with p(1-p) roughly constant across genes); the
    decode-accuracy marginal is unweighted.
    """
    if "Identified fraction" not in results_df.columns:
        print("Skipping decode-vs-identified jointplot: 'Identified fraction' column missing")
        return

    if "lambda" in metrics_df.columns:
        baselines = metrics_df.loc[metrics_df["lambda"].isna()]
    else:
        baselines = metrics_df
    baselines = baselines.loc[baselines["mean_identified_fraction"].notna()]
    if baselines.empty:
        print("Skipping decode-vs-identified jointplot: no baseline rows with crowding metrics")
        return

    baseline_row = baselines.loc[baselines["mean_identified_fraction"].idxmax()]
    baseline_method = str(baseline_row["Method"])

    duet_method = _select_matched_duet_lambda(metrics_df, baseline_method)
    if duet_method is None:
        print("Skipping decode-vs-identified jointplot: no DUET rows in metrics_df")
        return

    duet_id = float(metrics_df.loc[metrics_df["Method"] == duet_method, "mean_identified_fraction"].iloc[0])
    if duet_id < float(baseline_row["mean_identified_fraction"]):
        print(
            f"WARNING: no DUET lambda satisfies identified_fraction >= "
            f"{baseline_row['mean_identified_fraction']:.4f} (baseline {baseline_method}); "
            f"falling back to DUET with highest identified fraction ({duet_method}, {duet_id:.4f})"
        )

    _render_decode_vs_identified_jointplot(
        results_df, baseline_method, duet_method, output_dir,
        "decode_vs_identified_jointplot",
    )


def sweep_decode_vs_identified_jointplot(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
    include: Callable[[Path], bool] | None = None,
) -> int:
    """Emit a decode-vs-identified jointplot for EVERY DUET lambda into a
    ``jointplot_sweep/`` sub-directory, each against the fixed
    highest-identified-fraction baseline.

    The canonical ``decode_vs_identified_jointplot`` shows only the matched
    lambda; this sweep renders the whole frontier so a reader can pick a lambda
    by eye. Filed in a sub-directory to keep the main figures directory tidy.
    ``include`` picks panels by SVG path (see ``_included_sweep_panels``); the
    Fig 4b/4c insets are single panels of this sweep. Returns the number of
    panels it skipped.
    """
    if "Identified fraction" not in results_df.columns:
        print("Skipping jointplot sweep: 'Identified fraction' column missing")
        return 0
    if "lambda" not in metrics_df.columns or "mean_identified_fraction" not in metrics_df.columns:
        print("Skipping jointplot sweep: needs 'lambda' + 'mean_identified_fraction' columns")
        return 0
    baselines = metrics_df.loc[
        metrics_df["lambda"].isna() & metrics_df["mean_identified_fraction"].notna()
    ]
    if baselines.empty:
        print("Skipping jointplot sweep: no baseline rows with crowding metrics")
        return 0
    baseline_method = str(baselines.loc[baselines["mean_identified_fraction"].idxmax(), "Method"])
    duet_rows = metrics_df.loc[metrics_df["lambda"].notna()].sort_values("lambda")
    if duet_rows.empty:
        print("Skipping jointplot sweep: no DUET rows")
        return 0
    sweep_dir = output_dir / "jointplot_sweep"
    kept, skipped = _included_sweep_panels(
        duet_rows, sweep_dir, "decode_vs_identified_jointplot", include
    )
    for row, stem in kept:
        _render_decode_vs_identified_jointplot(
            results_df, baseline_method, str(row["Method"]), sweep_dir, stem,
        )
    if kept:
        print(f"Saved {len(kept)} jointplot sweep panels to {sweep_dir}")
    return skipped


def _render_decode_vs_identified_jointplot(
    results_df: pd.DataFrame,
    baseline_method: str,
    duet_method: str,
    output_dir: Path,
    stem: str,
) -> None:
    """Render one baseline-vs-DUET jointplot (scatter + KDE contours + weighted
    marginals) to ``output_dir/<stem>.{svg,png}``.

    Factored out of ``plot_decode_vs_identified_jointplot`` so the canonical
    matched-lambda panel and the per-lambda sweep share one recipe; choosing the
    (baseline, duet_method) pair to draw stays with the callers.
    """
    method_to_color = {
        baseline_method: _method_color(baseline_method),
        duet_method: _method_color(duet_method),
    }
    df = results_df.loc[results_df["Method"].isin([baseline_method, duet_method])].copy()
    # Drop codewords with NaN identified fraction (genes that never produced
    # a transcript in any crowding trial). seaborn.kdeplot would otherwise
    # raise; scatterplot drops them silently which would make the marginal
    # disagree with the scatter density.
    df = df.dropna(subset=["Identified fraction"])
    # Drop genes whose identified fraction rests on too few simulated transcripts
    # (small-sample noise -- e.g. a single transcript that happened to conflict ->
    # a hard 0). Needs the realized per-gene count (n_transcripts), backfilled by
    # scripts/benchmark/recompute_crowding.py; skipped (with a note) if absent.
    if "n_transcripts" in df.columns:
        before = len(df)
        df = df[df["n_transcripts"] >= MIN_TRANSCRIPTS_FOR_JOINTPLOT]
        print(f"  jointplot: kept {len(df)}/{before} codewords with >= "
              f"{MIN_TRANSCRIPTS_FOR_JOINTPLOT} transcripts (dropped {before - len(df)} low-count)")
    else:
        print("  jointplot: 'n_transcripts' column absent -- skipping the low-count "
              "filter; run scripts/benchmark/recompute_crowding.py to enable it")
    if df.empty:
        print(f"Skipping decode-vs-identified jointplot: no per-codeword rows for {baseline_method} or {duet_method}")
        return

    g = sns.JointGrid(
        data=df,
        x="Decode accuracy", y="Identified fraction",
        height=FIGURE_WIDTHS["third_page"], ratio=2, space=0.2,
    )

    sns.scatterplot(
        data=df, x="Decode accuracy", y="Identified fraction",
        hue="Method", palette=method_to_color,
        s=15, alpha=0.6, edgecolor="white", linewidth=0.5,
        ax=g.ax_joint,
    )

    for method, color in method_to_color.items():
        method_data = df[df["Method"] == method]
        if len(method_data) < 5:
            continue
        sns.kdeplot(
            data=method_data, x="Decode accuracy", y="Identified fraction",
            ax=g.ax_joint, levels=5, color=color, linewidths=1.5, alpha=0.7,
        )

    # The identified-fraction (y) marginal is weighted by each gene's realized
    # transcript count, which makes it transcript-weighted -- and, because
    # var(per-codeword IF) ~ p(1-p)/n_transcripts with p(1-p) roughly constant
    # across genes, approximately inverse-variance -- so noisy low-count genes
    # contribute in proportion to their information and the marginal (approximately)
    # tracks the aggregate mean_identified_fraction over the retained genes. The
    # decode-accuracy (x) marginal is left unweighted: decode accuracy is a
    # Monte-Carlo estimate over a fixed num_samples, so its per-codeword precision
    # is constant and inverse-variance weighting there would be uniform.
    marg_y_weights = "n_transcripts" if "n_transcripts" in df.columns else None
    for method, color in method_to_color.items():
        method_data = df[df["Method"] == method]
        if method_data.empty:
            continue
        sns.histplot(
            data=method_data, x="Decode accuracy", ax=g.ax_marg_x,
            color=color, alpha=0.35, edgecolor=None,
            kde=True, stat="density",
            line_kws={"linewidth": 1.6, "alpha": 1.0},
            kde_kws={"cut": 0},
        )
        # A *weighted* KDE raises (ValueError / LinAlgError) on a single-survivor
        # or zero-variance subset, where the unweighted path merely skips it; gate
        # the y-marginal KDE on the subset actually having spread so a degenerate
        # method can't abort the whole figure. Latent on the real panels (hundreds
        # of positive-variance survivors), but the sibling kdeplot loop guards the
        # same way.
        y_has_spread = (
            len(method_data) >= 2
            and method_data["Identified fraction"].nunique() >= 2
        )
        sns.histplot(
            data=method_data, y="Identified fraction", ax=g.ax_marg_y,
            weights=marg_y_weights,
            # seaborn can't auto-bin weighted data (Freedman-Diaconis needs raw
            # counts); pin an explicit bin count to avoid its coarse bins=10 fallback.
            bins=30 if marg_y_weights else "auto",
            color=color, alpha=0.35, edgecolor=None,
            kde=y_has_spread, stat="density",
            line_kws={"linewidth": 1.6, "alpha": 1.0},
            kde_kws={"cut": 0},
        )

    # The scatter y-positions AND the 2D KDE contours are per-codeword
    # (gene-weighted); only the y-marginal above is transcript-weighted (weighted
    # by n_transcripts), so the marginal -- not the raw scatter/contour spread --
    # is what (approximately) tracks the bar chart's transcript-weighted
    # mean_identified_fraction.
    # Abbreviate the (long) method names so the legend fits inside the narrow
    # third_page panel without clipping; the underlying data labels are untouched.
    # framealpha < 1 keeps the (few) high-identified-fraction points under the
    # upper-left box visible; fontsize="small" matches the other legends in this
    # file and scales with the stylesheet rather than hardcoding a point size.
    # Show each method as a short line swatch rather than the scatter marker
    # seaborn puts in the legend by default. The series already reads as a
    # colored KDE contour *line* in the panel, and at the narrow third_page size
    # a line swatch is easier to distinguish from an actual data point than a
    # single dot in a box. Colors come from method_to_color (the same source the
    # scatter and KDE use), keyed by the un-abbreviated Method label seaborn
    # returned; a linewidth a touch above the 1.5 contour reads cleanly at this size.
    # Filter to labels we have a color for so a seaborn build that injects the
    # hue-variable name as a title row can't slip a junk entry (or KeyError) in.
    _, labels = g.ax_joint.get_legend_handles_labels()
    method_labels = [label for label in labels if label in method_to_color]
    line_handles = [
        Line2D([0], [0], color=method_to_color[label], linewidth=2.5)
        for label in method_labels
    ]
    short_labels = _abbrev_legend_labels(method_labels)
    g.ax_joint.legend(
        line_handles, short_labels,
        loc="upper left", fontsize="small", frameon=True, framealpha=0.6,
        edgecolor="none", title=None, borderpad=0.3, handletextpad=0.4,
        labelspacing=0.3,
    )
    g.set_axis_labels("Decode accuracy", "Resolved fraction")
    plt.tight_layout()
    _savefig_at_figsize(
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.png",
    )
    plt.close()

    print(f"Saved decode-vs-identified jointplot ({baseline_method} vs {duet_method}) to {output_dir}")


def _select_matched_baseline_and_duet(metrics_df: pd.DataFrame) -> tuple[str, str] | None:
    """The (baseline, DUET) pair the jointplot and its companion diagnostics
    compare: the baseline with the highest mean identified fraction, paired with
    the DUET lambda matched to it by ``_select_matched_duet_lambda``. Returns
    None if either is unavailable (mirrors the jointplot's own selection)."""
    if "mean_identified_fraction" not in metrics_df.columns:
        return None
    baselines = metrics_df.loc[metrics_df["lambda"].isna()] if "lambda" in metrics_df.columns else metrics_df
    baselines = baselines.loc[baselines["mean_identified_fraction"].notna()]
    if baselines.empty:
        return None
    baseline_method = str(baselines.loc[baselines["mean_identified_fraction"].idxmax(), "Method"])
    duet_method = _select_matched_duet_lambda(metrics_df, baseline_method)
    if duet_method is None:
        return None
    return baseline_method, duet_method


def plot_identified_fraction_vs_count(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Diagnostic: per-codeword identified fraction vs realized transcript count
    (log x), for the jointplot's baseline + matched DUET.

    Confirms whether the jointplot's low-identified-fraction tail is small-sample
    noise: low-count genes should show a wide downward spread that tightens as the
    count grows, and a binned median that climbs then flattens. Shown UNFILTERED
    (every gene with >= 1 transcript, including those the jointplot drops) with a
    dashed reference line at ``MIN_TRANSCRIPTS_FOR_JOINTPLOT`` so the filter's cut
    can be judged against the actual count->fraction relationship.
    """
    from matplotlib.transforms import blended_transform_factory

    if "Identified fraction" not in results_df.columns or "n_transcripts" not in results_df.columns:
        print("Skipping identified-fraction-vs-count: needs 'Identified fraction' + 'n_transcripts' columns")
        return
    sel = _select_matched_baseline_and_duet(metrics_df)
    if sel is None:
        print("Skipping identified-fraction-vs-count: no baseline/DUET comparison pair")
        return
    baseline_method, duet_method = sel

    method_to_color = {
        baseline_method: _method_color(baseline_method),
        duet_method: _method_color(duet_method),
    }
    df = results_df.loc[results_df["Method"].isin([baseline_method, duet_method])].copy()
    df = df.dropna(subset=["Identified fraction", "n_transcripts"])
    df = df[df["n_transcripts"] > 0]
    if df.empty:
        print("Skipping identified-fraction-vs-count: no rows with transcripts")
        return

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
        constrained_layout=True,
    )
    for method, color in method_to_color.items():
        md = df[df["Method"] == method]
        ax.scatter(
            md["n_transcripts"], md["Identified fraction"],
            s=8, alpha=0.5, color=color, edgecolors="white", linewidth=0.3,
            label=method, zorder=3,
        )

    # Pooled binned median (the count->fraction trend is about sample size, not
    # method): log-spaced bins so the low-count regime is resolved.
    counts = df["n_transcripts"].to_numpy(dtype=float)
    if counts.max() > counts.min():  # need a real range to bin (avoids duplicate edges)
        edges = np.logspace(np.log10(counts.min()), np.log10(counts.max()), 12)
        binned = (
            df.assign(_bin=pd.cut(df["n_transcripts"], edges, include_lowest=True))
            .groupby("_bin", observed=True)
            .agg(x=("n_transcripts", "median"), y=("Identified fraction", "median"))
            .dropna()
        )
        if len(binned) >= 2:
            ax.plot(binned["x"], binned["y"], color="black", linewidth=1.0,
                    marker="o", markersize=2, zorder=4, label="binned median")

    ax.set_xscale("log")
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(MIN_TRANSCRIPTS_FOR_JOINTPLOT, color="0.55", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(MIN_TRANSCRIPTS_FOR_JOINTPLOT, 0.03, f" jointplot filter (>= {MIN_TRANSCRIPTS_FOR_JOINTPLOT})",
            transform=trans, fontsize=6, color="0.4", va="bottom", ha="left")

    ax.set_xlabel("Transcripts per gene (summed over trials, log scale)")
    ax.set_ylabel("Resolved fraction")
    ax.set_title("Resolved fraction vs sample size")
    handles, labels = ax.get_legend_handles_labels()
    _boxed_legend(ax, handles, _abbrev_legend_labels(labels), loc="best")

    _savefig_at_figsize(
        output_dir / "identified_fraction_vs_count.svg",
        output_dir / "identified_fraction_vs_count.png",
    )
    plt.close(fig)
    print(f"Saved identified-fraction-vs-count diagnostic ({baseline_method} vs {duet_method}) to {output_dir}")


def plot_decode_accuracy_histogram_by_hamming_weight(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Per-codeword decode-accuracy histogram for the comparison DUET lambda,
    split by Hamming weight.

    Selects the DUET lambda "intended to be compared against the codebook" --
    the highest-decode-accuracy DUET sweep point whose mean identified fraction
    meets or beats the baseline -- via the same ``_select_matched_duet_lambda``
    rule used by ``plot_decode_vs_identified_jointplot``. For that single
    codebook it overlays the decode-accuracy distribution of its codewords
    grouped by Hamming weight (number of '1' bits), using the same HW color
    scheme as ``hamming_weight_distribution.svg``.

    Diagnostic intent: reveal whether the low-decode-accuracy codewords are
    concentrated at a particular Hamming weight.
    """
    if "mean_identified_fraction" not in metrics_df.columns:
        print("Skipping decode-accuracy-by-HW histogram: no crowding metrics "
              "(cannot select the comparison lambda)")
        return

    baselines = metrics_df.loc[metrics_df["lambda"].isna()] if "lambda" in metrics_df.columns else metrics_df
    baselines = baselines.loc[baselines["mean_identified_fraction"].notna()]
    if baselines.empty:
        print("Skipping decode-accuracy-by-HW histogram: no baseline rows with crowding metrics")
        return
    baseline_method = str(baselines.loc[baselines["mean_identified_fraction"].idxmax(), "Method"])

    duet_method = _select_matched_duet_lambda(metrics_df, baseline_method)
    if duet_method is None:
        print("Skipping decode-accuracy-by-HW histogram: no matched DUET lambda in metrics_df")
        return

    _render_decode_accuracy_histogram_by_hw(
        results_df, duet_method, output_dir,
        "decode_accuracy_histogram_by_hamming_weight",
    )


def sweep_decode_accuracy_histogram_by_hamming_weight(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
    include: Callable[[Path], bool] | None = None,
) -> int:
    """Emit a decode-accuracy-by-Hamming-weight histogram for EVERY DUET lambda
    into a ``decode_hist_by_hw_sweep/`` sub-directory.

    The canonical panel shows only the matched lambda; this sweep renders every
    lambda so a reader can see how the per-HW decode-accuracy distributions shift
    across the frontier. Filed in a sub-directory to keep the main figures
    directory tidy. ``include`` picks panels by SVG path (see
    ``_included_sweep_panels``). Returns the number of panels it skipped.
    """
    if "lambda" not in metrics_df.columns:
        print("Skipping decode-accuracy-by-HW sweep: no 'lambda' column")
        return 0
    duet_rows = metrics_df.loc[metrics_df["lambda"].notna()].sort_values("lambda")
    if duet_rows.empty:
        print("Skipping decode-accuracy-by-HW sweep: no DUET rows")
        return 0
    sweep_dir = output_dir / "decode_hist_by_hw_sweep"
    kept, skipped = _included_sweep_panels(
        duet_rows, sweep_dir, "decode_accuracy_histogram_by_hamming_weight", include
    )
    for row, stem in kept:
        _render_decode_accuracy_histogram_by_hw(results_df, str(row["Method"]), sweep_dir, stem)
    if kept:
        print(f"Saved {len(kept)} decode-accuracy-by-HW sweep panels to {sweep_dir}")
    return skipped


def _render_decode_accuracy_histogram_by_hw(
    results_df: pd.DataFrame,
    duet_method: str,
    output_dir: Path,
    stem: str,
) -> None:
    """Render the decode-accuracy-by-Hamming-weight histogram for one DUET
    codebook to ``output_dir/<stem>.{svg,png}``.

    Factored out of ``plot_decode_accuracy_histogram_by_hamming_weight`` so the
    canonical matched-lambda panel and the per-lambda sweep share one recipe.
    """
    sub = results_df.loc[results_df["Method"] == duet_method].copy()
    if sub.empty:
        print(f"Skipping decode-accuracy-by-HW histogram: no per-codeword rows for {duet_method}")
        return
    sub["HW"] = sub["Sequence"].astype(str).str.count("1")

    # Color by HW from the full results_df so the assignment matches
    # hamming_weight_distribution.svg even if the chosen lambda is missing an HW.
    hw_to_color = _hamming_weight_colors(results_df)
    groups = []
    for hw in sorted(sub["HW"].unique()):
        accs = sub.loc[sub["HW"] == hw, "Decode accuracy"].to_numpy()
        if accs.size == 0:
            continue
        groups.append((f"{hw} (n={accs.size})", accs, hw_to_color[hw]))

    if not groups:
        print(f"Skipping decode-accuracy-by-HW histogram: no codewords for {duet_method}")
        return

    fig, ax = plot_codeword_probabilities(
        groups,
        bins=50,
        kde=True,
        print_summary=True,
        figsize=(FIGURE_WIDTHS["third_page"], 2.0),
    )
    # plot_codeword_probabilities builds its own figure with no
    # constrained_layout kwarg; switch the engine post-hoc so the canvas
    # behaves like the other OPS-style panels (matches
    # plot_decode_accuracy_histogram).
    fig.set_layout_engine("constrained")

    ax.set_xlabel("Decode accuracy")
    ax.set_title(f"Decode accuracy by Hamming weight\n{_lambda_to_symbol(duet_method)}")
    ax.legend(title="Hamming weight", fontsize="small")

    _savefig_at_figsize(
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.png",
    )
    plt.close(fig)

    print(f"Saved decode-accuracy-by-HW histogram ({duet_method}) to {output_dir}")


def compute_round_expression_load(
    codebook_df: pd.DataFrame,
    expr_map: dict,
    *,
    seq_col: str = "Sequence",
    gene_col: str = "Gene",
) -> np.ndarray:
    """Per-round expression load β_r = Σ_genes expr[gene] · bit[gene, r].

    `codebook_df` has one row per gene with a binary `seq_col` string and a
    `gene_col` symbol. Genes absent from `expr_map` contribute 0. Returns a
    length-``seq_rounds`` vector.
    """
    seqs = codebook_df[seq_col].astype(str).tolist()
    bits = np.array([[1.0 if c == "1" else 0.0 for c in s] for s in seqs])
    expr = codebook_df[gene_col].map(expr_map).fillna(0.0).to_numpy(float)
    return (bits * expr[:, None]).sum(axis=0)


def select_best_lambda(metrics_df: pd.DataFrame) -> float:
    """λ maximizing (`Mean decode accuracy` + `mean_identified_fraction`) over DUET
    rows (non-null ``lambda``). Tie-break: largest λ (more weight on decoding).
    Raises if no DUET rows."""
    duet = metrics_df[metrics_df["lambda"].notna()].copy()
    if duet.empty:
        raise ValueError("metrics_df has no DUET rows (all 'lambda' are null)")
    duet["_score"] = duet["Mean decode accuracy"] + duet["mean_identified_fraction"]
    duet = duet.sort_values(["_score", "lambda"], ascending=[False, False])
    return float(duet.iloc[0]["lambda"])


def codebook_from_results(results_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """{Gene, Sequence} codebook for ``method``, one row per Gene.

    `results.csv` has per-trial rows, so dedup on Gene. Raises if absent."""
    sub = results_df[results_df["Method"] == method]
    if sub.empty:
        raise ValueError(f"method {method!r} not found in results_df")
    return sub.drop_duplicates("Gene")[["Gene", "Sequence"]].reset_index(drop=True)


def _hamming_weight_shade(base_hex: str, hw: int | None, hw_levels: list[int]) -> str:
    """White-tint ``base_hex`` by Hamming weight: lower HW -> lighter shade.

    ``hw_levels`` is the ascending list of distinct Hamming weights drawn in a
    panel. The highest HW keeps the family base color; each lower HW is a
    progressively lighter white-tint (via ``sequential_shades``, which ramps
    lightest -> base). ``hw is None`` (DUET, or a standalone baseline with no
    Hamming-weight token) or a single HW level returns the base unchanged.
    Replaces the former dashed(HW5)/solid(HW4) linestyle cue, which was
    indistinguishable in the narrow third-page legend.

    ``sequential_shades`` ramps from a 0.35 white-tint up to the base. We build a
    ramp of twice as many shades as levels and keep the odd entries: this drops
    the too-pale 0.35 floor (unreadable as a line) yet still pushes the lighter
    tint well clear of the base, so adjacent Hamming weights separate by a wider
    hue gap than the default ramp -- while the highest HW lands exactly on base.
    """
    if hw is None or len(hw_levels) <= 1 or hw not in hw_levels:
        return base_hex
    return sequential_shades(base_hex, 2 * len(hw_levels))[1::2][hw_levels.index(hw)]


def plot_round_expression_load(
    loads_by_method: dict[str, np.ndarray],
    *,
    sort: bool,
    y_from_zero: bool = False,
    ax=None,
):
    """One β_r line per method, comparing baselines against a DUET codebook.

    `sort=False` -> natural imaging-round order ("uniformity"); `sort=True` ->
    each series sorted descending by its own load ("sorted"). Color =
    METHOD_PALETTE via `_method_color`, then white-tinted by Hamming weight
    (lower HW = lighter shade of the family base; see `_hamming_weight_shade`),
    so a family's two HW variants are told apart by shade rather than a
    dashed/solid distinction that is near-invisible in the small legend. Every
    series is solid and carries a small circle marker sized to leave the
    connecting line visible between points (not a solid chain of circles). Each
    legend label carries the series' CV (= std/mean of β_r) on its own line and
    is abbreviated (``Hamming weight``->``HW``, ``lambda=``->``λ=``) to fit the
    third-page panel. ``y_from_zero`` pins the y-axis floor at 0 (an absolute
    read of the per-round loads); the default keeps matplotlib's autoscaled
    floor, which magnifies the between-method spread. Insertion order of `loads_by_method` is
    draw order (put DUET last to draw it on top). With ``ax=None`` the figure is
    a third-page slot panel (``slot_figure``, default margins); save it with
    ``duet.plotting.save_panel``. Returns the Figure.
    """
    if ax is None:
        fig, ax = slot_figure(
            FIGURE_WIDTHS["third_page"] * MM_PER_INCH, 2.2 * MM_PER_INCH
        )
    else:
        fig = ax.figure
    # Distinct Hamming weights present -> a shared shade ramp so the same HW
    # maps to the same tint across method families (see _hamming_weight_shade).
    hw_levels = sorted({
        hw for hw in (_baseline_family_and_hw(m)[1] for m in loads_by_method)
        if hw is not None
    })
    for method, beta in loads_by_method.items():
        _family, hw = _baseline_family_and_hw(method)
        color = _hamming_weight_shade(_method_color(method), hw, hw_levels)
        mean = float(np.mean(beta))
        cv = float(np.std(beta) / mean) if mean else float("nan")
        y = np.sort(beta)[::-1] if sort else beta
        ax.plot(
            np.arange(len(beta)), y, color=color, linestyle="-",
            linewidth=1.0, marker="o", markersize=1.5,
            label=f"{_lambda_to_symbol(method)}\n(CV={cv:.3f})",
        )
    if y_from_zero:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Round rank (sorted by load)" if sort else "Readout bit")
    ax.set_ylabel("Total expression")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, _abbrev_legend_labels(labels), fontsize=6, frameon=False)
    return fig


def plot_expression_by_hamming_weight(
    codebook_df: pd.DataFrame,
    expr_map: dict,
    *,
    kind: str = "violin",
    seq_col: str = "Sequence",
    gene_col: str = "Gene",
    ax=None,
):
    """Single-panel log10(expression)-by-Hamming-weight for one codebook (the
    selected DUET codebook), drawn with seaborn.

    ``kind="violin"`` -> a thin violin (``cut=0``, so the body stops at the data
    with no artificial tail) with a slim inner box + median; ``kind="box"`` -> a
    thin box with whiskers and outlier fliers. HW = count of '1' bits; groups
    are colored by ``_hamming_weight_colors`` (purple sequential ramp) so the
    panel matches the other HW-grouped panels. HW groups with <2 expressed genes
    are dropped (the violin KDE needs >=2 points). With ``ax=None`` the figure
    is a third-page slot panel (``slot_figure``, default margins); save it with
    ``duet.plotting.save_panel``. Returns the Figure.
    """
    if kind not in ("violin", "box"):
        raise ValueError(f"kind must be 'violin' or 'box', got {kind!r}")
    if ax is None:
        fig, ax = slot_figure(
            FIGURE_WIDTHS["third_page"] * MM_PER_INCH, 2.2 * MM_PER_INCH
        )
    else:
        fig = ax.figure

    df = codebook_df.copy()
    df["_hw"] = df[seq_col].astype(str).str.count("1")
    df["_logexpr"] = np.log10(df[gene_col].map(expr_map).clip(lower=1e-3))
    df = df.dropna(subset=["_logexpr"])
    keep = [h for h in sorted(df["_hw"].unique()) if int((df["_hw"] == h).sum()) >= 2]
    df = df[df["_hw"].isin(keep)].copy()
    df["HW"] = df["_hw"].map(lambda h: f"HW{h}")
    order = [f"HW{h}" for h in keep]

    hw_colors = _hamming_weight_colors(codebook_df)  # keyed off the "Sequence" column
    palette = {f"HW{h}": hw_colors[h] for h in keep}
    shared = dict(
        data=df, x="HW", y="_logexpr", hue="HW", order=order, hue_order=order,
        palette=palette, saturation=1.0, linewidth=1.0, legend=False, ax=ax,
    )
    if kind == "violin":
        sns.violinplot(inner="box", cut=0, density_norm="width", width=0.45, **shared)
    else:
        sns.boxplot(width=0.28, fliersize=1.5, **shared)

    ax.set_xlabel("")
    ax.set_ylabel(r"$\log_{10}$ expression (CPM)")
    return fig
