#!/usr/bin/env python
"""Step 5: tables and the Supp Fig S4b panel across all gene panels.

Reads <outdir>/<panel>/crowding/per_class_metrics.csv and <outdir>/<panel>/duet/metrics.csv
for every panel and writes into <outdir>:
  per_panel_class_metrics.csv   all panels' per-class rows, with a `panel` column
  panel_summary.csv             one row per panel: whole-brain identified fraction and
                                decode accuracy per method, DUET wins, min and median gap
  class_across_panels.csv       per class: mean, SD and SEM over panels of each method and
                                of DUET minus each baseline, and DUET's win count
  crowding cell type robustness.{svg,pdf,png}         (Supp Fig S4b)
      markers: mean over panels; bars: standard error of that mean (SD / sqrt(panels));
      dashed lines: each method's whole-brain value, averaged over panels

    python aggregate_panels.py --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from duet.plotting import apply_style, FIGURE_WIDTHS, METHOD_PALETTE, UNKNOWN_METHOD_COLOR
from duet.merfish_benchmark.cell_type_crowding import REFERENCE_LABEL

DUET = "DUET"
BASELINES = ["Bostrom et al. (Hamming weight 5)", "MERFISH MHD4 (Hamming weight 5)"]
METHODS = [DUET] + BASELINES
LABELS = {DUET: "DUET",   # replaced by the config's duet_label in main()
          "Bostrom et al. (Hamming weight 5)": "Boström HW5",
          "MERFISH MHD4 (Hamming weight 5)": "MERFISH MHD4 HW5"}
IDENT = "mean_identified_fraction"

# --- presentation knobs (as plot_crowding_cell_type_robustness.py) ---
TITLE = "Optical crowding improvement is robust to\nmismatched gene expression prior"
XLABEL = "Whole mouse brain major cell types"
XROT, XHA, XFS = 90, "center", 5
MS = 3
FIGSIZE = (FIGURE_WIDTHS["two_thirds_page"], 3.5)
DODGE = {DUET: 0.0, BASELINES[0]: -0.2, BASELINES[1]: 0.2}


def color(method):
    return METHOD_PALETTE.get(method, UNKNOWN_METHOD_COLOR)


def load(out, panels, lam):
    frames, decode = [], {}
    for pid in panels:
        df = pd.read_csv(out / pid / "crowding" / "per_class_metrics.csv")
        frames.append(df.assign(panel=pid))
        m = pd.read_csv(out / pid / "duet" / "metrics.csv").set_index("Method")
        decode[pid] = {DUET: m.loc[f"DUET (lambda={lam:.2f})", "Mean decode accuracy"],
                       **{b: m.loc[b, "Mean decode accuracy"] for b in BASELINES}}
    return pd.concat(frames, ignore_index=True), decode


def panel_summary(long, decode):
    rows = []
    for pid, df in long.groupby("panel", sort=False):
        ident = df.pivot(index="class_name", columns="method", values=IDENT)
        ref, per = ident.loc[REFERENCE_LABEL], ident.drop(index=REFERENCE_LABEL)
        gap = per[DUET] - per[BASELINES].max(axis=1)
        row = {"panel": pid}
        for m in METHODS:
            row[f"whole_brain_{LABELS[m]}"] = ref[m]
            row[f"decode_{LABELS[m]}"] = decode[pid][m]
        row.update(duet_wins=int((gap > 0).sum()), n_classes=len(gap),
                   min_gap=gap.min(), min_gap_class=gap.idxmin(), median_gap=gap.median(),
                   whole_brain_gap=ref[DUET] - ref[BASELINES].max())
        rows.append(row)
    return pd.DataFrame(rows)


def across_panels(long):
    ident = long.pivot_table(index=["class_name", "panel"], columns="method", values=IDENT)
    for b in BASELINES:
        ident[f"gap_{LABELS[b]}"] = ident[DUET] - ident[b]
    ident["gap_best_baseline"] = ident[DUET] - ident[BASELINES].max(axis=1)
    g = ident.groupby(level="class_name")
    agg = pd.concat({"mean": g.mean(), "sd": g.std(ddof=1), "sem": g.sem(ddof=1)}, axis=1)
    agg.columns = [f"{col}_{stat}" for stat, col in agg.columns]
    agg["duet_wins"] = g["gap_best_baseline"].apply(lambda s: int((s > 0).sum()))
    agg["n_panels"] = g.size()
    return agg


def draw_points(ax, agg, order, ref, n_panels):
    x = np.arange(len(order))
    for m in METHODS:
        c = color(m)
        ax.errorbar(x + DODGE[m], agg.loc[order, f"{m}_mean"], yerr=agg.loc[order, f"{m}_sem"],
                    fmt="o", ms=MS, color=c, ecolor=c, elinewidth=0.6,
                    capsize=1.2, capthick=0.6, ls="none")
        ax.axhline(ref[m], color=c, ls="--", lw=0.7, alpha=0.6)
    finish_axes(ax, order, "Mean resolved fraction")
    handles = [Line2D([0], [0], marker="o", ls="none", ms=4, color=color(m), label=LABELS[m])
               for m in METHODS]
    handles.append(Line2D([0], [0], color="0.55", ls="--", lw=0.9,
                          label="Optimized on\nwhole-brain expression"))
    # Outside the axes: the 34 classes leave no empty corner. The constrained layout
    # shrinks the axes to fit it, so the figure stays at its FIGURE_WIDTHS width.
    ax.figure.legend(handles=handles, frameon=False, loc="outside right center", fontsize=5.5,
                     title=f"Mean ± SEM over\n{n_panels} gene panels", title_fontsize=5.5,
                     alignment="left")


def finish_axes(ax, order, ylabel):
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=XROT, ha=XHA, fontsize=XFS)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(XLABEL)
    ax.set_title(TITLE)


def save(fig, out, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    base = args.config.resolve().parent
    cfg = yaml.safe_load(args.config.read_text())
    out = (base / cfg["outdir"]).resolve()
    LABELS[DUET] = cfg["duet_label"]
    panels = [panel["id"] for panel in cfg["panels"]]

    long, decode = load(out, panels, cfg["lambda"])
    long.to_csv(out / "per_panel_class_metrics.csv", index=False)
    summary = panel_summary(long, decode)
    summary.to_csv(out / "panel_summary.csv", index=False)
    agg = across_panels(long)
    agg.to_csv(out / "class_across_panels.csv")

    per_agg = agg.drop(index=REFERENCE_LABEL)
    order = sorted(per_agg.index)
    ref = {m: agg.loc[REFERENCE_LABEL, f"{m}_mean"] for m in METHODS}

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    draw_points(ax, per_agg, order, ref, len(panels))
    save(fig, out, "crowding cell type robustness")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(4).to_string(index=False))
    wins, pairs = int(per_agg["duet_wins"].sum()), int(per_agg["n_panels"].sum())
    worst = per_agg["gap_best_baseline_mean"].idxmin()
    print(f"DUET beats both baselines in {wins}/{pairs} (panel, class) pairs; "
          f"classes won in every panel: {int((per_agg['duet_wins'] == per_agg['n_panels']).sum())}"
          f"/{len(per_agg)}")
    print(f"gap vs best baseline over classes: median of panel medians "
          f"{summary['median_gap'].median():+.4f}; smallest class mean {worst} "
          f"{per_agg.loc[worst, 'gap_best_baseline_mean']:+.4f} "
          f"(SD {per_agg.loc[worst, 'gap_best_baseline_sd']:.4f})")
    print(f"median SD / SEM over panels, per class: " + ", ".join(
        f"{LABELS[m]} {per_agg[f'{m}_sd'].median():.4f} / {per_agg[f'{m}_sem'].median():.4f}"
        for m in METHODS))
    print(f"[ok] wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
