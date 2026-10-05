#!/usr/bin/env python
"""Fig 4d,e: per-round expression-load and expression-by-Hamming-weight figures
for the 2000-gene MERFISH run.

Trimmed copy of scripts/benchmark/archive/06-28-2026/make_figures.py:
  * k2000 only (the k1000 and zhang_2023 panels are not in the paper);
  * DUET at lambda = 0.90, the as-run override of 0.10 mirrored to the current
    convention (docs/adr/0001-lambda-weights-decoding-accuracy.md). Fig 4d and
    4e both use this codebook;
  * reads this experiment's outputs: the run directory and the expression table
    come from the experiment config, resolved exactly as the benchmark resolves
    them, instead of from hard-coded results/benchmark/06-28-2026 paths.
The plotting calls and output filenames are unchanged, so the shipped names
(round_expression_uniformity_y0_k2000.svg = Fig 4d,
expr_by_hw_violin_k2000.svg = Fig 4e) carry over. Only those two are written by
default; --debug-plots adds the three alternates (the autoscaled and sorted
load panels and the box-plot version of 4e).

Run with an env whose `duet` resolves to this repo:
    python make_figures.py --config config.yaml [--outdir DIR] [--lambda 0.90] [--debug-plots]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from duet.merfish_benchmark.config import MerfishBenchmarkConfig
from duet.merfish_benchmark.visualization import (
    codebook_from_results,
    compute_round_expression_load,
    plot_expression_by_hamming_weight,
    plot_round_expression_load,
    select_best_lambda,
)
from duet.plotting import apply_style, save_panel
from duet.plotting.tiers import add_debug_plots_arg

LABEL = "k2000"
DEFAULT_LAMBDA = 0.90  # as-run LAMBDA_OVERRIDE {"k2000": 0.10}, old convention


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="experiment YAML (its outdir holds results.csv and metrics.csv)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output dir (default: <config outdir>/figures/fig4de)")
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA,
                   help=f"DUET lambda to plot (default {DEFAULT_LAMBDA})")
    add_debug_plots_arg(p)
    args = p.parse_args(argv)

    cfg = MerfishBenchmarkConfig.from_yaml(args.config)
    results_dir = Path(cfg.outdir)
    outdir = args.outdir if args.outdir is not None else results_dir / "figures" / "fig4de"

    results = pd.read_csv(results_dir / "results.csv", dtype={"Sequence": str})
    metrics = pd.read_csv(results_dir / "metrics.csv")
    expr_df = pd.read_csv(cfg.expression.path)
    expr_map = dict(zip(expr_df[cfg.expression.gene_col],
                        expr_df[cfg.expression.expression_col]))

    apply_style()
    auto = select_best_lambda(metrics)
    lam = args.lam
    print(f"[{LABEL}] chosen lambda={lam:.2f} (auto={auto:.2f})")

    duet_method = f"DUET (lambda={lam:.2f})"
    baselines = metrics.loc[metrics["lambda"].isna(), "Method"].tolist()
    methods = baselines + [duet_method]  # DUET last -> drawn on top

    loads_by_method = {
        m: compute_round_expression_load(codebook_from_results(results, m), expr_map)
        for m in methods
    }

    outdir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    # (sort, y_from_zero, stem, paper panel). The "uniformity_y0" panel (Fig 4d)
    # is a companion to "uniformity": the same per-round loads on an absolute
    # (0-based) y-axis, so between-method offsets read against the true zero
    # rather than the autoscaled floor. The other two are debug plots.
    for sort, y_from_zero, stem, paper in [
        (False, False, "uniformity", False),
        (False, True, "uniformity_y0", True),
        (True, False, "sorted", False),
    ]:
        if not (paper or args.debug_plots):
            skipped += 1
            continue
        fig = plot_round_expression_load(loads_by_method, sort=sort, y_from_zero=y_from_zero)
        save_panel(
            fig,
            outdir / f"round_expression_{stem}_{LABEL}.svg",
            outdir / f"round_expression_{stem}_{LABEL}.png",
            dpi=150,
        )
        written += 1

    duet_cb = codebook_from_results(results, duet_method)
    # The violin is Fig 4e; the box plot is a debug alternate.
    for kind, paper in (("violin", True), ("box", False)):
        if not (paper or args.debug_plots):
            skipped += 1
            continue
        fig = plot_expression_by_hamming_weight(duet_cb, expr_map, kind=kind)
        save_panel(
            fig,
            outdir / f"expr_by_hw_{kind}_{LABEL}.svg",
            outdir / f"expr_by_hw_{kind}_{LABEL}.png",
            dpi=150,
        )
        written += 1
    print(f"[{LABEL}] wrote {written} figures to {outdir}")
    if skipped:
        print(f"[{LABEL}] skipped {skipped} debug plots; pass --debug-plots to write them")


if __name__ == "__main__":
    main()
