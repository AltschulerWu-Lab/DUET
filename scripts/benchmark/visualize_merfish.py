#!/usr/bin/env python3
"""MERFISH benchmark visualization from disk-persisted CSVs.

Reads ``results.csv`` and ``metrics.csv`` from the config's ``outdir``
(or from an explicit --input results.csv) and writes the MERFISH figure
set into ``<output_dir>/figures/``.

Crowding plots are emitted only when ``metrics.csv`` carries the
``mean_identified_fraction`` column.

By default only paper panels and diagnostics are written (see
``duet.plotting.tiers``): the crowding Pareto fronts and bar chart, the
decode-accuracy histograms, the Hamming-weight distribution and the mean and
5th-percentile decode-accuracy bar plots. ``--debug-plots`` adds the other
metric bar plots, the matched-λ jointplot, the identified-fraction-vs-count
check and the three per-λ sweeps (``figures/*_sweep/``). A debug plot listed
under ``visualization.paper_panels`` in the ``--config`` YAML (e.g. one λ of the
jointplot sweep, a paper inset) is written by default too.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from duet.merfish_benchmark import MerfishBenchmarkConfig
from duet.merfish_benchmark.visualization import (
    metric_barplot_stem,
    plot_crowding_bar_chart,
    plot_crowding_pareto_front,
    plot_crowding_pareto_front_lambda_labeled,
    plot_decode_accuracy_histogram,
    plot_decode_accuracy_histogram_by_hamming_weight,
    plot_decode_vs_identified_jointplot,
    plot_hamming_weight_distribution,
    plot_identified_fraction_vs_count,
    plot_metrics_barplots,
    summary_metric_columns,
    sweep_decode_accuracy_histogram,
    sweep_decode_accuracy_histogram_by_hamming_weight,
    sweep_decode_vs_identified_jointplot,
)
from duet.plotting import apply_style
from duet.plotting.tiers import PlotTiers, add_debug_plots_arg

# The metric bar plots written by default; the other summary stats are debug
# plots.
DIAGNOSTIC_METRICS = ("Mean decode accuracy", "5th percentile decode accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MERFISH benchmark figures from results.csv + metrics.csv.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=str,
        help="Path to results.csv produced by run_merfish.py "
             "(metrics.csv is expected alongside it).",
    )
    input_group.add_argument(
        "--config", type=str,
        help="Path to the MERFISH YAML config; reads CSVs from config's outdir.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=("Figure output directory (defaults to the config's outdir). "
              "Input CSVs are always read from the config's outdir; this "
              "controls only where figures land."),
    )
    add_debug_plots_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_config = None
    if args.config:
        config_path = Path(args.config)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        cfg = MerfishBenchmarkConfig.from_dict(raw_config, config_dir=config_path.parent)
        outdir = cfg.outdir
        results_csv = outdir / "results.csv"
        metrics_csv = outdir / "metrics.csv"
        output_dir = Path(args.output_dir) if args.output_dir else outdir
    else:
        results_csv = Path(args.input)
        if not args.output_dir:
            raise SystemExit("--output-dir is required when using --input")
        output_dir = Path(args.output_dir)
        metrics_csv = results_csv.parent / "metrics.csv"

    apply_style()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tiers = PlotTiers.from_config(output_dir, args.debug_plots, raw_config)

    results_df = pd.read_csv(results_csv)
    metrics_df = pd.read_csv(metrics_csv)

    all_metrics = summary_metric_columns(metrics_df)
    barplot_metrics = [
        m for m in all_metrics
        if m in DIAGNOSTIC_METRICS
        or tiers.writes_debug(figures_dir / f"{metric_barplot_stem(m)}.svg")
    ]
    skipped = len(all_metrics) - len(barplot_metrics)
    plot_metrics_barplots(metrics_df, figures_dir, metrics=barplot_metrics)
    plot_decode_accuracy_histogram(results_df, metrics_df, figures_dir)
    plot_hamming_weight_distribution(results_df, figures_dir)
    skipped += sweep_decode_accuracy_histogram(
        results_df, metrics_df, figures_dir, include=tiers.writes_debug
    )

    has_crowding = (
        "mean_identified_fraction" in metrics_df.columns
        and metrics_df["mean_identified_fraction"].notna().any()
    )
    if has_crowding:
        plot_crowding_pareto_front(metrics_df, figures_dir)
        plot_crowding_pareto_front_lambda_labeled(metrics_df, figures_dir)
        plot_crowding_bar_chart(metrics_df, figures_dir)
        for stem, plot in (
            ("decode_vs_identified_jointplot", plot_decode_vs_identified_jointplot),
            ("identified_fraction_vs_count", plot_identified_fraction_vs_count),
        ):
            if tiers.writes_debug(figures_dir / f"{stem}.svg"):
                plot(results_df, metrics_df, figures_dir)
            else:
                skipped += 1
        plot_decode_accuracy_histogram_by_hamming_weight(results_df, metrics_df, figures_dir)
        skipped += sweep_decode_vs_identified_jointplot(
            results_df, metrics_df, figures_dir, include=tiers.writes_debug
        )
        skipped += sweep_decode_accuracy_histogram_by_hamming_weight(
            results_df, metrics_df, figures_dir, include=tiers.writes_debug
        )
    else:
        print("Skipping crowding plots (no optical_crowding metrics in metrics.csv)")

    print(f"Figures written to {figures_dir}")
    if skipped:
        print(f"Skipped {skipped} debug plots; pass --debug-plots to write them")
    tiers.report_missing_paper_panels()


if __name__ == "__main__":
    main()
