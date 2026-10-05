#!/usr/bin/env python3
"""
Compare multiple benchmark runs with unified evaluation (CSV outputs only).

This script compares DUET configurations (and baselines) from multiple
run_benchmark.py outputs using a unified ground truth evaluator. All methods
are re-evaluated with the same evaluator configuration for fair comparison.

Outputs `reevaluated_results.csv` and `aggregated_metrics.csv` into
`config.outdir`. For Pareto-front and hypervolume plots from those CSVs, run
`visualize_comparison.py --config <same config>` after this script finishes.

Use case: Show that DUET with optimal noise channel / distance metric / decoding
rule dominates DUET variants with suboptimal components and baseline methods.

Usage:
    python compare_benchmarks.py   --config /path/to/compare_config.yaml
    python visualize_comparison.py --config /path/to/compare_config.yaml  # plots

Example config (the keys; experiments/ops_crispri_cross_eval/eval_position_varying_asymmetric.yaml
is a complete one, and scripts/benchmark/README.md describes each block):
    outdir: /path/to/comparison_output
    cache_dir: /path/to/cache
    seed: 42  # Must match seed from run_benchmark.py for cache hits
    reference_dir: /path/to/positional  # Contains trial_XX/guides.csv
    chemistry: dna  # Optional, defaults to 'dna'. Options: dna, two_color, three_color, four_color
    eval_mem_budget_gb: 20.0  # Optional, defaults to 20.0. Memory budget (GB) for
                              # the batched cross-eval matmul. Raise on big-memory
                              # machines (e.g. 64.0) for larger matmul batches.

    evaluator:
      noise_channel:
        type: position_varying
        epsilon: [0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2]
      decoding_metric:
        type: position_varying_nll
        epsilon: [0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2]
      decoding_rule:
        type: unique_minimum
      num_samples: 5000
      num_cpus: 20

    methods:
      - path: /path/to/positional/results.csv
        regex: "DUET.*"
        label: "DUET (Position-varying NLL)"

      - path: /path/to/positional/results.csv
        regex:
          - "Feldman et al.*"
          - "Maximum activity"
        label: "Feldman et al."

      - path: /path/to/uniform_duet/results.csv
        regex: "DUET.*"
        label: "DUET (Symmetric + Hamming)"
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparison import (
    ComparisonConfig,
    load_and_merge_results,
    load_candidates_from_csv,
    reevaluate_with_unified_evaluator,
    compute_aggregated_metrics,
)
from duet.dual_guide_factory import get_alphabet_sizes
from duet.evaluator_config import create_evaluator
from duet.utils import Stopwatch, log_memory


# =============================================================================
# Helpers
# =============================================================================


def _compute_se(x: pd.Series) -> float:
    """Compute standard error, returning 0 for single observations."""
    n = len(x)
    if n <= 1:
        return 0.0
    return x.std() / np.sqrt(n)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple benchmark runs with unified evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to comparison YAML config file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # Configure logging
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
    # Load configuration
    config = ComparisonConfig.from_yaml(Path(args.config))
    print(f"Output directory: {config.outdir}")
    print(f"Cache directory: {config.cache_dir}")
    print(f"Reference directory: {config.reference_dir}")
    print(f"Number of method sources: {len(config.methods)}")

    # Get alphabet sizes from chemistry
    transmitted_alphabet_size, observed_alphabet_size = get_alphabet_sizes(config.chemistry)
    print(f"Chemistry: {config.chemistry}")
    print(f"Transmitted alphabet size: {transmitted_alphabet_size}")

    # Create output directory
    config.outdir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load and merge results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading and merging results...")
    print("=" * 60)

    for source in config.methods:
        print(f"  {source.label}: {source.path}")
        print(f"    Regex filter: {source.regex}")

    merged_df = load_and_merge_results(config)
    merged_df = merged_df.loc[merged_df["Valid"]].reset_index(drop=True)
    print(f"\nMerged {len(merged_df)} rows from {len(config.methods)} sources")
    print(f"Labels: {sorted(merged_df['Label'].unique())}")
    print(f"Trials: {sorted(merged_df['Trial'].unique())}")

    # =========================================================================
    # Step 2: Load per-trial guide candidates and build union evaluators
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading guide candidates and building evaluators...")
    print("=" * 60)

    guide_csvs = config.get_trial_guide_csvs()
    num_trials = len(guide_csvs)
    print(f"Found {num_trials} trials in {config.reference_dir}")

    # Generate per-trial seeds (matching run_benchmark.py logic)
    rng = np.random.default_rng(config.seed)
    trial_seeds = rng.integers(0, 2**31 - 1, size=num_trials).tolist()

    # =========================================================================
    # Step 3: Re-evaluate all methods per trial (union-based)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Re-evaluating all methods with unified evaluator...")
    print("=" * 60)

    all_reevaluated = []

    for trial_idx, (guide_csv, trial_seed) in enumerate(zip(guide_csvs, trial_seeds), start=1):
        print(f"\nTrial {trial_idx}:")
        print(f"  Guide CSV: {guide_csv}")
        print(f"  Seed: {trial_seed}")

        # Load candidates for this trial (full pool, for sequence/score lookups)
        with Stopwatch("  Load candidates"):
            candidates = load_candidates_from_csv(guide_csv)
            print(f"    Library size: {candidates.pool_size}")
            print(f"    Sequence length: {candidates.seq_length}")

        # Filter merged_df to this trial
        trial_df = merged_df[merged_df["Trial"] == trial_idx].copy()

        if trial_df.empty:
            print(f"  WARNING: No data found for trial {trial_idx}")
            continue

        # Compute union of all selected indices for this trial
        all_pool_indices = set()
        for _, row_group in trial_df.groupby(["Label", "Method"]):
            all_pool_indices.update(row_group["Index"].values.astype(int))
        union_pool_indices = sorted(all_pool_indices)
        print(f"    Union size: {len(union_pool_indices)} (of {candidates.pool_size} total)")

        # Build pool-to-union mapping
        pool_to_union = {pool_idx: union_idx
                         for union_idx, pool_idx in enumerate(union_pool_indices)}

        # Remap indices in trial_df from pool space to union space
        remapped_df = trial_df.copy()
        remapped_df["Index"] = remapped_df["Index"].map(pool_to_union)

        # Build union-indexed scores
        union_scores = candidates.scores[union_pool_indices]

        # Build evaluator over union sequences only (bypass EvaluatorProvider for GPU support)
        union_sequences = [candidates.sequences[i] for i in union_pool_indices]
        with Stopwatch("  Initialize evaluator"):
            log_memory("before union evaluator creation")
            evaluator = create_evaluator(
                union_sequences, config.evaluator, alphabet_size=transmitted_alphabet_size
            )
            evaluator.initialize_cache(
                n_jobs=config.evaluator.num_cpus, device=config.device
            )
            log_memory("after union evaluator initialization")

        with Stopwatch("  Re-evaluate"):
            reevaluated = reevaluate_with_unified_evaluator(
                remapped_df,
                evaluator,
                union_scores,
                mem_budget_gb=config.eval_mem_budget_gb,
            )
            all_reevaluated.append(reevaluated)

    # Combine all trials
    results_df = pd.concat(all_reevaluated, ignore_index=True)

    # =========================================================================
    # Step 4: Compute aggregated metrics
    # =========================================================================
    print("\n" + "=" * 60)
    print("Computing aggregated metrics...")
    print("=" * 60)

    agg_df = compute_aggregated_metrics(results_df)

    # Save results
    results_df.to_csv(config.outdir / "reevaluated_results.csv", index=False)
    agg_df.to_csv(config.outdir / "aggregated_metrics.csv", index=False)
    print(f"Saved reevaluated_results.csv and aggregated_metrics.csv")

    # =========================================================================
    # Step 5: Print summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    summary = (
        agg_df.groupby("Label")
        .agg(
            **{
                "Mean decode accuracy": ("Mean decode accuracy", "mean"),
                "Mean decode accuracy SE": ("Mean decode accuracy", _compute_se),
                "Mean activity score": ("Mean activity score", "mean"),
                "Mean activity score SE": ("Mean activity score", _compute_se),
            }
        )
        .round(4)
    )
    print(summary)

    # Copy config to output
    config_dest = config.outdir / "compare_config.yaml"
    shutil.copy2(args.config, config_dest)
    print(f"\nConfiguration saved to {config_dest}")
    print(f"All outputs saved to {config.outdir}")


if __name__ == "__main__":
    main()