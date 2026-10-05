"""
Comparison utilities for multi-variant benchmark analysis.

This module provides infrastructure for loading, merging, and re-evaluating
results from multiple benchmark runs with different DUET configurations.

The key use case is comparing DUET variants that use different noise channels,
distance metrics, or decoding rules against a unified ground truth evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from duet.evaluator_config import EvaluatorConfig
from duet.codebook_evaluator import CodebookEvaluator
from duet.candidate_pool import CandidatePool


# =============================================================================
# Helper Functions
# =============================================================================


def load_candidates_from_csv(csv_path: Path) -> CandidatePool:
    """Load CandidatePool from a CSV file.

    The CSV should have columns: Group, Sequence, Score, Quota.
    This is the format output by run_benchmark.py in each trial directory.
    Also supports legacy column names: Gene, Activity score, Num selections.

    Args:
        csv_path: Path to CSV file.

    Returns:
        CandidatePool instance.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return CandidatePool.from_dataframe(df)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class MethodSource:
    """Configuration for a single method source.

    Attributes:
        path: Path to results.csv file.
        regex: Regex pattern(s) to filter methods from the results file.
            Can be a single string or list of strings. All patterns are
            combined with OR logic. Examples:
            - "DUET.*" matches all DUET variants
            - ["Feldman et al.*", "Maximum activity"] matches all Feldman
              edit distances plus the Maximum activity baseline
        label: Display label for this method group in plots.
        color: Optional matplotlib-parseable color string (e.g. hex
            "#08306b"). When set, overrides the auto-assigned palette in
            visualize_comparison.generate_palette. Defaults to None.
    """

    path: Path
    regex: List[str]
    label: str
    color: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MethodSource":
        """Create MethodSource from dictionary."""
        regex = d["regex"]
        # Normalize to list for consistent handling
        if isinstance(regex, str):
            regex = [regex]
        return cls(
            path=Path(d["path"]),
            regex=regex,
            label=d["label"],
            color=d.get("color"),
        )


@dataclass
class ComparisonConfig:
    """Complete configuration for cross-benchmark comparison.

    Attributes:
        outdir: Output directory for comparison results and plots.
        cache_dir: Cache directory for evaluators.
        evaluator: Configuration for the unified ground truth evaluator.
        reference_dir: Path to a benchmark output directory containing trial_XX/guides.csv files.
            Used to load per-trial guide candidates.
        methods: List of method sources to compare.
        seed: Random seed for evaluator initialization (should match the seed used in run_benchmark.py).
        chemistry: Observation chemistry. One of 'dna', 'two_color', 'three_color', 'four_color'.
            Defaults to 'dna'.
        device: Computation device for evaluator cache initialization.
            One of 'cpu', 'gpu', 'gpu:all', 'gpu:0,1', etc. Defaults to 'cpu'.
        eval_mem_budget_gb: Memory budget in GB for batch evaluator matmul
            in reevaluate_with_unified_evaluator. Defaults to 20.0.
            Raise on machines with more RAM (e.g. 64.0) to get larger
            matmul batches.
    """

    outdir: Path
    cache_dir: Path | None
    evaluator: EvaluatorConfig
    reference_dir: Path
    methods: List[MethodSource]
    seed: int
    chemistry: str = "dna"
    device: str = "cpu"
    eval_mem_budget_gb: float = 20.0

    @classmethod
    def from_yaml(cls, path: Path) -> "ComparisonConfig":
        """Load ComparisonConfig from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            ComparisonConfig instance.
        """
        with open(path) as f:
            d = yaml.safe_load(f)

        return cls(
            outdir=Path(d["outdir"]),
            cache_dir=Path(d["cache_dir"]) if d.get("cache_dir") else None,
            evaluator=EvaluatorConfig.from_dict(d["evaluator"]),
            reference_dir=Path(d["reference_dir"]),
            methods=[MethodSource.from_dict(m) for m in d["methods"]],
            seed=d["seed"],
            chemistry=d.get("chemistry", "dna"),
            device=d.get("device", "cpu"),
            eval_mem_budget_gb=d.get("eval_mem_budget_gb", 20.0),
        )

    def get_trial_guide_csvs(self) -> List[Path]:
        """Get sorted list of guides.csv paths from reference_dir.

        Returns:
            List of paths to guides.csv files, sorted by trial number.

        Raises:
            FileNotFoundError: If reference_dir doesn't exist or has no trial directories.
        """
        if not self.reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {self.reference_dir}")

        # Glob for trial directories
        trial_dirs = sorted(self.reference_dir.glob("trial_*"))
        if not trial_dirs:
            raise FileNotFoundError(
                f"No trial directories (trial_*) found in {self.reference_dir}"
            )

        guide_csvs = []
        for trial_dir in trial_dirs:
            guide_csv = trial_dir / "guides.csv"
            if not guide_csv.exists():
                raise FileNotFoundError(f"guides.csv not found in {trial_dir}")
            guide_csvs.append(guide_csv)

        return guide_csvs

    @property
    def num_trials(self) -> int:
        """Number of trials based on reference_dir structure."""
        return len(self.get_trial_guide_csvs())


# =============================================================================
# Data Loading and Validation
# =============================================================================


def load_method_results(source: MethodSource) -> pd.DataFrame:
    """Load and filter results for a single method source.

    Args:
        source: MethodSource configuration.

    Returns:
        Filtered DataFrame with added 'Label' column.

    Raises:
        FileNotFoundError: If results file does not exist.
        ValueError: If no matching methods found in results.
    """
    import re

    if not source.path.exists():
        raise FileNotFoundError(f"Results file not found: {source.path}")

    df = pd.read_csv(source.path)

    # Build combined regex pattern (OR of all patterns)
    combined_pattern = "|".join(f"({p})" for p in source.regex)

    # Filter using regex matching
    mask = df["Method"].str.match(combined_pattern, na=False)
    filtered = df[mask].copy()

    if filtered.empty:
        available = df["Method"].unique().tolist()
        raise ValueError(
            f"No methods matching regex {source.regex} found in {source.path}. "
            f"Available methods: {available}"
        )

    # Add label column for grouping in plots
    filtered["Label"] = source.label

    return filtered


def validate_group_consistency(dataframes: List[pd.DataFrame], labels: List[str]) -> None:
    """Validate that all results use the same group set per trial.

    Compares the unique set of groups across all dataframes for each trial.

    Args:
        dataframes: List of DataFrames to compare.
        labels: List of labels corresponding to each DataFrame.

    Raises:
        ValueError: If group sets differ between any two sources.
    """
    if len(dataframes) < 2:
        return

    # Get all trials present
    all_trials = set()
    for df in dataframes:
        all_trials.update(df["Trial"].unique())

    # Determine column name (support both new and legacy)
    group_col = "Group" if "Group" in dataframes[0].columns else "Gene"

    for trial in sorted(all_trials):
        group_sets = []
        present_labels = []

        for df, label in zip(dataframes, labels):
            trial_df = df[df["Trial"] == trial]
            if trial_df.empty:
                continue
            col = "Group" if "Group" in df.columns else "Gene"
            groups = set(trial_df[col].unique())
            group_sets.append(groups)
            present_labels.append(label)

        if len(group_sets) < 2:
            continue

        # Compare all pairs
        reference_groups = group_sets[0]
        reference_label = present_labels[0]

        for groups, label in zip(group_sets[1:], present_labels[1:]):
            if groups != reference_groups:
                missing_in_ref = groups - reference_groups
                missing_in_other = reference_groups - groups
                raise ValueError(
                    f"Group set mismatch in Trial {trial} between '{reference_label}' "
                    f"and '{label}'.\n"
                    f"  Groups in '{label}' but not '{reference_label}': {missing_in_ref}\n"
                    f"  Groups in '{reference_label}' but not '{label}': {missing_in_other}"
                )


def load_and_merge_results(config: ComparisonConfig) -> pd.DataFrame:
    """Load and merge results from all method sources.

    Performs the following:
    1. Load results from each method source
    2. Filter to specified method names
    3. Validate gene consistency across sources
    4. Merge into single DataFrame

    Args:
        config: ComparisonConfig with method sources.

    Returns:
        Merged DataFrame with columns:
        - Original columns from results.csv
        - 'Label': Display label for the method
        - 'Source': Path to the source file
    """
    dataframes = []
    labels = []

    for source in config.methods:
        df = load_method_results(source)
        df["Source"] = str(source.path)
        dataframes.append(df)
        labels.append(source.label)

    # Validate group consistency
    validate_group_consistency(dataframes, labels)

    # Merge all results
    merged = pd.concat(dataframes, ignore_index=True)

    return merged


# =============================================================================
# Re-evaluation
# =============================================================================


def reevaluate_with_unified_evaluator(
    merged_df: pd.DataFrame,
    evaluator: CodebookEvaluator,
    scores: np.ndarray,
    mem_budget_gb: float = 20.0,
) -> pd.DataFrame:
    """Re-evaluate every (Label, Method, Trial) codebook with a single
    batched matmul on the shared union evaluator.

    Each benchmark run may have used a different per-source evaluator
    (different noise channel / distance / decoding rule). To compare fairly
    every row is re-scored under one ground-truth evaluator. The per-codeword
    accuracy and the three error-correction categories (No_error / Corrected /
    Failed) are all refreshed in the same matmul pass.

    Args:
        merged_df: Merged DataFrame from load_and_merge_results(). Expected
            columns include Label, Method, Trial, Index, Decode accuracy,
            and a score column ("Score" or "Activity score").
        evaluator: Initialized CodebookEvaluator built over the per-trial
            union of indices referenced by merged_df.
        scores: Array of candidate scores, indexed by evaluator-local
            (union-space) index.
        mem_budget_gb: Memory budget in GB for the batch matmul. Default 20.0.

    Returns:
        A new DataFrame with Decode accuracy, the score column, and
        No_error/Corrected/Failed refreshed from the unified evaluator.
        Input merged_df is not mutated. No_error/Corrected/Failed are
        created if absent in the input, overwritten otherwise.
    """
    score_col = "Score" if "Score" in merged_df.columns else "Activity score"

    groups = list(merged_df.groupby(["Label", "Method", "Trial"], sort=False))
    if not groups:
        out = merged_df.copy()
        for col in ("No_error", "Corrected", "Failed"):
            if col not in out.columns:
                out[col] = pd.Series(dtype=np.float64)
        return out

    codebooks = [g["Index"].values.astype(int) for (_, g) in groups]

    cw_acc_list, metrics_list = evaluator.batch_get_accuracy_with_error_metrics(
        codebooks, mem_budget_gb=mem_budget_gb,
    )

    updated_rows = []
    for i, (_, group) in enumerate(groups):
        g = group.copy()
        g["Decode accuracy"] = cw_acc_list[i]
        g[score_col]         = scores[codebooks[i]]
        g["No_error"]        = metrics_list[i].codeword_no_error
        g["Corrected"]       = metrics_list[i].codeword_corrected
        g["Failed"]          = metrics_list[i].codeword_failed
        updated_rows.append(g)

    return pd.concat(updated_rows, ignore_index=True)


# =============================================================================
# Aggregation Utilities
# =============================================================================


def compute_aggregated_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated metrics per method/trial.

    Args:
        results_df: Raw results DataFrame with per-candidate rows.

    Returns:
        Aggregated DataFrame with one row per (Label, Method, Trial).
    """
    # Determine score column name (support both new and legacy)
    score_col = "Activity score" if "Activity score" in results_df.columns else "Score"

    agg_df = (
        results_df.groupby(["Label", "Method", "Trial"])
        .agg(
            **{
                "Mean decode accuracy": ("Decode accuracy", "mean"),
                "Standard deviation decode accuracy": ("Decode accuracy", "std"),
                "Mean activity score": (score_col, "mean"),
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

    agg_df["95th over 5th percentile decode accuracy"] = (
        agg_df["95th percentile decode accuracy"]
        / agg_df["5th percentile decode accuracy"]
    )

    return agg_df