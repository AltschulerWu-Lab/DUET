# src/duet/ops_benchmark/runner.py
"""OPS benchmark runner: multi-trial orchestration over DUET + baselines."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from duet.benchmark.metrics import evaluate_solutions
from duet.candidate_pool_factory import create_pool_from_source
from duet.dual_guide_factory import get_alphabet_sizes
from duet.runner.ops import run_duet_ops
from duet.initialization import InitializationStrategy
from duet.ops_benchmark.baselines import (
    encode_for_sivanandan,
    run_sivanandan_for_eds,
    run_feldman_for_eds,
)
from duet.ops_benchmark.config import OpsBenchmarkConfig
from duet.ops_benchmark.init import parse_init, SivanandanWarmStart, FeldmanWarmStart
from duet.utils import Stopwatch


def run_ops_benchmark(
    config: OpsBenchmarkConfig,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Run the full OPS benchmark and write outputs.

    Args:
        config: Parsed OpsBenchmarkConfig.
        config_path: Original YAML path (copied to outdir/config.yaml if given).

    Returns:
        results_df: Per-codebook-position results across trials and methods.
    """
    config.outdir.mkdir(parents=True, exist_ok=True)

    seed = config.seed
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
        print(f"Using random seed: {seed}")

    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31 - 1, size=config.trials).tolist()

    chemistry = config.candidate_pool.chemistry
    transmitted_alphabet_size, _observed = get_alphabet_sizes(chemistry)

    print(f"Output directory: {config.outdir}")
    print(f"Cache directory: {config.cache_dir}")
    print(f"Running methods: {', '.join(config.methods)}")
    print(f"Number of trials: {config.trials}")
    print(f"Chemistry: {chemistry}")

    init_strategy = parse_init(config.duet_initialization_raw) if config.run_duet else None

    results_df = pd.DataFrame()
    for t, trial_seed in enumerate(trial_seeds, start=1):
        print(f"\n{'=' * 60}\nTrial {t}/{config.trials} (seed={trial_seed})\n{'=' * 60}")
        trial_dir = config.outdir / f"trial_{t:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_df = _run_single_trial(
            config=config,
            trial_index=t,
            trial_seed=trial_seed,
            chemistry=chemistry,
            transmitted_alphabet_size=transmitted_alphabet_size,
            init_strategy=init_strategy,
            trial_dir=trial_dir,
        )
        results_df = pd.concat([results_df, trial_df], ignore_index=True)

    print(f"\n{'=' * 60}\nFinished all trials. Saving results...\n{'=' * 60}")
    results_df.to_csv(config.outdir / "results.csv", index=False)
    print(f"Results saved to {config.outdir / 'results.csv'}")

    if config_path is not None:
        dest = config.outdir / "config.yaml"
        try:
            shutil.copy2(config_path, dest)
            print(f"Configuration saved to {dest}")
        except shutil.SameFileError:
            pass

    print("\n=== Summary Statistics ===")
    summary = (
        results_df.loc[results_df["Valid"]].groupby("Method").agg({
            "Decode accuracy": ["mean", "std"],
            "Activity score": ["mean", "std"],
            "No_error": ["mean", "std"],
            "Corrected": ["mean", "std"],
            "Failed": ["mean", "std"],
        })
    )
    print(summary)
    return results_df


def _run_single_trial(
    config: OpsBenchmarkConfig,
    trial_index: int,
    trial_seed: int,
    chemistry: str,
    transmitted_alphabet_size: int,
    init_strategy: InitializationStrategy | None,
    trial_dir: Path,
) -> pd.DataFrame:
    """Execute one trial: build candidates, run warm-start + DUET + baselines, evaluate."""
    print("\nLoading candidate pool...")
    cp_cfg = config.candidate_pool
    candidates = create_pool_from_source(
        source=cp_cfg.source,
        seq_rounds=cp_cfg.seq_rounds,
        quota=cp_cfg.quota,
        num_controls=cp_cfg.num_controls,
        min_rank=cp_cfg.min_rank,
        num_groups=cp_cfg.num_groups,
        seed=trial_seed,
        pairing_strategy=cp_cfg.pairing_strategy,
        max_control_pairs=cp_cfg.max_control_pairs,
        csv_path=cp_cfg.csv_path,
        control_quotas=cp_cfg.control_quotas,
        score_method=cp_cfg.score_method,
        candidates_per_group=cp_cfg.candidates_per_group,
        alphabet_size=cp_cfg.alphabet_size,
    )
    print(f"  Pool size: {candidates.pool_size}")
    print(f"  Number of groups: {candidates.num_groups}")
    print(f"  Sequence length: {candidates.seq_length}")
    print(f"  Total selections: {candidates.total_selections}")
    candidates.to_dataframe().to_csv(trial_dir / "guides.csv", index=False)

    baseline_results: Dict[str, Dict[int, np.ndarray]] = {}

    # Sivanandan encoding pre-computed once, reused by warm-start + baseline runs.
    encoded_for_sivanandan = None
    if config.run_sivanandan:
        encoded_for_sivanandan = encode_for_sivanandan(candidates, chemistry)

    # Warm-start source runs FIRST if DUET requires it.
    needs_warm_start = isinstance(init_strategy, (SivanandanWarmStart, FeldmanWarmStart))
    if needs_warm_start:
        if isinstance(init_strategy, SivanandanWarmStart):
            if not config.run_sivanandan:
                raise ValueError("DUET initialization.type='sivanandan' requires sivanandan section.")
            print(f"\nRunning Sivanandan baseline for warm start (ED={init_strategy.edit_distance})...")
            baseline_results["sivanandan"] = run_sivanandan_for_eds(
                candidates, config.sivanandan, encoded_for_sivanandan,
            )
        else:  # FeldmanWarmStart
            if not config.run_feldman:
                raise ValueError("DUET initialization.type='feldman' requires feldman section.")
            if chemistry != "dna":
                raise ValueError(
                    f"DUET initialization.type='feldman' requires DNA chemistry, got '{chemistry}'."
                )
            print(f"\nRunning Feldman baseline for warm start (ED={init_strategy.edit_distance})...")
            feldman_results = run_feldman_for_eds(
                candidates, config.feldman, config.candidate_pool.seq_rounds,
                config.candidate_pool.quota, config.candidate_pool.num_controls,
            )
            if not feldman_results:
                raise ValueError("Feldman warm start requested but Feldman returned no results (no edit distances).")
            baseline_results["feldman"] = feldman_results

    # DUET.
    if config.run_duet:
        print("\nInitializing DUET...")
        init, meta = init_strategy.get_initial_indices(
            candidates, trial_seed, baseline_results=baseline_results,
        )
        print(f"  Initialization: {meta.description}" + (f" (+{meta.num_swaps} swaps)" if meta.num_swaps else ""))

        duet_cache_dir = config.duet_cache_dir or config.cache_dir
        pep_config = config.duet_pep

        with Stopwatch("DUET"):
            duet_results = run_duet_ops(
                candidates=candidates,
                pep_config=pep_config,
                optimizer_config=config.duet_optimizer,
                init=init,
                alphabet_size=transmitted_alphabet_size,
                seed=trial_seed,
                cache_dir=duet_cache_dir,
                use_mmap=config.duet_use_mmap,
                device=config.duet_device,
                force_rebuild=config.duet_force_rebuild,
                sym_mem_budget_gb=config.duet_sym_mem_budget_gb,
            )

    # Sivanandan (if not already run for warm start).
    if config.run_sivanandan and "sivanandan" not in baseline_results:
        baseline_results["sivanandan"] = run_sivanandan_for_eds(
            candidates, config.sivanandan, encoded_for_sivanandan,
        )

    # Feldman (if not already run for warm start).
    if config.run_feldman and "feldman" not in baseline_results:
        if chemistry != "dna":
            print(f"WARNING: Feldman requires DNA chemistry, got '{chemistry}'. Skipping.")
        else:
            baseline_results["feldman"] = run_feldman_for_eds(
                candidates, config.feldman, config.candidate_pool.seq_rounds,
                config.candidate_pool.quota, config.candidate_pool.num_controls,
            )

    # Max-activity baseline.
    max_score_idx = candidates.sample_initial_selection(strategy="best_score")

    solutions = _build_solutions_dict(
        duet_results=duet_results if config.run_duet else None,
        baseline_results=baseline_results,
        max_score_idx=max_score_idx,
    )

    with Stopwatch("Evaluate all solutions"):
        batch = evaluate_solutions(
            solutions, candidates, config.evaluator,
            transmitted_alphabet_size,
            n_jobs=config.evaluator.num_cpus,
            device=config.eval_device,
            mem_budget_gb=config.eval_mem_budget_gb,
        )
    return _assemble_results_df(batch, solutions, candidates, trial_index)


def _build_solutions_dict(
    *,
    duet_results: Dict | None,
    baseline_results: Dict[str, Dict[int, np.ndarray]],
    max_score_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Collect all solution indices keyed by human-readable method label."""
    solutions: Dict[str, np.ndarray] = {}
    if duet_results is not None:
        for i, lambda_ in enumerate(duet_results["lambdas"]):
            solutions[f"DUET (lambda={lambda_:.2f})"] = duet_results["best_indices"][i]
    if "sivanandan" in baseline_results:
        for ed, idx in baseline_results["sivanandan"].items():
            solutions[f"Sivanandan et al. (ED={ed})"] = idx
    if "feldman" in baseline_results:
        for ed, idx in baseline_results["feldman"].items():
            solutions[f"Feldman et al. (ED={ed})"] = idx
    solutions["Maximum activity"] = max_score_idx
    return solutions


def _is_valid_codebook(indices, candidates) -> bool:
    """Structural validity of a produced codebook.

    A codebook is valid iff:
      1. it has exactly ``total_selections`` codewords (cardinality);
      2. no candidate index is selected more than once (distinct indices);
      3. each group is filled to exactly its quota (per-group quota).

    Duplicate *sequences* within a group are allowed: two distinct candidates
    that share a (possibly truncated) codeword both decode to the same gene, so
    reusing a barcode within a group is not a degeneracy. The failure the old
    cardinality-only check missed — e.g. the lambda=0.0 duplicate-stacking case
    that fills a group's quota with copies of one codeword — is the same
    candidate index selected twice, which rule 2 rejects directly.

    Per-group counts are read from a flat candidate->group map inverted from
    ``group_to_candidates``. Every pool the OPS pipeline feeds here assigns each
    candidate to exactly one group — a ``from_dataframe`` library pool (one
    candidate per row) and a directly constructed ``synthetic`` pool
    (partitioned indices) both do — so that inversion is lossless. The map is
    last-writer-wins and would mis-attribute a candidate belonging to several
    groups, but no OPS source produces such multi-group membership (it would
    take a directly constructed pool that deliberately shares an index across
    groups). An out-of-pool index maps to no group and is rejected (membership /
    crash-safety).
    """
    indices = [int(c) for c in indices]
    if len(indices) != candidates.total_selections:
        return False  # cardinality
    if len(set(indices)) != len(indices):
        return False  # distinct candidate indices
    candidate_to_group = {
        c: g for g, cs in candidates.group_to_candidates.items() for c in cs
    }
    counts: Dict[str, int] = {}
    for c in indices:
        group = candidate_to_group.get(c)
        if group is None:
            return False  # membership: index not in any group
        counts[group] = counts.get(group, 0) + 1
    for group, quota in candidates.quotas.items():
        if counts.get(group, 0) != quota:
            return False  # per-group quota
    return True


def _assemble_results_df(
    batch: Dict[str, tuple],
    solutions: Dict[str, np.ndarray],
    candidates,
    trial_index: int,
) -> pd.DataFrame:
    """Turn batch-evaluator output into a flat per-codeword DataFrame."""
    candidate_to_group = {
        c: g
        for g, cs in candidates.group_to_candidates.items()
        for c in cs
    }
    rows: List[pd.DataFrame] = []
    for method_name, (decode_acc, score, err) in batch.items():
        indices = solutions[method_name]
        # Reconstruct each codeword's group from candidate membership. An index
        # outside the pool maps to "<unknown>" (Group and Sequence) so it flags
        # the codebook invalid below instead of crashing the whole trial here.
        groups = [candidate_to_group.get(int(i), "<unknown>") for i in indices]
        sequences = [
            candidates.sequences[int(i)] if g != "<unknown>" else "<unknown>"
            for i, g in zip(indices, groups)
        ]
        rows.append(pd.DataFrame({
            "Method": method_name,
            "Index": indices,
            "Group": groups,
            "Sequence": sequences,
            "Decode accuracy": decode_acc,
            "Activity score": score,
            "No_error": err.codeword_no_error,
            "Corrected": err.codeword_corrected,
            "Failed": err.codeword_failed,
            "Trial": trial_index,
            "Valid": _is_valid_codebook(indices, candidates),
        }))
    return pd.concat(rows, ignore_index=True)
