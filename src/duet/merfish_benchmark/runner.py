# src/duet/merfish_benchmark/runner.py
"""MERFISH benchmark runner: orchestrates DUET + Chen baselines + crowding sim."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

from duet.runner.merfish import run_duet_merfish
from duet.initialization import InitializationStrategy
from duet.merfish_benchmark.baselines import load_baseline_codebook
from duet.merfish_benchmark.config import (
    FromCsvGenesConfig,
    MerfishBenchmarkConfig,
)
from duet.merfish_benchmark.evaluation import (
    lookup_expression_values,
    evaluate_crowding_simulation,
    load_expression,
    select_genes,
    CrowdingEvaluation,
)
from duet.merfish_benchmark.init import parse_init
from duet.benchmark.metrics import compute_metrics, evaluate_codebooks_by_sequence, CodebookEvalResult
from duet.candidate_pool import CandidatePool
from duet.merfish_factory import MERFISHFactory
from duet.utils import Stopwatch
from duet.visualization import generate_pep_debug_visualizations


logger = logging.getLogger(__name__)


@dataclass
class SolutionEvaluation:
    """One evaluated codebook (DUET lambda or baseline) for output assembly."""
    method_label: str
    sequences: list
    eval: CodebookEvalResult                 # accuracy + error metrics from the union pass
    lambda_: float | None = None             # DUET only
    genes: list | None = None                # gene label per codeword position
    valid: bool = True                       # DUET: cardinality check; baseline: True
    best_indices: list | None = None         # DUET pool indices (for selected_codewords csv)
    crowding: "CrowdingEvaluation | None" = None


_CROWDING_FLOAT_FIELDS = (
    "mean_conflict_fraction", "std_conflict_fraction",
    "mean_identified_fraction", "std_identified_fraction",
)


def _spread_crowding(row: dict, ce: CrowdingEvaluation) -> None:
    """Mutate `row` in place with the aggregate float fields of `ce`."""
    for k in _CROWDING_FLOAT_FIELDS:
        row[k] = getattr(ce, k)


@dataclass
class MerfishCandidates:
    """The candidate pool and the panel inputs built from a MERFISH config.

    Returned by :func:`build_candidates` (steps 1-5 of
    :func:`run_merfish_benchmark`). The pool's sequence order is what the PEP
    cache fingerprint hashes, so code that must predict a run's PEP cache hit
    (for example experiments/merfish_expression_prior/check_pep_cache.py)
    builds the pool through this function rather than a copy of it.
    """
    candidates: CandidatePool
    baseline_dfs: list[Tuple[str, pd.DataFrame, str, str]]  # (name, df, gene_col, seq_col)
    baseline_seqs_by_label: dict[str, set[str]]
    full_expr_df: pd.DataFrame | None
    gene_names: list[str] | None
    expression: np.ndarray | None


def build_candidates(config: MerfishBenchmarkConfig, seed: int) -> MerfishCandidates:
    """Read the baselines, warm start and genes, and build the candidate pool.

    Steps 1-5 of :func:`run_merfish_benchmark`: baseline and warm-start CSVs,
    gene assignment, the warm-start/gene alignment check, positional
    required_codewords, MERFISHFactory, and the pool-membership assertion.
    Writes nothing.
    """
    # === Step 1: Read baseline + warm-start CSVs. ===
    # DataFrames are kept (not just sequence sets) so positional
    # required_codewords below can do gene->codeword lookups against
    # gene_names.
    print("\n" + "=" * 60 + "\nGenerating candidate codewords...\n" + "=" * 60)

    baseline_dfs: list[Tuple[str, pd.DataFrame, str, str]] = []  # (name, df, gene_col, seq_col)
    baseline_seqs_by_label: dict[str, set[str]] = {}
    for bl in config.baselines:
        df, seqs, _genes = load_baseline_codebook(bl)
        baseline_dfs.append((bl.name, df, bl.gene_col, bl.sequence_col))
        baseline_seqs_by_label[bl.name] = set(seqs)
        print(f"Baseline {bl.name}: {len(baseline_seqs_by_label[bl.name])} sequences will be required in pool")

    warm_seqs: list[str] | None = None
    warm_init_df: pd.DataFrame | None = None
    if config.initialization.type == "warm_start":
        warm_init_df = pd.read_csv(
            config.initialization.path,
            dtype={config.initialization.sequence_col: str},
        )
        warm_seqs = warm_init_df[config.initialization.sequence_col].tolist()
        if len(warm_seqs) != config.candidates.codebook_size:
            raise ValueError(
                f"Warm-start CSV {config.initialization.path} has {len(warm_seqs)} rows; "
                f"expected codebook_size={config.candidates.codebook_size}"
            )
        print(f"Warm-start: {len(warm_seqs)} sequences (row-positional) will anchor pos_i's samples")

    # === Step 2: Gene assignment. ===
    # Moved from its prior position (later in the function) to here, so the
    # positional required_codewords block below can translate baseline
    # (gene -> codeword) mappings into (position -> codeword) anchors before
    # the factory is constructed.
    full_expr_df = None
    gene_names: list[str] | None = None
    expression = None
    if config.expression is not None and config.genes is not None:
        print("\n" + "=" * 60 + "\nLoading expression data\n" + "=" * 60)
        full_expr_df = load_expression(config.expression)
        rng_genes = np.random.default_rng(seed)
        gene_names, expression = select_genes(
            genes_cfg=config.genes,
            full_expr_df=full_expr_df,
            expression_cfg=config.expression,
            codebook_size=config.candidates.codebook_size,
            rng=rng_genes,
        )
        print(f"Assigned {len(gene_names)} genes; expression range [{expression.min():.1f}, {expression.max():.1f}]")

    # === Step 3: Warm-start <-> gene_names alignment check (defense-in-depth). ===
    # The positional required_codewords construction below uses two
    # indexing schemes that must agree:
    #   - warm_seqs[i] anchors pos_i  (CSV row index)
    #   - gene_names[i] -> baseline_codeword anchors pos_i  (gene-name lookup)
    # When the warm-start CSV happens to carry the same gene column as
    # config.genes, we can verify alignment row-by-row. If the CSV doesn't
    # carry that column, we accept the row-positional contract (the existing
    # CodebookWarmStart implementation has always relied on row order).
    if (
        warm_init_df is not None
        and gene_names is not None
        and isinstance(config.genes, FromCsvGenesConfig)
        and config.genes.gene_col in warm_init_df.columns
    ):
        warm_genes = warm_init_df[config.genes.gene_col].astype(str).tolist()
        if warm_genes != gene_names:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(warm_genes, gene_names)) if a != b),
                min(len(warm_genes), len(gene_names)),
            )
            raise ValueError(
                f"Warm-start CSV gene order does not match gene_names "
                f"(first divergence at index {first_diff}: "
                f"warm-start={warm_genes[first_diff]!r}, genes={gene_names[first_diff]!r}). "
                f"Misalignment would silently mis-anchor every position. "
                f"Reorder the warm-start CSV or the genes CSV to match."
            )

    # === Step 4: Build positional required_codewords. ===
    # Warm-start contributes by CSV row index (row i -> pos_i).
    # Baselines contribute by gene lookup (gene_names[i] -> baseline codeword).
    required_codewords: list[set[str]] = [set() for _ in range(config.candidates.codebook_size)]
    if warm_seqs is not None:
        for i, seq in enumerate(warm_seqs):
            required_codewords[i].add(seq)
    if gene_names is not None:
        for bl_name, df, gene_col, seq_col in baseline_dfs:
            gene_to_seq = dict(zip(df[gene_col].astype(str), df[seq_col]))
            covered = 0
            for i, gene in enumerate(gene_names):
                if gene in gene_to_seq:
                    required_codewords[i].add(gene_to_seq[gene])
                    covered += 1
            print(f"Baseline {bl_name}: anchored {covered} of {len(gene_names)} positions")
    elif baseline_dfs:
        # No gene_names available (config has no genes/expression block).
        # Baselines cannot be anchored positionally without a position->gene
        # mapping. Behavior split by sampling mode:
        #   - per_codeword_sample_size set: refuse early. Positional anchoring
        #     is structurally required; silently dropping baselines would
        #     produce a misleading pool-membership assertion failure later.
        #   - per_codeword_sample_size unset: dump all baseline seqs into
        #     pos 0. The HW-cap path consumes _required_flat (union across
        #     positions), so positional placement doesn't matter — only
        #     pool membership does. This preserves today's behavior.
        if config.candidates.per_codeword_sample_size is not None:
            raise ValueError(
                "per_codeword_sample_size cannot be used with baselines when "
                "config.genes / config.expression is not set: positional "
                "anchoring requires a gene -> position mapping. Configure "
                "config.genes or drop the baselines block."
            )
        for bl_name, df, gene_col, seq_col in baseline_dfs:
            for seq in df[seq_col]:
                required_codewords[0].add(seq)
            print(f"Baseline {bl_name}: {len(baseline_seqs_by_label[bl_name])} sequences pooled at pos_0 for HW-cap preservation")

    # `subsample_seed: 0` is a valid seed and must NOT silently fall back
    # to the top-level seed. Use `is None` instead of `or` to preserve 0.
    effective_subsample_seed = (
        config.candidates.subsample_seed
        if config.candidates.subsample_seed is not None
        else seed
    )
    effective_sample_seed = (
        config.candidates.sample_seed
        if config.candidates.sample_seed is not None
        else seed
    )

    factory = MERFISHFactory(
        seq_rounds=config.candidates.seq_rounds,
        codebook_size=config.candidates.codebook_size,
        hamming_weights=config.candidates.hamming_weights,
        hamming_weight_limits=config.candidates.hamming_weight_limits,
        subsample_seed=effective_subsample_seed,
        required_codewords=required_codewords,
        per_codeword_sample_size=config.candidates.per_codeword_sample_size,
        sample_seed=effective_sample_seed,
    )
    candidates = factory.create()
    print(f"Generated {candidates.pool_size} candidates; selecting {candidates.total_selections}")

    # === Step 5: Relaxed pool-membership assertion. ===
    # Original assertion checked every baseline sequence was in the pool.
    # Under positional required_codewords, only baseline rows whose gene is
    # in gene_names get anchored — rows for "extra" genes aren't used
    # downstream (baseline rows with extra genes are unused), so we relax
    # to "every anchored sequence is in the pool".
    candidate_set = set(
        candidates.sequences.tolist() if isinstance(candidates.sequences, np.ndarray)
        else candidates.sequences
    )
    if gene_names is not None:
        for bl_name, df, gene_col, seq_col in baseline_dfs:
            gene_to_seq = dict(zip(df[gene_col].astype(str), df[seq_col]))
            anchored_seqs = {gene_to_seq[g] for g in gene_names if g in gene_to_seq}
            missing = anchored_seqs - candidate_set
            assert not missing, (
                f"{bl_name}: factory failed to preserve anchored baseline sequences "
                f"(missing {len(missing)})"
            )
            print(f"{bl_name}: all {len(anchored_seqs)} anchored sequences found in candidate set")
    else:
        # No gene_names: keep the old "all baseline seqs in pool" check as
        # a safety net. The Step 4 block above ensured per_codeword_sample_size
        # is None in this branch, and dumped all baseline seqs into pos_0,
        # so the factory's _required_flat covers them via HW-cap preservation.
        for label, baseline_seqs in baseline_seqs_by_label.items():
            missing = baseline_seqs - candidate_set
            assert not missing, (
                f"{label}: factory failed to preserve required codewords "
                f"(missing {len(missing)})"
            )
            print(f"{label}: all {len(baseline_seqs)} sequences found in candidate set")

    return MerfishCandidates(
        candidates=candidates,
        baseline_dfs=baseline_dfs,
        baseline_seqs_by_label=baseline_seqs_by_label,
        full_expr_df=full_expr_df,
        gene_names=gene_names,
        expression=expression,
    )


def run_merfish_benchmark(
    config: MerfishBenchmarkConfig,
    args: argparse.Namespace,
) -> None:
    """Run the MERFISH benchmark and write all outputs.

    args carries CLI-only state (debug_pep, config path, etc.). YAML-level
    overrides (init-source/noise/seed/outdir) must already be applied to
    the config by the thin CLI wrapper before calling this function.
    """
    config.outdir.mkdir(parents=True, exist_ok=True)

    seed = config.seed
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
        print(f"Using random seed: {seed}")

    print(f"Output directory: {config.outdir}")
    print(f"Sequence rounds: {config.candidates.seq_rounds}")
    print(f"Codebook size: {config.candidates.codebook_size}")
    print(f"Hamming weights: {config.candidates.hamming_weights or 'all'}")

    prepared = build_candidates(config, seed)
    candidates = prepared.candidates
    candidates.to_dataframe().to_csv(config.outdir / "candidates.csv", index=False)
    baseline_dfs = prepared.baseline_dfs
    baseline_seqs_by_label = prepared.baseline_seqs_by_label
    full_expr_df = prepared.full_expr_df
    gene_names = prepared.gene_names
    expression = prepared.expression
    codewords_array = None  # populated lazily later if expression/genes absent

    alphabet_size = 2  # MERFISH uses binary

    # === Step 6: Initialization ===
    init_strategy: InitializationStrategy = parse_init(config.initialization)
    init, init_meta = init_strategy.get_initial_indices(candidates, seed)
    if init_meta.num_swaps > 0:
        print(f"\nInitialization: {init_meta.description} ({init_meta.num_swaps} swaps)")
    else:
        print(f"\nInitialization: {init_meta.description}")

    # Keep init_sequences for the union eval diagnostic pass.
    init_sequences = [candidates.sequences[i] for i in init]

    # === Step 7: DUET — single call, lambda sweep inside ===
    print("\n" + "=" * 60 + "\nRunning DUET (lambda sweep inside run_duet_merfish)\n" + "=" * 60)
    duet_cache_dir = config.duet_cache_dir or config.cache_dir
    pep_config = config.duet_pep  # required by config; no evaluator fallback (2026-06-21 schema)

    # Build codewords + expression even when expression/genes absent --
    # run_duet_merfish requires them. Provide placeholders when absent.
    # Expression uses ones (not zeros) so the CrowdingSwapCache satisfies its
    # Candidate-C preconditions (e_i > 0 for every selected position, which
    # together with HW(c_i) > 0 implies C(S_0) > 0). Without a `crowding`
    # block, MerfishBenchmarkConfig restricts the sweep to lambda = 1, where
    # the crowding weight (1 - lambda) is zero and the crowding deltas are
    # multiplied by zero in the optimizer, so any positive placeholder is
    # equivalent. (A lambda < 1 with no expression/crowding config is rejected
    # at config validation -- the placeholder would yield nonsensical
    # "crowding" on uniform expression.)
    if codewords_array is None:
        codewords_array = candidates.get_sequences_as_array(alphabet_size=2).astype(np.float64)
    if expression is None:
        expression = np.ones(config.candidates.codebook_size, dtype=np.float64)

    with Stopwatch("DUET (MERFISH)"):
        duet_results = run_duet_merfish(
            candidates=candidates,
            pep_config=pep_config,
            optimizer_config=config.duet_optimizer,
            init=init,
            codewords=codewords_array,
            expression=expression,
            alphabet_size=alphabet_size,
            seed=seed,
            cache_dir=duet_cache_dir,
            use_mmap=config.duet_use_mmap,
            device=config.duet_device,
            force_rebuild=config.duet_force_rebuild,
            sym_mem_budget_gb=config.duet_sym_mem_budget_gb,
        )

    # === Steps 8+9: Ground-truth evaluation of ALL codebooks via ONE union pass ===
    print("\n" + "=" * 60 + "\nEvaluating all codebooks (union)\n" + "=" * 60)

    # DUET solutions as sequence lists (keep best_idx for validity + selected_codewords csv).
    duet_specs = []  # (lambda_, best_idx_list, sel_seqs)
    for lambda_, best_idx in zip(duet_results["lambdas"], duet_results["best_indices"]):
        if isinstance(candidates.sequences, np.ndarray):
            sel_seqs = candidates.sequences[best_idx].tolist()
        else:
            sel_seqs = [candidates.sequences[i] for i in best_idx]
        duet_specs.append((lambda_, list(best_idx), sel_seqs))

    # Baseline solutions as sequence lists, sourced from the already-loaded baseline_dfs.
    baseline_specs = []  # (name, seqs, genes)
    for bl_name, df, gene_col, seq_col in baseline_dfs:
        baseline_specs.append((bl_name, df[seq_col].tolist(), df[gene_col].astype(str).tolist()))

    # Positional codebook list: index 0 = init diagnostic, then DUET, then baselines.
    codebooks = (
        [init_sequences]
        + [seqs for _, _, seqs in duet_specs]
        + [seqs for _, seqs, _ in baseline_specs]
    )
    with Stopwatch("Ground-truth union evaluation"):
        eval_results = evaluate_codebooks_by_sequence(
            codebooks,
            eval_config=config.evaluator,
            alphabet_size=alphabet_size,
            n_jobs=config.evaluator.num_cpus,
            device=config.eval_device,
            mem_budget_gb=config.eval_mem_budget_gb,
        )

    init_eval = eval_results[0]
    initial_accuracy = float(init_eval.mean_accuracy)
    print(f"Initial decode accuracy: {initial_accuracy:.4f}")

    n_duet = len(duet_specs)
    duet_res = eval_results[1:1 + n_duet]
    baseline_res = eval_results[1 + n_duet:]

    solution_evals: List[SolutionEvaluation] = []
    for (lambda_, best_idx, sel_seqs), res in zip(duet_specs, duet_res):
        solution_evals.append(SolutionEvaluation(
            method_label=f"DUET (lambda={lambda_:.2f})",
            sequences=sel_seqs, eval=res, lambda_=lambda_,
            genes=gene_names, valid=(len(best_idx) == config.candidates.codebook_size),
            best_indices=best_idx,
        ))
        print(f"  DUET (lambda={lambda_:.2f}): true decode={res.mean_accuracy:.4f}")
    for (bl_name, seqs, genes), res in zip(baseline_specs, baseline_res):
        solution_evals.append(SolutionEvaluation(
            method_label=bl_name, sequences=seqs, eval=res, lambda_=None,
            genes=genes, valid=True, best_indices=None,
        ))
        print(f"  baseline {bl_name}: mean accuracy={res.mean_accuracy:.4f}")

    # === Step 9b: PEP debug visualizations (optional CLI flag) ===
    if args.debug_pep:
        print("\n" + "=" * 60 + "\nGenerating PEP debug visualizations\n" + "=" * 60)
        debug_dir = config.outdir / "figures" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        pep_matrix = duet_results["pep_count_matrix"]
        n_samples_eff = duet_results["n_samples"]
        first_best_indices = np.array(duet_results["best_indices"][0])
        _baseline_results_legacy = {
            rec.method_label: (
                rec.sequences, rec.genes, rec.eval.codeword_accuracy,
                float(rec.eval.mean_accuracy), rec.eval.error_metrics,
            )
            for rec in solution_evals if rec.lambda_ is None
        }
        pep_summary_df = generate_pep_debug_visualizations(
            candidates=candidates,
            evaluator_config=config.evaluator,
            duet_pep_matrix=pep_matrix,
            best_indices=first_best_indices,
            baseline_results=_baseline_results_legacy,
            output_dir=debug_dir,
            alphabet_size=alphabet_size,
            n_samples=n_samples_eff,
        )
        print("\nPEP Sum Summary:")
        print(pep_summary_df.to_string(index=False))
        pep_summary_df.to_csv(debug_dir / "pep_summary.csv", index=False)

    # === Step 10: Crowding simulation (if configured) ===
    if config.crowding is not None:
        print("\n" + "=" * 60 + "\nEvaluating optical crowding\n" + "=" * 60)
        baseline_uses_own_genes = isinstance(config.genes, FromCsvGenesConfig)
        for rec in solution_evals:
            print(f"  Simulating crowding for {rec.method_label}...")
            if rec.lambda_ is not None:                  # DUET
                sim_gene_names = gene_names
            elif baseline_uses_own_genes:                # baseline w/ own genes
                sim_gene_names, _ = lookup_expression_values(
                    rec.genes, full_expr_df, config.expression,
                    source_label=f"baseline {rec.method_label}",
                )
            else:
                sim_gene_names = gene_names
            rec.crowding = evaluate_crowding_simulation(
                sequences=rec.sequences, gene_names=sim_gene_names,
                full_expr_df=full_expr_df, sim_config=config.crowding,
                expression_col=config.expression.expression_col,
                gene_col=config.expression.gene_col, seed=seed,
            )

    # === Step 11: results.csv ===
    print("\n" + "=" * 60 + "\nBuilding results.csv\n" + "=" * 60)
    results_rows = []
    for rec in solution_evals:
        cw_id_frac = rec.crowding.per_codeword_identified_fraction if rec.crowding is not None else None
        cw_count = rec.crowding.per_codeword_count if rec.crowding is not None else None
        em = rec.eval.error_metrics
        for i, seq in enumerate(rec.sequences):
            gene = rec.genes[i] if rec.genes is not None else "MERFISH"
            results_rows.append({
                "Method": rec.method_label, "Index": i, "Gene": gene, "Sequence": seq,
                "Decode accuracy": float(rec.eval.codeword_accuracy[i]),
                "Identified fraction": float(cw_id_frac[i]) if cw_id_frac is not None else float("nan"),
                # Realized total transcripts behind 'Identified fraction' (sum over trials);
                # the per-codeword sample size, used to filter noisy low-count genes.
                "n_transcripts": int(cw_count[i]) if cw_count is not None else float("nan"),
                "No_error": float(em.codeword_no_error[i]),
                "Corrected": float(em.codeword_corrected[i]),
                "Failed": float(em.codeword_failed[i]),
                "Trial": 1, "Valid": rec.valid,
            })
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(config.outdir / "results.csv", index=False)
    print(f"Saved results.csv with {len(results_df)} rows")

    # === Step 12: metrics.csv ===
    print("\n" + "=" * 60 + "\nComputing metrics\n" + "=" * 60)
    metrics_rows = []
    for rec in solution_evals:
        row = {"Method": rec.method_label, **compute_metrics(rec.eval.codeword_accuracy)}
        # SE of the mean decode accuracy = std over codewords / sqrt(#codewords).
        # Tiny (~1e-4) vs the identified-fraction SE, but persisted so the Pareto
        # front can draw both axes' error bars from one source.
        n_codewords = len(rec.eval.codeword_accuracy)
        row["se_decode_accuracy"] = (
            row["Std decode accuracy"] / np.sqrt(n_codewords) if n_codewords > 0 else float("nan")
        )
        if rec.lambda_ is not None:
            row["lambda"] = rec.lambda_
        if rec.crowding is not None:
            _spread_crowding(row, rec.crowding)
            # SE of the mean identified fraction = std over trials / sqrt(#trials).
            row["se_identified_fraction"] = (
                rec.crowding.std_identified_fraction / np.sqrt(config.crowding.n_trials)
                if config.crowding.n_trials > 0 else float("nan")
            )
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(config.outdir / "metrics.csv", index=False)
    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))

    # === Step 13: selected codewords + summary ===
    for rec in solution_evals:
        if rec.lambda_ is None:
            continue
        tag = f"lambda{rec.lambda_:.2f}"
        sel_df = {"Index": rec.best_indices, "Sequence": rec.sequences}
        if gene_names is not None:
            sel_df["Gene"] = gene_names
        pd.DataFrame(sel_df).to_csv(config.outdir / f"selected_codewords_{tag}.csv", index=False)

    # Decode-only arm = the largest lambda in the sweep (lambda weights decode).
    duet_evals = [rec for rec in solution_evals if rec.lambda_ is not None]
    decode_only_true_acc = max(duet_evals, key=lambda rec: rec.lambda_).eval.mean_accuracy
    summary = {
        "seed": seed,
        "seq_rounds": config.candidates.seq_rounds,
        "codebook_size": config.candidates.codebook_size,
        "hamming_weights": config.candidates.hamming_weights,
        "num_candidates": candidates.pool_size,
        "initial_accuracy": initial_accuracy,
        "lambdas": list(duet_results["lambdas"]),
        "true_accuracy_last_lambda": float(decode_only_true_acc),
        "baselines_evaluated": [rec.method_label for rec in solution_evals if rec.lambda_ is None],
        "init_description": init_meta.description,
        "init_num_swaps": init_meta.num_swaps,
    }
    if config.crowding is not None:
        summary["n_crowding_trials"] = config.crowding.n_trials
    with open(config.outdir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    for lambda_, history in zip(duet_results["lambdas"], duet_results["histories"]):
        if history:
            tag = f"lambda{lambda_:.2f}"
            pd.DataFrame(history).to_csv(config.outdir / f"optimization_history_{tag}.csv", index=False)

    try:
        shutil.copy2(args.config, config.outdir / "config.yaml")
    except (shutil.SameFileError, AttributeError):
        pass

    print(f"\nResults saved to {config.outdir}")
