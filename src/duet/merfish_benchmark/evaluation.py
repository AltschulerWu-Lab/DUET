# src/duet/merfish_benchmark/evaluation.py
"""MERFISH-specific evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from duet.optical_crowding import simulate_optical_crowding
from duet.merfish_benchmark.config import (
    CrowdingConfig,
    ExpressionConfig,
    FromCsvGenesConfig,
    GenesConfigT,
    RandomGenesConfig,
)


logger = logging.getLogger(__name__)


def load_expression(cfg: ExpressionConfig) -> pd.DataFrame:
    return pd.read_csv(cfg.path)


def select_genes(
    genes_cfg: GenesConfigT,
    full_expr_df: pd.DataFrame,
    expression_cfg: ExpressionConfig,
    codebook_size: int,
    rng: np.random.Generator,
) -> Tuple[List[str], np.ndarray]:
    if isinstance(genes_cfg, RandomGenesConfig):
        return _select_random(genes_cfg, full_expr_df, expression_cfg, codebook_size, rng)
    if isinstance(genes_cfg, FromCsvGenesConfig):
        return _select_from_csv(genes_cfg, full_expr_df, expression_cfg, codebook_size)
    raise TypeError(f"Unsupported genes_cfg type: {type(genes_cfg).__name__}")


def _select_random(cfg, full_expr_df, expression_cfg, codebook_size, rng):
    pool = full_expr_df[full_expr_df[expression_cfg.expression_col] > 0].copy()
    if cfg.exclude_controls:
        before = len(pool)
        pool = pool[~pool[expression_cfg.gene_col].str.startswith("ERCC-")]
        logger.info("Excluded %d ERCC controls (%d -> %d)", before - len(pool), before, len(pool))
    pool = pool.sort_values(expression_cfg.expression_col, ascending=False)
    if cfg.pool_filter == "top_k":
        pool = pool.head(cfg.pool_filter_value)
    elif cfg.pool_filter == "top_percentile":
        cutoff = int(len(pool) * cfg.pool_filter_value / 100.0)
        pool = pool.head(cutoff)
    else:
        raise ValueError(f"Unknown pool_filter: {cfg.pool_filter}")
    if len(pool) < codebook_size:
        raise ValueError(f"Gene pool has {len(pool)} genes after filtering but need {codebook_size}")
    idx = rng.choice(len(pool), size=codebook_size, replace=False)
    selected = pool.iloc[idx]
    return (
        selected[expression_cfg.gene_col].tolist(),
        selected[expression_cfg.expression_col].values.astype(np.float64),
    )


def _select_from_csv(cfg, full_expr_df, expression_cfg, codebook_size):
    csv_df = pd.read_csv(cfg.path)
    # gene_col validated at config load; row count checked below.
    gene_names = csv_df[cfg.gene_col].astype(str).tolist()
    if len(gene_names) != codebook_size:
        raise ValueError(
            f"genes.path CSV has {len(gene_names)} rows but codebook_size={codebook_size}"
        )
    return lookup_expression_values(gene_names, full_expr_df, expression_cfg, source_label=str(cfg.path))


def lookup_expression_values(gene_names, full_expr_df, expression_cfg, source_label):
    """Look up expression values for `gene_names`; missing → 0.0 with WARNING.

    Also reports, at INFO level, the count of panel genes that are *present* in
    the expression CSV but carry expression <= 0 (typically zero counts). These
    positions are crowding-neutral in CrowdingSwapCache (they contribute nothing
    to TEV) so the count is purely diagnostic, but it explains away most of the
    non-positive expression entries downstream consumers care about.
    """
    gene_to_expr = dict(zip(
        full_expr_df[expression_cfg.gene_col].astype(str),
        full_expr_df[expression_cfg.expression_col].astype(float),
    ))
    expression = np.array([gene_to_expr.get(g, 0.0) for g in gene_names], dtype=np.float64)
    n_missing = sum(1 for g in gene_names if g not in gene_to_expr)
    if n_missing:
        logger.warning(
            "%d / %d genes from %s not present in expression CSV (set to 0.0)",
            n_missing, len(gene_names), source_label,
        )
    n_present_zero = sum(
        1 for g in gene_names if g in gene_to_expr and not (gene_to_expr[g] > 0.0)
    )
    if n_present_zero:
        logger.info(
            "%d / %d genes from %s present in expression CSV but have "
            "expression <= 0 (crowding-neutral positions)",
            n_present_zero, len(gene_names), source_label,
        )
    return gene_names, expression


@dataclass
class CrowdingEvaluation:
    """Aggregate + per-codeword crowding metrics across trials.

    per_codeword_identified_fraction is mean over trials (np.nanmean), indexed
    positionally by the codebook (length == codebook_size). Genes that produce
    zero transcripts in a given trial contribute NaN for that trial so the
    final mean only reflects trials in which the gene actually appeared.

    per_codeword_count is the TOTAL number of simulated transcripts for each
    codeword summed across all trials (a simple sum, not a fraction), also
    indexed positionally. It is the realized sample size behind each
    per_codeword_identified_fraction value: a gene whose fraction is averaged
    over only 1-2 transcripts is noise, so downstream per-codeword plots filter
    on this count (e.g. require >= 25). It cannot be recovered from the fraction
    -- the fraction is a quality ratio, not an abundance.

    NOTE on semantics: mean_identified_fraction (aggregate) is
    transcript-weighted -- high-expression genes dominate because they
    contribute more transcripts. per_codeword_identified_fraction is
    gene-weighted -- each gene contributes equally regardless of expression.
    The two summaries disagree whenever expression varies across the panel,
    which is the common case. In the jointplot the raw scatter y-positions are
    these gene-weighted per_codeword_identified_fraction values, but the
    identified-fraction *marginal* is weighted by n_transcripts, so it is
    (approximately) transcript-weighted and does track the bar chart's
    mean_identified_fraction over the retained genes; the raw scatter/contour
    spread does not.
    """

    mean_conflict_fraction: float
    std_conflict_fraction: float
    mean_identified_fraction: float
    std_identified_fraction: float
    per_codeword_identified_fraction: np.ndarray
    per_codeword_count: np.ndarray


def evaluate_crowding_simulation(
    sequences: List[str],
    gene_names: List[str],
    full_expr_df: pd.DataFrame,
    sim_config: CrowdingConfig,
    expression_col: str,
    gene_col: str,
    seed: int,
) -> CrowdingEvaluation:
    """Evaluate crowding via spatial simulation over multiple trials."""
    codebook_df = pd.DataFrame({"Gene": gene_names, "Sequence": sequences})

    conflict_fracs = []
    identified_fracs = []
    # per_codeword[t, i] = identified fraction for codeword i in trial t (NaN if no transcripts)
    per_codeword = np.full((sim_config.n_trials, len(gene_names)), np.nan, dtype=np.float64)
    # per_codeword_count[i] = total transcripts for codeword i summed over all trials.
    per_codeword_count = np.zeros(len(gene_names), dtype=np.int64)
    gene_to_pos = {g: i for i, g in enumerate(gene_names)}

    for trial in range(sim_config.n_trials):
        result = simulate_optical_crowding(
            codebook_df=codebook_df,
            gene_expression_df=full_expr_df,
            expression_col=expression_col,
            gene_col=gene_col,
            total_reads=sim_config.total_reads,
            cell_size_um=sim_config.cell_size_um,
            wavelength_nm=sim_config.wavelength_nm,
            numerical_aperture=sim_config.numerical_aperture,
            expansion_factor=sim_config.expansion_factor,
            diffraction_model=sim_config.diffraction_model,
            neighborhood=sim_config.neighborhood,
            # placement stays multinomial (the default), intentionally not exposed on
            # CrowdingConfig: for a fixed real panel the cell distributes total_reads
            # stochastically over the transcriptome. "deterministic" placement is a
            # Fig-4 oracle-parity knob (round(reads*prop) per gene) used only by the
            # reproduction in optical_crowding.run_crowding_sweep, not a post-hoc model.
            seed=seed + trial,
        )
        conflict_fracs.append(result.conflict_fraction)
        identified_fracs.append(result.identified_fraction)

        # 1 - mean(conflicted) per gene, positionally aligned to gene_names; the
        # group size is the realized transcript count for that gene this trial.
        grouped = result.transcripts_df.groupby("gene")["conflicted"]
        per_gene = grouped.mean()
        per_gene_size = grouped.size()
        for gene, conflict_rate in per_gene.items():
            pos = gene_to_pos.get(gene)
            if pos is not None:
                per_codeword[trial, pos] = 1.0 - conflict_rate
                per_codeword_count[pos] += int(per_gene_size[gene])

    # All-NaN columns (genes that never produced a transcript across any trial)
    # would emit RuntimeWarning("Mean of empty slice") and yield NaN. Both are
    # the right downstream behavior — caller can dropna or skip those codewords.
    with np.errstate(invalid="ignore"):
        per_codeword_mean = np.nanmean(per_codeword, axis=0)

    return CrowdingEvaluation(
        mean_conflict_fraction=float(np.mean(conflict_fracs)),
        std_conflict_fraction=float(np.std(conflict_fracs)),
        mean_identified_fraction=float(np.mean(identified_fracs)),
        std_identified_fraction=float(np.std(identified_fracs)),
        per_codeword_identified_fraction=per_codeword_mean,
        per_codeword_count=per_codeword_count,
    )
