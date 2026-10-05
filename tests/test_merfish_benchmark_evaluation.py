"""Tests for select_genes + load_expression (post-2026-05-25 refactor)."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.merfish_benchmark.config import (
    ExpressionConfig,
    RandomGenesConfig,
    FromCsvGenesConfig,
)
from duet.merfish_benchmark.evaluation import load_expression, select_genes, lookup_expression_values


@pytest.fixture
def expr_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "gene_name": ["GeneA", "GeneB", "GeneC", "GeneD", "ERCC-001"],
        "mean_raw_counts": [100.0, 50.0, 10.0, 0.0, 30.0],
    })
    p = tmp_path / "expr.csv"
    df.to_csv(p, index=False)
    return p


def test_load_expression_returns_full_dataframe(expr_csv):
    cfg = ExpressionConfig(path=expr_csv)
    df = load_expression(cfg)
    assert len(df) == 5
    assert set(df.columns) == {"gene_name", "mean_raw_counts"}


def test_select_genes_random_top_percentile(expr_csv):
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = RandomGenesConfig(pool_filter="top_percentile", pool_filter_value=50, exclude_controls=True)
    names, expr = select_genes(genes_cfg, full, expr_cfg, codebook_size=1, rng=np.random.default_rng(42))
    assert names == ["GeneA"]
    assert expr.tolist() == [100.0]


def test_select_genes_random_top_k(expr_csv):
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = RandomGenesConfig(pool_filter="top_k", pool_filter_value=10, exclude_controls=True)
    names, expr = select_genes(genes_cfg, full, expr_cfg, codebook_size=3, rng=np.random.default_rng(42))
    assert set(names) == {"GeneA", "GeneB", "GeneC"}
    assert all(e > 0 for e in expr)


def test_select_genes_random_keeps_controls_when_disabled(expr_csv):
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = RandomGenesConfig(pool_filter="top_k", pool_filter_value=10, exclude_controls=False)
    names, _ = select_genes(genes_cfg, full, expr_cfg, codebook_size=4, rng=np.random.default_rng(42))
    assert set(names) == {"GeneA", "GeneB", "GeneC", "ERCC-001"}


def test_select_genes_random_pool_too_small(expr_csv):
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = RandomGenesConfig(pool_filter="top_k", pool_filter_value=2, exclude_controls=True)
    with pytest.raises(ValueError, match="pool"):
        select_genes(genes_cfg, full, expr_cfg, codebook_size=3, rng=np.random.default_rng(42))


def test_select_genes_from_csv_assigns_expression(expr_csv, tmp_path):
    cb_csv = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["GeneA", "GeneB"], "Sequence": ["0000", "1111"]}).to_csv(cb_csv, index=False)
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = FromCsvGenesConfig(path=cb_csv)
    names, expr = select_genes(genes_cfg, full, expr_cfg, codebook_size=2, rng=np.random.default_rng(42))
    assert names == ["GeneA", "GeneB"]
    assert expr.tolist() == [100.0, 50.0]


def test_select_genes_from_csv_size_mismatch_errors(expr_csv, tmp_path):
    cb_csv = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["GeneA", "GeneB"], "Sequence": ["0000", "1111"]}).to_csv(cb_csv, index=False)
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = FromCsvGenesConfig(path=cb_csv)
    with pytest.raises(ValueError, match="rows but codebook_size"):
        select_genes(genes_cfg, full, expr_cfg, codebook_size=3, rng=np.random.default_rng(42))


def test_select_genes_from_csv_missing_gene_warns_and_zeros(expr_csv, tmp_path, caplog):
    cb_csv = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["GeneA", "NotInExpr"], "Sequence": ["0000", "1111"]}).to_csv(cb_csv, index=False)
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = FromCsvGenesConfig(path=cb_csv)
    with caplog.at_level(logging.WARNING):
        names, expr = select_genes(genes_cfg, full, expr_cfg, codebook_size=2, rng=np.random.default_rng(42))
    assert names == ["GeneA", "NotInExpr"]
    assert expr.tolist() == [100.0, 0.0]
    assert any("not present" in r.message for r in caplog.records)


def test_select_genes_from_csv_custom_gene_col(expr_csv, tmp_path):
    cb_csv = tmp_path / "cb.csv"
    pd.DataFrame({"name": ["GeneA", "GeneB"], "Sequence": ["0000", "1111"]}).to_csv(cb_csv, index=False)
    expr_cfg = ExpressionConfig(path=expr_csv)
    full = load_expression(expr_cfg)
    genes_cfg = FromCsvGenesConfig(path=cb_csv, gene_col="name")
    names, _ = select_genes(genes_cfg, full, expr_cfg, codebook_size=2, rng=np.random.default_rng(42))
    assert names == ["GeneA", "GeneB"]


# ---------------------------------------------------------------------------
# Tests for CrowdingEvaluation + evaluate_crowding_simulation per-codeword
# ---------------------------------------------------------------------------

from unittest.mock import patch

from duet.merfish_benchmark.config import CrowdingConfig
from duet.merfish_benchmark.evaluation import (
    CrowdingEvaluation,
    evaluate_crowding_simulation,
)
from duet.optical_crowding import CrowdingResult


def _make_crowding_result(
    gene_to_conflict_count: dict[str, tuple[int, int]],
) -> CrowdingResult:
    """Build a CrowdingResult whose transcripts_df has the given (n_conflicted, n_total) per gene."""
    rows = []
    for gene, (n_conflicted, n_total) in gene_to_conflict_count.items():
        for i in range(n_total):
            rows.append({"gene": gene, "x": 0.0, "y": 0.0, "conflicted": i < n_conflicted})
    transcripts = pd.DataFrame(rows)
    n_total_all = len(transcripts)
    n_conflicted_all = int(transcripts["conflicted"].sum())
    return CrowdingResult(
        conflict_fraction=n_conflicted_all / n_total_all if n_total_all else 0.0,
        identified_fraction=1.0 - (n_conflicted_all / n_total_all) if n_total_all else 1.0,
        n_transcripts=n_total_all,
        n_conflicted=n_conflicted_all,
        transcripts_df=transcripts,
        barcode_length=16,
        hamming_weight=4,
    )


def test_evaluate_crowding_simulation_returns_per_codeword_fraction():
    """Per-codeword identified fraction is groupby('gene') over conflicted, averaged across trials."""
    sequences = ["0001", "0010", "0100"]
    gene_names = ["GeneA", "GeneB", "GeneC"]
    full_expr_df = pd.DataFrame({"gene_name": gene_names, "mean_raw_counts": [10.0, 10.0, 10.0]})
    sim_config = CrowdingConfig(
        n_trials=2,
        total_reads=100,
        cell_size_um=100.0,
        wavelength_nm=500.0,
        numerical_aperture=1.4,
        expansion_factor=1.0,
    )

    # Trial 0: GeneA 2/10 conflicted, GeneB 5/10, GeneC 0/10
    # Trial 1: GeneA 4/10 conflicted, GeneB 5/10, GeneC 10/10
    # Expected per-codeword identified fraction (mean across trials):
    #   GeneA = mean(1-0.2, 1-0.4) = 0.7
    #   GeneB = mean(1-0.5, 1-0.5) = 0.5
    #   GeneC = mean(1-0.0, 1-1.0) = 0.5
    trial_results = [
        _make_crowding_result({"GeneA": (2, 10), "GeneB": (5, 10), "GeneC": (0, 10)}),
        _make_crowding_result({"GeneA": (4, 10), "GeneB": (5, 10), "GeneC": (10, 10)}),
    ]

    with patch(
        "duet.merfish_benchmark.evaluation.simulate_optical_crowding",
        side_effect=trial_results,
    ):
        out = evaluate_crowding_simulation(
            sequences=sequences,
            gene_names=gene_names,
            full_expr_df=full_expr_df,
            sim_config=sim_config,
            expression_col="mean_raw_counts",
            gene_col="gene_name",
            seed=42,
        )

    assert isinstance(out, CrowdingEvaluation)
    assert out.per_codeword_identified_fraction.shape == (3,)
    np.testing.assert_allclose(out.per_codeword_identified_fraction, [0.7, 0.5, 0.5])
    # Realized transcript count is a simple sum over trials (10 per gene per trial,
    # 2 trials each) -- NOT derivable from the fraction above.
    np.testing.assert_array_equal(out.per_codeword_count, [20, 20, 20])


def test_evaluate_crowding_simulation_handles_missing_gene_with_nan():
    """A gene that has zero transcripts in a trial uses np.nanmean across trials."""
    sequences = ["0001", "0010"]
    gene_names = ["GeneA", "GeneB"]
    full_expr_df = pd.DataFrame({"gene_name": gene_names, "mean_raw_counts": [10.0, 10.0]})
    sim_config = CrowdingConfig(
        n_trials=2, total_reads=100, cell_size_um=100.0,
        wavelength_nm=500.0, numerical_aperture=1.4, expansion_factor=1.0,
    )

    # Trial 0: only GeneA appears (10 transcripts, 1 conflicted) -> GeneA=0.9
    # Trial 1: both appear; GeneA 0/10 conflicted -> 1.0; GeneB 5/10 -> 0.5
    # Expected: GeneA = nanmean(0.9, 1.0) = 0.95; GeneB = nanmean(nan, 0.5) = 0.5
    trial_results = [
        _make_crowding_result({"GeneA": (1, 10)}),
        _make_crowding_result({"GeneA": (0, 10), "GeneB": (5, 10)}),
    ]

    with patch(
        "duet.merfish_benchmark.evaluation.simulate_optical_crowding",
        side_effect=trial_results,
    ):
        out = evaluate_crowding_simulation(
            sequences=sequences, gene_names=gene_names, full_expr_df=full_expr_df,
            sim_config=sim_config, expression_col="mean_raw_counts",
            gene_col="gene_name", seed=42,
        )

    np.testing.assert_allclose(out.per_codeword_identified_fraction, [0.95, 0.5])
    # GeneA appears in both trials (10+10), GeneB only in trial 1 (10); the count
    # sums realized transcripts and is independent of the NaN-aware fraction mean.
    np.testing.assert_array_equal(out.per_codeword_count, [20, 10])
