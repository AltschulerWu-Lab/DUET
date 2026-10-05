"""Tests for load_baseline_codebook."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import pandas as pd
import pytest

from duet.merfish_benchmark.baselines import load_baseline_codebook
from duet.merfish_benchmark.config import BaselineConfig


def _make_baseline_csv(path: Path, gene_col="Gene", seq_col="Sequence") -> None:
    # Leading-zero codeword "0011" must survive as a string, not become int 11.
    pd.DataFrame({gene_col: ["G1", "G2"], seq_col: ["0011", "1100"]}).to_csv(path, index=False)


def test_load_returns_df_sequences_genes(tmp_path):
    cb = tmp_path / "cb.csv"
    _make_baseline_csv(cb)
    bl = BaselineConfig(name="chen", path=cb)
    df, sequences, genes = load_baseline_codebook(bl)
    assert sequences == ["0011", "1100"]      # str dtype preserved (no int corruption)
    assert genes == ["G1", "G2"]
    assert list(df.columns) == ["Gene", "Sequence"]


def test_load_custom_columns(tmp_path):
    cb = tmp_path / "cb.csv"
    _make_baseline_csv(cb, gene_col="name", seq_col="barcode")
    bl = BaselineConfig(name="bil", path=cb, gene_col="name", sequence_col="barcode")
    _, sequences, genes = load_baseline_codebook(bl)
    assert sequences == ["0011", "1100"]
    assert genes == ["G1", "G2"]


def test_load_missing_sequence_col_raises(tmp_path):
    cb = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["G1"], "BAD": ["0011"]}).to_csv(cb, index=False)
    bl = BaselineConfig(name="bad", path=cb)
    with pytest.raises(ValueError, match="Sequence"):
        load_baseline_codebook(bl)


def test_load_missing_gene_col_raises(tmp_path):
    cb = tmp_path / "cb.csv"
    pd.DataFrame({"BAD": ["G1"], "Sequence": ["0011"]}).to_csv(cb, index=False)
    bl = BaselineConfig(name="bad", path=cb)
    with pytest.raises(ValueError, match="gene"):
        load_baseline_codebook(bl)
