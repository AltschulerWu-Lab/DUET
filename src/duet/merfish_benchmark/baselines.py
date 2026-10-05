# src/duet/merfish_benchmark/baselines.py
"""Baseline codebook loading."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from duet.merfish_benchmark.config import BaselineConfig


def load_baseline_codebook(
    bl_cfg: BaselineConfig,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Read a baseline codebook CSV and return (df, sequences, genes).

    The sequence column is read with dtype=str so leading-zero binary codewords
    are preserved (otherwise "0011" would parse to int 11). The returned df is
    reused by the runner for positional gene-anchoring; sequences/genes feed the
    union evaluator.
    """
    df = pd.read_csv(bl_cfg.path, dtype={bl_cfg.sequence_col: str})
    if bl_cfg.sequence_col not in df.columns:
        raise ValueError(
            f"Baseline {bl_cfg.name}: CSV {bl_cfg.path} missing sequence column {bl_cfg.sequence_col!r}"
        )
    if bl_cfg.gene_col not in df.columns:
        raise ValueError(
            f"Baseline {bl_cfg.name}: CSV {bl_cfg.path} missing gene column {bl_cfg.gene_col!r}"
        )
    sequences = df[bl_cfg.sequence_col].tolist()
    genes = df[bl_cfg.gene_col].astype(str).tolist()
    return df, sequences, genes


