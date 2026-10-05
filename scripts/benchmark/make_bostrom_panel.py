#!/usr/bin/env python
"""Preprocess a Boström codebook into a {Gene, Sequence} warm-start panel CSV.

Genes are sorted by descending expression in `--expression` and paired with the
codebook's codewords in file order, so the highest-expression gene gets the
first codeword. The output CSV is referenced by BOTH the `genes:` (from_csv) and
`initialization:` (warm_start) blocks of a MERFISH benchmark YAML. The
`--expression` file MUST be identical to the run's `expression:` file, or panel
genes silently get 0.0 expression and the crowding objective is wrong.

By default the panel has a third column, named after `--expression-col`, with
each gene's value from `--expression`. The MERFISH runner reads only Gene and
Sequence. `--no-expression-col` leaves the third column out, for example when
the expression table may not be redistributed with the panel.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from duet.bostrom_codebook import (
    assign_codewords_to_genes,
    load_bostrom_codebook,
    parse_bostrom_filename,
    verify_codebook,
)

logger = logging.getLogger("make_bostrom_panel")


def select_panel_genes(expr_df, gene_col, expr_col, *, top_k=None,
                       genes_csv=None, genes_col="Gene", exclude_controls=True):
    """Return (genes: list[str], expression: np.ndarray) for the panel."""
    if (top_k is None) == (genes_csv is None):
        raise ValueError("Provide exactly one of top_k or genes_csv")
    if top_k is not None:
        pool = expr_df[expr_df[expr_col] > 0].copy()
        if exclude_controls:
            pool = pool[~pool[gene_col].astype(str).str.startswith("ERCC-")]
        pool = pool.sort_values(expr_col, ascending=False, kind="stable")
        if len(pool) < top_k:
            raise ValueError(f"top_k={top_k} but only {len(pool)} genes available after filtering")
        pool = pool.head(top_k)
        return pool[gene_col].astype(str).tolist(), pool[expr_col].to_numpy(dtype=float)
    genes = pd.read_csv(genes_csv)[genes_col].astype(str).tolist()
    gene_to_expr = dict(zip(expr_df[gene_col].astype(str), expr_df[expr_col].astype(float)))
    missing = [g for g in genes if g not in gene_to_expr]
    if missing:
        logger.warning("%d/%d panel genes absent from expression CSV (set to 0.0): %s",
                       len(missing), len(genes), missing[:5])
    expr = np.array([gene_to_expr.get(g, 0.0) for g in genes], dtype=float)
    return genes, expr


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codebook", required=True, type=Path)
    p.add_argument("--barcode-length", type=int, default=None,
                   help="Defaults to the bit count parsed from the codebook filename.")
    p.add_argument("--fmt", choices=["auto", "set", "binary"], default="auto",
                   help="Defaults to the filename's Binary/Set token, else content auto-detect.")
    p.add_argument("--expression", required=True, type=Path,
                   help="MUST be byte-identical to the run's expression: file.")
    p.add_argument("--gene-col", default="gene_name", help="Gene column in --expression.")
    p.add_argument("--expression-col", default="mean_raw_counts")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--top-k", type=int, help="Top-K genes by expression (controls excluded).")
    grp.add_argument("--genes-csv", type=Path, help="Explicit gene-panel CSV.")
    p.add_argument("--genes-col", default="Gene", help="Gene column in --genes-csv.")
    p.add_argument("--no-exclude-controls", dest="exclude_controls",
                   action="store_false", default=True,
                   help="Keep ERCC- control probes in the --top-k pool.")
    p.add_argument("--shuffle-assignment", action="store_true", default=False,
                   help="Random (expression-agnostic) gene->codeword assignment, "
                        "imitating standard MERFISH. Holds the codeword set fixed; "
                        "only the gene->codeword pairing is permuted.")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Seed for --shuffle-assignment (default 42).")
    p.add_argument("--no-expression-col", dest="write_expression",
                   action="store_false", default=True,
                   help="Write only the Gene and Sequence columns (by default a "
                        "third column, named after --expression-col, holds each "
                        "gene's expression value).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    args = p.parse_args(argv)

    meta = parse_bostrom_filename(args.codebook)
    barcode_length = args.barcode_length or meta.get("bits")
    if barcode_length is None:
        p.error("--barcode-length is required (filename did not encode the bit count)")
    if meta.get("bits") and meta["bits"] != barcode_length:
        p.error(f"--barcode-length {barcode_length} != filename bits {meta['bits']}")
    fmt = args.fmt
    if fmt == "auto" and meta.get("fmt"):
        fmt = meta["fmt"]

    codewords = load_bostrom_codebook(args.codebook, barcode_length, fmt=fmt)
    if args.verify:
        logger.info("Codebook verified: %s", verify_codebook(codewords, expected_hw=meta.get("hw")))

    expr_df = pd.read_csv(args.expression)
    genes, expression = select_panel_genes(
        expr_df, args.gene_col, args.expression_col,
        top_k=args.top_k, genes_csv=args.genes_csv, genes_col=args.genes_col,
        exclude_controls=args.exclude_controls,
    )
    panel = assign_codewords_to_genes(
        codewords, genes, expression,
        shuffle=args.shuffle_assignment,
        seed=args.shuffle_seed if args.shuffle_assignment else None,
    )
    if args.write_expression:
        panel[args.expression_col] = panel["Gene"].map(dict(zip(genes, expression)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    logger.info(
        "Wrote %d rows to %s (codebook finalsize=%d, dropped %d). Point BOTH "
        "genes: and initialization: at this file; the run's expression: file "
        "must be %s.",
        len(panel), args.out, len(codewords), len(codewords) - len(panel), args.expression,
    )


if __name__ == "__main__":
    main()
