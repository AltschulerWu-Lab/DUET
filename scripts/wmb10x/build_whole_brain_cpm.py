#!/usr/bin/env python
"""Build the whole-brain WMB-10X expression table that the MERFISH experiments read.

Writes one row per gene symbol, with the columns gene_symbol and mean_cpm, sorted by
descending mean_cpm. By default it writes
examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv, the `expression:` table
of experiments/merfish_zhang2023_v2, experiments/merfish_2000_genes and
experiments/merfish_expression_prior, and the expression input of their panel
builders.

The values come from the Allen Brain Cell Atlas WMB-10X data (Yao et al. 2023,
Nature, doi:10.1038/s41586-023-06812-z), which is licensed CC BY-NC 4.0. The table is
therefore not distributed with this repository; this script rebuilds it.

Inputs. Each is found by resolve_large_input() in download_wmb_metadata.py, which
looks in data/raw/WMB-10X/ and downloads the file from the ABC bucket if it is not
there. --stats and --genes give other paths instead.
  wmb_precomputed_stats.h5  1.4 GB. "sum" is the (5,322 clusters x 32,285 genes)
                            matrix of mean log2(CPM+1); "col_names" holds the
                            Ensembl gene ids in column order.
  wmb_gene.csv              2.3 MB. Ensembl id (gene_identifier) to gene_symbol.

Steps:
  1. Per cluster and gene, log2(CPM+1) back to CPM as 2^x - 1, clamped at 0.
  2. Per gene, the unweighted mean over the 5,322 clusters.
  3. Ensembl id to symbol (an id with no symbol keeps the id). Keep the ids with a
     mean above 0 and sort by descending mean (30,622 rows).
  4. Write that table as CSV text and read it back with pandas' default float
     parser. The paper's table was made in two steps, through such a CSV file, and
     the parser can change the last digit of a value (1 ulp). Repeating the round
     trip makes the output byte-identical to the paper's table; without it, 174
     of the 30,599 values differ in the last digits (at most 1.5e-13 relative).
  5. A symbol on more than one row (22 symbols) becomes one row whose value is the
     SUM of its rows. Summing keeps the total CPM, so a panel's share of all reads
     is unchanged.
  6. Keep mean_cpm > 0 and sort by descending mean_cpm (30,599 rows).

Caveats: 2^(mean of x) is not the mean of 2^x, so absolute CPM values are
approximate; and clusters are averaged without weighting by cell count.

The table used in the paper has 30,599 rows and the sha256 in EXPECTED_SHA256. The
script compares its output with that hash and prints a warning if they differ.

Runs in the scanpy environment (environments/scanpy.yml; needs h5py).

Usage:
  python scripts/wmb10x/build_whole_brain_cpm.py
  python scripts/wmb10x/build_whole_brain_cpm.py --stats PATH --genes PATH --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script, not a package
from download_wmb_metadata import resolve_large_input  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # scripts/wmb10x/x.py -> repo root
DEFAULT_OUT = ROOT / "examples" / "data" / "processed" / "WMB-10X" / "whole_brain_per_gene_cpm.csv"

# sha256 of the table the paper's MERFISH runs read (30,599 rows).
EXPECTED_SHA256 = "a8314155531fefb3e7abfb4993f1f4c184ece9cdbf541fdf38cc2503584fd8e4"


def build_whole_brain_cpm(stats_h5: Path, gene_csv: Path) -> pd.DataFrame:
    """Return the per-symbol whole-brain mean CPM table (see the module docstring)."""
    with h5py.File(stats_h5, "r") as f:
        cols = json.loads(np.asarray(f["col_names"][()]).item())
        cols = list(cols.keys()) if isinstance(cols, dict) else list(cols)
        cpm = np.maximum(2.0 ** np.asarray(f["sum"][()]) - 1.0, 0.0)   # (clusters, genes)
    gene_cpm = cpm.mean(axis=0)
    sym = pd.read_csv(gene_csv).set_index("gene_identifier")["gene_symbol"].to_dict()
    per_id = pd.DataFrame({"gene_symbol": [sym.get(c, c) for c in cols], "mean_cpm": gene_cpm})
    per_id = (per_id[per_id["mean_cpm"] > 0]
              .sort_values("mean_cpm", ascending=False).reset_index(drop=True))

    # Step 4: the CSV round trip of the original two-step build (see the docstring).
    # "high" is pandas' default float parser; it is named so that a change of the
    # default cannot change the output.
    tab = pd.read_csv(io.StringIO(per_id.to_csv(index=False)), float_precision="high")

    n_dup = int((tab["gene_symbol"].value_counts() > 1).sum())
    print(f"[info] {len(cols)} Ensembl ids, {len(tab)} with a mean above 0; {n_dup} "
          f"symbols are on more than one row and are summed into one")
    # Sum, not max or last: the panel capture ratio sum(panel)/sum(all) keeps its
    # denominator, and a panel gene carries its symbol's full read mass.
    tab = tab.groupby("gene_symbol", as_index=False)["mean_cpm"].sum()
    tab = tab[tab["mean_cpm"] > 0].sort_values("mean_cpm", ascending=False).reset_index(drop=True)
    return tab


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stats", type=Path, default=None,
                        help="wmb_precomputed_stats.h5 (default: data/raw/WMB-10X/, "
                             "downloaded if absent)")
    parser.add_argument("--genes", type=Path, default=None,
                        help="wmb_gene.csv (default: data/raw/WMB-10X/, downloaded if absent)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output CSV (default: {DEFAULT_OUT.relative_to(ROOT)})")
    args = parser.parse_args(argv)

    stats_h5 = args.stats if args.stats is not None else resolve_large_input("wmb_precomputed_stats.h5")
    gene_csv = args.genes if args.genes is not None else resolve_large_input("wmb_gene.csv")
    print(f"[read] {stats_h5}")
    print(f"[read] {gene_csv}")

    tab = build_whole_brain_cpm(stats_h5, gene_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out, index=False, lineterminator="\n")
    digest = sha256_of(args.out)
    print(f"[ok  ] wrote {len(tab)} genes to {args.out}")
    print(f"[sha ] {digest}")
    if digest == EXPECTED_SHA256:
        print("[ok  ] matches the table used in the paper")
    else:
        print(f"WARNING: sha256 differs from the table used in the paper "
              f"({EXPECTED_SHA256}). Check the ABC release of the inputs and the "
              f"pandas and numpy versions (environments/scanpy.yml).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
