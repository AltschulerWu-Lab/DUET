#!/usr/bin/env python
"""Build the CRISPick candidate table from three CRISPick downloads.

The table holds the first 20 CRISPick picks for every human gene, plus the
CRISPick negative-control guides. experiments/ops_crispick_genome_wide reads
it (candidate_pool.csv_path), and so does duet.crispick_factory.CRISPickFactory
when no csv_path is given. It is not shipped with DUET (see Licence below), so
build it yourself. Run from the repository root:

    python scripts/data_processing/build_crispick_candidates.py --raw-dir data/raw/CRISPick

Inputs
------
Download these three files from the Broad Institute GPP CRISPick portal,
https://portals.broadinstitute.org/gppx/crispick/public, and put them in
--raw-dir (default data/raw/CRISPick). The design file is CRISPick's
genome-wide design for human (taxon 9606, GRCh38), SpCas9 (SpyoCas9),
CRISPRko, rule sets RS3seq-Chen2013 + RS3target, Ensembl gene IDs and
aggregate CFD off-target scoring, dated 2025-11-21. The two control files
hold 2,000 guides each, one sequence per line.

    sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_RS3seq-Chen2013+RS3target_Ensembl_aggrCFD_20251121.txt
        2,116,333,376 bytes
        sha256 5dd4da45440dc095e14755680a2d0ac9051467eef92aa613725ba2ba12195a24
    sgrna-negcontrols-nosite-9606-GRCh38-SpyoCas9-2000.txt
        42,000 bytes
        sha256 e86fce65335216b4bf0606a512494f2d5229610fb5f1cf96b218ce75589fbb85
    sgrna-negcontrols-onesite-9606-GRCh38-SpyoCas9-2000.txt
        42,000 bytes
        sha256 dd350b75d89c341e67387229129b7b77256946ccbc13f341750fef2d3a62d3d6

Licence
-------
CRISPick output is provided by the Broad Institute Genetic Perturbation
Platform (GPP) under its terms of use, for research use. Neither the raw files
nor the table built from them are redistributed with DUET: download the files
yourself and accept the portal's terms. Do not commit the built table to a
public repository. If you use it, cite CRISPick: DeWeirdt et al. 2022,
Nat Commun 13:5255, doi:10.1038/s41467-022-33024-2. The MIT licence of DUET
covers this script only.

Transformation
--------------
1. Read seven columns of the design file (tab separated): Target Gene ID,
   Target Gene Symbol, sgRNA Sequence, Aggregate CFD Score, On-Target Efficacy
   Score, Pick Order and Picking Round. Pick Order and Picking Round are read
   as nullable integers, so they are written as 1, 2, ... and not 1.0, 2.0.
2. Drop the guides that CRISPick did not pick (empty Pick Order).
3. Sort by Target Gene ID, then Pick Order, and keep the first 20 rows of each
   gene (fewer for genes with fewer picks).
4. Append the no-site controls, then the one-site intergenic controls, each in
   file order. Their Target Gene ID and Target Gene Symbol are NO_SITE or
   ONE_SITE_INTERGENIC, and their scores, Pick Order and Picking Round are
   empty.
5. Add a Source column (Ensembl, NO_SITE or ONE_SITE_INTERGENIC) and write a
   CSV with no index and LF line endings. The columns keep the order of the
   design file, then Source.

With the files above, the result has 405,059 rows (401,059 guides for 20,116
genes, then 4,000 controls), 27,708,495 bytes, sha256
e49d92641f932c2610e8f8817428532386a0f56425d306e50141d56fb531a484. That is the
table the paper used, byte for byte (checked with pandas 3.0.0 and
numpy 2.4.2). A build takes about a minute and about 1.6 GB of memory.

The "MAX" quirk
---------------
The design file writes "MAX" instead of a number in Aggregate CFD Score for
some guides. The original build read the text "MAX" as missing in every
column, not only that one, and this script does the same. In the 2025-11-21
file that has one visible effect: the gene named MAX (ENSG00000125952) gets an
empty Target Gene Symbol in its 20 rows. No guide with a "MAX" CFD score is
among the first 20 picks of its gene. Nothing in DUET reads the symbol column,
so the quirk is kept to match the paper's table.

Why row order matters
---------------------
CRISPickFactory takes genes in the order they first appear in the table, and
its seeded gene sampling and the order of the candidates follow that order. A
table with the same rows in another order therefore gives other gene draws and
other simulated reads from the same seed. Keep the sort above.

Checks
------
The script checks the sha256 of the three inputs and of the output against
the values above. A mismatch prints a warning but does not stop the build.
The CRISPick portal may serve a newer release than 2025-11-21. A newer release
gives a table built the same way, but with other guides and scores, so the
experiment's results should agree with the paper's statistically, not exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DESIGN_FILE = (
    "sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_RS3seq-Chen2013+RS3target_"
    "Ensembl_aggrCFD_20251121.txt"
)
NOSITE_FILE = "sgrna-negcontrols-nosite-9606-GRCh38-SpyoCas9-2000.txt"
ONESITE_FILE = "sgrna-negcontrols-onesite-9606-GRCh38-SpyoCas9-2000.txt"

# sha256 of the files the paper's table was built from, and of that table.
RAW_SHA256 = {
    DESIGN_FILE: "5dd4da45440dc095e14755680a2d0ac9051467eef92aa613725ba2ba12195a24",
    NOSITE_FILE: "e86fce65335216b4bf0606a512494f2d5229610fb5f1cf96b218ce75589fbb85",
    ONESITE_FILE: "dd350b75d89c341e67387229129b7b77256946ccbc13f341750fef2d3a62d3d6",
}
EXPECTED_SHA256 = "e49d92641f932c2610e8f8817428532386a0f56425d306e50141d56fb531a484"

PORTAL_URL = "https://portals.broadinstitute.org/gppx/crispick/public"
# Defaults resolve against the repository root, whatever the working folder.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "CRISPick"
DEFAULT_OUT = ROOT / "data" / "processed" / "CRISPick" / "crispick_ensembl_aggrCFD_top20_candidates.csv"
TOP_N = 20

# Controls: (file, label used as Target Gene ID, Target Gene Symbol and Source).
CONTROLS = [(NOSITE_FILE, "NO_SITE"), (ONESITE_FILE, "ONE_SITE_INTERGENIC")]

USECOLS = [
    "Target Gene ID",
    "Target Gene Symbol",
    "sgRNA Sequence",
    "On-Target Efficacy Score",
    "Pick Order",
    "Picking Round",
    "Aggregate CFD Score",
]
DTYPE = {
    "Target Gene ID": str,
    "Target Gene Symbol": str,
    "sgRNA Sequence": str,
    "On-Target Efficacy Score": float,
    "Pick Order": "Int64",  # nullable integer: written as 1, not 1.0
    "Picking Round": "Int64",
    "Aggregate CFD Score": float,
}


def read_design(path: Path, top_n: int = TOP_N) -> pd.DataFrame:
    """Read the design file and keep the first ``top_n`` picks of each gene."""
    design = pd.read_csv(
        path,
        sep="\t",
        usecols=USECOLS,
        dtype=DTYPE,
        # "MAX" marks some Aggregate CFD Scores. Applied to every column, as in
        # the original build, so the symbol of the gene MAX is read as missing.
        # Kept for byte identity with the paper's table (see the module notes).
        na_values=["MAX"],
    )
    design = design.dropna(subset=["Pick Order"])
    # A sort on two columns is stable in pandas; the 2025-11-21 file has no
    # tied (gene, Pick Order) pairs in any case.
    design = design.sort_values(["Target Gene ID", "Pick Order"])
    design = design.groupby("Target Gene ID").head(top_n)
    design["Source"] = "Ensembl"
    return design


def read_controls(path: Path, label: str) -> pd.DataFrame:
    """Read a control file (one sequence per line) as rows with empty scores."""
    seqs = path.read_text().strip().splitlines()
    n = len(seqs)
    return pd.DataFrame(
        {
            "Target Gene ID": label,
            "Target Gene Symbol": label,
            "sgRNA Sequence": seqs,
            "On-Target Efficacy Score": np.nan,
            "Pick Order": pd.array([pd.NA] * n, dtype="Int64"),
            "Picking Round": pd.array([pd.NA] * n, dtype="Int64"),
            "Aggregate CFD Score": np.nan,
            "Source": label,
        }
    )


def build(raw_dir: Path, top_n: int = TOP_N, design_file: str = DESIGN_FILE) -> pd.DataFrame:
    """Build the candidate table from the three CRISPick files in ``raw_dir``.

    ``design_file`` names the design download; another CRISPick release has
    another date in its name.
    """
    raw_dir = Path(raw_dir)
    parts = [read_design(raw_dir / design_file, top_n)]
    parts += [read_controls(raw_dir / name, label) for name, label in CONTROLS]
    return pd.concat(parts, ignore_index=True)


def write_table(table: pd.DataFrame, out: Path) -> None:
    """Write the table as CSV with LF line endings on every platform."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    files = "\n".join(f"  {name}" for name in RAW_SHA256)
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            f"Put these three files from the CRISPick portal ({PORTAL_URL})\n"
            f"in --raw-dir:\n{files}\n\n"
            "The output is the paper's table byte for byte when the inputs are\n"
            "the 2025-11-21 release (the script checks the sha256 values).\n"
            "The table is under the Broad GPP CRISPick terms (research use):\n"
            "do not redistribute it. See the module docstring for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help="folder with the three CRISPick files "
                             "(default: data/raw/CRISPick in the repository)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output CSV (default: data/processed/CRISPick/"
                             "crispick_ensembl_aggrCFD_top20_candidates.csv in the "
                             "repository, where the genome-wide configs read it)")
    parser.add_argument("--design-file", default=DESIGN_FILE,
                        help="name of the design download in --raw-dir, for a CRISPick "
                             "release other than the paper's (default: %(default)s)")
    args = parser.parse_args(argv)

    needed = [args.design_file, NOSITE_FILE, ONESITE_FILE]
    missing = [name for name in needed if not (args.raw_dir / name).is_file()]
    if missing:
        print(f"error: missing in {args.raw_dir}:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        print(f"Download them from the CRISPick portal, {PORTAL_URL}, "
              "or pass --raw-dir. See --help.", file=sys.stderr)
        return 1

    raw_matches = args.design_file == DESIGN_FILE
    if not raw_matches:
        _warn(f"{args.design_file} is not the design file the paper's table was "
              f"built from ({DESIGN_FILE}).")
    for name, expected in RAW_SHA256.items():
        if name == DESIGN_FILE and not raw_matches:
            continue
        if sha256_of(args.raw_dir / name) != expected:
            raw_matches = False
            _warn(f"{name} is not the file the paper's table was built from "
                  "(its sha256 differs).")
    if not raw_matches:
        _warn("A different CRISPick release gives a table built the same way but "
              "with other guides and scores, so results will agree with the "
              "paper's statistically, not exactly.")

    table = build(args.raw_dir, design_file=args.design_file)
    write_table(table, args.out)

    digest = sha256_of(args.out)
    n_genes = table.loc[table["Source"] == "Ensembl", "Target Gene ID"].nunique()
    print(f"Wrote {args.out}: {len(table):,} rows ({n_genes:,} genes plus controls), "
          f"sha256 {digest}")
    if digest == EXPECTED_SHA256:
        print("This is the paper's table, byte for byte.")
    else:
        _warn(f"the output differs from the paper's table (sha256 {EXPECTED_SHA256}).")
        if raw_matches:
            _warn("The inputs match, so the difference comes from the software "
                  "(for example another pandas version). Please report it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
