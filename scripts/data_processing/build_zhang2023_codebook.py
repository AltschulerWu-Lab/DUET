#!/usr/bin/env python
"""Convert the Zhang et al. (2023) MERFISH codebook to DUET's codebook format.

experiments/merfish_zhang2023_v2 reads the converted codebook (its gene list,
warm start and published-codebook baseline), and its make_panels.sh takes the
gene list of the Boström baseline panels from it. Run from the repository
root:

    python scripts/data_processing/build_zhang2023_codebook.py --download

Source
------
Zhang M. et al. (2023), Nature, doi:10.1038/s41586-023-06808-9. Dataset:
Zhuang X, Jung W, Zhang M (2023), Brain Image Library, doi:10.35077/act-bag,
file additional_files/codebook_32bit_v2.csv (92,715 bytes, sha256
7e45c01cab5bfa398d86031748ff9769ad3829906ef57c16c5e287842e222667):

    https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/additional_files/codebook_32bit_v2.csv

With --download the script fetches that file to --raw when it is absent
(default data/raw/BIL_MERFISH/additional_files/codebook_32bit_v2.csv).

Licence
-------
The dataset is shared under CC BY-SA 4.0. The converted codebook is an
adaptation of codebook_32bit_v2.csv (blank barcodes dropped, readout bits
joined into one string per gene), so it is under CC BY-SA 4.0 as well:
attribute Zhang et al. (2023) and the dataset above, and share adaptations
under the same licence. The MIT licence of DUET covers this script only.
This work used data from the Brain Image Library (RRID:SCR_017272), which is supported by the
National Institutes of Mental Health of the National Institutes of Health
under award number R24-MH-114793.

Transformation
--------------
1. Read the CSV. Its columns are name, id and 32 readout bits named RS####,
   in an order that is not sorted by readout name.
2. Drop the 93 blank barcodes (names starting with "blank-").
3. Join each gene's 32 bits, in file column order, into a 32-character string
   of 0s and 1s.
4. Write a CSV with the columns Gene and Sequence, no index, LF line endings.

With the file above, the result has 1,147 genes, each with four bits on,
45,167 bytes, sha256
3a6159f8ce074c78db13dbce00cb422f716611764ea9409683ff015a29dde779, the paper's
codebook byte for byte (checked with pandas 3.0.0). The script checks the
sha256 of the input and of the output and prints a warning if either differs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW_URL = (
    "https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/"
    "additional_files/codebook_32bit_v2.csv"
)
RAW_SHA256 = "7e45c01cab5bfa398d86031748ff9769ad3829906ef57c16c5e287842e222667"
EXPECTED_SHA256 = "3a6159f8ce074c78db13dbce00cb422f716611764ea9409683ff015a29dde779"

# Defaults resolve against the repository root, whatever the working folder.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw" / "BIL_MERFISH" / "additional_files" / "codebook_32bit_v2.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "BIL_MERFISH" / "codebooks" / "zhang_2023_v2_processed.csv"
BLANK_PREFIX = "blank-"
BIT_PREFIX = "RS"


def convert(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the BIL codebook table into DUET's Gene/Sequence codebook."""
    raw = raw[~raw["name"].astype(str).str.startswith(BLANK_PREFIX)]
    bits = [c for c in raw.columns if c.startswith(BIT_PREFIX)]  # readout bits, in file order
    if not bits:
        raise ValueError(f"no readout-bit columns (named {BIT_PREFIX}####) in the codebook")
    values = raw[bits].astype(int)
    if not values.isin([0, 1]).all().all():
        raise ValueError("the readout-bit columns must hold only 0 and 1")
    return pd.DataFrame({
        "Gene": raw["name"].astype(str),
        "Sequence": values.astype(str).agg("".join, axis=1),
    })


def write_codebook(codebook: pd.DataFrame, out: Path) -> None:
    """Write the codebook as CSV with LF line endings on every platform."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    codebook.to_csv(out, index=False, lineterminator="\n")


def download(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest`` through a temporary file, so a failed download leaves nothing."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            f"Source file: {RAW_URL}\n"
            "Data: Brain Image Library, doi:10.35077/act-bag, CC BY-SA 4.0. The\n"
            "converted codebook is an adaptation under the same licence. See the\n"
            "module docstring for details."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help="codebook_32bit_v2.csv from the Brain Image Library "
                             "(default: data/raw/BIL_MERFISH/additional_files/ in the repository)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output CSV (default: data/processed/BIL_MERFISH/codebooks/"
                             "zhang_2023_v2_processed.csv in the repository, where "
                             "merfish_zhang2023_v2 reads it)")
    parser.add_argument("--download", action="store_true",
                        help="download the source file to --raw if it is absent")
    args = parser.parse_args(argv)

    if not args.raw.is_file():
        if not args.download:
            print(f"error: {args.raw} not found. Rerun with --download to fetch it from\n"
                  f"  {RAW_URL}\nor pass --raw.", file=sys.stderr)
            return 1
        print(f"Downloading {RAW_URL}\n  to {args.raw}")
        try:
            download(RAW_URL, args.raw)
        except OSError as exc:  # urllib.error.URLError is an OSError
            print(f"error: download failed: {exc}", file=sys.stderr)
            return 1

    if sha256_of(args.raw) != RAW_SHA256:
        _warn(f"{args.raw} is not the file the paper used (its sha256 differs from "
              f"{RAW_SHA256}); the codebook may differ.")

    codebook = convert(pd.read_csv(args.raw))
    write_codebook(codebook, args.out)

    digest = sha256_of(args.out)
    print(f"Wrote {args.out}: {len(codebook):,} genes, sha256 {digest}")
    if digest == EXPECTED_SHA256:
        print("This is the paper's codebook, byte for byte.")
    else:
        _warn(f"the output differs from the paper's codebook (sha256 {EXPECTED_SHA256}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
