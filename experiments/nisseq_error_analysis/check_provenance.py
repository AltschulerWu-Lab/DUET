"""Compare one subsample size's regenerated NIS-seq matrices with the reference ones.

For subsample size N, the notebook writes
<matrices-dir>/nisseq_hela_{kind}_pctl50_subsample{N}.npy for the four kinds.
The config's ``provenance`` map names, per N, the reference file each one must
equal byte for byte. Both the bytes (sha256) and the arrays (np.array_equal,
dtype, shape) are compared. Exit status 1 on any mismatch or on a missing
regenerated file. A size with no entry in ``provenance`` is reported as skipped
(exit 0). So is a size whose four reference files are all missing from this
copy (exit 0 if nothing else fails): the NIS-seq channel matrices are included
only with the permission of the authors of Fandrey et al. (2025). Some but not
all four missing means a wrong path or a deleted file, and fails.

Usage (run.sh calls it from this folder, so relative paths resolve here):
    python check_provenance.py --config config.yaml --n 50000 \
        --matrices-dir ../../results/experiments/nisseq_error_analysis/subsample50000/noise_matrices
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

KINDS = ("uniform", "positional", "channel", "positional_channel")
# Notebook cells 12 and 29: SELECTED_PERCENTILE = 50, SUFFIX = f'_subsample{N}'.
REGENERATED = "nisseq_hela_{kind}_pctl50_subsample{n}.npy"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", required=True, type=int, help="reads per file (SUBSAMPLE_N)")
    ap.add_argument("--matrices-dir", required=True, type=Path)
    args = ap.parse_args()

    refs = yaml.safe_load(open(args.config)).get("provenance") or {}
    refs = {int(k): v for k, v in refs.items()}
    if args.n not in refs:
        print(f"PROVENANCE CHECK SKIPPED: {args.config} lists no reference matrices for N={args.n}")
        return 0

    failures = []
    skipped = []  # kinds whose reference file is not in this copy
    print(f"Provenance check, N = {args.n:,} reads per file")
    for kind in KINDS:
        new = args.matrices_dir / REGENERATED.format(kind=kind, n=args.n)
        ref = Path(refs[args.n].format(kind=kind))
        print(f"  {kind}")
        print(f"    regenerated {new.resolve()}")
        print(f"    reference   {ref.resolve()}")
        if not new.is_file():
            failures.append(f"{kind}: missing regenerated file {new}")
            print(f"    MISSING: {new}")
            continue
        if not ref.is_file():
            skipped.append(kind)
            print(f"    PROVENANCE CHECK SKIPPED: reference {ref} is not in this copy "
                  "(the NIS-seq channel matrices are included only with the permission "
                  "of the authors of Fandrey et al. 2025)")
            continue
        h_new, h_ref = sha256(new), sha256(ref)
        a_new, a_ref = np.load(new), np.load(ref)
        same_bytes = h_new == h_ref
        same_array = (a_new.dtype == a_ref.dtype and a_new.shape == a_ref.shape
                      and np.array_equal(a_new, a_ref))
        print(f"    sha256 {h_new} / {h_ref}")
        print(f"    shape {a_new.shape} {a_new.dtype}; bytes {'identical' if same_bytes else 'DIFFER'}; "
              f"np.array_equal {same_array}")
        if not (same_bytes and same_array):
            diff = (float(np.max(np.abs(a_new - a_ref)))
                    if a_new.shape == a_ref.shape else float("nan"))
            failures.append(f"{kind}: bytes identical {same_bytes}, arrays equal {same_array}, "
                            f"max |diff| {diff:.3g}")

    if skipped and len(skipped) < len(KINDS):
        # The matrices ship or are dropped as a set, so a partial set means a
        # wrong path in config.yaml or a deleted file, not a copy without them.
        failures.append(f"reference matrices missing for {', '.join(skipped)} while the "
                        "others are present (check provenance: in config.yaml)")
    if failures:
        print(f"PROVENANCE CHECK FAILED for N={args.n}:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    if len(skipped) == len(KINDS):
        print(f"PROVENANCE CHECK SKIPPED: none of the 4 reference matrices for N={args.n:,} "
              "is in this copy, so nothing was compared")
    else:
        print(f"PROVENANCE CHECK PASSED: all 4 matrices for N={args.n:,} are byte-identical to the references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
