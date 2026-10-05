#!/usr/bin/env python
"""Idempotently fetch the ABC Atlas metadata this analysis needs (~16 MB total).

Pulls five small files from the Allen Brain Cell Atlas public S3 bucket into
data/raw/WMB-10X/ (gitignored):

  cluster_annotation_term.csv  taxonomy terms: class/subclass names + Allen's
                               official color_hex_triplet per term
  cluster.csv                  per-cluster cell counts (sums to 4,042,976)
  membership_pivoted.csv       cluster_alias -> neurotransmitter/class/subclass/supertype
  term_with_counts.csv         taxonomy terms annotated with cluster/cell counts
  mouse_markers_230821.json    MapMyCells marker genes per taxonomy node (union 6,558)

Files that already exist are skipped, so reruns are free. This script exists for
reproducibility: it records exactly which release of which object each local file
came from.

THE TWO LARGE INPUTS ARE NOT PART OF THAT ~16 MB FETCH. wmb_precomputed_stats.h5
(1.4 GB, the 5,322 x 32,285 centroid matrix) and wmb_gene.csv (2.3 MB, Ensembl ->
symbol) are resolved lazily by resolve_large_input(), which build_centroid_matrix.py
and build_whole_brain_cpm.py call, IN THIS ORDER:

  1. data/raw/WMB-10X/<name>            the local copy, if it exists
  2. download to data/raw/WMB-10X/      only if there is no local copy

The resolved path is printed at runtime, so which copy fed a given run is never a
guess. Cell-level expression (94.7 GB to 150 GB) is deliberately never downloaded by
any step of this pipeline.

Reproduce:  python download_wmb_metadata.py          # skips what is already there
            python download_wmb_metadata.py --force  # re-fetch everything
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/wmb10x/x.py -> repo root
DATA = ROOT / "data" / "raw" / "WMB-10X"

S3_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"

# local filename -> key within the bucket
FILES: dict[str, str] = {
    "cluster_annotation_term.csv":
        "metadata/WMB-taxonomy/20231215/cluster_annotation_term.csv",
    "cluster.csv":
        "metadata/WMB-taxonomy/20231215/cluster.csv",
    "membership_pivoted.csv":
        "metadata/WMB-taxonomy/20231215/views/"
        "cluster_to_cluster_annotation_membership_pivoted.csv",
    "term_with_counts.csv":
        "metadata/WMB-taxonomy/20231215/views/cluster_annotation_term_with_counts.csv",
    "mouse_markers_230821.json":
        "mapmycells/WMB-10X/20240831/mouse_markers_230821.json",
}

# The two large inputs. Same bucket, but NOT fetched by main(): they are resolved on
# demand by resolve_large_input() below, and downloaded only if there is no local copy.
LARGE_FILES: dict[str, str] = {
    "wmb_precomputed_stats.h5":
        "mapmycells/WMB-10X/20240831/precomputed_stats_ABC_revision_230821.h5",
    "wmb_gene.csv":
        "metadata/WMB-10X/20241115/gene.csv",
}


def resolve_large_input(name: str) -> Path:
    """Locate a large input in data/raw/WMB-10X/, downloading it there if absent.

    Prints the copy it settled on, so a run always says which bytes it read.
    """
    if name not in LARGE_FILES:
        raise KeyError(f"{name} is not a known large input: {sorted(LARGE_FILES)}")

    local = DATA / name
    if local.exists():
        print(f"[source] {name}: {local}  "
              f"({local.stat().st_size / 1e6:.1f} MB, local copy)")
        return local

    print(f"[source] {name}: no copy in {DATA}, fetching from ABC")
    DATA.mkdir(parents=True, exist_ok=True)
    fetch(LARGE_FILES[name], local)
    return local


def fetch(key: str, dest: Path, force: bool = False) -> bool:
    """Download S3_BASE/key to dest. Returns True if a download happened."""
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"[skip] {dest.name} already present ({size_mb:.2f} MB)")
        return False

    url = f"{S3_BASE}/{key}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"[fail] could not fetch {url}: {exc}") from exc
    tmp.replace(dest)  # atomic: a partial download never masquerades as complete
    print(f"[ok  ] {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    n_new = sum(fetch(key, DATA / name, force=args.force) for name, key in FILES.items())
    print(f"[done] {len(FILES)} files in {DATA} ({n_new} downloaded, "
          f"{len(FILES) - n_new} already present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
