#!/usr/bin/env python
"""Build the WMB-10X cluster-centroid matrix and its row/column annotations.

Points in every figure downstream of this script are the 5,322 transcriptomic
CLUSTER CENTROIDS of Yao et al. 2023, one point per cell type, not the ~4.04M
individual cells (cell-level data, 94.7 GB to 150 GB, was never downloaded).
Within-cluster spread is therefore absent from the data by construction.

Reads:
  wmb_precomputed_stats.h5  1.4 GB. resolve_large_input() looks for it in
      data/raw/WMB-10X/ and downloads it there from ABC if it is absent.
      The path actually used is printed at the top of every run.
      "sum"            (5322 clusters, 32285 genes) float64, mean log2(CPM+1)
      "cluster_to_row" cluster id -> row index of "sum"
      "col_names"      Ensembl gene ids, in the column order of "sum"
      "taxonomy_tree"  CLAS -> SUBC -> SUPT -> CLUS hierarchy + name mapper
      (the h5's "n_cells" is a uniform placeholder of 1s and is NOT used; real
       cell counts come from cluster.csv)
  wmb_gene.csv            Ensembl -> symbol. Resolved the same way.
  data/raw/WMB-10X/cluster_annotation_term.csv  class names + Allen's color_hex_triplet
  data/raw/WMB-10X/cluster.csv                  per-cluster cell counts
  data/raw/WMB-10X/membership_pivoted.csv       independent alias -> cluster-name check
  data/raw/WMB-10X/mouse_markers_230821.json    MapMyCells markers per taxonomy node

Writes (all gitignored, all regenerable by rerunning this script):
  data/processed/WMB-10X/centroids.npy   (5322, 32285) float32, ~687 MB
  data/processed/WMB-10X/genes.csv       ensembl_id, gene_symbol, in column order
  data/processed/WMB-10X/marker_genes.txt  6,558 Ensembl marker ids in the h5 columns
  data/processed/WMB-10X/cluster_metadata.csv  one row per cluster, in centroids.npy
                                               ROW ORDER. The join product, not a
                                               result: every downstream script keys
                                               on it.

THE CELL-COUNT JOIN IS A TRAP, so read this before touching it. cluster.csv's `label`
looks like the h5's cluster id with the level token dropped (CS20230722_0001 vs the
h5's CS20230722_CLUS_0001), but it is NOT: `label` is just cluster_alias zero-padded,
and the alias is not the cluster index. Alias 1 is CS20230722_CLUS_0326, not
CLUS_0001. Joining on that string transform silently attaches the WRONG cell count to
about 5,030 of the 5,322 clusters and only fails outright on 292 of them.

The sound key is the `alias` field carried by each cluster in the h5's taxonomy
name_mapper, joined to cluster.csv's `cluster_alias`. That join is bijective, and it
is cross-checked here against membership_pivoted.csv's independent cluster names
(0/5322 mismatches) so the mapping cannot silently regress.

Reproduce:
  cd scripts/wmb10x
  PYTHONPATH=../../src python build_centroid_matrix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script, not a package
from download_wmb_metadata import resolve_large_input  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # scripts/wmb10x/x.py -> repo root
RAW = ROOT / "data" / "raw" / "WMB-10X"
PROC = ROOT / "data" / "processed" / "WMB-10X"

TERM_CSV = RAW / "cluster_annotation_term.csv"
CLUSTER_CSV = RAW / "cluster.csv"
MEMBERSHIP_CSV = RAW / "membership_pivoted.csv"
MARKERS_JSON = RAW / "mouse_markers_230821.json"

CENTROIDS_NPY = PROC / "centroids.npy"
GENES_CSV = PROC / "genes.csv"
MARKERS_TXT = PROC / "marker_genes.txt"
CLUSTER_META_CSV = PROC / "cluster_metadata.csv"

CLAS, SUBC, SUPT, CLUS = (
    "CCN20230722_CLAS", "CCN20230722_SUBC", "CCN20230722_SUPT", "CCN20230722_CLUS",
)

N_CLUSTERS = 5322
N_CLASSES = 34
N_GENES = 32285
N_MARKERS = 6558
TOTAL_CELLS = 4_042_976


def load_h5(h5_path: Path) -> tuple[np.ndarray, dict[str, int], list[str], dict]:
    """Return (centroids float32, cluster_id -> row, gene ids, taxonomy tree)."""
    with h5py.File(h5_path, "r") as f:
        centroids = np.asarray(f["sum"][()], dtype=np.float32)   # mean log2(CPM+1)
        cluster_to_row = json.loads(np.asarray(f["cluster_to_row"][()]).item())
        gene_ids = json.loads(np.asarray(f["col_names"][()]).item())
        taxonomy = json.loads(np.asarray(f["taxonomy_tree"][()]).item())
    gene_ids = list(gene_ids.keys()) if isinstance(gene_ids, dict) else list(gene_ids)
    return centroids, cluster_to_row, gene_ids, taxonomy


def walk_taxonomy(
    tax: dict,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
    """Walk CLAS -> SUBC -> SUPT -> CLUS.

    Returns (lineage, names, aliases) where lineage[cluster_id] holds the
    class/subclass/supertype ids, names[level][term_id] is the display name, and
    aliases[cluster_id] is the integer cluster_alias (the only sound key into
    cluster.csv; see the module docstring).
    """
    lineage: dict[str, dict[str, str]] = {}
    for class_id, subclasses in tax[CLAS].items():
        for subclass_id in subclasses:
            for supertype_id in tax[SUBC][subclass_id]:
                for cluster_id in tax[SUPT][supertype_id]:
                    if cluster_id in lineage:
                        raise AssertionError(
                            f"cluster {cluster_id} reached from >1 lineage: "
                            f"{lineage[cluster_id]} and class {class_id}"
                        )
                    lineage[cluster_id] = {
                        "class_id": class_id,
                        "subclass_id": subclass_id,
                        "supertype_id": supertype_id,
                    }
    names = {
        level: {tid: entry["name"] for tid, entry in tax["name_mapper"][level].items()}
        for level in (CLAS, SUBC, SUPT, CLUS)
    }
    aliases = {
        cid: int(entry["alias"])
        for cid, entry in tax["name_mapper"][CLUS].items()
    }
    assert len(set(aliases.values())) == len(aliases), (
        "cluster aliases are not unique in the h5 taxonomy; the cluster.csv join "
        "would be ambiguous"
    )
    return lineage, names, aliases


def marker_union(gene_ids: list[str]) -> list[str]:
    """Union of MapMyCells marker genes over taxonomy nodes, in h5 column order.

    The JSON holds one list of Ensembl ids per taxonomy node plus a single
    non-list "metadata" entry (a provenance dict), which is skipped.
    """
    raw = json.loads(MARKERS_JSON.read_text())
    nodes = {k: v for k, v in raw.items() if isinstance(v, list)}
    union: set[str] = set()
    for genes in nodes.values():
        union.update(genes)
    in_h5 = [g for g in gene_ids if g in union]  # h5 column order, deterministic
    missing = union - set(gene_ids)
    assert not missing, (
        f"{len(missing)} marker genes are absent from the h5 gene columns: "
        f"{sorted(missing)[:5]}"
    )
    print(f"[markers] {len(nodes)} taxonomy nodes -> union {len(union)} genes, "
          f"{len(in_h5)} present in the h5 columns")
    return in_h5


def build_cluster_metadata(
    cluster_to_row: dict[str, int],
    lineage: dict[str, dict[str, str]],
    names: dict[str, dict[str, str]],
    aliases: dict[str, int],
) -> pd.DataFrame:
    """One row per cluster, in centroids.npy row order, fully annotated."""
    assert set(cluster_to_row) == set(lineage), (
        "cluster id sets differ between the h5 'sum' rows and the h5 taxonomy tree: "
        f"{len(set(cluster_to_row) ^ set(lineage))} ids do not match"
    )

    # Allen's official class palette. The 34-class taxonomy is the one case the
    # repo's shared palettes do not cover, so the published hexes are used verbatim.
    terms = pd.read_csv(TERM_CSV)
    classes = terms[terms["cluster_annotation_term_set_label"] == CLAS]
    assert len(classes) == N_CLASSES, (
        f"expected {N_CLASSES} class terms in {TERM_CSV.name}, got {len(classes)}"
    )
    class_color = classes.set_index("label")["color_hex_triplet"].to_dict()
    class_name = classes.set_index("label")["name"].to_dict()

    tax_class_ids = {lin["class_id"] for lin in lineage.values()}
    assert tax_class_ids == set(class_color), (
        "class ids in the h5 taxonomy do not match the class labels in "
        f"{TERM_CSV.name}: symmetric difference "
        f"{sorted(tax_class_ids ^ set(class_color))}"
    )
    no_color = [cid for cid, hexv in class_color.items()
                if not isinstance(hexv, str) or not hexv.startswith("#")]
    assert not no_color, f"classes without a usable color hex: {no_color}"

    rows = []
    for cluster_id, row_index in cluster_to_row.items():
        lin = lineage[cluster_id]
        rows.append({
            "row_index": row_index,
            "cluster_id": cluster_id,
            "cluster_alias": aliases[cluster_id],
            "cluster_name": names[CLUS][cluster_id],
            "supertype_name": names[SUPT][lin["supertype_id"]],
            "subclass_name": names[SUBC][lin["subclass_id"]],
            "class_id": lin["class_id"],
            "class_name": class_name[lin["class_id"]],
            "class_color": class_color[lin["class_id"]],
        })
    meta = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    assert meta["row_index"].tolist() == list(range(N_CLUSTERS)), (
        "cluster_to_row is not a permutation of 0..n-1; centroids.npy row order "
        "cannot be reconstructed"
    )

    # Join cell counts on cluster_alias. NOT on cluster.csv's `label`: see the module
    # docstring, that string transform mis-joins ~5,030 of 5,322 clusters in silence.
    counts = pd.read_csv(CLUSTER_CSV)
    merged = meta.merge(
        counts[["cluster_alias", "number_of_cells"]],
        on="cluster_alias", how="left", validate="one_to_one",
    )
    unmatched = merged["number_of_cells"].isna()
    assert not unmatched.any(), (
        f"cluster.csv cell-count join is NOT total: {int(unmatched.sum())} clusters "
        f"unmatched, e.g. {merged.loc[unmatched, 'cluster_id'].head(5).tolist()}"
    )
    leftover = set(counts["cluster_alias"]) - set(meta["cluster_alias"])
    assert not leftover, (
        f"{len(leftover)} cluster.csv aliases have no h5 row, e.g. {sorted(leftover)[:5]}"
    )

    # Independent confirmation that the alias join lands on the RIGHT cluster: the
    # cluster names Allen publishes per alias must equal the h5 taxonomy's names.
    membership = pd.read_csv(MEMBERSHIP_CSV)[["cluster_alias", "cluster", "class"]]
    check = merged.merge(membership, on="cluster_alias", how="left", validate="one_to_one")
    bad_name = check["cluster_name"] != check["cluster"]
    assert not bad_name.any(), (
        f"alias join is total but WRONG: {int(bad_name.sum())} clusters disagree with "
        f"membership_pivoted.csv, e.g. h5 '{check.loc[bad_name, 'cluster_name'].iloc[0]}' "
        f"vs Allen '{check.loc[bad_name, 'cluster'].iloc[0]}'"
    )
    bad_class = check["class_name"] != check["class"]
    assert not bad_class.any(), (
        f"{int(bad_class.sum())} clusters land in a different class than "
        "membership_pivoted.csv assigns them"
    )
    print(f"[join] cell counts matched on cluster_alias for all {len(merged):,} clusters; "
          f"names and classes agree with membership_pivoted.csv (0 mismatches)")

    merged["n_cells"] = merged["number_of_cells"].astype(np.int64)
    return merged[[
        "row_index", "cluster_id", "cluster_name", "supertype_name", "subclass_name",
        "class_id", "class_name", "class_color", "n_cells",
    ]]


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    h5_path = resolve_large_input("wmb_precomputed_stats.h5")
    gene_csv = resolve_large_input("wmb_gene.csv")

    centroids, cluster_to_row, gene_ids, taxonomy = load_h5(h5_path)
    assert centroids.shape == (N_CLUSTERS, N_GENES), (
        f"expected a ({N_CLUSTERS}, {N_GENES}) centroid matrix, got {centroids.shape}"
    )
    assert len(cluster_to_row) == N_CLUSTERS, (
        f"expected {N_CLUSTERS} clusters, got {len(cluster_to_row)}"
    )
    assert len(gene_ids) == N_GENES, f"expected {N_GENES} genes, got {len(gene_ids)}"
    assert np.isfinite(centroids).all(), "centroid matrix contains NaN or inf"

    lineage, names, aliases = walk_taxonomy(taxonomy)
    assert len(taxonomy[CLAS]) == N_CLASSES, (
        f"expected {N_CLASSES} classes in the h5 taxonomy, got {len(taxonomy[CLAS])}"
    )
    assert len(lineage) == N_CLUSTERS, (
        f"taxonomy walk reached {len(lineage)} clusters, expected {N_CLUSTERS} "
        "(every cluster must map to exactly one class)"
    )

    meta = build_cluster_metadata(cluster_to_row, lineage, names, aliases)
    assert len(meta) == N_CLUSTERS, f"cluster metadata has {len(meta)} rows"
    assert meta["class_id"].nunique() == N_CLASSES, (
        f"cluster metadata spans {meta['class_id'].nunique()} classes, "
        f"expected {N_CLASSES}"
    )
    total_cells = int(meta["n_cells"].sum())
    assert total_cells == TOTAL_CELLS, (
        f"cell counts sum to {total_cells:,}, expected {TOTAL_CELLS:,}"
    )

    id2sym = pd.read_csv(gene_csv).set_index("gene_identifier")["gene_symbol"].to_dict()
    symbols = [id2sym.get(g) for g in gene_ids]
    n_missing_sym = sum(1 for s in symbols if not isinstance(s, str) or not s)
    symbols = [s if isinstance(s, str) and s else g for s, g in zip(symbols, gene_ids)]
    genes = pd.DataFrame({"ensembl_id": gene_ids, "gene_symbol": symbols})

    markers = marker_union(gene_ids)
    assert len(markers) == N_MARKERS, (
        f"marker union intersected with the h5 columns is {len(markers)} genes, "
        f"expected {N_MARKERS}"
    )

    np.save(CENTROIDS_NPY, centroids)
    genes.to_csv(GENES_CSV, index=False)
    MARKERS_TXT.write_text("\n".join(markers) + "\n")
    meta.to_csv(CLUSTER_META_CSV, index=False)

    top = (meta.groupby(["class_id", "class_name"], as_index=False)["n_cells"].sum()
              .sort_values("n_cells", ascending=False).head(5))

    print()
    print("=" * 72)
    print("WMB-10X centroid build: ALL ASSERTIONS PASSED")
    print("=" * 72)
    print(f"  clusters (points)   : {N_CLUSTERS:,}   (cell-type centroids, not cells)")
    print(f"  classes             : {meta['class_id'].nunique()}")
    print(f"  subclasses          : {meta['subclass_name'].nunique():,}")
    print(f"  supertypes          : {meta['supertype_name'].nunique():,}")
    print(f"  genes               : {N_GENES:,}   "
          f"({n_missing_sym} without a symbol, fell back to the Ensembl id)")
    print(f"  marker genes        : {len(markers):,}")
    print(f"  cells represented   : {total_cells:,}")
    print(f"  centroids.npy       : {centroids.dtype}, "
          f"{CENTROIDS_NPY.stat().st_size / 1e6:.0f} MB")
    print(f"  source h5           : {h5_path}")
    print(f"  source gene table   : {gene_csv}")
    print("  5 largest classes by cell count:")
    for _, r in top.iterrows():
        print(f"    {r['class_name']:<28} {r['n_cells']:>9,} cells")
    print("  wrote:")
    for path in (CENTROIDS_NPY, GENES_CSV, MARKERS_TXT, CLUSTER_META_CSV):
        print(f"    {path}")


if __name__ == "__main__":
    main()
