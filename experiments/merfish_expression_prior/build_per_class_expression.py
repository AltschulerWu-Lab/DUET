#!/usr/bin/env python
"""Roll WMB-10X per-cluster stats up to per-CLASS genome-wide CPM profiles.

Parameterized copy of the original exploratory script
(build_wmb_per_class_expression.py, not part of this repository): the input and
output paths come from the experiment config (resolved against its folder)
instead of fixed repo locations, and the capture check reads the rebuilt k2000
panel. The roll-up itself is unchanged.

Self-contained: runs in scanpy_env (h5py + numpy + pandas + pyyaml; NOT duet). Writes,
under the config's outdir:
  per_class_cpm_wide.csv   gene_symbol + one linear-CPM column per class (34 classes)
  class_summary.csv        class_name, n_clusters
Per-class profiles are an equal-weight mean over the class's clusters (the h5's
n_cells is a uniform placeholder, so cell-abundance weighting is unavailable);
this matches the unweighted whole-brain prior. Then verifies the roll-up against it.

    python build_per_class_expression.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


def build_cluster_to_class(tax):
    """Walk CLAS->SUBC->SUPT->CLUS -> {cluster_id: class_id}, plus {class_id: name}."""
    clas, subc, supt, _clus = tax["hierarchy"]
    c2class = {}
    for clas_id, subclasses in tax[clas].items():
        for subc_id in subclasses:
            for supt_id in tax[subc][subc_id]:
                for clus_id in tax[supt][supt_id]:
                    c2class[clus_id] = clas_id
    names = {cid: tax["name_mapper"][clas][cid]["name"] for cid in tax[clas]}
    return c2class, names


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    base = args.config.resolve().parent
    H5 = base / cfg["abc_stats_h5"]
    GENE_CSV = base / cfg["abc_gene_csv"]
    PRIOR_CSV = base / cfg["prior_csv"]
    PANEL_CSV = base / cfg["panel_csv"]
    OUT = (base / cfg["outdir"]).resolve()

    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5, "r") as f:
        sum_log2 = np.asarray(f["sum"][()])                       # (clusters, genes) log2(CPM+1)
        cluster_to_row = json.loads(np.asarray(f["cluster_to_row"][()]).item())
        gene_ids = json.loads(np.asarray(f["col_names"][()]).item())
        gene_ids = list(gene_ids.keys()) if isinstance(gene_ids, dict) else list(gene_ids)
        tax = json.loads(np.asarray(f["taxonomy_tree"][()]).item())

    cpm = np.maximum(2.0 ** sum_log2 - 1.0, 0.0)                  # linear CPM per (cluster, gene)
    c2class, class_names = build_cluster_to_class(tax)

    assert set(cluster_to_row) == set(c2class), "cluster set mismatch: stats vs taxonomy"
    classes = sorted(set(c2class.values()))
    assert len(classes) == 34, f"expected 34 classes, got {len(classes)}"

    id2sym = pd.read_csv(GENE_CSV).set_index("gene_identifier")["gene_symbol"].to_dict()
    symbols = [id2sym.get(g, g) for g in gene_ids]

    wide = {"gene_symbol": symbols}
    summary = []
    for clas_id in classes:
        rows = [cluster_to_row[c] for c, cc in c2class.items() if cc == clas_id]
        # Equal-weight mean CPM over the class's transcriptomic clusters. The h5's
        # n_cells is a uniform placeholder (all 1) in this ABC means-freeze release,
        # so cell-abundance weighting is unavailable; this matches the unweighted
        # whole-brain prior built by wmb10x_brain_reference.py.
        prof = cpm[rows].mean(axis=0)
        wide[class_names[clas_id]] = prof
        summary.append({"class_name": class_names[clas_id], "n_clusters": len(rows)})
    pd.DataFrame(wide).to_csv(OUT / "per_class_cpm_wide.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "class_summary.csv", index=False)

    # SMOKE CHECK: whole-brain (all-cluster) mean capture ~ prior capture on the panel.
    panel = set(pd.read_csv(PANEL_CSV)["Gene"].astype(str))
    wb = cpm.mean(axis=0)
    wb_df = pd.DataFrame({"gene_symbol": symbols, "mean_cpm": wb})
    cap_wb = wb_df[wb_df.gene_symbol.isin(panel)].mean_cpm.sum() / wb_df.mean_cpm.sum()
    prior = pd.read_csv(PRIOR_CSV)
    cap_prior = prior[prior.gene_symbol.isin(panel)].mean_cpm.sum() / prior.mean_cpm.sum()
    rel = abs(cap_wb - cap_prior) / cap_prior
    print(f"[check] panel capture unweighted WB {cap_wb:.4f} vs prior {cap_prior:.4f} "
          f"(rel diff {rel*100:.1f}%)")
    assert rel < 0.25, "whole-brain capture diverges from prior beyond tolerance"
    print(f"[ok] wrote {len(classes)} class columns x {len(symbols)} genes to {OUT}")


if __name__ == "__main__":
    main()
