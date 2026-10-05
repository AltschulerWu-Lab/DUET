#!/usr/bin/env python
"""Step 2: build every panel's gene set, baseline panels and per-panel configs.

For each entry of the config's `panels` list, writes into <outdir>/<id>/:
  genes.csv                 the panel's 2,000 genes: a uniform draw with its gene_seed
                            over the non-zero, non-ERCC genes of prior_csv
  bostrom_hw5_panel.csv     make_bostrom_panel.py (expression-ordered assignment)
  mhd4_hw5_panel.csv        make_bostrom_panel.py --shuffle-assignment (mhd4_shuffle_seed)
  run_config.yaml           duet_template with the per-panel overrides (see config.yaml)
  crowding_config.yaml      config for evaluate_crowding_by_cell_type.py (step 4)
  crowding/                 its outdir, with step 1's per-class tables symlinked in

The baseline panels are built with the same make_bostrom_panel.py calls as
experiments/merfish_2000_genes/make_panels.sh. prior_csv must be the template's
expression file, as make_bostrom_panel.py requires; this is checked.

    python make_panels.py --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
MAKE_PANEL = HERE.parents[1] / "scripts" / "benchmark" / "make_bostrom_panel.py"
BOSTROM = "Bostrom et al. (Hamming weight 5)"
MHD4 = "MERFISH MHD4 (Hamming weight 5)"


def draw_genes(expr, gene_col, expr_col, n, seed):
    """Uniform draw of n genes without replacement over the non-zero, non-ERCC genes."""
    pool = expr[(expr[expr_col] > 0) & ~expr[gene_col].astype(str).str.startswith("ERCC-")]
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return pool.iloc[np.sort(idx)][gene_col].astype(str).tolist()


def build_panel(out, genes_csv, cfg, gene_col, expr_col, shuffle):
    cmd = [sys.executable, str(MAKE_PANEL), "--codebook", str(cfg["bostrom_codebook"]),
           "--expression", str(cfg["prior_csv"]), "--gene-col", gene_col,
           "--expression-col", expr_col, "--genes-col", "Gene",
           "--genes-csv", str(genes_csv), "--out", str(out)]
    if shuffle:
        cmd += ["--shuffle-assignment", "--shuffle-seed", str(cfg["mhd4_shuffle_seed"])]
    subprocess.run(cmd, check=True)


def run_config(template, template_dir, pdir, bostrom, mhd4, gpu, cfg):
    """duet_template with only the overrides listed in config.yaml."""
    t = copy.deepcopy(template)
    scratch = str(cfg["scratch_dir"] / pdir.name)
    t["outdir"] = str(pdir / "duet")
    t["cache_dir"] = str((template_dir / t["cache_dir"]).resolve())   # reuse the template's PEP
    t["expression"]["path"] = str((template_dir / t["expression"]["path"]).resolve())
    t["genes"]["path"] = str(bostrom)
    t["initialization"]["path"] = str(bostrom)
    names = [b["name"] for b in t["baselines"]]
    if names != [BOSTROM, MHD4]:
        raise ValueError(f"template baselines are {names}, expected {[BOSTROM, MHD4]}")
    t["baselines"][0]["path"], t["baselines"][1]["path"] = str(bostrom), str(mhd4)
    t["evaluator"].update(device=f"gpu:{gpu}", num_cpus=cfg["evaluator_num_cpus"],
                          scratch_dir=scratch)
    t["duet"].update({"device": f"gpu:{gpu}", "num_cpus": 1, "lambda": [cfg["lambda"]]})
    t["duet"]["pep"]["scratch_dir"] = scratch
    return t


def crowding_config(pdir, bostrom, mhd4, cfg):
    lam = f"{cfg['lambda']:.2f}"
    return {
        "outdir": str(pdir / "crowding"),
        "prior_csv": str(cfg["prior_csv"]),
        "codebooks": {"DUET": str(pdir / "duet" / f"selected_codewords_lambda{lam}.csv"),
                      BOSTROM: str(bostrom), MHD4: str(mhd4)},
        "n_trials": cfg["n_trials"], "seed": cfg["seed"], "classes": cfg["classes"],
        "anchor": {"metrics_csv": str(pdir / "duet" / "metrics.csv"),
                   "methods": {"DUET": f"DUET (lambda={lam})", BOSTROM: BOSTROM, MHD4: MHD4},
                   "tol": cfg["anchor_tol"]},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    base = args.config.resolve().parent
    cfg = yaml.safe_load(args.config.read_text())
    for k in ("outdir", "scratch_dir", "prior_csv", "bostrom_codebook", "duet_template"):
        cfg[k] = (base / cfg[k]).resolve()
    out = cfg["outdir"]

    template = yaml.safe_load(cfg["duet_template"].read_text())
    run_expr = (cfg["duet_template"].parent / template["expression"]["path"]).resolve()
    if run_expr != cfg["prior_csv"]:
        raise ValueError(f"prior_csv {cfg['prior_csv']} is not the template's expression file "
                         f"{run_expr}; make_bostrom_panel.py needs the two to be the same file")
    gene_col = template["expression"]["gene_col"]
    expr_col = template["expression"]["expression_col"]
    expr = pd.read_csv(cfg["prior_csv"])
    wide = pd.read_csv(out / "per_class_cpm_wide.csv", usecols=["gene_symbol"])
    class_genes = set(wide["gene_symbol"].astype(str))

    ids = [panel["id"] for panel in cfg["panels"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate panel ids: {ids}")
    seeds = [panel["gene_seed"] for panel in cfg["panels"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate gene seeds: {seeds}")
    gene_sets = {}
    for i, panel in enumerate(cfg["panels"]):
        pdir = out / panel["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        genes = draw_genes(expr, gene_col, expr_col, cfg["panel_size"], panel["gene_seed"])
        if len(set(genes)) != cfg["panel_size"]:
            raise ValueError(f"{panel['id']}: {len(set(genes))} distinct genes, "
                             f"expected {cfg['panel_size']}")
        absent = sorted(set(genes) - class_genes)
        if absent:
            raise ValueError(f"{panel['id']}: {len(absent)} genes missing from the per-class "
                             f"tables, e.g. {absent[:5]}")
        gene_sets[panel["id"]] = set(genes)
        pd.DataFrame({"Gene": genes}).to_csv(pdir / "genes.csv", index=False)

        bostrom, mhd4 = pdir / "bostrom_hw5_panel.csv", pdir / "mhd4_hw5_panel.csv"
        build_panel(bostrom, pdir / "genes.csv", cfg, gene_col, expr_col, shuffle=False)
        build_panel(mhd4, pdir / "genes.csv", cfg, gene_col, expr_col, shuffle=True)

        gpu = cfg["gpus"][i % len(cfg["gpus"])]
        with open(pdir / "run_config.yaml", "w") as f:
            yaml.safe_dump(run_config(template, cfg["duet_template"].parent, pdir, bostrom,
                                      mhd4, gpu, cfg), f, sort_keys=False, allow_unicode=True)
        with open(pdir / "crowding_config.yaml", "w") as f:
            yaml.safe_dump(crowding_config(pdir, bostrom, mhd4, cfg), f, sort_keys=False)
        (pdir / "crowding").mkdir(exist_ok=True)
        for name in ("per_class_cpm_wide.csv", "class_summary.csv"):
            link = pdir / "crowding" / name
            link.unlink(missing_ok=True)
            link.symlink_to(out / name)
        print(f"[ok] {panel['id']}: {len(genes)} genes, gpu:{gpu}, configs in {pdir}")

    overlaps = {(a, b): len(gene_sets[a] & gene_sets[b])
                for j, a in enumerate(ids) for b in ids[j + 1:]}
    if overlaps:
        print(f"gene overlap between panels: min {min(overlaps.values())}, "
              f"max {max(overlaps.values())} of {cfg['panel_size']}")


if __name__ == "__main__":
    main()
