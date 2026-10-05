#!/usr/bin/env python
"""Step 4: matched-density optical-crowding evaluation of one panel's three codebooks
(DUET, Bostrom, MHD4) across the 34 WMB-10X major cell types.

run.sh calls it once per panel, with the <outdir>/<panel>/crowding_config.yaml that
make_panels.py writes. Parameterized copy of the original exploratory script of
the same name (not part of this repository):
  * codebooks, prior, n_trials, seed and outdir come from that config
    (paths resolved against its folder) instead of the vendored codebooks/;
  * the hard-coded fidelity anchor (0.831 / 0.778 / 0.776) is replaced by the
    config's `anchor` block, checked AFTER per_class_metrics.csv is written:
    each whole-brain reference must match the source run's metrics.csv within
    `tol`, and each codebook must equal that method's rows in the source run's
    results.csv;
  * an optional `classes` list restricts the per-class evaluation (smoke runs).
The simulation (SIM_KW, CrowdingConfig, evaluate_crowding_simulation, matched
density via duet.merfish_benchmark.cell_type_crowding) is unchanged.

    python evaluate_crowding_by_cell_type.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from duet.merfish_benchmark.config import CrowdingConfig
from duet.merfish_benchmark.evaluation import evaluate_crowding_simulation
from duet.merfish_benchmark.cell_type_crowding import (
    load_codebook, assert_same_panel, run_evaluations, REFERENCE_LABEL,
)

SIM_KW = dict(cell_size_um=12.0, wavelength_nm=500.0, numerical_aperture=1.4,
              expansion_factor=1.0, diffraction_model="abbe", neighborhood="box")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    base = args.config.resolve().parent
    OUT = (base / cfg["outdir"]).resolve()
    n_trials, seed = int(cfg["n_trials"]), int(cfg["seed"])

    OUT.mkdir(parents=True, exist_ok=True)
    codebooks = {name: load_codebook(base / path) for name, path in cfg["codebooks"].items()}
    panel = assert_same_panel(codebooks)
    prior = pd.read_csv(base / cfg["prior_csv"])
    wide = pd.read_csv(OUT / "per_class_cpm_wide.csv")
    summary = pd.read_csv(OUT / "class_summary.csv").set_index("class_name")
    if cfg.get("classes"):
        missing = set(cfg["classes"]) - set(wide.columns)
        if missing:
            raise ValueError(f"unknown classes in config: {sorted(missing)}")
        wide = wide[["gene_symbol"] + list(cfg["classes"])]

    def eval_fn(cb, expr_df, total_reads):
        ce = evaluate_crowding_simulation(
            sequences=cb["Sequence"].tolist(), gene_names=cb["Gene"].tolist(),
            full_expr_df=expr_df,
            sim_config=CrowdingConfig(total_reads=int(total_reads), n_trials=n_trials, **SIM_KW),
            expression_col="mean_cpm", gene_col="gene_symbol", seed=seed)
        return ce.mean_identified_fraction, ce.std_identified_fraction

    rows = run_evaluations(codebooks, wide, panel, prior, summary,
                           n_trials=n_trials, seed=seed, eval_fn=eval_fn)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_class_metrics.csv", index=False)
    print(f"[ok] wrote {OUT / 'per_class_metrics.csv'} ({len(df)} rows)")

    # Fidelity anchor: whole-brain reference reproduces the source run's own numbers.
    ref = df[df.class_name == REFERENCE_LABEL].set_index("method")["mean_identified_fraction"]
    anchor = cfg.get("anchor")
    if not anchor:
        print("[skip] no fidelity anchor configured")
        return
    metrics_csv = base / anchor["metrics_csv"]
    m = pd.read_csv(metrics_csv).set_index("Method")["mean_identified_fraction"]
    # The source run's results.csv (beside metrics.csv) holds each method's codewords;
    # checking them catches a codebook/method mismatch that the identified fractions,
    # e.g. 0.8339 vs 0.8334 for adjacent DUET lambdas, are too close to reveal.
    results = pd.read_csv(metrics_csv.parent / "results.csv", dtype=str)
    tol, failed = float(anchor["tol"]), []
    for name, source_method in anchor["methods"].items():
        got, exp = float(ref[name]), float(m[source_method])
        src = results.loc[results["Method"] == source_method, ["Gene", "Sequence"]]
        same_cb = (set(map(tuple, src.values))
                   == set(map(tuple, codebooks[name][["Gene", "Sequence"]].values)))
        ok = abs(got - exp) < tol and same_cb
        print(f"[{'ok' if ok else 'FAIL'}] {name}: whole-brain {got:.4f} vs "
              f"{source_method} {exp:.4f} (tol {tol:g}); "
              f"codebook {'matches' if same_cb else 'DIFFERS FROM'} its results.csv rows")
        failed += [] if ok else [name]
    if failed:
        sys.exit(f"fidelity anchor failed for {failed}")


if __name__ == "__main__":
    main()
