#!/usr/bin/env python3
"""Re-run ONLY the optical-crowding simulation for an existing MERFISH result.

Use this to (a) backfill new crowding columns (``n_transcripts`` in results.csv,
``se_identified_fraction`` / ``se_decode_accuracy`` in metrics.csv) and/or
(b) re-evaluate crowding at a different number of trials (``--n-trials``) --
both WITHOUT the multi-hour PEP build + DUET optimization + decode evaluation.
Every codebook (DUET lambda or baseline) is read back from the persisted
``results.csv`` (its Gene + Sequence columns, in Index order) and re-simulated
with the same config + seed; only the crowding step re-runs.

More trials = more simulated cells = lower-variance per-gene identified fractions
(it does NOT change the crowding density -- unlike raising total_reads, which
packs more transcripts into the same cell and inflates crowding). When the trial
count matches the recorded run, per-codeword fractions reproduce exactly (gate
ON); at a different count they shift toward their true values (gate OFF).

The decode-accuracy columns are untouched (read from the existing CSVs and kept).

Usage:
    conda run -n <env> python scripts/benchmark/recompute_crowding.py --config path/to/config.yaml [--n-trials 1000]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from duet.merfish_benchmark import MerfishBenchmarkConfig
from duet.merfish_benchmark.evaluation import evaluate_crowding_simulation, load_expression


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="MERFISH YAML config (reads CSVs from its outdir).")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override crowding n_trials (default: the config's value). Re-evaluates at "
                        "more simulated cells without a config edit or PEP rebuild.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write updated CSVs (default: the config's outdir, in place).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = MerfishBenchmarkConfig.from_yaml(Path(args.config))
    outdir = cfg.outdir
    out = Path(args.output_dir) if args.output_dir else outdir
    out.mkdir(parents=True, exist_ok=True)

    n_trials = args.n_trials if args.n_trials is not None else cfg.crowding.n_trials
    crowding_cfg = dataclasses.replace(cfg.crowding, n_trials=n_trials)

    # summary.yaml records the trial count (for the exact-match gate) AND the seed
    # that actually produced results.csv -- the runner resolves a null config seed
    # to a random int at runtime and records it here (runner.py). Prefer the
    # recorded seed so reproduction is honest and we never pass None into
    # evaluate_crowding_simulation's `seed + trial` (which would crash).
    summary_path = outdir / "summary.yaml"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = yaml.safe_load(summary_path.read_text()) or {}
        except Exception:
            summary = {}
    stored_n_trials = summary.get("n_crowding_trials")
    seed = summary.get("seed")
    if seed is None:
        seed = cfg.seed
    if seed is None:
        raise SystemExit(
            "No usable crowding seed: config 'seed' is null and summary.yaml records no 'seed'. "
            "Cannot reproduce results.csv -- supply a config with an explicit seed."
        )
    is_reproduction = stored_n_trials == n_trials
    if is_reproduction:
        print(f"Reproducing crowding at n_trials={n_trials} (matches recorded run); exact-match gate ON.")
    else:
        print(f"Re-estimating crowding at n_trials={n_trials} (recorded run used {stored_n_trials}); "
              "per-codeword fractions shift toward true values, exact-match gate OFF.")

    # Sequence is a pure-digit binary string -> force str so leading zeros survive.
    results_df = pd.read_csv(outdir / "results.csv", dtype={"Sequence": str})
    metrics_df = pd.read_csv(outdir / "metrics.csv")
    full_expr_df = load_expression(cfg.expression)

    results_df["n_transcripts"] = np.nan
    # Crowding aggregates we recompute per method (keyed by Method in metrics.csv).
    agg: dict[str, dict] = {}

    # Preserve first-appearance method order; re-simulate each codebook.
    for method in results_df["Method"].drop_duplicates():
        sub = results_df[results_df["Method"] == method].sort_values("Index")
        seqs = sub["Sequence"].tolist()
        genes = sub["Gene"].astype(str).tolist()
        ce = evaluate_crowding_simulation(
            sequences=seqs, gene_names=genes, full_expr_df=full_expr_df,
            sim_config=crowding_cfg, expression_col=cfg.expression.expression_col,
            gene_col=cfg.expression.gene_col, seed=seed,
        )
        idx = sub.index  # rows in Index order
        recomputed = ce.per_codeword_identified_fraction
        if is_reproduction:
            # Exact reproduction expected -- catch seed/config drift.
            stored = sub["Identified fraction"].to_numpy()
            both = ~(np.isnan(stored) | np.isnan(recomputed))
            if both.any() and not np.allclose(stored[both], recomputed[both], atol=1e-9):
                md = float(np.max(np.abs(stored[both] - recomputed[both])))
                print(f"  WARNING [{method}]: recomputed identified fraction differs from stored "
                      f"(max abs diff {md:.2e}) -- seed/config mismatch?")
        results_df.loc[idx, "n_transcripts"] = ce.per_codeword_count
        results_df.loc[idx, "Identified fraction"] = recomputed
        n_codewords = len(seqs)
        agg[method] = {
            "mean_conflict_fraction": ce.mean_conflict_fraction,
            "std_conflict_fraction": ce.std_conflict_fraction,
            "mean_identified_fraction": ce.mean_identified_fraction,
            "std_identified_fraction": ce.std_identified_fraction,
            "se_identified_fraction": (
                ce.std_identified_fraction / np.sqrt(n_trials) if n_trials > 0 else float("nan")
            ),
            "n_codewords": n_codewords,
        }
        print(f"  {method:34s} n_codewords={n_codewords:5d}  "
              f"mean_IF={ce.mean_identified_fraction:.4f}  "
              f"median transcripts/codeword={int(np.median(ce.per_codeword_count))}")

    results_df["n_transcripts"] = results_df["n_transcripts"].astype("Int64")

    # Update metrics.csv crowding aggregates + SE columns (decode SE from std/sqrt(K)).
    for col in ("mean_conflict_fraction", "std_conflict_fraction",
                "mean_identified_fraction", "std_identified_fraction",
                "se_identified_fraction"):
        metrics_df[col] = metrics_df["Method"].map(lambda m: agg.get(m, {}).get(col, np.nan))
    metrics_df["se_decode_accuracy"] = metrics_df.apply(
        lambda r: r["Std decode accuracy"] / np.sqrt(agg[r["Method"]]["n_codewords"])
        if r["Method"] in agg and agg[r["Method"]]["n_codewords"] > 0 else np.nan,
        axis=1,
    )

    results_df.to_csv(out / "results.csv", index=False)
    metrics_df.to_csv(out / "metrics.csv", index=False)

    # Keep the run record's trial count in sync with what crowding was evaluated at.
    if summary_path.exists():
        try:
            summary = yaml.safe_load(summary_path.read_text()) or {}
            summary["n_crowding_trials"] = n_trials
            (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
        except Exception as e:
            print(f"  (could not update summary.yaml n_crowding_trials: {e})")

    print(f"\nUpdated results.csv (+ n_transcripts) and metrics.csv (+ se_* columns) "
          f"at n_trials={n_trials} in {out}")


if __name__ == "__main__":
    main()
