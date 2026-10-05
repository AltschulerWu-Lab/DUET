"""Per-cell-type crowding-robustness helpers (supplementary figure).

DUET-side pure logic only: codebook loading, panel validation, matched-density
calibration, and plot-frame assembly. The WMB h5 -> per-class CPM roll-up lives in
the self-contained scanpy_env script
experiments/merfish_expression_prior/build_per_class_expression.py (no h5py here,
so this module imports cleanly with the core DUET install).
"""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd


def load_codebook(path) -> pd.DataFrame:
    """Load a codebook CSV -> DataFrame[Gene, Sequence] (Sequence as str)."""
    df = pd.read_csv(path, dtype=str)
    missing = {"Gene", "Sequence"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: codebook missing columns {sorted(missing)} (has {list(df.columns)})"
        )
    out = df[["Gene", "Sequence"]].copy()
    return out.reset_index(drop=True)


def assert_same_panel(codebooks: Dict[str, pd.DataFrame]) -> List[str]:
    """Assert every codebook covers the identical gene set; return the sorted panel."""
    items = list(codebooks.items())
    ref_name, ref_df = items[0]
    ref = set(ref_df["Gene"])
    for name, df in items[1:]:
        g = set(df["Gene"])
        if g != ref:
            raise ValueError(
                f"panel mismatch: {name} has {len(g)} genes, {ref_name} has {len(ref)}; "
                f"symmetric difference {len(g ^ ref)}"
            )
    return sorted(ref)


def panel_capture(expr_df, panel_genes, gene_col="gene_symbol", expr_col="mean_cpm") -> float:
    """Fraction of total expression mass carried by the panel genes."""
    expr = expr_df[[gene_col, expr_col]].copy()
    expr[expr_col] = expr[expr_col].clip(lower=0.0)
    total = float(expr[expr_col].sum())
    if total <= 0:
        raise ValueError("total expression mass is zero; cannot compute capture")
    panel = set(map(str, panel_genes))
    on = float(expr[expr[gene_col].astype(str).isin(panel)][expr_col].sum())
    return on / total


def target_panel_spots(prior_df, panel_genes, base_total_reads=8700,
                       gene_col="gene_symbol", expr_col="mean_cpm") -> int:
    """T = round(base_total_reads * capture(prior)) — whole-brain panel-spot count."""
    cap = panel_capture(prior_df, panel_genes, gene_col, expr_col)
    return int(round(base_total_reads * cap))


def matched_total_reads(target_spots, capture, cap=200_000) -> int:
    """total_reads that yields ~target_spots panel spots at the given capture."""
    if capture <= 0:
        raise ValueError("capture must be > 0 to hold spot density constant")
    return int(min(round(target_spots / capture), cap))


REFERENCE_LABEL = "(whole-brain prior)"
BASE_TOTAL_READS = 8700


def run_evaluations(codebooks, wide, panel, prior, summary, n_trials, seed, eval_fn):
    """Assemble per-(method x class) crowding rows at matched density.

    eval_fn(cb_df, expr_df, total_reads) -> (mean_identified, std_identified). Injected
    so this driver is unit-testable without the heavy simulator. Reference rows evaluate
    each codebook on the whole-brain prior at BASE_TOTAL_READS.
    """
    T = target_panel_spots(prior, panel, BASE_TOTAL_READS)
    cap_prior = panel_capture(prior, panel)
    rows = []
    n_clusters_total = int(summary["n_clusters"].sum())
    for name, cb in codebooks.items():
        m, s = eval_fn(cb, prior, BASE_TOTAL_READS)
        rows.append(dict(method=name, class_name=REFERENCE_LABEL, capture=cap_prior,
                         total_reads=BASE_TOTAL_READS, mean_identified_fraction=m,
                         std_identified_fraction=s, n_clusters=n_clusters_total,
                         n_trials=n_trials))
    for clas in [c for c in wide.columns if c != "gene_symbol"]:
        expr = wide[["gene_symbol", clas]].rename(columns={clas: "mean_cpm"})
        cap = panel_capture(expr, panel)
        tr = matched_total_reads(T, cap)
        for name, cb in codebooks.items():
            m, s = eval_fn(cb, expr, tr)
            rows.append(dict(method=name, class_name=clas, capture=cap, total_reads=tr,
                             mean_identified_fraction=m, std_identified_fraction=s,
                             n_clusters=int(summary.loc[clas, "n_clusters"]), n_trials=n_trials))
    return rows


def load_plot_frame(metrics_df):
    """Split the metrics table into (identified, se, reference) for plotting.

    identified: DataFrame, index=class_name (reference rows dropped), columns=method,
        values = mean identified fraction.
    se: DataFrame of the same shape — standard error of the mean, std / sqrt(n_trials).
    reference: dict method -> whole-brain identified fraction (the optimization target,
        drawn from the REFERENCE_LABEL rows).
    """
    ref = (metrics_df[metrics_df["class_name"] == REFERENCE_LABEL]
           .set_index("method")["mean_identified_fraction"].to_dict())
    per = metrics_df[metrics_df["class_name"] != REFERENCE_LABEL]
    identified = per.pivot(index="class_name", columns="method",
                           values="mean_identified_fraction")
    std = per.pivot(index="class_name", columns="method", values="std_identified_fraction")
    n_trials = int(per["n_trials"].iloc[0])
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    se = std / math.sqrt(n_trials)
    return identified, se, ref


def duet_gap(identified, baseline_methods, duet_method="DUET"):
    """Per-class DUET-minus-best-baseline identified fraction (positive = DUET wins)."""
    best_base = identified[list(baseline_methods)].max(axis=1)
    return (identified[duet_method] - best_base).rename("duet_gap")
