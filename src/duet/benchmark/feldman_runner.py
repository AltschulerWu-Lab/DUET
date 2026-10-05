from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
import json
import os
import subprocess
import shlex
import tempfile

import numpy as np
import pandas as pd

@dataclass
class FeldmanParams:
    # Where the separate OPS environment lives
    conda_env: Optional[str] = None  # if set, we'll use 'conda run -n <env>'
    # edit distances to test; Feldman et al. typically consider 1 or 2
    edit_distances: Tuple[int, ...] = (1, 2)


def _validate_feldman_input(df: pd.DataFrame, seq_rounds: int):
    seqs = df['Sequence'].values
    if not all(len(s) == seq_rounds for s in seqs):
        raise ValueError(f"All sequences must be of length {seq_rounds} for Feldman et al.")
    if len(seqs[0]) < 19:
        raise ValueError("sgRNA sequences are expected to be full-length (at least length 19) for Feldman et al.")
    if len(set(seqs)) != len(seqs):
        # output warning instead of error
        frac_dupes = 1 - len(set(seqs)) / len(seqs)
        print(
            f"Warning: Full-length input sequences contain duplicates ({frac_dupes * 100: .2f}%); "
            f"Feldman et al. with edit distance = 1 may provide unexpected results."
        )
    if len(seqs[0]) < seq_rounds:
        raise ValueError(f"All sequences must be at least length {seq_rounds} for Feldman et al.")


# The OPS wrapper maps Feldman's selected rows back to input rows by merging on
# these columns, so they must identify each input row uniquely.
_OPS_KEY_COLUMNS = ["Gene", "Sequence", "Rank"]


def _check_unique_ops_keys(in_df: pd.DataFrame) -> None:
    """Raise if two input rows share a (Gene, Sequence, Rank) key.

    The wrapper's merge-back would map one selected row to every input row
    with its key: a repeated candidate index, or an extra index OPS never
    selected that overfills the gene. NaN ranks count as equal, as they do in
    the merge.
    """
    dup = in_df.duplicated(_OPS_KEY_COLUMNS, keep=False)
    if dup.any():
        examples = in_df.loc[dup, _OPS_KEY_COLUMNS].head(6).to_dict("records")
        raise ValueError(
            f"Feldman et al. input has {int(dup.sum())} rows whose "
            f"(Gene, Sequence, Rank) key is not unique, e.g. {examples}. "
            "The OPS wrapper merges its selection back on these keys, so "
            "duplicates would become repeated or extra candidate indices."
        )


def run_feldman(
    guides_df: pd.DataFrame,
    seq_rounds: int,
    guides_per_gene: int,
    num_ntc: int,
    params: FeldmanParams,
    tmpdir: Optional[Path] = None,
    validate_input: bool = False,
    group_quotas: Optional[Mapping[str, int]] = None,
) -> Dict[int, List[int]]:
    """
    Call the external OPS wrapper to select sgRNAs per gene under different edit_distance
    constraints. We construct df_genes per the example notebook and treat non-targeting
    controls identically to genes.

    Per-gene quotas: with ``group_quotas`` (group name -> quota, e.g. a
    CandidatePool's ``quotas``), every gene in ``guides_df`` gets its own quota
    from the map, so several control groups (CRISPick's ``NO_SITE`` and
    ``ONE_SITE_INTERGENIC``) keep their separate quotas. Without it, the
    ``negative_control`` gene gets ``num_ntc`` and every other gene
    ``guides_per_gene``.
    Returns a mapping edit_distance -> selected global indices. (Feldman et al.)
    """
    if tmpdir is None:
        tmpdir_cm = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_cm.name)
    else:
        tmpdir_cm = None

    if validate_input:
        _validate_feldman_input(guides_df, seq_rounds)

    # Prepare minimal inputs for the wrapper
    in_df = guides_df[_OPS_KEY_COLUMNS].copy()
    _check_unique_ops_keys(in_df)

    genes = sorted(in_df["Gene"].unique())
    if group_quotas is not None:
        missing = [g for g in genes if g not in group_quotas]
        if missing:
            raise ValueError(
                f"group_quotas has no quota for {len(missing)} gene(s) in the "
                f"Feldman et al. input, e.g. {missing[:5]}"
            )

    in_df.to_csv(tmpdir / "df_sgRNAs_for_ops.csv", index=False)

    # Build df_genes: one row per gene; the 'nontargeting' design label is
    # informational (OPS never reads the design column).
    gene_rows = []
    for g in genes:
        if group_quotas is not None:
            n = group_quotas[g]
        elif g == "negative_control":
            n = num_ntc
        else:
            n = guides_per_gene
        gene_rows.append({
            "design": "nontargeting" if g == "negative_control" else "X",
            "gene_id": g,
            "sgRNAs_per_gene": int(n),
            "prefix_length": int(seq_rounds),
        })
    df_genes = pd.DataFrame(gene_rows)
    df_genes.to_csv(tmpdir / "df_genes_for_ops.csv", index=False)

    results: Dict[int, np.ndarray] = {}
    for ed in params.edit_distances:
        df_genes_ed = df_genes.copy()
        df_genes_ed["edit_distance"] = int(ed)
        df_genes_ed.to_csv(tmpdir / f"df_genes_ed{ed}.csv", index=False)

        out_json = tmpdir / f"ops_out_ed{ed}.json"

        # Build command; use conda run only if an env is specified
        cmd = []
        if params.conda_env:
            cmd += ["conda", "run", "-n", params.conda_env]
        cmd += [
            "python", str(Path(__file__).parent / "external_ops" / "ops_wrapper.py"),
            "--df-genes", str(tmpdir / f"df_genes_ed{ed}.csv"),
            "--df-sgrnas", str(tmpdir / "df_sgRNAs_for_ops.csv"),
            "--seq-rounds", str(seq_rounds),
            "--out", str(out_json),
        ]

        # Pin the child's str hashing: the OPS package's maxy_clique_groups
        # (ops/pool_design.py) picks ED=2 guides with set.pop() over gene-ID
        # strings, so the selection otherwise changes with PYTHONHASHSEED.
        p = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"OPS wrapper failed (edit_distance={ed}).\n"
                f"CMD: {' '.join(shlex.quote(c) for c in cmd)}\n"
                f"STDERR:\n{p.stderr}"
            )

        # Read JSON payload from file produced by the wrapper
        try:
            with out_json.open() as f:
                payload = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read/parse OPS output (edit_distance={ed}).\n"
                f"CMD: {' '.join(shlex.quote(c) for c in cmd)}\n"
                f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\n"
                f"Error: {e}"
            )

        results[ed] = np.array(payload["selected_indices"], dtype=int)

    if tmpdir_cm is not None:
        tmpdir_cm.cleanup()

    return results
