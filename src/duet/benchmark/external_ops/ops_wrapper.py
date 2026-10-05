"""Thin CLI that runs inside the Feldman&Funk (OPS) environment.

It reads two CSV inputs:
  - df_sgRNAs: columns ['Gene' (gene_id), 'Sequence' (sgRNA), 'Rank']
  - df_genes: columns ['gene_id', 'sgRNAs_per_gene', 'prefix_length', 'edit_distance', 'design']

and writes a JSON with the global indices of the chosen guides to --out:
  { "selected_indices": [ ... ] }

We rely on `ops.pool_design.select_prefix_group`, with one patch (below) so that
edit distance 2 runs on pools of any size.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


import ops.pool_design as pool
from ops.constants import SUBPOOL, GENE_ID, RANK, SGRNA


# _select_prefixes_edit_distance_any_size below is a modified copy of
# select_prefixes_edit_distance from OpticalPooledScreens
# (https://github.com/feldman4/OpticalPooledScreens, ops/pool_design.py at
# commit 5417bf4), used under the MIT License:
#
#   Copyright 2021 David Feldman, Luke Funk
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the
#   "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to permit
#   persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#   USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modifications for DUET: the function is renamed; the branch for more than
# 80,000 sequences is removed, so the <=80,000 branch runs at every size; the
# upstream module's helpers (build_khash, sparse_dist_parallel,
# distance_prefix, sparse_view, maxy_clique_groups) are called through the
# imported ops.pool_design module; whitespace is tidied. The note below
# explains why.
#
# Patch: run Feldman's <=80k edit-distance code path at any pool size.
#
# `pool.select_prefixes_edit_distance` (ED=2 only; ED=1 never calls it) takes a
# different branch above 80,000 sequences. That branch looks for a precomputed
# distance file under the authors' own project (`design/pool2/...pkl`); without
# it, it stores the hash buckets on the module and stops at `assert False`, so
# its own distance computation can never run. Since upstream commit bf86654 it
# even raises NameError first (`ops` is no longer imported there). It has been
# this way since the first public commit (be64e3c); no upstream branch or fork
# lacks it. The <=80k branch has no size dependence: it hashes the prefixes,
# computes the sparse edit distances in parallel and runs the same greedy
# (`maxy_clique_groups`). The function below is the upstream function
# (ops/pool_design.py, master 5417bf4) with the >80k branch removed, i.e. the
# <=80k path at every size. For pools of at most 80,000 sequences it is the code
# upstream runs, so their selections are unchanged. The OPS clone itself is not
# edited (several conda envs import it). Genome-wide (305,202 CRISPick
# sequences) it takes about 25 min and under 4 GB with the default 4 workers.
def _select_prefixes_edit_distance_any_size(sequences, group_ids, prefix_length,
                                            min_distance):
    if min_distance != 2:
        msg = 'prefix distance only correct for single edits'
        raise NotImplementedError(msg)

    # remove duplicate prefixes immediately
    prefix_series = (pd.Series(list(sequences))
        .str[:prefix_length + 1]
        .drop_duplicates())
    index_map = np.array(prefix_series.index)
    prefixes = list(prefix_series)

    group_ids = np.array(group_ids)[index_map]
    print(len(sequences))
    hash_buckets = pool.build_khash(prefixes, min_distance)
    print('hashed {} prefixes into {} buckets'
        .format(len(prefixes), len(hash_buckets)))
    D = pool.sparse_dist_parallel(hash_buckets, threshold=min_distance,
            distance=pool.distance_prefix)

    cm = pool.sparse_view(prefixes, D)
    print('built approximate distance matrix:', cm.shape)
    index = pool.maxy_clique_groups(cm, group_ids, verbose=True)
    print('selected sgRNAs:', len(index_map[index]))

    return list(index_map[index])


def patch_ops_pool_design():
    """Route `select_guides` (ED=2) through the any-size function above."""
    pool.select_prefixes_edit_distance = _select_prefixes_edit_distance_any_size

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--df-genes', type=Path, required=True)
    p.add_argument('--df-sgrnas', type=Path, required=True)
    p.add_argument('--seq-rounds', type=int, required=True, help='prefix length / sequence length to enforce')
    p.add_argument('--out', type=Path, required=True, help='path to write JSON payload')
    return p.parse_args()


def main():
    args = parse_args()
    patch_ops_pool_design()
    df_genes = pd.read_csv(args.df_genes)
    df_sg = pd.read_csv(args.df_sgrnas)

    # Rename to match OPS expectations (gene_id, sgRNA, rank) + subpool column
    df_sg = df_sg.rename(columns={'Gene': GENE_ID, 'Sequence': SGRNA, 'Rank': RANK})
    df_sg[SUBPOOL] = 'subpool'

    # The merge below maps selected rows back to input rows on these keys; a
    # duplicate key would map one selected row to two input rows.
    merge_cols = [GENE_ID, SGRNA, RANK]
    n_dup = int(df_sg.duplicated(merge_cols, keep=False).sum())
    if n_dup:
        raise ValueError(
            '{} input rows share a (gene_id, sgRNA, rank) key; the merge-back '
            'needs unique keys'.format(n_dup))

    # Ensure subpool column exists (single group by prefix length for this run)
    if 'group' not in df_genes.columns:
        df_genes['group'] = 'group'

    # The OpticalPooledScreens example notebook uses
    # df_genes.groupby('group').apply(select_prefix_group, df_sgRNAs), so we will too
    selected = (
        df_genes
        .groupby('group')
        .apply(pool.select_prefix_group, df_sg)
        .reset_index(drop=True)
    )

    # Selected table should include rows with original df_sg entries. Recover indices by merge.
    merged = (
        selected.merge(
            df_sg.reset_index().rename(columns={'index':'_orig_idx'}),
            on=merge_cols, how='left', validate='many_to_one'
        )
    )
    idx = merged['_orig_idx'].dropna().astype(int).tolist()
    if len(idx) != len(selected) or len(set(idx)) != len(idx):
        raise ValueError(
            'merge-back mismatch: {} selected rows gave {} indices ({} unique)'
            .format(len(selected), len(idx), len(set(idx))))

    # Write JSON payload to the requested file
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        json.dump({'selected_indices': idx}, f)

if __name__ == '__main__':
    main()
