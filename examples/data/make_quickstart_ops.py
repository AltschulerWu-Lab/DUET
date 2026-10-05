"""Rebuild examples/data/quickstart_ops.csv from the repository's hCRISPRi-v2.1 table.

Run from the repository root: python examples/data/make_quickstart_ops.py
See examples/data/README.md for the source, license and selection rule.
"""

import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/Horlbeck_2016/CRISPRi_v2_1.csv", low_memory=False)
tgt = df[(df.Gene != "negative_control") & (df.Rank <= 10)]
ok = tgt.groupby("Gene", sort=False)["Activity score"].agg(
    lambda s: len(s) == 10 and s.between(0, 1).all()
)
eligible = ok[ok].index.to_numpy()  # first-occurrence (table) order
rng = np.random.default_rng(0)
genes = set(rng.choice(eligible, size=50, replace=False))
sub = tgt[tgt.Gene.isin(genes)]  # keeps the table's row order
ctrl = df[df.Gene == "negative_control"].iloc[:50]
out = pd.concat([sub, ctrl])
out = pd.DataFrame({
    "sgID": out["sgID"],
    "gene": out["Gene"],
    "gene_name": out["Gene name"],
    "sequence": out["Sequence"],
    "activity": out["Activity score"].fillna(1.0),  # controls: 1.0, as in the paper's pool
})
out.to_csv("examples/data/quickstart_ops.csv", index=False)
