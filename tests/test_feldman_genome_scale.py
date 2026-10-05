"""Feldman et al. at genome scale: no size guard, per-group control quotas.

The unit tests need no OPS install. The tests marked with the ``ops_env``
fixture run the real OPS wrapper in the ``ops`` conda env (Feldman et al.'s
``ops`` package, Python 3.7) and are skipped where that env is missing.
"""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from duet.crispick_factory import CRISPickFactory
from duet.ops_benchmark.baselines import run_feldman_for_eds
from duet.ops_benchmark.config import FeldmanConfig
from duet.ops_benchmark.runner import _is_valid_codebook

OPS_ENV = "ops"
WRAPPER_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "duet" / "benchmark" / "external_ops"
)


@pytest.fixture(scope="session")
def ops_env():
    """Skip unless `conda run -n ops` can import Feldman et al.'s package."""
    conda = shutil.which("conda")
    if conda is None:
        pytest.skip("conda is not on PATH")
    p = subprocess.run(
        [conda, "run", "-n", OPS_ENV, "python", "-c", "import ops.pool_design"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        pytest.skip(f"conda env '{OPS_ENV}' with the OPS package is not available")
    return conda


def test_pool_above_80k_sequences_is_not_skipped():
    """The old guard returned {} above 80,000 sequences, for ED=1 and ED=2."""
    source_df = pd.DataFrame({"Gene": ["G"], "Sequence": ["A" * 20], "Rank": [1]})
    quotas = {"G": 3, "NO_SITE": 100, "ONE_SITE_INTERGENIC": 900}
    pool = SimpleNamespace(
        pool_size=305_202, quotas=quotas, metadata={"source_dataframe": source_df},
    )
    with patch(
        "duet.ops_benchmark.baselines.run_feldman",
        return_value={1: np.arange(3), 2: np.arange(2)},
    ) as run:
        out = run_feldman_for_eds(
            pool, FeldmanConfig(edit_distances=[1, 2], conda_env="ops"),
            seq_rounds=14, quota=3, num_controls=0,
        )
    assert sorted(out) == [1, 2]
    run.assert_called_once()
    assert run.call_args.kwargs["group_quotas"] == quotas
    assert run.call_args.args[0] is source_df


def _random_seq(rng, n=20):
    return "".join(rng.choice(list("ACGT"), size=n))


def _tiny_crispick_csv(path: Path) -> Path:
    """40 genes x 5 candidates, NO_SITE x 2 and ONE_SITE_INTERGENIC x 6 controls.

    One NO_SITE control shares its 10-base prefix with GENE_00's rank-2
    candidate. NO_SITE has exactly its quota (2) of candidates, so it is filled
    at ED=1 only if the control wins that prefix tie. Feldman's ED=1 path sorts
    by rank, then by the group's candidate count, so the control wins at rank
    1 or 2 (NO_SITE has fewer candidates than GENE_00) and loses at NaN, which
    sorts last.
    """
    rng = np.random.default_rng(7)
    rows = []
    for g in range(40):
        for rank in range(1, 6):
            rows.append((f"GENE_{g:02d}", _random_seq(rng), rank, 1 + (rank - 1) // 2))
    tied = rows[1][1]  # GENE_00, rank 2
    # Same 10-base prefix; a different 11th base keeps the 11-mers distinct,
    # since ED=2 drops repeated (prefix_length + 1)-mers before it starts.
    base11 = "C" if tied[10] != "C" else "G"
    rows.append(("NO_SITE", tied[:10] + base11 + _random_seq(rng, 9), np.nan, np.nan))
    rows.append(("NO_SITE", _random_seq(rng), np.nan, np.nan))
    for _ in range(6):
        rows.append(("ONE_SITE_INTERGENIC", _random_seq(rng), np.nan, np.nan))
    df = pd.DataFrame(
        rows, columns=["Target Gene ID", "sgRNA Sequence", "Pick Order", "Picking Round"],
    )
    df.to_csv(path, index=False)
    return path


def test_crispick_pool_gets_valid_feldman_codebooks(tmp_path, ops_env):
    """End to end through the OPS env: both control groups filled at both EDs.

    CRISPick configs set num_controls: 0. Before the fix, both control groups
    were merged into one `negative_control` gene with quota num_controls, so
    Feldman picked no controls and neither codebook was valid.
    """
    factory = CRISPickFactory(
        csv_path=_tiny_crispick_csv(tmp_path / "crispick.csv"),
        seq_rounds=10, quota=2, min_rank=5,
        control_quotas={"NO_SITE": 2, "ONE_SITE_INTERGENIC": 3},
    )
    pool = factory.create()
    out = run_feldman_for_eds(
        pool, FeldmanConfig(edit_distances=[1, 2], conda_env=OPS_ENV),
        seq_rounds=10, quota=2, num_controls=0,
    )
    assert sorted(out) == [1, 2]
    groups = pool.to_dataframe()["Group"].to_numpy()
    for ed, idx in out.items():
        counts = pd.Series(groups[idx]).value_counts()
        assert counts.get("NO_SITE", 0) == 2, ed
        assert counts.get("ONE_SITE_INTERGENIC", 0) == 3, ed
        assert _is_valid_codebook(idx, pool), ed


_PATCH_CHECK = textwrap.dedent(r'''
    import inspect, json, sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, sys.argv[1])
    import ops.pool_design as pool
    import ops_wrapper

    # 60 genes x 6 candidates; every second candidate is one substitution
    # away (in the 15-base prefix) from the one before, so ED=2 has to drop some.
    rng = np.random.RandomState(0)
    seqs, genes = [], []
    for g in range(60):
        for k in range(6):
            if k % 2:
                s = list(seqs[-1])
                i = rng.randint(15)
                s[i] = [b for b in 'ACGT' if b != s[i]][rng.randint(3)]
                s = ''.join(s)
            else:
                s = ''.join(rng.choice(list('ACGT'), 20))
            seqs.append(s)
            genes.append('G%02d' % g)

    upstream = pool.select_prefixes_edit_distance
    src = inspect.getsource(upstream)
    assert src.count('80000') == 1, 'upstream function changed'
    scope = {}
    exec(src.replace('80000', '100'), pool.__dict__, scope)
    lowered = scope['select_prefixes_edit_distance']

    # 1. Above the (lowered) threshold, upstream's own branch halts.
    try:
        lowered(pd.Series(seqs), genes, 14, 2)
    except (NameError, AssertionError) as e:
        print('lowered upstream fails:', type(e).__name__)
    else:
        raise SystemExit('upstream >threshold branch did not fail')

    # 2. The patch is upstream's <=80k path: same selection on the same pool.
    expected = upstream(pd.Series(seqs), genes, 14, 2)
    got = ops_wrapper._select_prefixes_edit_distance_any_size(pd.Series(seqs), genes, 14, 2)
    assert list(got) == list(expected), (got, expected)
    assert len(expected) < len(seqs)

    # 3. The wrapper's entry point applies the patch itself. With the lowered
    #    upstream installed, main() still runs, and it returns the rows that
    #    unpatched upstream selects below 80k.
    df_sg = pd.DataFrame({'gene_id': genes, 'sgRNA': seqs,
                          'rank': [1 + i % 6 for i in range(len(seqs))],
                          'subpool': 'subpool'})
    df_genes = pd.DataFrame({'gene_id': sorted(set(genes)), 'sgRNAs_per_gene': 2,
                             'prefix_length': 14, 'edit_distance': 2, 'group': 'group'})
    reference = (df_genes.groupby('group').apply(pool.select_prefix_group, df_sg)
                 .reset_index(drop=True))
    row_of = df_sg.reset_index().set_index(['gene_id', 'sgRNA', 'rank'])['index']
    expected_idx = sorted(row_of.loc[list(zip(
        reference['gene_id'], reference['sgRNA'], reference['rank']))].tolist())

    (df_sg.drop(columns='subpool')
     .rename(columns={'gene_id': 'Gene', 'sgRNA': 'Sequence', 'rank': 'Rank'})
     .to_csv('sg.csv', index=False))
    df_genes.drop(columns='group').assign(design='X').to_csv('genes.csv', index=False)
    pool.select_prefixes_edit_distance = lowered
    sys.argv = ['ops_wrapper.py', '--df-genes', 'genes.csv', '--df-sgrnas', 'sg.csv',
                '--seq-rounds', '14', '--out', 'out.json']
    ops_wrapper.main()
    assert pool.select_prefixes_edit_distance is ops_wrapper._select_prefixes_edit_distance_any_size
    with open('out.json') as fh:
        got_idx = sorted(json.load(fh)['selected_indices'])
    assert got_idx == expected_idx, (len(got_idx), len(expected_idx))
    print('OK', len(expected_idx))
''')


def test_ops_wrapper_patch_runs_above_the_threshold(tmp_path, ops_env):
    """The wrapper's patch is upstream's <=80k path, at any size.

    Upstream's threshold (80,000) is lowered to 100 in a copy of its function,
    so a 360-sequence pool takes the >threshold branch. The wrapper's main()
    must still run on that pool, so this also fails if main() stops applying
    the patch.
    """
    script = tmp_path / "patch_check.py"
    script.write_text(_PATCH_CHECK)
    p = subprocess.run(
        [ops_env, "run", "-n", OPS_ENV, "python", str(script), str(WRAPPER_DIR)],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "lowered upstream fails" in p.stdout
    assert "OK" in p.stdout


def test_ops_wrapper_rejects_duplicate_keys(tmp_path, ops_env):
    """The wrapper refuses input whose (gene_id, sgRNA, rank) keys repeat."""
    pd.DataFrame({
        "Gene": ["A", "A", "B"],
        "Sequence": ["ACGTACGTACGTACGTACGT"] * 2 + ["TTGTACGTACGTACGTACGT"],
        "Rank": [1, 1, 1],
    }).to_csv(tmp_path / "sg.csv", index=False)
    pd.DataFrame({
        "design": "X", "gene_id": ["A", "B"], "sgRNAs_per_gene": 1,
        "prefix_length": 10, "edit_distance": 1,
    }).to_csv(tmp_path / "genes.csv", index=False)
    p = subprocess.run(
        [ops_env, "run", "-n", OPS_ENV, "python", str(WRAPPER_DIR / "ops_wrapper.py"),
         "--df-genes", str(tmp_path / "genes.csv"), "--df-sgrnas", str(tmp_path / "sg.csv"),
         "--seq-rounds", "10", "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True,
    )
    assert p.returncode != 0
    assert "share a (gene_id, sgRNA, rank) key" in p.stderr
    assert not (tmp_path / "out.json").exists()


def test_ops_wrapper_rejects_a_merge_back_mismatch(tmp_path, ops_env):
    """A gene listed twice in df_genes doubles its selected rows; refuse that."""
    pd.DataFrame({
        "Gene": ["A", "A", "B"],
        "Sequence": ["ACGTACGTACGTACGTACGT", "GGGTACGTACGTACGTACGT", "TTGTACGTACGTACGTACGT"],
        "Rank": [1, 2, 1],
    }).to_csv(tmp_path / "sg.csv", index=False)
    pd.DataFrame({
        "design": "X", "gene_id": ["A", "A", "B"], "sgRNAs_per_gene": 2,
        "prefix_length": 10, "edit_distance": 1,
    }).to_csv(tmp_path / "genes.csv", index=False)
    p = subprocess.run(
        [ops_env, "run", "-n", OPS_ENV, "python", str(WRAPPER_DIR / "ops_wrapper.py"),
         "--df-genes", str(tmp_path / "genes.csv"), "--df-sgrnas", str(tmp_path / "sg.csv"),
         "--seq-rounds", "10", "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True,
    )
    assert p.returncode != 0
    assert "merge-back mismatch" in p.stderr
    assert not (tmp_path / "out.json").exists()
