"""Tests for the Feldman et al. baseline runner (duet.benchmark.feldman_runner)."""
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from duet.benchmark.feldman_runner import FeldmanParams, run_feldman


def test_ops_wrapper_runs_with_fixed_hash_seed(tmp_path, monkeypatch):
    """The OPS wrapper subprocess gets PYTHONHASHSEED=0 on top of the parent env.

    The OPS package's maxy_clique_groups picks ED=2 guides with set.pop() over
    gene-ID strings, so an unpinned hash seed makes the selection vary per run.
    """
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    monkeypatch.setenv("FELDMAN_TEST_SENTINEL", "kept")
    guides = pd.DataFrame({
        "Gene": ["GENE_A", "GENE_A", "negative_control"],
        "Sequence": ["ACGTACGTACGTACGTACGT", "TTGTACGTACGTACGTACGT", "GGGTACGTACGTACGTACGT"],
        "Rank": [1, 2, 1],
    })
    child_envs = []

    def fake_run(cmd, **kwargs):
        child_envs.append(kwargs.get("env"))
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"selected_indices": [0, 2]}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("duet.benchmark.feldman_runner.subprocess.run", side_effect=fake_run):
        results = run_feldman(
            guides, seq_rounds=10, guides_per_gene=1, num_ntc=1,
            params=FeldmanParams(conda_env="ops", edit_distances=(1, 2)),
            tmpdir=tmp_path,
        )

    assert sorted(results) == [1, 2]
    assert len(child_envs) == 2
    for env in child_envs:
        assert env is not None
        assert env["PYTHONHASHSEED"] == "0"
        assert env["FELDMAN_TEST_SENTINEL"] == "kept"
        assert env["PATH"] == os.environ["PATH"]


def _capture_ops_inputs(tmp_path, guides, **kwargs):
    """Run run_feldman with a fake wrapper; return (results, df_genes, df_sgRNAs)."""
    def fake_run(cmd, **_):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"selected_indices": [0]}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("duet.benchmark.feldman_runner.subprocess.run", side_effect=fake_run):
        results = run_feldman(
            guides, seq_rounds=10, params=FeldmanParams(edit_distances=(1,)),
            tmpdir=tmp_path, **kwargs,
        )
    return (
        results,
        pd.read_csv(tmp_path / "df_genes_for_ops.csv"),
        pd.read_csv(tmp_path / "df_sgRNAs_for_ops.csv"),
    )


def _crispick_like_guides() -> pd.DataFrame:
    return pd.DataFrame({
        "Gene": ["GENE_A", "GENE_A", "NO_SITE", "ONE_SITE_INTERGENIC", "ONE_SITE_INTERGENIC"],
        "Sequence": [
            "ACGTACGTACGTACGTACGT", "TTGTACGTACGTACGTACGT", "GGGTACGTACGTACGTACGT",
            "CCGTACGTACGTACGTACGT", "CAGTACGTACGTACGTACGT",
        ],
        "Rank": [1, 2, 1, 1, 1],
    })


def test_group_quotas_give_each_control_group_its_own_quota(tmp_path):
    """CRISPick's two control groups keep separate quotas (not num_ntc = 0)."""
    quotas = {"GENE_A": 2, "NO_SITE": 1, "ONE_SITE_INTERGENIC": 2}
    _, df_genes, df_sg = _capture_ops_inputs(
        tmp_path, _crispick_like_guides(),
        guides_per_gene=3, num_ntc=0, group_quotas=quotas,
    )
    assert dict(zip(df_genes["gene_id"], df_genes["sgRNAs_per_gene"])) == quotas
    assert set(df_sg["Gene"]) == set(quotas)


def test_group_quotas_reproduce_the_negative_control_fallback(tmp_path):
    """Weissman path: the pool's quotas give byte-identical OPS inputs.

    A Weissman pool's quotas are `quota` per gene and `num_controls` for
    negative_control, which is what the fallback builds, so passing them
    changes nothing for the existing Weissman experiments.
    """
    guides = pd.DataFrame({
        "Gene": ["GENE_B", "GENE_A", "GENE_A", "negative_control", "negative_control"],
        "Sequence": [
            "ACGTACGTACGTACGTACGT", "TTGTACGTACGTACGTACGT", "GGGTACGTACGTACGTACGT",
            "CCGTACGTACGTACGTACGT", "CAGTACGTACGTACGTACGT",
        ],
        "Rank": [1, 1, 2, float("nan"), float("nan")],
    })
    old_dir, new_dir = tmp_path / "fallback", tmp_path / "quotas"
    old_dir.mkdir()
    new_dir.mkdir()
    _capture_ops_inputs(old_dir, guides, guides_per_gene=2, num_ntc=2)
    _capture_ops_inputs(
        new_dir, guides, guides_per_gene=2, num_ntc=2,
        group_quotas={"GENE_A": 2, "GENE_B": 2, "negative_control": 2},
    )
    for name in ("df_genes_for_ops.csv", "df_sgRNAs_for_ops.csv", "df_genes_ed1.csv"):
        assert (old_dir / name).read_bytes() == (new_dir / name).read_bytes()


def test_negative_control_fallback_without_group_quotas(tmp_path):
    guides = _crispick_like_guides().replace({"NO_SITE": "negative_control"})
    guides = guides[guides["Gene"] != "ONE_SITE_INTERGENIC"]
    _, df_genes, _ = _capture_ops_inputs(tmp_path, guides, guides_per_gene=2, num_ntc=5)
    got = dict(zip(df_genes["gene_id"], df_genes["sgRNAs_per_gene"]))
    assert got == {"GENE_A": 2, "negative_control": 5}
    assert df_genes.set_index("gene_id").loc["negative_control", "design"] == "nontargeting"


def test_group_quotas_missing_a_gene_raises(tmp_path):
    with patch("duet.benchmark.feldman_runner.subprocess.run") as run:
        with pytest.raises(ValueError, match="no quota"):
            run_feldman(
                _crispick_like_guides(), seq_rounds=10, guides_per_gene=2, num_ntc=0,
                params=FeldmanParams(edit_distances=(1,)), tmpdir=tmp_path,
                group_quotas={"GENE_A": 2, "NO_SITE": 1},
            )
    run.assert_not_called()


@pytest.mark.parametrize("rank", [3, float("nan")])
def test_duplicate_ops_keys_raise_before_the_wrapper_runs(tmp_path, rank):
    """A repeated (Gene, Sequence, Rank) key would become a repeated or extra index.

    When the CRISPick table was grouped by gene symbol (an earlier version), ASMTL and
    IL9R had such rows; the wrapper's merge-back then returned 10,002 indices
    for 9,998 unique rows. NaN ranks (controls) compare equal, as in the merge.
    """
    guides = pd.DataFrame({
        "Gene": ["GENE_A", "GENE_A", "GENE_A"],
        "Sequence": ["ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT", "TTGTACGTACGTACGTACGT"],
        "Rank": [rank, rank, 1],
    })
    with patch("duet.benchmark.feldman_runner.subprocess.run") as run:
        with pytest.raises(ValueError, match="not unique"):
            run_feldman(
                guides, seq_rounds=10, guides_per_gene=2, num_ntc=0,
                params=FeldmanParams(edit_distances=(1, 2)), tmpdir=tmp_path,
            )
    run.assert_not_called()


def test_same_sequence_in_two_genes_is_allowed(tmp_path):
    """Keys are per gene: a paralog's shared sgRNA is two distinct rows."""
    guides = pd.DataFrame({
        "Gene": ["GENE_A", "GENE_B"],
        "Sequence": ["ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT"],
        "Rank": [1, 1],
    })
    results, _, df_sg = _capture_ops_inputs(tmp_path, guides, guides_per_gene=1, num_ntc=0)
    assert len(df_sg) == 2 and sorted(results) == [1]
