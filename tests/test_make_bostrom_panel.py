# tests/test_make_bostrom_panel.py
"""End-to-end + warm-start-contract tests for the Boström panel CLI."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

from duet.bostrom_codebook import assign_codewords_to_genes
from duet.initialization import CodebookWarmStart
from duet.merfish_factory import MERFISHFactory

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "benchmark" / "make_bostrom_panel.py"
SET_FIXTURE = REPO / "examples" / "data" / "bostrom" / "10Bit_HW4_HD4_finalsize30Set.csv"


def test_cli_end_to_end_top_k(tmp_path):
    expr = tmp_path / "expr.csv"
    # ERCC control has the highest expression but must be excluded.
    expr.write_text(
        "gene_name,mean_raw_counts\n"
        "GeneA,100\nGeneB,50\nGeneC,25\nGeneD,5\nERCC-1,999\n"
    )
    out = tmp_path / "panel.csv"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--codebook", str(SET_FIXTURE),   # --barcode-length / --fmt inferred from filename
         "--expression", str(expr),
         "--gene-col", "gene_name", "--expression-col", "mean_raw_counts",
         "--top-k", "3",
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out, dtype={"Sequence": str})
    assert len(df) == 3
    assert list(df.columns)[:2] == ["Gene", "Sequence"]
    assert df["Gene"].tolist() == ["GeneA", "GeneB", "GeneC"]  # ERCC excluded, desc expr
    assert all(len(s) == 10 and s.count("1") == 4 for s in df["Sequence"])
    assert df["Sequence"].iloc[0] == "1110001000"             # first codeword of the fixture
    assert df["mean_raw_counts"].tolist() == [100.0, 50.0, 25.0]


def test_warm_start_contract(tmp_path):
    # The panel CSV must satisfy the runner's warm-start path: every sequence
    # resolves to a pool index (no "Sequence not found"), len == codebook_size.
    codewords = ["1110001000", "1101100000", "1100010001"]    # HW4, 10-bit
    panel = assign_codewords_to_genes(codewords, ["g0", "g1", "g2"], [3.0, 2.0, 1.0])
    panel_csv = tmp_path / "panel.csv"
    panel.to_csv(panel_csv, index=False)

    required = [{s} for s in panel["Sequence"].tolist()]      # positional anchors
    factory = MERFISHFactory(
        seq_rounds=10, codebook_size=3, hamming_weights=[4],
        required_codewords=required,
    )
    pool = factory.create()

    ws = CodebookWarmStart(codebook_path=panel_csv, sequence_col="Sequence")
    idx, meta = ws.get_initial_indices(pool, seed=0)
    assert len(idx) == 3
    seqs = list(pool.sequences)
    assert [seqs[i] for i in idx] == panel["Sequence"].tolist()


def test_cli_shuffle_assignment_repairs_same_set(tmp_path):
    """--shuffle-assignment keeps the same genes + codeword set but re-pairs them."""
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location("make_bostrom_panel", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    main = mod.main

    codebook = tmp_path / "8Bit_HW2_HD4_finalsize6Binary.csv"
    codebook.write_text(
        "1 1 0 0 0 0 0 0\n"
        "0 0 1 1 0 0 0 0\n"
        "0 0 0 0 1 1 0 0\n"
        "0 0 0 0 0 0 1 1\n"
        "1 0 0 0 0 1 0 0\n"
        "0 1 0 0 0 0 1 0\n"
    )
    expr = tmp_path / "expr.csv"
    expr.write_text(
        "gene_name,mean_raw_counts\n"
        "g0,6\ng1,5\ng2,4\ng3,3\ng4,2\ng5,1\n"
    )
    genes = tmp_path / "genes.csv"
    genes.write_text("Gene\ng0\ng1\ng2\ng3\ng4\ng5\n")

    common = [
        "--codebook", str(codebook), "--barcode-length", "8", "--fmt", "binary",
        "--no-verify", "--expression", str(expr),
        "--gene-col", "gene_name", "--expression-col", "mean_raw_counts",
        "--genes-csv", str(genes), "--genes-col", "Gene",
    ]
    out_base = tmp_path / "base.csv"
    out_shuf = tmp_path / "shuf.csv"
    main(common + ["--out", str(out_base)])
    main(common + ["--shuffle-assignment", "--shuffle-seed", "42", "--out", str(out_shuf)])

    base = pd.read_csv(out_base, dtype={"Sequence": str})
    shuf = pd.read_csv(out_shuf, dtype={"Sequence": str})
    assert list(shuf["Gene"]) == list(base["Gene"])             # same gene rows
    assert set(shuf["Sequence"]) == set(base["Sequence"])       # same codeword set
    assert list(shuf["Sequence"]) != list(base["Sequence"])     # different pairing


def test_cli_no_expression_col_keeps_rows(tmp_path):
    """--no-expression-col writes Gene,Sequence only, with the default run's rows."""
    expr = tmp_path / "expr.csv"
    expr.write_text("gene_name,mean_raw_counts\nGeneA,100\nGeneB,50\nGeneC,25\n")
    common = [
        sys.executable, str(SCRIPT), "--codebook", str(SET_FIXTURE),
        "--expression", str(expr), "--gene-col", "gene_name",
        "--expression-col", "mean_raw_counts", "--top-k", "3",
    ]
    full, bare = tmp_path / "full.csv", tmp_path / "bare.csv"
    for extra, out in (([], full), (["--no-expression-col"], bare)):
        result = subprocess.run(common + extra + ["--out", str(out)],
                                capture_output=True, text=True, cwd=str(REPO))
        assert result.returncode == 0, result.stderr
    with_expr = pd.read_csv(full, dtype={"Sequence": str})
    without = pd.read_csv(bare, dtype={"Sequence": str})
    assert list(with_expr.columns) == ["Gene", "Sequence", "mean_raw_counts"]
    assert list(without.columns) == ["Gene", "Sequence"]
    pd.testing.assert_frame_equal(without, with_expr[["Gene", "Sequence"]])
