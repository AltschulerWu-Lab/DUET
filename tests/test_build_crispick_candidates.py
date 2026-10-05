"""Tests for scripts/data_processing/build_crispick_candidates.py.

The real CRISPick design file is 2 GB and is not redistributed, so these tests
write a few genes in its format and check the details that make the real build
match the paper's table byte for byte: the first 20 picks per gene in numeric
Pick Order, integer Pick Order in the output, the "MAX" quirk, the controls in
file order after the genes, the column order and LF line endings.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The script is not an installed module (scripts/data_processing has no
# __init__.py); load it from its path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_processing" / "build_crispick_candidates.py"
_spec = importlib.util.spec_from_file_location("build_crispick_candidates", _SCRIPT)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)

GENE_A = "ENSG00000000001"  # 3 picks, written after GENE_B in the file
GENE_B = "ENSG00000000002"  # 25 picks, written out of order
GENE_MAX = "ENSG00000125952"  # the real gene named MAX
GENE_UNPICKED = "ENSG00000000009"  # no picked guide

# A subset of the real header, in the real order, with columns the script
# does not read (Input, Quota, Picking Notes) around the ones it does.
HEADER = [
    "Input", "Quota", "Target Gene ID", "Target Gene Symbol", "sgRNA Sequence",
    "Aggregate CFD Score", "On-Target Efficacy Score", "Pick Order", "Picking Round",
    "Picking Notes",
]
OUT_HEADER = (
    "Target Gene ID,Target Gene Symbol,sgRNA Sequence,Aggregate CFD Score,"
    "On-Target Efficacy Score,Pick Order,Picking Round,Source"
)
NOSITE = ["CCCCNOSITE3", "AAAANOSITE1", "GGGGNOSITE2"]  # not sorted: file order must stay
ONESITE = ["TTTTONESITE2", "CCCCONESITE1"]


def _row(gene, symbol, seq, cfd, eff, pick, rnd):
    # Quota is "MAX" in the real file; the script does not read it.
    return [gene, "MAX", gene, symbol, seq, cfd, eff, pick, rnd, "note"]


def _design_rows():
    rows = []
    # GENE_B: picks 25 down to 1, so file order is the reverse of Pick Order,
    # and a text sort would put 10 before 2. Two guides were not picked.
    for pick in range(25, 0, -1):
        rnd = 1 if pick <= 10 else 2
        rows.append(_row(GENE_B, "GENEB", f"SEQ_B_{pick:02d}", "0.25", "0.5", str(pick), str(rnd)))
    rows.append(_row(GENE_B, "GENEB", "SEQ_B_UNPICKED1", "0.1", "0.9", "", ""))
    rows.append(_row(GENE_B, "GENEB", "SEQ_B_UNPICKED2", "0.1", "0.9", "", ""))
    # GENE_A: three picks; the second has a "MAX" CFD score. One unpicked guide.
    rows.append(_row(GENE_A, "GENEA", "SEQ_A_3", "0.0", "0.3", "3", "2"))
    rows.append(_row(GENE_A, "GENEA", "SEQ_A_UNPICKED", "0.0", "0.99", "", ""))
    rows.append(_row(GENE_A, "GENEA", "SEQ_A_1", "0.0", "0.8", "1", "1"))
    rows.append(_row(GENE_A, "GENEA", "SEQ_A_2", "MAX", "0.6", "2", "1"))
    # The gene MAX: its symbol is read as missing too.
    rows.append(_row(GENE_MAX, "MAX", "SEQ_MAX_2", "0.5", "0.4", "2", "1"))
    rows.append(_row(GENE_MAX, "MAX", "SEQ_MAX_1", "0.5", "0.7", "1", "1"))
    rows.append(_row(GENE_UNPICKED, "GENEU", "SEQ_U_1", "0.5", "0.7", "", ""))
    return rows


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    lines = ["\t".join(HEADER)] + ["\t".join(r) for r in _design_rows()]
    (d / bc.DESIGN_FILE).write_text("\n".join(lines) + "\n")
    (d / bc.NOSITE_FILE).write_text("\n".join(NOSITE) + "\n")
    (d / bc.ONESITE_FILE).write_text("\n".join(ONESITE) + "\n")
    return d


@pytest.fixture
def lines(raw_dir, tmp_path):
    out = tmp_path / "out" / "table.csv"
    bc.write_table(bc.build(raw_dir), out)
    data = out.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")  # LF line endings
    return data.decode().split("\n")[:-1]


def _fields(lines):
    return [line.split(",") for line in lines[1:]]


def test_header_keeps_the_design_file_column_order(lines):
    assert lines[0] == OUT_HEADER


def test_first_20_picks_per_gene_sorted_by_gene_then_pick_order(lines):
    rows = [f for f in _fields(lines) if f[7] == "Ensembl"]
    genes = [f[0] for f in rows]
    assert genes == [GENE_A] * 3 + [GENE_B] * 20 + [GENE_MAX] * 2
    b_rows = [f for f in rows if f[0] == GENE_B]
    assert [f[5] for f in b_rows] == [str(p) for p in range(1, 21)]  # integers, numeric order
    assert [f[2] for f in b_rows] == [f"SEQ_B_{p:02d}" for p in range(1, 21)]
    assert [f[2] for f in rows if f[0] == GENE_A] == ["SEQ_A_1", "SEQ_A_2", "SEQ_A_3"]


def test_unpicked_guides_and_genes_are_dropped(lines):
    text = "\n".join(lines)
    assert "UNPICKED" not in text
    assert GENE_UNPICKED not in text


def test_exact_lines_of_a_gene(lines):
    assert lines[1:4] == [
        f"{GENE_A},GENEA,SEQ_A_1,0.0,0.8,1,1,Ensembl",
        f"{GENE_A},GENEA,SEQ_A_2,,0.6,2,1,Ensembl",  # "MAX" CFD score read as missing
        f"{GENE_A},GENEA,SEQ_A_3,0.0,0.3,3,2,Ensembl",
    ]


def test_max_quirk_blanks_the_symbol_of_the_gene_max(lines):
    max_rows = [f for f in _fields(lines) if f[0] == GENE_MAX]
    assert [f[1] for f in max_rows] == ["", ""]
    assert [f[2] for f in max_rows] == ["SEQ_MAX_1", "SEQ_MAX_2"]
    assert all(f[1] == "GENEB" for f in _fields(lines) if f[0] == GENE_B)


def test_controls_follow_the_genes_in_file_order(lines):
    tail = lines[-(len(NOSITE) + len(ONESITE)):]
    assert tail == (
        [f"NO_SITE,NO_SITE,{s},,,,,NO_SITE" for s in NOSITE]
        + [f"ONE_SITE_INTERGENIC,ONE_SITE_INTERGENIC,{s},,,,,ONE_SITE_INTERGENIC" for s in ONESITE]
    )
    assert len(lines) == 1 + 25 + len(NOSITE) + len(ONESITE)


def test_main_builds_and_warns_that_the_inputs_are_not_the_papers(raw_dir, tmp_path, capsys):
    out = tmp_path / "cli" / "table.csv"
    assert bc.main(["--raw-dir", str(raw_dir), "--out", str(out)]) == 0
    assert out.read_text().splitlines()[0] == OUT_HEADER
    captured = capsys.readouterr()
    assert "30 rows (3 genes plus controls)" in captured.out  # 25 guides, 5 controls
    assert f"{bc.DESIGN_FILE} is not the file the paper's table was built from" in captured.err
    assert "statistically" in captured.err
    assert "differs from the paper's table" in captured.err


def test_main_names_missing_inputs(tmp_path, capsys):
    (tmp_path / bc.NOSITE_FILE).write_text("ACGT\n")
    out = tmp_path / "table.csv"
    assert bc.main(["--raw-dir", str(tmp_path), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert bc.DESIGN_FILE in err and bc.ONESITE_FILE in err and bc.NOSITE_FILE not in err
    assert not out.exists()


def test_help_names_the_three_crispick_files(capsys):
    with pytest.raises(SystemExit) as exc:
        bc.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in (bc.DESIGN_FILE, bc.NOSITE_FILE, bc.ONESITE_FILE):
        assert name in out


def test_defaults_resolve_at_the_repository_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert bc.DEFAULT_RAW_DIR.is_absolute() and bc.DEFAULT_OUT.is_absolute()
    assert bc.DEFAULT_OUT == bc.ROOT / "data/processed/CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv"
    assert (bc.ROOT / "scripts" / "data_processing" / "build_crispick_candidates.py").is_file()


def test_other_release_through_design_file(raw_dir, tmp_path, capsys):
    (raw_dir / bc.DESIGN_FILE).rename(raw_dir / "newer_design_20270101.txt")
    out = tmp_path / "out.csv"
    assert bc.main(["--raw-dir", str(raw_dir), "--out", str(out),
                    "--design-file", "newer_design_20270101.txt"]) == 0
    err = capsys.readouterr().err
    assert "newer_design_20270101.txt is not the design file the paper's table was built from" in err
    assert out.is_file()
