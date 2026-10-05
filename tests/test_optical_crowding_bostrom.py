import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.optical_crowding import (
    compute_diffraction_limit,
    detect_conflicts,
    load_set_codebook,
    run_crowding_sweep,
)

# The published-codebook tests need the Set-format codebooks of Boström et al.
# (2025), which are not in this repository. Download the HW4, HW5 and HW6
# codebooks from Dryad (doi:10.5061/dryad.zkh1893m5), unzip them into one
# folder so that it holds HW4/, HW5/ and HW6/ (for example
# HW4/30Bit_HW4_HD4_finalsize1005Set.csv), and set DUET_BOSTROM_DATA to that
# folder. The tests skip when the variable is unset.
_BOSTROM_ENV = os.environ.get("DUET_BOSTROM_DATA", "")
BOSTROM = Path(_BOSTROM_ENV).expanduser() if _BOSTROM_ENV else None
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPR_CSV = REPO_ROOT / "examples/data/processed/Kastriki_et_al_2022/sensory_neuron_mean_expression.csv"
HAS_CODEBOOKS = BOSTROM is not None
NO_CODEBOOKS = (
    "set DUET_BOSTROM_DATA to a folder with the HW4/, HW5/ and HW6/ Set codebooks "
    "of Boström et al. (2025), Dryad doi:10.5061/dryad.zkh1893m5"
)


def test_abbe_is_default_and_correct():
    # Abbe: lambda/(2*NA) = 0.5/(2*1.4) = 0.178571 um, default model
    assert compute_diffraction_limit() == pytest.approx(0.5 / (2 * 1.4), rel=1e-9)
    assert compute_diffraction_limit(model="abbe") == pytest.approx(0.178571, abs=1e-5)


def test_rayleigh_still_available():
    assert compute_diffraction_limit(model="rayleigh") == pytest.approx(0.61 * 0.5 / 1.4, rel=1e-9)


def test_expansion_divides_limit():
    assert compute_diffraction_limit(model="abbe", expansion_factor=3.0) == pytest.approx(
        (0.5 / (2 * 1.4)) / 3.0, rel=1e-9
    )


def test_box_neighborhood_is_default_catches_diagonal_pair():
    # Two transcripts of the SAME gene at L-inf distance ~0.099 (inside the box,
    # OUTSIDE a disk of radius 0.1). Box must flag; disk must not.
    r = 0.1
    cb = pd.DataFrame({"Gene": ["G"], "Sequence": ["1100"]})
    tx = pd.DataFrame({"gene": ["G", "G"], "x": [0.0, 0.099], "y": [0.0, 0.099]})
    # L2 distance = 0.140 > 0.1 ; L-inf distance = 0.099 < 0.1
    box = detect_conflicts(tx, cb, r, neighborhood="box")
    disk = detect_conflicts(tx, cb, r, neighborhood="disk")
    assert box.tolist() == [True, True]   # default-equivalent flags the box pair
    assert disk.tolist() == [False, False]


def test_conflict_requires_shared_probe():
    # Disjoint codewords within distance -> no conflict even though spatially close.
    cb = pd.DataFrame({"Gene": ["A", "B"], "Sequence": ["1100", "0011"]})
    tx = pd.DataFrame({"gene": ["A", "B"], "x": [0.0, 0.01], "y": [0.0, 0.01]})
    out = detect_conflicts(tx, cb, 0.1, neighborhood="box")
    assert out.tolist() == [False, False]


def test_box_is_the_default_neighborhood():
    # Calling without neighborhood= must behave like 'box'.
    r = 0.1
    cb = pd.DataFrame({"Gene": ["G"], "Sequence": ["1100"]})
    tx = pd.DataFrame({"gene": ["G", "G"], "x": [0.0, 0.099], "y": [0.0, 0.099]})
    assert detect_conflicts(tx, cb, r).tolist() == [True, True]


def test_load_set_codebook_parses_positions(tmp_path):
    f = tmp_path / "toy_Set.csv"
    f.write_text("2 4\n1 3\n")  # HW2, L>=4, 1-indexed positions
    cb = load_set_codebook(f, barcode_length=4)
    # position p (1-indexed) -> bit (p-1)
    assert cb["Sequence"].tolist() == ["0101", "1010"]


@pytest.mark.skipif(not HAS_CODEBOOKS, reason=NO_CODEBOOKS)
def test_load_published_hw4_l30_is_constant_weight():
    cb = load_set_codebook(BOSTROM / "HW4/30Bit_HW4_HD4_finalsize1005Set.csv", 30)
    assert len(cb) == 1005
    assert all(s.count("1") == 4 and len(s) == 30 for s in cb["Sequence"])


def test_build_panel_codebook_highest_expression_first():
    from duet.optical_crowding import build_panel_codebook
    genes_desc = ["hi", "mid", "lo"]   # already sorted descending by expression
    seqs = ["111", "110", "101", "011"]
    cb = build_panel_codebook(genes_desc, seqs)
    assert cb["Gene"].tolist() == ["hi", "mid", "lo"]
    assert cb["Sequence"].tolist() == ["111", "110", "101"]  # first len(genes) rows


def test_unknown_diffraction_model_raises():
    with pytest.raises(ValueError):
        compute_diffraction_limit(model="fraunhofer")


def test_unknown_neighborhood_raises():
    cb = pd.DataFrame({"Gene": ["G"], "Sequence": ["1100"]})
    tx = pd.DataFrame({"gene": ["G"], "x": [0.0], "y": [0.0]})
    with pytest.raises(ValueError):
        detect_conflicts(tx, cb, 0.1, neighborhood="hexagon")


def _toy_expr():
    # 5 genes: 3 detectable at 1000 reads, 2 sub-threshold.
    return pd.DataFrame({
        "gene_name": ["A", "B", "C", "D", "E"],
        "mean_raw_counts": [500.0, 300.0, 200.0, 0.001, 0.0],
    })


def test_simulate_deterministic_counts_match_expectation():
    from duet.optical_crowding import simulate_optical_crowding
    cb = pd.DataFrame({"Gene": ["A", "B", "C"], "Sequence": ["110", "101", "011"]})
    res = simulate_optical_crowding(
        cb, _toy_expr(), expression_col="mean_raw_counts", gene_col="gene_name",
        total_reads=1000, placement="deterministic", seed=0,
    )
    # prop over all = [.5,.3,.2,~0,0]; round(1000*prop) = [500,300,200,0,0]
    assert res.n_transcripts == 1000  # only A,B,C are in the codebook
    counts = res.transcripts_df["gene"].value_counts().to_dict()
    assert counts == {"A": 500, "B": 300, "C": 200}


def test_run_crowding_sweep_detectable_pool_smaller_than_all_expressed():
    expr = _toy_expr()
    # detectable pool (count>=1) = {A,B,C} = 3 genes; all_expressed = 4 (D has expr>0)
    out_det = run_crowding_sweep(
        expr, n_genes_list=[3], hamming_weights=[2], barcode_lengths={(3, 2): 3},
        n_trials=1, total_reads=1000, panel_pool="detectable", seed=1,
    )
    assert len(out_det) == 1  # succeeds: exactly 3 detectable genes
    with pytest.raises(ValueError):
        run_crowding_sweep(
            expr, n_genes_list=[4], hamming_weights=[2], barcode_lengths={(4, 2): 3},
            n_trials=1, total_reads=1000, panel_pool="detectable", seed=1,
        )  # only 3 detectable genes, requested 4


def test_run_crowding_sweep_all_expressed_pool_includes_subthreshold():
    expr = _toy_expr()
    # all_expressed = {A,B,C,D} = 4 genes -> requesting 4 succeeds under all_expressed.
    # L=4 is the smallest length encoding 4 HW2 codewords (C(4,2)=6 >= 4; C(3,2)=3 < 4).
    out = run_crowding_sweep(
        expr, n_genes_list=[4], hamming_weights=[2], barcode_lengths={(4, 2): 4},
        n_trials=1, total_reads=1000, panel_pool="all_expressed", seed=1,
    )
    assert len(out) == 1


def test_run_crowding_sweep_honors_custom_gene_col():
    # A frame whose gene column is NOT "gene_name" must work when gene_col is set,
    # threading through to simulate_optical_crowding (matches the sibling path
    # evaluate_crowding_simulation, which exposes gene_col).
    expr = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "mean_raw_counts": [500.0, 300.0, 200.0],
    })
    out = run_crowding_sweep(
        expr, n_genes_list=[3], hamming_weights=[2], barcode_lengths={(3, 2): 3},
        n_trials=1, total_reads=1000, panel_pool="all_expressed",
        gene_col="symbol", seed=0,
    )
    assert len(out) == 1


# --- End-to-end Boström Fig 4D/4G/4J reproduction (library path) ---------------

FIGURE = {"4D": (5.6, 9.1, 11.6), "4G": (23.2, 32.3, 42.1), "4J": (3.6, 4.6, 6.1)}
ORACLE = {"4D": (5.3, 9.0, 10.8), "4G": (23.7, 31.8, 41.4), "4J": (2.9, 4.1, 5.7)}
FIG_TOL = {"4D": 1.5, "4G": 2.0, "4J": 1.5}
ORACLE_TOL = 1.5
BARCODE_LENGTHS = {(1000, 4): 30, (1000, 5): 21, (1000, 6): 18,
                   (5000, 4): 51, (5000, 5): 30, (5000, 6): 23}
CONDITIONS = {"4D": (1000, 1.0), "4G": (5000, 1.0), "4J": (5000, 3.0)}
N_SEEDS = 8
REPRO_SEED = 20260614


@pytest.fixture(scope="module")
def repro_means():
    """Fixed-Python-sim Fig4 means (% missed) per condition, computed once."""
    if not HAS_CODEBOOKS:
        pytest.skip(NO_CODEBOOKS)
    expr = pd.read_csv(EXPR_CSV)
    expr = expr[~expr["gene_name"].str.startswith("ERCC")].reset_index(drop=True)
    out = {}
    for label, (n_genes, expansion) in CONDITIONS.items():
        df = run_crowding_sweep(
            expr, n_genes_list=[n_genes], hamming_weights=[4, 5, 6],
            barcode_lengths=BARCODE_LENGTHS, n_trials=N_SEEDS, total_reads=100_000,
            expansion_factor=expansion, panel_pool="detectable",
            codebook_source="published", codebook_dir=BOSTROM,
            assign_by_expression=True, diffraction_model="abbe", neighborhood="box",
            placement="deterministic", seed=REPRO_SEED,
        )
        m = df.groupby("hamming_weight")["conflict_fraction"].mean() * 100.0
        out[label] = [float(m.loc[hw]) for hw in (4, 5, 6)]
    print("\nFixed-sim Fig4 means (% missed):")
    for k in ("4D", "4G", "4J"):
        print(f"  {k}: {[round(v, 1) for v in out[k]]}  (figure {FIGURE[k]}, oracle {ORACLE[k]})")
    return out


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CODEBOOKS, reason=NO_CODEBOOKS)
@pytest.mark.parametrize("label", ["4D", "4G", "4J"])
def test_reproduces_bostrom_figure(repro_means, label):
    got = repro_means[label]
    for g, f in zip(got, FIGURE[label]):
        assert abs(g - f) <= FIG_TOL[label], f"{label}: {got} vs figure {FIGURE[label]}"
    for g, o in zip(got, ORACLE[label]):
        assert abs(g - o) <= ORACLE_TOL, f"{label}: {got} vs oracle {ORACLE[label]}"
    assert got[0] < got[1] < got[2], f"{label}: not monotone in HW: {got}"


@pytest.mark.slow  # shares the ~136 s repro_means fixture
@pytest.mark.skipif(not HAS_CODEBOOKS, reason=NO_CODEBOOKS)
def test_panel_ordering_4G_gt_4D_gt_4J(repro_means):
    for k in range(3):
        assert repro_means["4G"][k] > repro_means["4D"][k] > repro_means["4J"][k], \
            f"ordering broke at HW{k+4}: {repro_means}"


@pytest.mark.skipif(not HAS_CODEBOOKS, reason=NO_CODEBOOKS)
def test_published_set_path_picks_sufficient_codebook():
    from duet.optical_crowding import _published_set_path
    p = _published_set_path(BOSTROM, n_genes=5000, hw=4, barcode_length=51)
    assert p.name == "51Bit_HW4_HD4_finalsize5100Set.csv"


def test_assign_by_expression_puts_highest_first_in_sweep(monkeypatch):
    # With assign_by_expression=True the panel is sorted desc by expression;
    # build_panel_codebook puts the highest-expression gene on the FIRST codeword.
    # Verify the order reaching the codebook is descending by capturing it.
    from duet import optical_crowding as oc
    expr = pd.DataFrame({"gene_name": ["lo", "mid", "hi"],
                         "mean_raw_counts": [10.0, 100.0, 1000.0]})
    captured = {}
    real = oc.simulate_optical_crowding
    def spy(codebook_df, **kw):
        captured["genes"] = list(codebook_df["Gene"])
        return real(codebook_df, **kw)
    monkeypatch.setattr(oc, "simulate_optical_crowding", spy)
    oc.run_crowding_sweep(
        expr, n_genes_list=[3], hamming_weights=[2], barcode_lengths={(3, 2): 3},
        n_trials=1, total_reads=100_000, panel_pool="all_expressed",
        assign_by_expression=True, seed=0,
    )
    assert captured["genes"] == ["hi", "mid", "lo"]
