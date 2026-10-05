from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()


import pandas as pd
import pytest

from duet.merfish_benchmark.cell_type_crowding import (
    load_codebook,
    assert_same_panel,
    panel_capture,
    target_panel_spots,
    matched_total_reads,
    run_evaluations,
    REFERENCE_LABEL,
    load_plot_frame,
    duet_gap,
)

def _write_cb(path, genes, seqs, extra_first_col=True):
    data = {}
    if extra_first_col:
        data["Index"] = list(range(len(genes)))
    data["Sequence"] = seqs
    data["Gene"] = genes
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_codebook_selects_gene_sequence(tmp_path):
    p = tmp_path / "cb.csv"
    _write_cb(p, ["G1", "G2"], ["1100", "0011"])
    df = load_codebook(p)
    assert list(df.columns) == ["Gene", "Sequence"]
    assert df["Gene"].tolist() == ["G1", "G2"]
    assert df["Sequence"].tolist() == ["1100", "0011"]


def test_load_codebook_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Gene": ["G1"], "Barcode": ["1100"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_codebook(p)


def test_assert_same_panel_ok():
    a = pd.DataFrame({"Gene": ["G1", "G2"], "Sequence": ["11", "10"]})
    b = pd.DataFrame({"Gene": ["G2", "G1"], "Sequence": ["01", "00"]})
    assert assert_same_panel({"A": a, "B": b}) == ["G1", "G2"]


def test_assert_same_panel_mismatch_raises():
    a = pd.DataFrame({"Gene": ["G1", "G2"], "Sequence": ["11", "10"]})
    b = pd.DataFrame({"Gene": ["G1", "G3"], "Sequence": ["01", "00"]})
    with pytest.raises(ValueError, match="panel mismatch"):
        assert_same_panel({"A": a, "B": b})


def _expr(symbols, vals):
    return pd.DataFrame({"gene_symbol": symbols, "mean_cpm": vals})


def test_panel_capture_is_panel_mass_over_total():
    df = _expr(["A", "B", "C", "D"], [10.0, 30.0, 60.0, 0.0])
    # panel {A,B} mass 40 of total 100
    assert panel_capture(df, ["A", "B"]) == pytest.approx(0.40)


def test_panel_capture_zero_total_raises():
    df = _expr(["A", "B"], [0.0, 0.0])
    with pytest.raises(ValueError, match="zero"):
        panel_capture(df, ["A"])


def test_target_panel_spots_rounds():
    df = _expr(["A", "B", "C"], [3.0, 97.0, 0.0])  # capture(A) = 0.03
    assert target_panel_spots(df, ["A"], base_total_reads=8700) == 261


def test_matched_total_reads_roundtrips_to_base():
    # capture 0.03, target 261 -> ~8700
    assert matched_total_reads(261, 0.03) == 8700


def test_matched_total_reads_caps_and_guards():
    assert matched_total_reads(1000, 0.0001, cap=200_000) == 200_000
    with pytest.raises(ValueError, match="capture"):
        matched_total_reads(261, 0.0)


def test_run_evaluations_assembles_rows_with_matched_reads():
    # panel {A,B}; two codebooks; one class where capture differs from prior.
    cbs = {
        "DUET": pd.DataFrame({"Gene": ["A", "B"], "Sequence": ["11", "10"]}),
        "Base": pd.DataFrame({"Gene": ["A", "B"], "Sequence": ["01", "00"]}),
    }
    panel = ["A", "B"]
    prior = _expr(["A", "B", "C"], [1.0, 2.0, 97.0])          # capture = 0.03 -> T=261
    wide = pd.DataFrame({"gene_symbol": ["A", "B", "C"],
                         "Astro": [5.0, 5.0, 90.0]})           # capture_Astro = 0.10
    summary = pd.DataFrame({"class_name": ["Astro"], "n_clusters": [3]}).set_index("class_name")

    # eval_fn just echoes total_reads so we can assert wiring deterministically.
    calls = []

    def fake_eval(cb_df, expr_df, total_reads):
        calls.append((set(cb_df["Gene"]), total_reads))
        return (0.9, 0.01) if len(cb_df) else (0.0, 0.0)

    rows = run_evaluations(cbs, wide, panel, prior, summary, n_trials=7, seed=42, eval_fn=fake_eval)
    df = pd.DataFrame(rows)
    # 2 reference rows + 2 methods x 1 class = 4 rows
    assert len(df) == 4
    ref = df[df.class_name == REFERENCE_LABEL]
    assert set(ref.method) == {"DUET", "Base"}
    assert (ref.total_reads == 8700).all()                     # base_total_reads on the prior
    astro = df[df.class_name == "Astro"]
    assert (astro.total_reads == 2610).all()                   # T=261 / capture 0.10 = 2610
    assert (astro.n_clusters == 3).all()


def _metrics(rows, n_trials=100, std=0.02):
    """Build a metrics-CSV-shaped frame; load_plot_frame needs std + n_trials."""
    return pd.DataFrame([dict(**r, std_identified_fraction=std, n_trials=n_trials) for r in rows])


def test_load_plot_frame_splits_reference_and_computes_se():
    m = _metrics([
        dict(method="DUET", class_name=REFERENCE_LABEL, mean_identified_fraction=0.83),
        dict(method="Base", class_name=REFERENCE_LABEL, mean_identified_fraction=0.78),
        dict(method="DUET", class_name="Astro", mean_identified_fraction=0.90),
        dict(method="Base", class_name="Astro", mean_identified_fraction=0.85),
    ], n_trials=100, std=0.02)
    ident, se, ref = load_plot_frame(m)
    assert set(ident.index) == {"Astro"}                       # reference rows excluded
    assert ref == {"DUET": 0.83, "Base": 0.78}
    assert ident.loc["Astro", "DUET"] == pytest.approx(0.90)
    assert se.loc["Astro", "DUET"] == pytest.approx(0.02 / 10)  # std / sqrt(100)


def test_load_plot_frame_rejects_nonpositive_n_trials():
    m = _metrics([dict(method="DUET", class_name="Astro", mean_identified_fraction=0.9)], n_trials=0)
    with pytest.raises(ValueError, match="n_trials"):
        load_plot_frame(m)


def test_duet_gap_signs():
    ident = pd.DataFrame({"DUET": [0.90, 0.88], "Base": [0.85, 0.89]}, index=["Astro", "Oligo"])
    gap = duet_gap(ident, baseline_methods=("Base",), duet_method="DUET")
    assert gap.loc["Astro"] == pytest.approx(0.05)
    assert gap.loc["Oligo"] == pytest.approx(-0.01)            # DUET loses this one
