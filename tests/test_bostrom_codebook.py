# tests/test_bostrom_codebook.py
"""Tests for duet.bostrom_codebook (Boström codebook loading + assignment)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.bostrom_codebook import load_bostrom_codebook

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "data" / "bostrom"
BINARY = FIXTURE / "10Bit_HW4_HD4_finalsize30Binary.csv"
SET = FIXTURE / "10Bit_HW4_HD4_finalsize30Set.csv"


def test_set_and_binary_encodings_agree():
    s = load_bostrom_codebook(SET, 10, fmt="set")
    b = load_bostrom_codebook(BINARY, 10, fmt="binary")
    assert s == b
    assert len(s) == 30
    assert all(len(x) == 10 and x.count("1") == 4 for x in s)
    # Set row "1 2 3 7" -> 1-indexed positions 1,2,3,7 set.
    assert s[0] == "1110001000"


def test_auto_detect_picks_format():
    assert load_bostrom_codebook(SET, 10, "auto") == load_bostrom_codebook(SET, 10, "set")
    assert load_bostrom_codebook(BINARY, 10, "auto") == load_bostrom_codebook(BINARY, 10, "binary")


def test_binary_wrong_barcode_length_raises():
    with pytest.raises(ValueError, match="columns"):
        load_bostrom_codebook(BINARY, 8, fmt="binary")


def test_whitespace_is_robust(tmp_path):
    f = tmp_path / "toy_Set.csv"
    f.write_text("1  2 3   7\n")  # double/triple spaces
    assert load_bostrom_codebook(f, 10, "set")[0] == "1110001000"


def test_set_position_out_of_range_raises(tmp_path):
    f = tmp_path / "bad_Set.csv"
    f.write_text("1 2 3 11\n")  # position 11 > barcode_length 10
    with pytest.raises(ValueError, match="out of range"):
        load_bostrom_codebook(f, 10, "set")


def test_unknown_fmt_raises():
    with pytest.raises(ValueError, match="Unknown fmt"):
        load_bostrom_codebook(SET, 10, fmt="nonsense")


from duet.bostrom_codebook import parse_bostrom_filename


def test_parse_plain_binary_name():
    assert parse_bostrom_filename("32Bit_HW4_HD4_finalsize1240Binary.csv") == {
        "bits": 32, "hw": 4, "hd": 4, "finalsize": 1240, "fmt": "binary",
    }


def test_parse_set_name():
    assert parse_bostrom_filename("10Bit_HW4_HD4_finalsize30Set.csv")["fmt"] == "set"


def test_parse_reordered_name():
    d = parse_bostrom_filename("32Bit_HW4_HD4_finalsize1240Binary_reordered40.csv")
    assert d["bits"] == 32 and d["finalsize"] == 1240 and d["fmt"] == "binary"


def test_parse_nonmatching_returns_empty():
    assert parse_bostrom_filename("not_a_bostrom_file.csv") == {}


def test_parse_rejects_trailing_junk():
    assert parse_bostrom_filename("32Bit_HW4_HD4_finalsize1240BinaryXXX.csv") == {}


from duet.bostrom_codebook import assign_codewords_to_genes, verify_codebook


def test_assign_highest_expression_first_and_truncates():
    codewords = ["1110001000", "1101100000", "1100010001", "1010100010"]  # 4 codewords
    genes = ["lo", "hi", "mid"]            # deliberately unsorted
    expr = [1.0, 100.0, 50.0]
    panel = assign_codewords_to_genes(codewords, genes, expr)
    assert panel["Gene"].tolist() == ["hi", "mid", "lo"]            # ranked desc
    assert panel["Sequence"].tolist() == codewords[:3]             # surplus dropped
    assert list(panel.columns) == ["Gene", "Sequence"]


def test_assign_raises_when_more_genes_than_codewords():
    with pytest.raises(ValueError, match="genes but only"):
        assign_codewords_to_genes(["1100", "1010"], ["a", "b", "c"], [3.0, 2.0, 1.0])


def test_verify_constant_hw_and_minhd_on_fixture():
    cw = load_bostrom_codebook(SET, 10, "set")
    summary = verify_codebook(cw, expected_hw=4)
    assert summary["hw"] == 4 and summary["n"] == 30 and summary["min_hd"] == 4


def test_verify_nonconstant_hw_raises():
    with pytest.raises(ValueError, match="Non-constant"):
        verify_codebook(["1100", "1110"])


def test_verify_skips_hd_above_cap():
    cw = load_bostrom_codebook(SET, 10, "set")
    summary = verify_codebook(cw, max_n_for_hd=5)
    assert "min_hd" not in summary


def test_verify_single_codeword_skips_hd():
    summary = verify_codebook(["1111000000"])
    assert summary["n"] == 1 and "min_hd" not in summary


def test_verify_empty_codebook_raises():
    with pytest.raises(ValueError, match="Empty"):
        verify_codebook([])


def test_mislabeled_format_raises():
    # Binary file declared 'set' -> hits a 0 position (out of range 1..N).
    with pytest.raises(ValueError):
        load_bostrom_codebook(BINARY, 10, fmt="set")
    # Set file declared 'binary' -> 4 columns != barcode_length 10.
    with pytest.raises(ValueError):
        load_bostrom_codebook(SET, 10, fmt="binary")


def _distinct_codewords(n):
    """n distinct 32-bit codeword strings (content irrelevant to assignment)."""
    return [format(i, "032b") for i in range(1, n + 1)]


def test_assign_default_is_expression_ordered():
    cw = _distinct_codewords(5)
    genes = ["g1", "g2", "g3"]
    expr = [10.0, 30.0, 20.0]
    df = assign_codewords_to_genes(cw, genes, expr)
    # genes sorted by DESCENDING expression: g2(30), g3(20), g1(10)
    assert list(df["Gene"]) == ["g2", "g3", "g1"]
    assert list(df["Sequence"]) == cw[:3]


def test_shuffle_same_genes_same_set_different_pairing():
    cw = _distinct_codewords(20)
    genes = [f"g{i}" for i in range(10)]
    expr = [float(i) for i in range(10)]
    base = assign_codewords_to_genes(cw, genes, expr)
    shuf = assign_codewords_to_genes(cw, genes, expr, shuffle=True, seed=42)
    # identical Gene column (same descending-expression row order)
    assert list(shuf["Gene"]) == list(base["Gene"])
    # identical Sequence SET (same first-N codewords, just re-paired)
    assert set(shuf["Sequence"]) == set(base["Sequence"])
    # pairing differs for at least one gene
    assert list(shuf["Sequence"]) != list(base["Sequence"])


def test_shuffle_is_deterministic_by_seed():
    cw = _distinct_codewords(20)
    genes = [f"g{i}" for i in range(10)]
    expr = [float(i) for i in range(10)]
    a = assign_codewords_to_genes(cw, genes, expr, shuffle=True, seed=7)
    b = assign_codewords_to_genes(cw, genes, expr, shuffle=True, seed=7)
    assert list(a["Sequence"]) == list(b["Sequence"])


def test_shuffle_requires_explicit_seed():
    cw = _distinct_codewords(20)
    genes = [f"g{i}" for i in range(10)]
    expr = [float(i) for i in range(10)]
    with pytest.raises(ValueError):
        assign_codewords_to_genes(cw, genes, expr, shuffle=True)


# --- Full-reorder codebook (steelmanned Boström baseline) -------------------
# regenerated/32Bit_HW4_HD4_finalsize1240Binary_reordered1240.csv is the authors'
# own CodebookHDSorter.R run to completion. Their published _reordered40 variant
# stops after 40 of 1240 codewords and leaves the rest in lexicographic
# construction order, which shares low-index bits; under their
# descending-expression assignment that piles the most abundant genes onto the
# earliest imaging rounds. These tests pin the property that the full reorder
# fixes. The regenerated file is not distributed with the public release, so
# the tests that read it skip when it is absent.

REORDER40 = FIXTURE / "32Bit_HW4_HD4_finalsize1240Binary_reordered40.csv"
REORDER_FULL = FIXTURE / "regenerated" / "32Bit_HW4_HD4_finalsize1240Binary_reordered1240.csv"
needs_full_reorder = pytest.mark.skipif(
    not REORDER_FULL.is_file(),
    reason=f"{REORDER_FULL.relative_to(FIXTURE.parents[2])} is not part of this copy",
)


def _bits(codewords):
    return np.array([[int(c) for c in s] for s in codewords])


@needs_full_reorder
def test_full_reorder_is_a_permutation_of_the_published_codebook():
    """The fix reorders; it must not add, drop, or alter any codeword."""
    pub = load_bostrom_codebook(REORDER40, 32)
    full = load_bostrom_codebook(REORDER_FULL, 32)
    assert len(full) == len(pub) == 1240
    assert set(full) == set(pub)          # identical codeword SET
    assert full != pub                    # but a different order


@needs_full_reorder
def test_full_reorder_preserves_hw4_and_min_hd4():
    full = load_bostrom_codebook(REORDER_FULL, 32)
    assert verify_codebook(full, expected_hw=4) == {"hw": 4, "n": 1240, "min_hd": 4}


@needs_full_reorder
@pytest.mark.parametrize("panel_size", [1000, 1147])
def test_full_reorder_prefix_is_per_bit_balanced(panel_size):
    """A truncated prefix must spread its ON bits evenly across imaging rounds.

    This is what the published _reordered40 ordering fails to do: lexicographic
    truncation over-selects codewords carrying the low-index bits.
    """
    pub = _bits(load_bostrom_codebook(REORDER40, 32))[:panel_size]
    full = _bits(load_bostrom_codebook(REORDER_FULL, 32))[:panel_size]
    ideal = panel_size * 4 / 32

    pub_usage, full_usage = pub.sum(axis=0), full.sum(axis=0)
    # The published prefix is badly skewed; the full reorder is within ~2 of ideal.
    assert pub_usage.max() - pub_usage.min() > 20
    assert full_usage.max() - full_usage.min() <= 3
    assert abs(full_usage.mean() - ideal) < 1e-6
