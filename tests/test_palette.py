"""Tests for src/duet/plotting/palette.py."""
from duet.plotting.palette import METHOD_PALETTE, OKABE_ITO


def test_method_palette_has_chen_baselines():
    """Chen 2015 codebooks must be in METHOD_PALETTE for MERFISH plots."""
    assert "Codebook 1 (Chen 2015)" in METHOD_PALETTE
    assert "Codebook 2 (Chen 2015)" in METHOD_PALETTE
    # Reuse the same Okabe-Ito hues as the OPS prior-art baselines.
    assert METHOD_PALETTE["Codebook 1 (Chen 2015)"] == OKABE_ITO["bluish_green"]
    assert METHOD_PALETTE["Codebook 2 (Chen 2015)"] == OKABE_ITO["orange"]


def test_fig4_baseline_palette_entries_grouped_by_family():
    """Fig-4 baselines share one color per method family (Hamming weight is the
    family's only lever, rendered as a 2-point mini-Pareto): both Bostrom
    weights one hue, both MERFISH-MHD4 weights another, the two families
    distinct from each other and from DUET."""
    from duet.plotting.palette import METHOD_PALETTE
    bos4 = METHOD_PALETTE["Bostrom et al. (Hamming weight 4)"]  # KeyError if missing
    bos5 = METHOD_PALETTE["Bostrom et al. (Hamming weight 5)"]
    mhd4 = METHOD_PALETTE["MERFISH MHD4 (Hamming weight 4)"]
    mhd5 = METHOD_PALETTE["MERFISH MHD4 (Hamming weight 5)"]
    assert bos4 == bos5                          # one color per family...
    assert mhd4 == mhd5
    assert bos4 != mhd4                          # ...families distinct
    assert METHOD_PALETTE["DUET"] not in (bos4, mhd4)  # distinct from DUET
