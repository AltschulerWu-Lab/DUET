"""Color palettes and canonical figure sizes for DUET publication figures.

All palettes are derived from the Okabe-Ito colorblind-safe palette
(Okabe & Ito, 2002). Method and error-category palettes assign
semantically-meaningful subsets to recurring plot categories.

Figure widths are the slot widths of the 180 mm page with 4 mm gaps between
panels (see duet.plotting.layout).
"""
from __future__ import annotations

import matplotlib as _mpl
import matplotlib.colors as _mcolors
import numpy as _np

from duet.plotting.layout import MM_PER_INCH as _MM_PER_INCH
from duet.plotting.layout import slot_width_mm as _slot_width_mm


OKABE_ITO: dict[str, str] = {
    "orange":          "#E69F00",
    "sky_blue":        "#56B4E9",
    "bluish_green":    "#009E73",
    "yellow":          "#F0E442",
    "blue":            "#0072B2",
    "vermillion":      "#D55E00",
    "reddish_purple":  "#CC79A7",
    "black":           "#000000",
}


METHOD_PALETTE: dict[str, str] = {
    "DUET":              OKABE_ITO["blue"],
    "Feldman et al.":    OKABE_ITO["bluish_green"],
    "Sivanandan et al.": OKABE_ITO["orange"],
    "Maximum activity":  "#999999",
    # 2-D synthetic benchmark methods.
    # Greedy-Hamming-MO uses reddish_purple to avoid colliding with
    # ERROR_CATEGORY_PALETTE["Failed"] (vermillion) in the 1-D analysis
    # plots, which display both palettes side-by-side.
    "Greedy-Hamming-MO": OKABE_ITO["reddish_purple"],
    # Greedy-NLL-MO uses vermillion. The 2-D synthetic visualizer does
    # not import ERROR_CATEGORY_PALETTE (only the 1-D analysis visualizer
    # does, and that one doesn't render greedy MO methods), so the
    # ERROR_CATEGORY collision constraint doesn't apply here. Sky_blue
    # was previously used but read as visually similar to DUET's dark
    # blue, defeating the goal of method-distinguishability.
    "Greedy-NLL-MO":     OKABE_ITO["vermillion"],
    "Exhaustive":        OKABE_ITO["black"],
    # MERFISH prior-art baselines. Reuse the same Okabe-Ito hues as the
    # OPS prior-art baselines (Feldman et al., Sivanandan et al.) -- Chen
    # 2015's two codebooks play the same semantic role in MERFISH that
    # those papers' methods play in OPS. MERFISH and OPS methods never
    # coexist in the same panel, so the cross-domain color reuse is
    # intentional, not an accidental collision.
    "Codebook 1 (Chen 2015)": OKABE_ITO["bluish_green"],  # parallels Feldman et al.
    "Codebook 2 (Chen 2015)": OKABE_ITO["orange"],        # parallels Sivanandan et al.
    # Zhang 2023 MERFISH baselines. "#2" is the published Zhang codebook shown
    # in the Figure-4 crowding panels as the standard-MERFISH comparator, so it
    # takes the same cool hue (bluish_green) as the "MERFISH MHD4" family below
    # -- it plays that comparator role and the two never coexist in one panel.
    # (Left orange it would now collide with the Bostrom family.) "#1" is unused
    # in the Figure-4 panels; if #1 and #2 ever share a panel, re-hue one.
    "Zhang et al. codebook #1": OKABE_ITO["bluish_green"],
    "Zhang et al. codebook #2": OKABE_ITO["bluish_green"],
    # 16-bit codebook-comparison archive baselines (Chen 2015 vs Bostrom 2025
    # set-exchange, HW4 and HW5). Distinct hues so the three codebook baselines
    # stay distinguishable in one crowding panel; without these the generic
    # UNKNOWN_METHOD_COLOR fallback collapses all three to a single grey. Chen
    # keeps bluish_green (consistent with the "Codebook 1 (Chen 2015)" entry
    # above); the two Bostrom weights take vermillion + reddish_purple.
    "Chen et al. 2015 (codebook_1)": OKABE_ITO["bluish_green"],
    "Bostrom HW4 (set-exchange)":    OKABE_ITO["vermillion"],
    "Bostrom HW5 (set-exchange)":    OKABE_ITO["reddish_purple"],
    # Figure 4 (MERFISH) baselines, grouped by method FAMILY so the two
    # Hamming-weight variants read as one method whose only lever is Hamming
    # weight -- a 2-point mini-Pareto on the crowding front (see
    # _draw_crowding_pareto). Both Bostrom (expression-aware set-exchange)
    # weights share one warm hue (orange); both expression-agnostic "MERFISH
    # MHD4" shuffled weights share one cool hue (bluish_green), paralleling the
    # OPS Sivanandan-orange / Feldman-green prior-art split. Within a family the
    # two points are told apart by marker shape + the connector line, not color.
    # Both families stay distinct from DUET (blue).
    "Bostrom et al. (Hamming weight 4)": OKABE_ITO["orange"],
    "Bostrom et al. (Hamming weight 5)": OKABE_ITO["orange"],
    "MERFISH MHD4 (Hamming weight 4)":   OKABE_ITO["bluish_green"],
    "MERFISH MHD4 (Hamming weight 5)":   OKABE_ITO["bluish_green"],
}


# Fallback color for methods not in METHOD_PALETTE. Distinct from
# METHOD_PALETTE["Maximum activity"] (#999999) so an unknown method renders
# visibly differently from the canonical "MA" baseline.
UNKNOWN_METHOD_COLOR: str = "#7f7f7f"

# Grey of the "N/A" text that stands in for a legend marker when a method has
# nothing plotted (duet.plotting.legend).
NOT_AVAILABLE_COLOR: str = "#777777"


# Marker shapes parallel to METHOD_PALETTE: when overlapping markers cluster
# at coincident (x, y) coordinates (common with discrete-codebook
# benchmarks), color alone alpha-blends into ambiguity. Shape acts as an
# orthogonal channel that survives both alpha-blending and B&W printing,
# and helps colorblind readers as a second cue beyond the (already
# colorblind-safe) Okabe-Ito hues. Only the methods that render as
# scatter or per-trial markers are listed; line-plot consumers
# (Exhaustive) don't need an entry.
METHOD_MARKERS: dict[str, str] = {
    "DUET":              "o",  # circle — headline method, "default" reading
    "Greedy-Hamming-MO": "s",  # square — most geometrically distinct from o
    "Greedy-NLL-MO":     "^",  # triangle-up — distinct corner profile
}


ERROR_CATEGORY_PALETTE: dict[str, str] = {
    "No error":  OKABE_ITO["bluish_green"],
    "Corrected": OKABE_ITO["blue"],
    "Failed":    OKABE_ITO["vermillion"],
}


# Per-objective palette for the baseline-objectives visualizers (the
# DUET-vs-baseline trial scatter and the cross-trial summary boxplot).
# Keyed by the OBJECTIVES column name so both visualizers pull from one
# source and the figures stay color-consistent. Vermillion is reserved
# for the "Optimal" reference line on the summary plots, so no objective
# uses it. Yellow is skipped — too pale to read on white.
BASELINE_OBJECTIVE_PALETTE: dict[str, str] = {
    "duet_objective":         OKABE_ITO["sky_blue"],
    "mean_pairwise_nll":      OKABE_ITO["bluish_green"],
    "min_pairwise_nll":       OKABE_ITO["orange"],
    "mean_pairwise_hamming":  OKABE_ITO["reddish_purple"],
    "min_pairwise_hamming":   OKABE_ITO["blue"],
}


# DNA-base palette for ATCG sequencing-error plots. Picks four hues from
# matplotlib's `tab10` palette that don't appear in METHOD_PALETTE — the
# two palettes are placed side-by-side in manuscript figures and need to
# read as different categorical concepts. The four hues are also
# maximally separated from each other (red / purple / pink / yellow-green)
# so adjacent ATCG lines stay visually distinct.
DNA_BASE_PALETTE: dict[str, str] = {
    "A": "#D62728",  # tab:red
    "T": "#9467BD",  # tab:purple
    "C": "#E377C2",  # tab:pink
    "G": "#BCBD22",  # tab:olive
}


# Widths in inches: the slot widths of the 180 mm page with a 4 mm gap between
# panels (duet.plotting.layout.slot_width_mm), so panels at these widths tile
# the page exactly. half_page (88 mm) is also the Nature Methods 1-column width.
# Heights are intentionally free — pick per-plot to fit content.
FIGURE_WIDTHS: dict[str, float] = {
    "third_page":      _slot_width_mm(3) / _MM_PER_INCH,          # 2.257"  57.33 mm — 1 of 3 per row
    "half_page":       _slot_width_mm(2) / _MM_PER_INCH,          # 3.465"  88 mm    — 1 of 2 per row
    "two_thirds_page": _slot_width_mm(3, span=2) / _MM_PER_INCH,  # 4.672"  118.67 mm — 2 of 3 columns
    "full_page":       _slot_width_mm(1) / _MM_PER_INCH,          # 7.087"  180 mm   — whole row
}


def sequential_shades(base_hex: str, n: int) -> list[str]:
    """Return ``n`` shades of ``base_hex``, lightest first, darkest last.

    Shades are white-tints of the base hue (mix fraction 0.35 -> 1.0), so the
    hue stays recognizable while luminance decreases monotonically. Use for a
    family of related fronts (e.g. one per imaging-round count) where the
    group ordering should read as light -> dark. ``n == 1`` returns the base.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    base = _np.array(_mcolors.to_rgb(base_hex))
    if n == 1:
        return [_mcolors.to_hex(base)]
    white = _np.ones(3)
    fracs = _np.linspace(0.35, 1.0, n)
    return [_mcolors.to_hex(white + f * (base - white)) for f in fracs]


# All-baselines overlay: the Figure 3 distribution stack that draws every OPS
# baseline and DUET on one panel. Two Sivanandan floors coexist there, so the
# weaker one (ED=1) takes a lighter tint of the family hue while ED=2 keeps
# METHOD_PALETTE["Sivanandan et al."]. The tint is the middle step of a
# three-step ramp to white; the first step of a two-step ramp was too pale to
# read where it overlapped the grey and green curves. Keyed by the full
# results.csv method label, because the tint belongs to one member of a
# family, not to the family. Methods absent from this table keep their
# METHOD_PALETTE hue.
OVERLAY_VARIANT_PALETTE: dict[str, str] = {
    "Sivanandan et al. (ED=1)": sequential_shades(OKABE_ITO["orange"], 3)[1],
}


def delta_diverging_cmap(name: str = "duet_delta") -> _mcolors.LinearSegmentedColormap:
    """Diverging colormap for signed differences, anchored on Okabe-Ito.

    Blue at the negative end, white at zero, vermillion at the positive end.
    Intended for heatmaps of a signed delta (e.g. DUET minus a baseline) where
    the sign carries meaning and zero must be visually neutral. Pair it with a
    symmetric normalization such as ``TwoSlopeNorm(vcenter=0)`` so white lands
    on zero rather than on the midpoint of the data range.
    """
    return _mcolors.LinearSegmentedColormap.from_list(
        name,
        [OKABE_ITO["blue"], "#FFFFFF", OKABE_ITO["vermillion"]],
    )


def gain_sequential_cmap(name: str = "duet_gain") -> _mcolors.LinearSegmentedColormap:
    """Single-hue sequential map for a metric where higher is better.

    Truncated ``Reds`` (the top 10% is dropped so the darkest cells stay legible
    under white annotation text). Pair it with ``vmin=0`` so colour encodes
    magnitude from "no difference" upward, matching the published Figure 3C
    convention: one hue per panel, chosen by the metric's direction rather than
    by the observed sign of the data.

    Only use this when every cell in the panel has the expected sign. A panel
    containing a wrong-sign cell must use :func:`delta_diverging_cmap` instead;
    clamping at zero would render a real loss identically to a true zero.
    """
    return _mcolors.LinearSegmentedColormap.from_list(
        name, _mpl.colormaps["Reds"](_np.linspace(0.0, 0.9, 256))
    )


def reduction_sequential_cmap(name: str = "duet_reduction") -> _mcolors.LinearSegmentedColormap:
    """Single-hue sequential map for a metric where lower is better.

    Reversed ``Blues``, so the most negative (largest reduction) cell is the
    darkest. Pair it with ``vmax=0``. The counterpart to
    :func:`gain_sequential_cmap`; the same wrong-sign caveat applies.
    """
    return _mcolors.LinearSegmentedColormap.from_list(
        name, _mpl.colormaps["Blues_r"](_np.linspace(0.1, 1.0, 256))
    )
