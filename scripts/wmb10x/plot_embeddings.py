"""Publication SVGs for the WMB-10X cell-type landscape.

Reads ONLY the precomputed tables in ``results/wmb10x/`` (plus the cluster metadata
in ``data/processed/WMB-10X/``). No embedding is ever recomputed here: this module is
pure presentation, so the figures cannot silently disagree with the coordinates that
``embed_centroids.py`` shipped.

Scientific framing, repeated wherever it matters
------------------------------------------------
Every point is one of the 5,322 transcriptomic CLUSTER CENTROIDS of the Allen
whole-mouse-brain 10x atlas (Yao et al. 2023), not one of its ~4.04M cells.
Cell-level data was never downloaded (94.7 to 150 GB). Within-cluster spread is
therefore absent by construction, so the islands here are sparser and cleaner
than the published Fig. 1B, which draws every cell.

THAT CAVEAT NOW LIVES IN THE CAPTION, NOT ON THE FIGURE. The on-figure footnote line
is gone from every figure, panel and diagnostic alike: a publication panel gets its
caveat from its caption, and a footnote baked into the SVG would be typeset twice on
the page and in a font the journal never asked for. results/wmb10x/captions.md is
therefore the ONLY place the centroid caveat is written, it is rewritten on every run
so it cannot drift from the coordinates, and every caption here carries it in full.

The publication panel is half page and carries no size encoding
---------------------------------------------------------------
wmb_umap_panel.svg is the primary artefact: FIGURE_WIDTHS["half_page"] (89 mm), for
assembly beside other panels in Inkscape. NO panel letter is drawn: letters are added
when the panels are assembled into a figure. This script writes only under
results/wmb10x/, never into a compiled figure.

Its points are one fixed size. The full-page figures map point DIAMETER affinely onto
log10(cell count), and at 183 mm that reads; at 89 mm it does not, and it was rendered
both ways and looked at before it was cut. Clusters run 9 to 264,669 cells but 91.4% of
them hold under 1,000, so the encoding spends nearly half its diameter ramp (1.1 to 3.3
of 1.1 to 6.0 pt) on 8.6% of the points: at 89 mm the whole bulk of the data lands
inside one barely-discriminable step, while the fat dots of the 53 clusters over 10,000
cells merge their neighbours and destroy the streak texture that the uniform dot
resolves. An encoding a reader cannot decode is not information, it is just ink, and it
was costing a size key besides. So the panel uses a uniform PANEL_DOT and spends the
freed room on a bigger, cleaner dot. Cluster size is in the tables. The full-page
wmb_pca_classes.svg keeps the encoding, and its size key, unchanged.

Resemblance to the published Fig. 1B is QUALITATIVE, judged by eye against the printed
panel. The published per-cell coordinates live in the cell-level metadata that was never
downloaded, so no point-for-point comparison was made and none is claimed.

The 34-class palette is Allen's official ``color_hex_triplet``, the documented
exception to the repo's shared-palette rule (the shared palettes do not cover a
34-class taxonomy).

The metric is ours, and the topology claim is generated from measurements
--------------------------------------------------------------------------
The Yao et al. Methods specify ``n_neighbors = 25`` and ``min_dist = 0.4`` and never
state a distance metric. cosine is OUR choice.

Every sentence this module writes about the SHAPE of the layout is interpolated from
``embedding_summary.json["island_structure"]``, which ``embed_centroids.py`` measures
(single-linkage components, continent fraction, each island's gap as a fraction of the
layout span). It is not written from an impression of the picture. An earlier pass wrote
it from an impression and shipped a caption claiming cosine puts 01 IT-ET Glut inside the
main mass; 98.8% of the 01 centroids are in fact in a detached island. Numbers, then
prose, and an assertion between them.

What the measurements say, and what the captions therefore say: BOTH metrics detach the
same handful of groups, because at centroid level those groups are nearly disconnected in
the kNN graph. cosine is the main embedding because it keeps the gaps PROPORTIONATE (01 at
23% of the layout span, the non-neuronal block at 7%), so the continent of neuronal
territories survives as the dominant structure, as in Fig. 1B. euclidean blows the same
gaps out to 75% and 53%, and those islands then set the layout scale and crush the
neuronal core into a blob. Where we still differ from Fig. 1B (01 IT-ET Glut and 10 LSX
GABA are interior there and islands here) is stated in every caption, not glossed.

No class labels are drawn inside the point clouds
-------------------------------------------------
Classes are identified by the 34-entry colour key alone, exactly as Fig. 1B does.
An earlier pass stamped the two-digit class codes into the clouds, which cost a
knot of overlapping boxes in the PCA core, a row of labels banished under the plot
trailing a fan of leader lines, and a "04" sitting on class 01's cloud that read as
a mislabel even though it was correctly anchored. All of that machinery is gone.
The only text inside an axes is the two keys and the UMAP axis arrows. The variants
grid's one explanatory note hangs under its euclidean row, outside every panel.

Allen's palette holds near-degenerate pairs, and the captions name them all. They are
computed here (CIE76 dE over all 561 class pairs), together with how far apart the two
classes actually sit in the main embedding, because "position disambiguates them" is
true for a pair on opposite sides of the layout and false for the brown and grey
non-neuronal classes, which are both confusable AND adjacent in the same rim satellite.

Layout: text is placed in measured emptiness, and that is asserted
------------------------------------------------------------------
Where a key sits is decided by measurement, never by eye. ``measure_legend_inches`` sizes a
legend in inches before the figure exists, ``corner_void`` measures the tallest empty
rectangle in a named corner of the layout IN DATA UNITS, and ``assert_no_ink_under`` then
proves that not one of the 5,322 centroids lies beneath any key or annotation that was
placed inside an axes.

On the 89 mm panel that machinery is what puts the class key BELOW the scatter rather than
inside it. At 7 pt the 34-entry key is 2.1 x 2.2 in at two columns and 3.2 x 1.5 in at
three, against an axes 3.4 x 3.1 in: no corner of this layout holds a void that size, and
``fig_umap_panel`` computes the shortfall and asserts it rather than asserting the
placement it happens to have chosen. Type is never shrunk below the stylesheet to force a
fit. A taller panel is cheap; illegible type is not.

The variants grid used to write its euclidean note inside panel c, threaded through the
corridor between the flung-out islands and the crushed continent. That corridor is too
tight for it: at 180 mm the centred rows the search tried cleared the data by at most
2.4 pt, and the only in-panel positions that held the 3 pt floor sat flush against the
panel frame. The note now hangs directly under the euclidean row, where it cannot reach a
centroid, and ``assert_text_clears_axes`` holds it the same 3 pt off every panel frame,
title, letter and the key, on the finished, frozen layout.

Display transforms of the UMAPs
-------------------------------
A UMAP embedding has no canonical orientation and no meaningful absolute scale:
rotation and reflection are isometries, and the size of an embedding is a free
parameter of the optimisation, not a property of the data.

ROTATION (all UMAP panels). Each embedding is rotated to put its principal axis
horizontal, and in the variants grid each panel is additionally reflected into the
main panel's orientation. Isometric: an assertion checks that every pairwise
distance survives.

SCALE (variants grid). Each ROW gets one frame SHAPE, so the three panels of an
n_neighbors sweep are directly comparable and none is letterboxed against its
neighbours, and each panel is scaled to fill its frame. The two rows differ because
the euclidean layouts are wide thin streaks and the cosine ones are nearly square;
one shape for all six either squashes the cosine panels into the middle of a wide
frame or hangs a band of air over every euclidean panel. The tempting alternative,
rescaling all six to a common RMS radius so that apparent compactness is comparable,
was implemented and rendered and is worse: the euclidean embeddings place two islands
far from a compact core, so their RMS is set by the islands, and on any single linear
scale that still shows those islands the core collapses to an indistinct blob. Since
no linear scale can show both, the panels are scaled independently and the caption
says so in as many words: apparent cluster size is NOT comparable across panels, only
topology is.

The axes are drawn bare precisely because the numbers carry no meaning.

Outputs, all under results/wmb10x/ (gitignored, regenerable)
------------------------------------------------------------
THE PANEL (half page, 89 mm; the publication artefact)
figures/wmb_umap_panel.svg        cosine UMAP + the 34-entry class key. Self-contained.
figures/wmb_umap_panel_nokey.svg  the same scatter, alone: points and axis arrows only.
figures/wmb_umap_key.svg          the 34-entry class key, alone, on its own 89 mm canvas.

The panel is shipped whole AND split, because a compiled supplementary figure may want one
shared key serving several panels. The split pair is drawn from the same arrays as the
whole panel, and ``assert_identical_scatter`` compares the rendered collections point for
point (offsets, areas, RGBA) so the two can never drift apart.

THE DIAGNOSTICS (full page; not publication panels)
figures/wmb_pca_classes.svg    PC1 vs PC2, all-gene PCA, equal aspect, size key
figures/wmb_pca_scree.svg      all-gene bars + BOTH cumulative curves (all-gene and
                               marker-gene, the latter being what the UMAP consumed)
figures/wmb_umap_variants.svg  metric x n_neighbors grid, one frame shape per row

captions.md                    the caption text, persisted so it cannot drift, and the
                               ONLY place the centroid caveat now lives

Reproduce:
  cd scripts/wmb10x
  PYTHONPATH=../../src python plot_embeddings.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from duet.plotting import FIGURE_WIDTHS, OKABE_ITO, apply_style

apply_style()

ROOT = Path(__file__).resolve().parents[2]  # scripts/wmb10x/x.py -> repo root
PROC = ROOT / "data" / "processed" / "WMB-10X"  # cluster_metadata.csv (the join product)
OUTPUTS = ROOT / "results" / "wmb10x"           # coordinate tables + captions.md
FIGURES = OUTPUTS / "figures"                   # the SVG panels
PNG_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "wmb_figure_review"

RNG_SEED = 0
SVG_DPI = 600  # the scatter layers are rasterized; 300 stair-steps 1-2 px discs

# Marker DIAMETER in points, mapped affinely from log10(cell count). Diameter, not area:
# an affine map onto AREA over any range small enough to keep 5,322 points legible gives
# the key three swatches a reader cannot tell apart (the previous [1.5, 13] pt^2 map bought
# 0.85 pt of diameter for a 100x change in cell count). Diameter buys 1.1 pt per decade,
# which is a distinction the ink can actually make.
DIA_MIN, DIA_MAX = 1.1, 6.0
# The key brackets the data. The clusters run from 9 to 264,669 cells with a MEDIAN of 122,
# and 91.4% of them hold under 1,000; a key that starts at 1,000 explains the top 8.6% of
# the figure and nothing else.
SIZE_KEY_COUNTS = (10, 100, 1_000, 10_000, 100_000)
VARIANT_DOT = 1.6 ** 2  # fixed area: the variants grid carries no size key, so no encoding

# The 89 mm publication panel: ONE fixed dot, and no size key. See the module docstring for
# why (rendered both ways at 89 mm and looked at: 91.4% of clusters sit inside 1.1 to 3.3 pt
# of the 1.1 to 6.0 pt ramp, which is not a step a reader can resolve at print size, while
# the fat end merges neighbouring centroids and blurs the streak texture away). 2.2 pt is
# also chosen by looking: 1.8 goes faint at print, 2.6 starts closing the gaps inside the
# continent. matplotlib's ``s`` is an AREA in pt^2, hence the square.
PANEL_DIA = 2.2
PANEL_DOT = PANEL_DIA ** 2
PANEL_KEY_NCOL = 3  # measured, not guessed: 3.16 in wide at 7 pt, inside the 3.50 in page

# A colour pair closer than this in CIE76 is called out in the captions by name.
DE_CONFUSABLE = 25.0
# Two classes closer than this (as a fraction of the layout span) are ADJACENT, so
# position cannot be offered as the thing that tells them apart. The cut is not arbitrary:
# the measured separations of the confusable pairs are strongly bimodal (all of them are
# either under 5.1% of span or over 12.3%, with nothing in between), so 8% falls in an
# empty gap. AMBIGUOUS_BAND asserts that the gap is still there; if a pair ever lands in
# it, the binary "position rescues them / it does not" claim is no longer supported by the
# data and the run must fail rather than ship it.
ADJACENT_FRAC_OF_SPAN = 0.08
AMBIGUOUS_BAND = (0.06, 0.10)
# The glial, vascular and immune classes. Confusable in colour AND sharing one small rim
# satellite, which is the one place on the figure where colour is genuinely not enough.
NON_NEURONAL = ("30 Astro-Epen", "31 OPC-Oligo", "32 OEC", "33 Vascular", "34 Immune")

# There is deliberately no FOOTNOTE constant any more. The centroid caveat is a CAPTION,
# and it lives in captions.md alone; see the module docstring.


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_tables() -> dict:
    # cluster_metadata.csv is the join product and lives with the matrix it indexes,
    # in data/processed/WMB-10X; everything else here is a result of this analysis.
    meta = pd.read_csv(PROC / "cluster_metadata.csv")
    pca = pd.read_csv(OUTPUTS / "pca_coords.csv")
    var = pd.read_csv(OUTPUTS / "pca_variance.csv")
    umap = pd.read_csv(OUTPUTS / "umap_coords.csv")
    variants = pd.read_csv(OUTPUTS / "umap_variants.csv")
    with open(OUTPUTS / "embedding_summary.json") as fh:
        summary = json.load(fh)

    # The coordinate tables must be in the same row order as the metadata, or
    # every colour on every figure is a lie. Assert it rather than trust it.
    for name, df in (("pca_coords", pca), ("umap_coords", umap)):
        assert len(df) == len(meta), f"{name}: {len(df)} rows vs meta {len(meta)}"
        assert (df["row_index"].to_numpy() == meta["row_index"].to_numpy()).all()
        assert (df["cluster_id"].to_numpy() == meta["cluster_id"].to_numpy()).all(), name
    for variant, sub in variants.groupby("variant"):
        assert (sub["cluster_id"].to_numpy() == meta["cluster_id"].to_numpy()).all(), variant

    # No unqualified variance column may exist to be grabbed by mistake: the whole point
    # of the two-curve scree is that the all-gene PCs are NOT what the UMAP consumed.
    for banned in ("cumulative", "explained_variance_ratio"):
        assert banned not in var.columns, (
            f"pca_variance.csv carries an unqualified '{banned}' column again. It holds "
            "the ALL-GENE curve, and an unqualified name is exactly what a caption "
            "reaches for when it means the marker-gene curve the UMAP actually used."
        )
    assert "cumulative_variance_100pc" not in summary, (
        "embedding_summary.json carries an unqualified cumulative_variance_100pc again; "
        "use all_gene_pca_* or marker_pca_*, which name the space they belong to"
    )

    # Which variant is the headline embedding is DATA, not a constant in this file:
    # the recompute stage stamps it into the summary and flags it in the variants
    # table. Read it from both and make them agree, so the framed panel of the grid
    # can never drift away from the embedding that wmb_umap_panel.svg actually draws.
    main = summary["main_variant"]
    flagged = sorted(variants.loc[variants["is_main"].astype(bool), "variant"].unique())
    assert flagged == [main], (
        f"embedding_summary.json calls {main!r} the main variant but umap_variants.csv "
        f"flags {flagged}; the figures would disagree with the tables"
    )
    ref = variants.loc[variants["variant"] == main, ["UMAP1", "UMAP2"]].to_numpy()
    assert np.allclose(ref, umap[["UMAP1", "UMAP2"]].to_numpy(), atol=1e-5), (
        f"umap_coords.csv is not the {main!r} variant of umap_variants.csv"
    )

    classes = (
        meta.groupby(["class_id", "class_name", "class_color"], as_index=False)
        .agg(n_clusters=("cluster_id", "size"), n_cells=("n_cells", "sum"))
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    assert len(classes) == 34, len(classes)
    assert classes["class_color"].nunique() == 34, "class colours are not unique"
    return {
        "meta": meta,
        "pca": pca,
        "var": var,
        "umap": umap,
        "variants": variants,
        "summary": summary,
        "classes": classes,
        "main_variant": main,
    }


def size_mapper(n_cells: np.ndarray, dia_lo: float = DIA_MIN, dia_hi: float = DIA_MAX):
    """Marker AREA (matplotlib's ``s``) for a diameter that is affine in log10(count).

    Returns ``s = d**2`` where ``d`` runs linearly from ``dia_lo`` to ``dia_hi`` across the
    observed range of log10(cell count). Affine, not proportional: the smallest cluster
    still gets a visible dot, so a caption must say "increases with", never "proportional
    to". Diameter rather than area is what makes the size key readable at all; see
    DIA_MIN / DIA_MAX above.
    """
    z = np.log10(n_cells.astype(float))
    z0, z1 = float(z.min()), float(z.max())

    def f(n):
        zz = (np.log10(np.asarray(n, dtype=float)) - z0) / (z1 - z0)
        return (dia_lo + (dia_hi - dia_lo) * zz) ** 2

    return f


def rigid_rotation(xy: np.ndarray, align_to: np.ndarray | None = None) -> np.ndarray:
    """Rotate 2-D coords so the principal axis is horizontal. Isometry, asserted."""
    c = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    out = c @ vt.T
    if align_to is not None:
        ref = align_to - align_to.mean(axis=0)
        for k in range(2):
            if np.dot(out[:, k], ref[:, k]) < 0:
                out[:, k] = -out[:, k]
    d0 = np.linalg.norm(xy[:120, None, :] - xy[None, :120, :], axis=-1)
    d1 = np.linalg.norm(out[:120, None, :] - out[None, :120, :], axis=-1)
    assert np.allclose(d0, d1, atol=1e-4), "rotation changed pairwise distances"
    return out


# --------------------------------------------------------------------------
# colour degeneracy in Allen's palette, measured
# --------------------------------------------------------------------------
def _srgb_to_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (m @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def confusable_pairs(classes: pd.DataFrame, xy: np.ndarray, meta: pd.DataFrame) -> list:
    """Class colour pairs a reader cannot separate, and whether position rescues them.

    Allen's 34-class palette is the sanctioned exception to the repo's shared-palette
    rule, and we do not repaint it: it is the palette of the published figure, and the
    same degeneracies are visible in Fig. 1B itself. What we owe the reader instead is an
    honest list. A caption that discloses one near-degenerate pair and quietly omits four
    others is worse than one that discloses none.

    "Position disambiguates them" is a claim that has to be checked, not assumed. It is
    true for two classes at opposite ends of the layout and FALSE for the brown and grey
    non-neuronal classes, which are confusable in colour AND abutting in the same rim
    satellite. So the separation is measured too, as a fraction of the layout span.
    """
    lab = {r.class_name: _srgb_to_lab(r.class_color) for r in classes.itertuples()}
    pts = {r.class_name: xy[(meta["class_name"] == r.class_name).to_numpy()]
           for r in classes.itertuples()}
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))

    out = []
    names = list(lab)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            de = float(np.linalg.norm(lab[a] - lab[b]))
            if de >= DE_CONFUSABLE:
                continue
            gap = float(np.linalg.norm(pts[a][:, None, :] - pts[b][None, :, :],
                                       axis=-1).min()) / span
            assert not (AMBIGUOUS_BAND[0] < gap < AMBIGUOUS_BAND[1]), (
                f"{a} / {b} sit {gap:.1%} of the layout span apart, inside the band the "
                "caption's binary claim assumes is empty. Either report the separations "
                "without bucketing them, or move the cut for a stated reason. Do not "
                "quietly widen it."
            )
            out.append({"a": a, "b": b, "delta_e": de, "separation_frac_of_span": gap,
                        "adjacent": gap < ADJACENT_FRAC_OF_SPAN})
    out.sort(key=lambda p: p["delta_e"])
    # The non-neuronal block is the case the reader will actually hit, so it is checked
    # by name rather than left to fall out of a threshold: these classes are browns and
    # greys and they share one small satellite.
    nn_de = [np.linalg.norm(lab[a] - lab[b])
             for i, a in enumerate(NON_NEURONAL) for b in NON_NEURONAL[i + 1:]]
    return {"pairs": out, "non_neuronal_min_delta_e": float(min(nn_de))}


# --------------------------------------------------------------------------
# placing text in measured emptiness
# --------------------------------------------------------------------------
def measure_legend_inches(fig_w: float, handles, ncol: int, fontsize: float,
                          **kw) -> tuple[float, float]:
    """Width and height of a legend, in inches, measured on a scratch figure.

    A legend's size depends on its text, its font and its column count, and on nothing
    about the axes it will eventually live in. So it can be measured BEFORE the real
    figure exists, which is what lets fig_umap size its axes to fit its keys instead of
    discovering the collision after the fact. The scratch figure is closed immediately and
    never written anywhere.
    """
    scratch = plt.figure(figsize=(fig_w, fig_w))
    ax = scratch.add_subplot(111)
    ax.axis("off")
    leg = ax.legend(handles=handles, ncol=ncol, fontsize=fontsize, **kw)
    scratch.canvas.draw()
    bb = leg.get_window_extent(scratch.canvas.get_renderer())
    w, h = bb.width / scratch.dpi, bb.height / scratch.dpi
    plt.close(scratch)
    return w, h


def measure_text_inches(fig_w: float, s: str, **text_kw) -> tuple[float, float]:
    """Width and height of a text block, in inches, measured on a scratch figure.

    The text counterpart of measure_legend_inches: the variants grid sizes the empty row
    under its euclidean panels from the note it carries, not from a typed constant. It is
    measured at the scratch figure's dpi, where hinting makes a 6 pt line come out about
    1 pt shorter than its ink at print resolution. That only sets a height ratio; the gap
    itself is checked on the finished figure.
    """
    scratch = plt.figure(figsize=(fig_w, fig_w))
    t = scratch.text(0.5, 0.5, s, ha="center", va="center", **text_kw)
    scratch.canvas.draw()
    bb = t.get_window_extent(scratch.canvas.get_renderer())
    w, h = bb.width / scratch.dpi, bb.height / scratch.dpi
    plt.close(scratch)
    return w, h


def corner_void(xy: np.ndarray, xlim, ylim, width: float, corner: str) -> float:
    """Height (in DATA units) of the tallest empty rectangle of the given width, anchored
    in the named corner of the axes.

    This is how the figure knows where its own emptiness is. The cosine layout has a large
    hole in its upper left and a smaller one in its upper right, and that is where the two
    keys go: not because a corner "looked empty", but because no centroid is in the
    rectangle the key will occupy.
    """
    if corner == "upper left":
        sel = xy[:, 0] <= xlim[0] + width
        return ylim[1] - (xy[sel, 1].max() if sel.any() else ylim[0])
    if corner == "upper right":
        sel = xy[:, 0] >= xlim[1] - width
        return ylim[1] - (xy[sel, 1].max() if sel.any() else ylim[0])
    raise ValueError(corner)


LABEL_CLEARANCE_PT = 3.0
"""Minimum gap, in POINTS ON THE PAGE, between an annotation's text box and the nearest dot.

A binary "does anything overlap" test passes just as happily at 0.1 pt as at 10, and 0.1 pt
is a label touching the data. So the check below measures the gap and holds it to a floor.
3 pt is a hair over one dot diameter (2.2 pt) at print size, which is the smallest gap that
still reads as a gap rather than as contact. The variants grid's euclidean note, which sits
outside every panel, is held to the same floor against the panels' frames, titles and
letters (assert_text_clears_axes), for the same reason.
"""


def _clearance_pt(art, disp: np.ndarray, radius: float, to_pt: float, rend) -> float:
    bb = art.get_window_extent(rend)
    dx = np.maximum(np.maximum(bb.x0 - disp[:, 0], disp[:, 0] - bb.x1), 0.0)
    dy = np.maximum(np.maximum(bb.y0 - disp[:, 1], disp[:, 1] - bb.y1), 0.0)
    return float(np.min(np.hypot(dx, dy))) * to_pt - radius


def _dot_radius_pt(sizes) -> float:
    return float(np.sqrt(np.max(np.asarray(sizes, dtype=float)) / np.pi))


def label_clearance_pt(fig, ax, xy: np.ndarray, sizes, artists: dict) -> dict:
    """Gap in points between each annotation's text box and the nearest centroid's INK.

    Measured on the rendered page, not in data coordinates, because points on the page is
    the unit the reader's eye works in and the only unit in which "nearly touching" means
    anything. The marker radius is subtracted, so this is the gap to the edge of the nearest
    dot and not to its centre. A negative value means the box is drawn over live data.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    disp = ax.transData.transform(xy)          # pixels
    to_pt = 72.0 / fig.dpi
    radius = _dot_radius_pt(sizes)
    return {name: _clearance_pt(art, disp, radius, to_pt, rend)
            for name, art in artists.items()}


def assert_text_clears_axes(fig, art, others: dict) -> dict:
    """A text block set OUTSIDE every panel must stay on the page and off every panel.

    For the euclidean note of the variants grid, which lives in a strip of its own between
    the two rows. It cannot sit on a centroid there (every centroid is inside a panel), so
    the check that still means something is the one against the panels themselves: the
    note's box must clear each panel's tight box (frame, title and panel letter; the
    whole key axes for the key) by LABEL_CLEARANCE_PT, measured in points on the FINISHED
    page, and it must not run off the canvas. Run it after the last axes exists and the
    layout is frozen: constrained_layout re-flows the whole grid whenever an axes is added,
    and again on every draw, so a check made on a partly built or still-flowing figure
    measures a layout that never ships.

    Returns the gap to each named axes, in pt, so the run can print them.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    tb = art.get_window_extent(rend)
    page = fig.bbox
    assert (tb.x0 >= page.x0 and tb.x1 <= page.x1
            and tb.y0 >= page.y0 and tb.y1 <= page.y1), (
        f"the note's box ({tb.width * to_pt:.1f} x {tb.height * to_pt:.1f} pt) runs off the "
        "canvas. Wrap it onto another line; do not shrink the font."
    )
    gaps = {}
    for name, ax in others.items():
        ob = ax.get_tightbbox(rend)
        dx = max(ob.x0 - tb.x1, tb.x0 - ob.x1)
        dy = max(ob.y0 - tb.y1, tb.y0 - ob.y1)
        # Apart along either direction: the gap is the distance between the two boxes.
        # Overlapping along both: negative, by the shallower of the two overlaps.
        gap = max(dx, dy) if (dx < 0 and dy < 0) else float(np.hypot(max(dx, 0), max(dy, 0)))
        gaps[name] = gap * to_pt
    for name, g in gaps.items():
        assert g > 0.0, (
            f"the note is drawn over {name} ({-g:.2f} pt of overlap). For panels a to c, "
            "check the note's offset under its row (note_gap_pt); for the cosine panels or "
            "the key, check the height the layout gave the empty row above them."
        )
        assert g >= LABEL_CLEARANCE_PT, (
            f"the note clears {name} by only {g:.2f} pt on the page, under the "
            f"{LABEL_CLEARANCE_PT:.1f} pt floor. For panels a to c, raise note_gap_pt; for "
            "the cosine panels or the key, give the empty row above them more room. Do not "
            "lower the floor."
        )
    return gaps


def assert_no_ink_under(fig, ax, xy: np.ndarray, sizes, artists: dict) -> dict:
    """No data point may lie beneath any key or annotation, and none may nearly lie under it.

    The whole point of the void-measuring above is that this passes for a reason rather than
    by luck. If the embedding ever changes, its holes move, and this fires instead of
    shipping a key stamped over live centroids.

    It used to be a binary overlap test. It is now a clearance floor, because on the 89 mm
    panel the UMAP1 label reaches toward the 02 NP-CT-L6b island and the true gap there is a
    couple of points: "no overlap" was passing on a margin nobody had ever measured. Returns
    the measured clearances so the run can print them.
    """
    clear = label_clearance_pt(fig, ax, xy, sizes, artists)
    for name, c in clear.items():
        assert c > 0.0, (
            f"'{name}' is drawn ON the data ({-c:.2f} pt of overlap with the nearest "
            "centroid). Either the embedding moved or the void it was placed in is gone."
        )
        assert c >= LABEL_CLEARANCE_PT, (
            f"'{name}' clears the nearest centroid by only {c:.2f} pt on the page, under the "
            f"{LABEL_CLEARANCE_PT:.1f} pt floor. It does not overlap, but at print size it "
            "reads as touching the data. Move the annotation; do not lower the floor."
        )
    return clear


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------
def scatter_by_class(ax, xy, meta, sizes, alpha=0.85):
    """Draw the centroids. Returns the PathCollection, which is the ONLY honest record of
    what actually landed on the canvas: assert_identical_scatter reads its offsets, areas
    and RGBA back out rather than re-deriving them from the inputs, so a divergence between
    two figures cannot hide behind an argument that merely looks the same."""
    rng = np.random.default_rng(RNG_SEED)
    order = rng.permutation(len(meta))  # no class systematically drawn on top
    return ax.scatter(
        xy[order, 0], xy[order, 1],
        s=np.asarray(sizes)[order] if np.ndim(sizes) else sizes,
        c=meta["class_color"].to_numpy()[order],
        linewidths=0, alpha=alpha,
        rasterized=True,  # keeps the SVG small; all TEXT stays vector
        zorder=2,
    )


def scatter_fingerprint(fig, ax, coll) -> dict:
    """What a scatter actually put on the page: positions, areas, colours, AND the physical
    size of the axes it was drawn into.

    That last field is the one that matters and it was missing. ``offsets`` are DATA
    coordinates and ``sizes`` are pt^2: both are invariant to how large the axes ends up on
    the page, so a fingerprint made of them alone cannot see a scatter that has been drawn at
    a different scale. It did not see one: the key-less panel used to come out 1.1% larger
    than the keyed one, because the two figures were handed to constrained_layout with
    different padding, and the check passed anyway. The axes rect in points is measured on
    the rendered canvas and compared, so the two files must now be the same picture at the
    same size on paper.
    """
    fig.canvas.draw()
    bb = ax.get_window_extent()
    to_pt = 72.0 / fig.dpi
    return {
        "offsets": np.asarray(coll.get_offsets(), dtype=float),
        "sizes": np.asarray(coll.get_sizes(), dtype=float),
        "colors": np.asarray(coll.get_facecolors(), dtype=float),
        # width, height and left edge. NOT the bottom edge: the keyed figure is taller,
        # because it carries the key underneath, so the axes necessarily sits higher up its
        # own canvas. What must not differ is the size of the axes and its inset from the
        # left, which together fix where every dot lands relative to the 89 mm page.
        "axes_wh_pt": np.array([bb.width, bb.height]) * to_pt,
        "axes_left_pt": bb.x0 * to_pt,
    }


def assert_identical_scatter(a: dict, b: dict, name_a: str, name_b: str) -> None:
    """The whole panel and the key-less panel must be the SAME picture, at the SAME SIZE.

    They are two files precisely so that a compiled figure can choose between one
    self-contained panel and a scatter served by a shared key. If the two disagree by even a
    point, the shared-key assembly is a lie about the panel it claims to key. They are drawn
    from one set of arrays, and then this checks the rendered collections rather than
    trusting that they were.

    The page-geometry half of the check is not decoration. Two scatters can plot identical
    data coordinates, identical marker areas and identical colours and still print at
    different sizes, which is exactly what these two did until the panel family stopped
    letting constrained_layout choose its own margins.
    """
    for field in ("offsets", "sizes", "colors"):
        assert a[field].shape == b[field].shape, (
            f"{name_a} and {name_b} differ in {field} shape "
            f"({a[field].shape} vs {b[field].shape})"
        )
        assert np.array_equal(a[field], b[field]), (
            f"{name_a} and {name_b} do not plot identical {field}. They are the same scatter "
            "with and without its key; if they have drifted apart, a shared key assembled "
            "over the key-less panel keys a picture that is not the one it was measured on."
        )
    dwh = np.abs(a["axes_wh_pt"] - b["axes_wh_pt"])
    dleft = abs(a["axes_left_pt"] - b["axes_left_pt"])
    assert dwh.max() < 0.01 and dleft < 0.01, (
        f"{name_a} draws its axes at {a['axes_wh_pt'][0]:.2f} x {a['axes_wh_pt'][1]:.2f} pt "
        f"(left edge {a['axes_left_pt']:.2f}) and {name_b} at {b['axes_wh_pt'][0]:.2f} x "
        f"{b['axes_wh_pt'][1]:.2f} pt (left edge {b['axes_left_pt']:.2f}). The same data at "
        "two scales is two different pictures: a shared key dropped beside the key-less "
        "panel would key a scatter drawn larger or smaller than the one it was measured on. "
        "Both panels must place their axes at the same explicitly computed rect."
    )


def class_handles(classes) -> list:
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=3.4,
               markerfacecolor=row.class_color, markeredgecolor="none",
               label=row.class_name)
        for row in classes.itertuples()
    ]


CLASS_KEY_KW = dict(handletextpad=0.35, columnspacing=0.9, labelspacing=0.42,
                    borderpad=0.0, borderaxespad=0.0, frameon=False)


def class_key(ax, classes, ncol, fontsize, loc="center left", **kw):
    """The 34-entry colour key. THE ONLY identification of a class on any figure.

    Fig. 1B carries no in-plot text either: it keys its classes by colour and lists
    them in a legend. The entries keep Allen's two-digit prefix ("01 IT-ET Glut"),
    which is how the class is named in the paper and in the taxonomy, not a pointer
    to a label stamped on the cloud.
    """
    opts = dict(CLASS_KEY_KW)
    opts.update(kw)
    return ax.legend(handles=class_handles(classes), loc=loc, ncol=ncol,
                     fontsize=fontsize, **opts)


def size_handles(f) -> list:
    return [
        Line2D([], [], marker="o", linestyle="none",
               markersize=float(np.sqrt(f(n))),  # scatter s is AREA in pt^2
               markerfacecolor="0.45", markeredgecolor="none", label=f"{n:,}")
        for n in SIZE_KEY_COUNTS
    ]


SIZE_KEY_KW = dict(frameon=False, labelspacing=0.75, handletextpad=0.6,
                   borderpad=0.2, borderaxespad=0.0)


def size_key(ax, f, fontsize, loc="upper left", **kw):
    """Key for the point-size channel.

    It brackets the DATA, not a round-number ladder above it: the median cluster holds 122
    cells and 91.4% hold under 1,000, so the old 1,000 / 10,000 / 100,000 key described
    the top 8.6% of the points and left the median dot smaller than its smallest swatch.
    """
    opts = dict(SIZE_KEY_KW)
    opts.update(kw)
    leg = ax.legend(handles=size_handles(f), loc=loc, fontsize=fontsize,
                    title="cells in cluster", **opts)
    leg.get_title().set_fontsize(fontsize)
    leg.set_zorder(5)
    return leg


def save(fig, name, require_strings=(), width_key="full_page", forbid_strings=()):
    """Write the SVG, a PNG review copy outside the repo, and verify the text.

    The stylesheet's ``savefig.bbox: tight`` is switched OFF here, in an rc_context, and
    the canvas width is then asserted. Tight cropping shrink-wraps the canvas to the ink,
    which produced four SVGs at three different widths (527.5 and 524.3 pt) when the whole
    point of FIGURE_WIDTHS is that a figure drops into a layout at 100%: scaling them to
    183 mm afterwards lands the same 7 pt key text at two different final sizes across a
    figure set that is meant to be typographically identical. The diagnostics manage their
    margins with constrained_layout and the three panel files place their axes at an explicit
    rect (see _panel_figure), so either way the canvas comes out exactly the width that was
    asked for and the type lands where the stylesheet intends. (Passing ``bbox_inches=None``
    is NOT enough: matplotlib reads that as "consult the rcParam".)

    ``width_key`` names the FIGURE_WIDTHS entry the canvas must come out at, and there are
    exactly two here: the three UMAP panel files are "half_page" (89 mm, the width they will
    be dropped into an Inkscape assembly at) and the three diagnostics are "full_page".

    ``require_strings`` must come out as editable SVG <text>, never as frozen <path> glyph
    outlines. This is the one text check worth keeping: the stylesheet sets
    svg.fonttype: none so that Inkscape sees real text boxes, but a stroke path effect (the
    obvious way to halo something) silently defeats it and the file still LOOKS perfect
    until someone opens it and finds vector shapes. What has to survive is everything a
    human reads: axis labels, the 34 key entries, the size key and on-figure annotations.
    The old gid-tagged class-label audit is gone with the labels it policed.

    ``forbid_strings`` is the other half of that check and it is what keeps the split panel
    honest: wmb_umap_panel_nokey.svg must contain NO class name and NO key text, or it is
    not the bare scatter it claims to be and a shared key placed beside it would be the
    second copy of a key already in the file.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    svg = FIGURES / f"{name}.svg"
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(svg, dpi=SVG_DPI)
        # Opaque PNG so the (correctly) transparent SVG background does not read as
        # a black rectangle in a viewer. Review copy only, never in the repo.
        fig.savefig(PNG_DIR / f"{name}.png", dpi=200, transparent=False,
                    facecolor="white")
    want_pt = FIGURE_WIDTHS[width_key] * 72
    fig_h = fig.get_size_inches()[1]
    plt.close(fig)

    content = svg.read_text()
    got_pt = float(re.search(r'width="([\d.]+)pt"', content).group(1))
    assert abs(got_pt - want_pt) < 1.0, (
        f"{svg}: canvas is {got_pt:.1f} pt wide, not the {want_pt:.1f} pt of "
        f"FIGURE_WIDTHS[{width_key!r}]. It cannot be dropped in at 100%, and rescaling it "
        "drifts the type away from the stylesheet target."
    )

    rendered = {t.strip() for t in re.findall(r"<text[^>]*>(.*?)</text>", content, flags=re.S)}
    missing = [s for s in require_strings if s not in rendered]
    assert not missing, (
        f"{svg}: {len(missing)} strings are not editable <text> (e.g. {missing[:4]}). "
        "Something rendered them as <path> outlines and broke svg.fonttype: none."
    )
    present = [s for s in forbid_strings if s in rendered]
    assert not present, (
        f"{svg}: {len(present)} strings that must not be on this figure are (e.g. "
        f"{present[:4]}). This file is meant to carry no key text at all."
    )
    assert not re.search(r'<g id="classlabel-', content), (
        f"{svg}: an in-plot class label came back; Fig. 1B has none and neither do we"
    )
    # The centroid caveat is a caption, not ink on the panel. If a footnote ever creeps back
    # onto a figure, it will be typeset twice on the page, in a font the journal did not set.
    assert not any("cluster centroids" in t for t in rendered), (
        f"{svg}: the centroid footnote is back on the figure. It belongs in captions.md, "
        "which is now the only place it is written."
    )

    emitted = content.count("<text")
    mb = svg.stat().st_size / 1e6
    assert emitted > 0, f"{svg}: no <text> elements at all"
    assert mb < 5.0, f"{svg}: {mb:.2f} MB is too large"
    print(f"  {svg}  {mb:.2f} MB  {got_pt:.1f} x {fig_h * 72:.1f} pt ({width_key})  "
          f"({emitted} <text>; {len(require_strings)} strings verified editable)")
    return {"path": str(svg), "size_mb": round(mb, 3), "n_text": emitted,
            "width_in": round(want_pt / 72, 3), "height_in": round(fig_h, 3)}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_pca(d):
    """PC1 vs PC2 of the ALL-GENE PCA. Colour key only, no labels in the cloud.

    PC1 vs PC2 collapses the subcortical neuronal classes onto one another. That is
    a fact about the projection, not a layout problem, and the previous attempt to
    label through it (a banished row of boxes under the cloud, each trailing a
    leader line) made the figure worse than the collapse it was documenting. The
    overlap is now simply shown, and the key names the colours.
    """
    meta, var = d["meta"], d["var"]
    xy = d["pca"][["PC1", "PC2"]].to_numpy()
    pc1 = 100 * var.loc[var.pc == 1, "explained_variance_ratio_all_genes"].item()
    pc2 = 100 * var.loc[var.pc == 2, "explained_variance_ratio_all_genes"].item()

    # EQUAL ASPECT. PC1 and PC2 are in the same units and their spreads are the
    # square roots of the eigenvalues; stretching PC2 to fill the box makes the
    # weaker axis look as important as the stronger one, which is the one thing a
    # PCA scatter must not do. The panel is therefore short and wide, and that
    # shape IS the 19.1% against 8.4%. The key rides under it: unlike the UMAP, this
    # panel is a wide letterbox with no large corner void to put a 34-entry key in.
    xspan = float(np.ptp(xy[:, 0])) * 1.06
    yspan = float(np.ptp(xy[:, 1])) * 1.12
    ax_w = FIGURE_WIDTHS["full_page"] - 0.75
    ax_h = ax_w * yspan / xspan
    key_h = 1.05
    # + 0.40 in for the x label and the layout's own padding. It was 0.55 while a footnote
    # sat under the x label; the footnote is gone, so the strip it was reserved goes too
    # rather than staying on as a band of dead paper.
    fig = plt.figure(figsize=(FIGURE_WIDTHS["full_page"], ax_h + key_h + 0.40),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(ax_h, key_h))
    ax = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[1, 0])

    f = size_mapper(meta["n_cells"].to_numpy())
    scatter_by_class(ax, xy, meta, f(meta["n_cells"].to_numpy()), alpha=0.82)
    mid = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    ax.set_xlim(mid[0] - xspan / 2, mid[0] + xspan / 2)
    ax.set_ylim(mid[1] - yspan / 2, mid[1] + yspan / 2)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)")

    ax_key.axis("off")
    class_key(ax_key, d["classes"], ncol=6, fontsize=plt.rcParams["legend.fontsize"],
              loc="upper center")
    size_key(ax, f, plt.rcParams["legend.fontsize"], loc="upper left",
             borderaxespad=0.4)
    return save(fig, "wmb_pca_classes", width_key="full_page",
                require_strings=list(d["classes"]["class_name"])
                + [f"PC1 ({pc1:.1f}% variance)", f"PC2 ({pc2:.1f}% variance)",
                   "cells in cluster"])


def fig_scree(d):
    """Variance explained, with BOTH cumulative curves on it.

    The figure previously plotted the all-gene PCA and annotated its 100-PC cumulative as
    "input to the UMAP". It is not. The UMAP consumed a SEPARATE PCA, over the 6,558 marker
    genes, whose 100 PCs capture a different fraction of a different total variance.
    Attributing one space's number to the other's role is a scientific error, so both
    curves are now drawn, each annotated with its own 100-PC value and each named for the
    thing it actually is. The bars remain the all-gene per-PC variance and say so.

    The legend names gene sets, not files. It used to read "plotted in
    wmb_pca_classes.svg" and "INPUT TO THE UMAP", which is a source filename from the
    authors' working directory plus an all-caps shout, on the face of a figure. The
    cross-reference belongs in the caption, where a sibling panel can be named as a panel.
    """
    var = d["var"]  # all 100 PCs: 100 is the number both PCAs hand to their consumer
    # 2.75 in, not 2.9: the bottom strip that used to be reserved for the footnote (the
    # rect= call that stood here) is no longer reserved, because there is no longer a
    # footnote to keep off the "Principal component" label.
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTHS["full_page"], 2.75),
                           constrained_layout=True)
    bar_c = OKABE_ITO["sky_blue"]
    all_c = OKABE_ITO["vermillion"]
    mark_c = OKABE_ITO["bluish_green"]

    ax.bar(var["pc"], 100 * var["explained_variance_ratio_all_genes"], width=0.9,
           color=bar_c, linewidth=0, label="per PC, all 32,285 genes")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained per PC (%)", color=bar_c)
    ax.tick_params(axis="y", colors=bar_c)
    # Room to the right of the PC=100 rule, so the rule reads as an annotation and not as
    # a second right spine drawn 1.5 data units from the first.
    ax.set_xlim(0.0, 106.0)

    ax2 = ax.twinx()
    ax2.plot(var["pc"], 100 * var["cumulative_all_genes"], color=all_c, linewidth=1.3,
             label="cumulative, all 32,285 genes")
    ax2.plot(var["pc"], 100 * var["cumulative_marker_genes"], color=mark_c, linewidth=1.3,
             linestyle=(0, (4, 1.6)),
             label="cumulative, 6,558 marker genes (the UMAP input)")
    # The right axis governs two series in two colours, so it cannot be colour-coded to
    # either of them; it stays neutral and the legend carries the mapping. The left axis
    # governs exactly one series and is coded to it.
    ax2.set_ylabel("Cumulative variance explained (%)", color="0.25")
    ax2.tick_params(axis="y", colors="0.25")
    # A cumulative fraction cannot exceed 100%, and neither curve passes 68. The old
    # (0, 108) axis implied headroom that cannot exist and squashed two curves 1.3 points
    # apart into its lower two thirds, which is where they need the room most.
    ax2.set_ylim(0, 75)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("0.25")
    ax2.spines["top"].set_visible(False)

    all100 = 100 * var.loc[var.pc == 100, "cumulative_all_genes"].item()
    mark100 = 100 * var.loc[var.pc == 100, "cumulative_marker_genes"].item()
    # Each curve must be the same run as the summary it will be quoted next to.
    # (1e-6, not 0: the CSV holds a float32 written as text, so a bit-exact
    # comparison would fail on the round-trip.)
    assert abs(all100 / 100 - d["summary"]["all_gene_pca_cumulative_variance_100pc"]) < 1e-6
    assert abs(mark100 / 100 - d["summary"]["marker_pca_cumulative_variance_100pc"]) < 1e-6
    assert d["summary"]["pca_100pc_is_input_to_umap"] == "marker_genes", (
        "the summary no longer says the marker-gene PCA is what the UMAP consumed; "
        "this figure's whole point is not to confuse the two spaces"
    )

    # The 100-PC cut is the number the analysis depends on, so it is ON the chart,
    # twice, once per space, never quoted from off the end of it.
    ax2.axvline(100, color="0.55", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    fs = plt.rcParams["xtick.labelsize"]
    for y, c, txt, dy in ((all100, all_c, f"all-gene, 100 PCs: {all100:.1f}%", 8),
                          (mark100, mark_c, f"marker-gene, 100 PCs: {mark100:.1f}%", -9)):
        ax2.plot([100], [y], marker="o", markersize=2.8, color=c, zorder=3)
        ax2.annotate(txt, xy=(100, y), xytext=(-8, dy), textcoords="offset points",
                     ha="right", va="bottom" if dy > 0 else "top", color=c, fontsize=fs)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=fs - 0.5,
                     frameon=False, handlelength=1.8, labelspacing=0.35,
                     borderaxespad=0.3)
    leg.set_zorder(6)

    return save(fig, "wmb_pca_scree", width_key="full_page",
                require_strings=["Principal component", "Variance explained per PC (%)",
                                 "Cumulative variance explained (%)",
                                 f"all-gene, 100 PCs: {all100:.1f}%",
                                 f"marker-gene, 100 PCs: {mark100:.1f}%",
                                 "cumulative, 6,558 marker genes (the UMAP input)",
                                 "cumulative, all 32,285 genes"])


def _bare_umap_axes(ax, xlim, ylim):
    """No ticks, no numbers, no spines: UMAP coordinates are not measurements."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")  # a UMAP with unequal axis scales is a distorted UMAP
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


ARROW_ANCHOR_INSET = -0.20
"""Where the axis-arrow corner sits, in arrow lengths from the bottom-left of the CLOUD.

Negative: the corner sits just outside the cloud, in the axes padding. It was +0.3, i.e.
tucked up into the cloud, and at 89 mm that left the UMAP1 label clearing the nearest ink of
the 02 NP-CT-L6b island by 1.3 pt, which prints as contact. Backing the anchor out by a fifth
of an arrow length takes the gap to 6.4 pt (measured, printed on every run, and held to a
floor by assert_no_ink_under) while keeping the whole indicator inside the axes and still
plainly attached to the corner of the cloud.
"""


def _umap_axis_arrows(ax, xy, xlim, ylim):
    """Short UMAP1/UMAP2 arrows, anchored just outside the corner of the POINT CLOUD.

    Anchored to the cloud, not to the corner of the axes: the axes corner is several
    centimetres from the nearest centroid in this layout, and an axis indicator marooned
    out there reads as an orphan rather than as a key to the panel it belongs to.
    """
    length = 0.05 * (xlim[1] - xlim[0])
    x0 = xy[:, 0].min() + ARROW_ANCHOR_INSET * length
    y0 = xy[:, 1].min() + ARROW_ANCHOR_INSET * length
    assert x0 >= xlim[0] and y0 >= ylim[0], (
        "the axis-arrow anchor has been backed out past the edge of the axes; it would be "
        "clipped. Either shrink ARROW_ANCHOR_INSET or widen the axes padding."
    )
    arrow = dict(arrowstyle="-|>,head_width=0.12,head_length=0.3", color="0.25",
                 linewidth=0.7, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0), arrowprops=arrow)
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0), arrowprops=arrow)
    fs = plt.rcParams["xtick.labelsize"]
    t1 = ax.text(x0 + 1.15 * length, y0, "UMAP1", va="center", ha="left", fontsize=fs,
                 color="0.25")
    t2 = ax.text(x0, y0 + 1.15 * length, "UMAP2", va="bottom", ha="left", fontsize=fs,
                 color="0.25")
    # Returned so the caller can prove they land on empty paper. On the 89 mm panel the
    # UMAP1 label reaches toward the 02 NP-CT-L6b island, which is the bottom-most thing in
    # the layout, and "it looks clear" is not a check.
    return {"the UMAP1 axis label": t1, "the UMAP2 axis label": t2}


# --------------------------------------------------------------------------
# THE PANEL: the cosine UMAP at 89 mm, whole and split
# --------------------------------------------------------------------------
PANEL_W = FIGURE_WIDTHS["half_page"]
PANEL_SIDE_PAD = 0.06  # inches the layout engine keeps clear at each side of the axes


def panel_layout(d) -> dict:
    """Everything the three panel files must agree on, computed exactly ONCE.

    wmb_umap_panel.svg and wmb_umap_panel_nokey.svg have to be the same picture, and the
    cheapest way to guarantee that is to give them no opportunity to differ: one rotation,
    one set of limits, one dot size, one colour column, handed to both. (They are then also
    compared after rendering, in assert_identical_scatter, because "computed once" is a
    claim about this function and not about what reached the canvas.)
    """
    xy = rigid_rotation(d["umap"][["UMAP1", "UMAP2"]].to_numpy())
    span = xy.max(axis=0) - xy.min(axis=0)
    pad = 0.035 * span
    xlim = [xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0]]
    ylim = [xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1]]
    ax_w = PANEL_W - 2 * PANEL_SIDE_PAD
    # EQUAL ASPECT, so the axes height follows from the data and is not a free choice:
    # a UMAP encodes distances and nothing else, and an anisotropic panel is a lie about
    # them. Every height below is derived from this one, never dialled to fill a hole.
    ax_h = ax_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    return {"xy": xy, "xlim": xlim, "ylim": ylim, "ax_w": ax_w, "ax_h": ax_h,
            "sizes": PANEL_DOT}


def panel_key_size(d) -> tuple[float, float]:
    """Measure the 34-entry key at the stylesheet's 7 pt, and check it fits the page."""
    fs = plt.rcParams["legend.fontsize"]
    w, h = measure_legend_inches(PANEL_W, class_handles(d["classes"]),
                                 ncol=PANEL_KEY_NCOL, fontsize=fs, loc="upper center",
                                 **CLASS_KEY_KW)
    assert w <= PANEL_W - 0.02, (
        f"the {PANEL_KEY_NCOL}-column class key is {w:.2f} in wide and the page is "
        f"{PANEL_W:.2f} in. Drop a column and let the key get taller. Do NOT reduce the "
        "font: 7 pt is the stylesheet's legend size and the floor for Nature Methods."
    )
    return w, h


def _draw_panel_scatter(ax, d, L):
    coll = scatter_by_class(ax, L["xy"], d["meta"], L["sizes"], alpha=0.85)
    _bare_umap_axes(ax, L["xlim"], L["ylim"])
    arrows = _umap_axis_arrows(ax, L["xy"], L["xlim"], L["ylim"])
    return coll, arrows


PANEL_KEY_GAP = 0.06  # inches between the bottom of the cloud and the top of the key


def _panel_figure(key_h: float, L: dict):
    """A panel canvas with its axes at an EXPLICITLY COMPUTED rect, and no layout engine.

    ``key_h`` is 0 for the key-less panel. Both panels get an axes of exactly
    ``L["ax_w"] x L["ax_h"]`` inches, inset ``PANEL_SIDE_PAD`` from the left of the same
    89 mm page, so the two files are the same picture at the same size on paper BY
    CONSTRUCTION, and assert_identical_scatter then confirms it on the rendered canvas.

    constrained_layout is deliberately not used here, and that is the fix for a real bug.
    The keyed panel tuned the engine (``w_pad=PANEL_SIDE_PAD``) and the key-less one took the
    engine's defaults, so the engine handed them axes of different widths and the key-less
    scatter printed 1.1% larger. An engine that is free to choose the margins is free to
    choose them differently for two figures that must agree, and the figure that must agree
    with another is the one figure that cannot be left to an engine. The diagnostics keep
    constrained_layout: they have tick labels, axis labels and multiple panes to reconcile,
    and no twin they must match.
    """
    ax_w, ax_h = L["ax_w"], L["ax_h"]
    key_block = (key_h + PANEL_KEY_GAP) if key_h else 0.0
    fig_h = ax_h + key_block + 2 * PANEL_SIDE_PAD
    fig = plt.figure(figsize=(PANEL_W, fig_h))
    ax = fig.add_axes([
        PANEL_SIDE_PAD / PANEL_W,
        (PANEL_SIDE_PAD + key_block) / fig_h,
        ax_w / PANEL_W,
        ax_h / fig_h,
    ])
    ax_key = None
    if key_h:
        # Full page width, so the key centres on the PAGE and not on the axes: the panel and
        # the standalone key file must put their columns at the same x, or the shared-key
        # assembly does not line up with the panel it replaces.
        ax_key = fig.add_axes([0.0, PANEL_SIDE_PAD / fig_h, 1.0, key_h / fig_h])
        ax_key.axis("off")
    return fig, ax, ax_key, fig_h


def fig_umap_panel(d, L, with_key: bool):
    """The 89 mm publication panel. cosine, n_neighbors = 25, uniform dots, no size key.

    NO PANEL LETTER. The compiled figure is assembled and lettered in Inkscape; a letter
    baked in here would end up either duplicated or in the wrong corner of whatever
    assembly this panel lands in.

    THE KEY GOES UNDER THE SCATTER, and the reason is measured rather than asserted. At
    89 mm the axes is about 3.4 x 3.1 in. The 34-entry key at the stylesheet's 7 pt is
    2.1 x 2.2 in at two columns: to sit inside the axes it would need a corner void 70% of
    the panel height, and the tallest void this layout has in either upper corner is
    printed below and is nowhere near that. The full-page figure could put its key in the
    void; this one cannot, and the honest fix is a taller panel, never smaller type.

    ``with_key`` writes the same scatter with the key (self-contained panel) or without it
    (for a compiled figure whose several panels share one key, which is what wmb_umap_key.svg
    is for).
    """
    fs = plt.rcParams["legend.fontsize"]
    ax_w, ax_h = L["ax_w"], L["ax_h"]

    if not with_key:
        fig, ax, _, fig_h = _panel_figure(0.0, L)
        coll, arrows = _draw_panel_scatter(ax, d, L)
        clear = assert_no_ink_under(fig, ax, L["xy"], L["sizes"], arrows)
        fp = scatter_fingerprint(fig, ax, coll)
        print(f"  panel (no key): {PANEL_W:.3f} x {fig_h:.2f} in (half_page). Same axes rect "
              f"as the keyed panel by construction: "
              f"{fp['axes_wh_pt'][0]:.2f} x {fp['axes_wh_pt'][1]:.2f} pt.")
        res = save(fig, "wmb_umap_panel_nokey", width_key="half_page",
                   require_strings=["UMAP1", "UMAP2"],
                   forbid_strings=list(d["classes"]["class_name"]) + ["cells in cluster"])
        return res, fp

    key_w, key_h = panel_key_size(d)
    fig, ax, ax_key, fig_h = _panel_figure(key_h, L)

    coll, arrows = _draw_panel_scatter(ax, d, L)
    class_key(ax_key, d["classes"], ncol=PANEL_KEY_NCOL, fontsize=fs, loc="upper center")

    # Why the key is not in the axes, in numbers, on every run. If the embedding ever
    # changes shape enough that a key WOULD fit inside it, this print says so and the
    # layout can be revisited on evidence.
    per_in = (L["xlim"][1] - L["xlim"][0]) / ax_w
    fs_ck_w, fs_ck_h = measure_legend_inches(PANEL_W, class_handles(d["classes"]), ncol=2,
                                             fontsize=fs, loc="upper left", **CLASS_KEY_KW)
    voids = {c: corner_void(L["xy"], L["xlim"], L["ylim"], (fs_ck_w + 0.12) * per_in, c)
             / per_in for c in ("upper left", "upper right")}
    assert max(voids.values()) < fs_ck_h, (
        "a 2-column 34-entry key now FITS inside the panel's axes, which it did not when "
        "the key was put underneath. Reconsider the layout on the evidence; do not leave "
        "the key below out of habit."
    )
    # The key is on its own axes, so the only text over the cloud is the axis indicator.
    clear = assert_no_ink_under(fig, ax, L["xy"], L["sizes"], arrows)
    fp = scatter_fingerprint(fig, ax, coll)
    print(f"  panel: {PANEL_W:.3f} x {fig_h:.2f} in (half_page). Axes {ax_w:.2f} x "
          f"{ax_h:.2f} in, equal aspect, placed at an explicit rect "
          f"({fp['axes_wh_pt'][0]:.2f} x {fp['axes_wh_pt'][1]:.2f} pt). Uniform "
          f"{PANEL_DIA} pt dots, no size encoding and no size key. Key BELOW the scatter, "
          f"{PANEL_KEY_NCOL} columns, {key_w:.2f} x {key_h:.2f} in at {fs:g} pt. It cannot "
          f"go inside: even at 2 columns it needs {fs_ck_h:.2f} in of void and the best "
          f"corner offers {max(voids.values()):.2f} in.")
    print("  axis labels clear the nearest dot by "
          + ", ".join(f"{n.split()[1]} {c:.1f} pt" for n, c in clear.items())
          + f" (floor {LABEL_CLEARANCE_PT:.1f} pt, measured on the page, dot radius "
            "subtracted).")
    res = save(fig, "wmb_umap_panel", width_key="half_page",
               require_strings=list(d["classes"]["class_name"]) + ["UMAP1", "UMAP2"])
    return res, fp


def fig_umap_key(d):
    """The 34-entry class key, alone, on its own 89 mm canvas.

    For the compiled figure where one key serves several panels: this file is dropped in
    once, beside wmb_umap_panel_nokey.svg. Same colours, same 7 pt, same three columns and
    the same page width as the panel it keys, so it lands in Inkscape at 100% like
    everything else. There is no size key here, because the panel carries no size encoding
    (see PANEL_DOT).
    """
    fs = plt.rcParams["legend.fontsize"]
    key_w, key_h = panel_key_size(d)
    # Full-width axes and the same "upper center" placement the panel uses, so the three
    # columns land at the same x on the page as they do inside wmb_umap_panel.svg. The
    # layout engine is kept out of the panel family entirely (see _panel_figure).
    fig = plt.figure(figsize=(PANEL_W, key_h + 0.04))
    ax = fig.add_axes([0.0, 0.0, 1.0, key_h / (key_h + 0.04)])
    ax.axis("off")
    class_key(ax, d["classes"], ncol=PANEL_KEY_NCOL, fontsize=fs, loc="upper center")
    print(f"  key: standalone, {PANEL_W:.3f} x {key_h + 0.04:.2f} in (half_page), "
          f"{PANEL_KEY_NCOL} columns at {fs:g} pt, 34 entries, no size key.")
    return save(fig, "wmb_umap_key", width_key="half_page",
                require_strings=list(d["classes"]["class_name"]))


def fig_variants(d):
    """The metric x n_neighbors grid. Six panels, six letters, six frames.

    NO SIZE ENCODING HERE, hence no size key. The panels are 2.3 inches wide and the only
    claim they make is topological, so a size channel would be decoration; worse, it used
    to run on a THIRD area scale, different from both other figures, under a shared caption
    sentence that pointed at a size key this figure never drew. Fixed dot, one claim.
    """
    meta, variants, s = d["meta"], d["variants"], d["summary"]
    main_variant = d["main_variant"]
    metrics = ["euclidean", "cosine"]
    neighbors = [15, 25, 50]

    main_xy = rigid_rotation(
        variants.loc[variants["variant"] == main_variant, ["UMAP1", "UMAP2"]].to_numpy()
    )

    # ONE FRAME SHAPE PER ROW, and each panel scaled to fill it.
    #
    # The three panels of a row are the n_neighbors sweep at a fixed metric, which is
    # the comparison a reader actually makes side by side, so those three share a
    # frame exactly and none is letterboxed into a fraction of another's ink area.
    #
    # The two rows do NOT share a frame shape, and forcing them to would waste half
    # the figure. The euclidean layouts are wide thin streaks (aspect 0.27 to 0.39):
    # two flung-out islands stretch them along one axis. The cosine layouts are nearly
    # square (0.53 to 1.02). One shape for all six either squashes the cosine panels
    # into the middle third of a wide frame (the median, 0.46, does this, and it leaves
    # the framed headline panel mostly white) or hangs a band of air above and below
    # every euclidean panel (the main panel's own 0.92 does this). Both were rendered.
    #
    # The scale is per panel, and that is a deliberate choice with a caveat attached.
    # The honest-looking alternative, rescaling all six to a common RMS radius, was
    # implemented and rendered: it is comparable but unreadable. The euclidean
    # embeddings place two islands (01 IT-ET and the non-neuronal group) far from a
    # compact core, so their RMS radius is set by the islands and, on any single linear
    # scale that still shows those islands, the core collapses to an indistinct blob.
    # No linear scale can show both. So each panel is scaled to fill its frame, and the
    # caption states plainly that apparent cluster size is NOT comparable across panels.
    panels = {}
    for metric in metrics:
        for nn in neighbors:
            key = f"{metric}_nn{nn}"
            sub = variants[variants["variant"] == key]
            panels[key] = rigid_rotation(sub[["UMAP1", "UMAP2"]].to_numpy(),
                                         align_to=main_xy)
    assert main_variant in panels, f"{main_variant} is not one of the six grid panels"

    spans = {k: (np.ptp(v[:, 0]), np.ptp(v[:, 1])) for k, v in panels.items()}
    row_asp = {
        metric: float(np.median([spans[f"{metric}_nn{nn}"][1] / spans[f"{metric}_nn{nn}"][0]
                                 for nn in neighbors]))
        for metric in metrics
    }

    limits = {}
    for metric in metrics:
        asp = row_asp[metric]
        for nn in neighbors:
            k = f"{metric}_nn{nn}"
            v = panels[k]
            w, h = spans[k]
            # Pad the short side to the row's frame shape. PAD, never stretch: an
            # anisotropic scale would distort the embedding, so the aspect stays equal
            # inside every panel and the padding just adds margin.
            bw, bh = max(w, h / asp), max(h, w * asp)
            mid = 0.5 * (v.min(axis=0) + v.max(axis=0))
            pad = 1.06
            limits[k] = ((mid[0] - bw * pad / 2, mid[0] + bw * pad / 2),
                         (mid[1] - bh * pad / 2, mid[1] + bh * pad / 2))
            got = (limits[k][1][1] - limits[k][1][0]) / (limits[k][0][1] - limits[k][0][0])
            assert abs(got - asp) < 1e-9, f"{k}: frame shape drifted ({got} vs {asp})"
            if k == main_variant:  # the framed panel must carry no dead margin of its own
                assert abs(bw - w) < 1e-9 and abs(bh - h) < 1e-9, (
                    "the MAIN panel does not fill its frame, so the headline panel of "
                    "the grid is letterboxed into white"
                )

    panel_w = (FIGURE_WIDTHS["full_page"] - 0.35) / 3
    title_h = 0.30  # title plus the panel letter above it
    key_h = 0.95
    fs = plt.rcParams["legend.fontsize"]
    letters = "abcdef"

    # The euclidean row looks like a rendering failure, and it is not: those panels are
    # 98% white because the two outlying groups sit most of the layout span away and set
    # the scale. That IS the finding. A row a reader takes for a broken render argues
    # nothing, so the figure says what the row is showing, in numbers taken from the
    # measured structure (euclidean at 25 neighbours, the only euclidean layout measured).
    eu = s["island_structure"]["euclidean_nn25"]
    eu_gaps = sorted((i["gap_to_continent_frac_of_span"] for i in eu["islands"]),
                     reverse=True)
    # ONE line, hung directly under the euclidean row, outside every panel. It used to sit
    # inside panel c, threaded through the corridor between the islands and the crushed
    # continent, and that corridor is too tight for it: at 180 mm the centred rows cleared
    # the data by at most 2.4 pt, and by 1.1 pt once the finished layout had re-flowed.
    # Under the row the note cannot reach a centroid, and it speaks for all three euclidean
    # panels rather than for one of them.
    eu_note = (
        "Not a rendering failure. Under euclidean the two outlying groups land "
        f"{eu_gaps[0]:.0%} and {eu_gaps[1]:.0%} of the span away, so they set the scale "
        "and crush the continent."
    )
    note_kw = dict(fontsize=fs - 1, color="0.35")
    # The note hangs this far under the euclidean frames: one point over the clearance
    # floor, so the check below does not sit on its own threshold.
    note_gap_pt = LABEL_CLEARANCE_PT + 1.0
    # The note itself hangs in the dead paper under the width-limited euclidean frames. An
    # empty grid row between the two panel rows, sized from the note plus that gap (as a
    # height ratio, so it comes out a little shorter on the page), pushes the cosine titles
    # further down. The canvas does NOT grow for it: both panel rows are width-limited by
    # set_box_aspect and already leave dead paper under their frames, so the row's share of
    # the height comes out of that and every panel keeps its size. What guarantees the gap
    # on the page is assert_text_clears_axes, not the row.
    note_w, note_h = measure_text_inches(FIGURE_WIDTHS["full_page"], eu_note, **note_kw)
    strip_h = note_h + note_gap_pt / 72

    # + 0.12 in of layout padding. It was 0.25 while the footnote sat in the strip under the
    # key; the strip goes with the footnote rather than staying on as dead paper.
    row_h = [panel_w * row_asp[m] + title_h for m in metrics]
    fig = plt.figure(
        figsize=(FIGURE_WIDTHS["full_page"], sum(row_h) + key_h + 0.12),
        constrained_layout=True,
    )
    # Grid rows: euclidean panels, the (empty) note strip, cosine panels, the class key.
    gs = fig.add_gridspec(4, 3, height_ratios=(row_h[0], strip_h, row_h[1], key_h))
    grid_row = {"euclidean": 0, "cosine": 2}

    panel_axes = {}
    for r, metric in enumerate(metrics):
        for c, nn in enumerate(neighbors):
            key = f"{metric}_nn{nn}"
            ax = fig.add_subplot(gs[grid_row[metric], c])
            panel_axes[letters[r * 3 + c]] = ax
            scatter_by_class(ax, panels[key], meta, VARIANT_DOT, alpha=0.9)
            _bare_umap_axes(ax, *limits[key])
            ax.set_box_aspect(row_asp[metric])  # identical frame shape within the row
            main = key == main_variant
            # "n_neighbors" is a code identifier, not a caption word; "MAIN" in capitals is
            # not how emphasis is set in a journal. The frame does the emphasis.
            # The title is LEFT-aligned and pushed right, so the panel letter sits beside it
            # on the same baseline instead of landing under a centred title (the bold main
            # title is long enough to run straight into a letter placed at x = 0).
            ax.set_title(f"{metric}, {nn} neighbours" + (" (main)" if main else ""),
                         loc="left", x=0.055,
                         fontweight="bold" if main else "normal",
                         color="black" if main else "0.35", pad=3)
            # Same 3 pt offset as the title's pad, so letter and title sit on one line.
            ax.annotate(letters[r * 3 + c], xy=(0.0, 1.0), xycoords="axes fraction",
                        xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                        fontsize=plt.rcParams["font.size"] + 1, fontweight="bold")
            # EVERY panel gets a frame. With only the main one framed, the "one frame shape
            # per row" design was invisible and neighbouring clouds bled into one another.
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1.0 if main else 0.5)
                sp.set_color("black" if main else "0.8")

    # Hung from the bottom edge of the middle euclidean panel, so it follows that row's
    # frames wherever constrained_layout puts them and reads as the row's note rather than
    # as a heading for the cosine row. (Set inside the strip itself, it drifted down onto
    # the cosine titles, because the width-limited euclidean frames do not reach the bottom
    # of their cell.) The columns are equal, so centred on panel b is centred on the row.
    # in_layout=False: counted as part of panel b, a line this wide would make the layout
    # squeeze panel b's column to fit it. It also keeps the note out of panel b's tight
    # box, which assert_text_clears_axes measures the note against.
    note = panel_axes["b"].annotate(
        eu_note, xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -note_gap_pt),
        textcoords="offset points", ha="center", va="top", **note_kw)
    note.set_in_layout(False)

    ax_key = fig.add_subplot(gs[3, :])
    ax_key.axis("off")
    class_key(ax_key, d["classes"], ncol=6, fontsize=fs, loc="upper center")

    # Checked on the FINISHED figure, once every axes and the key exist, and on the layout
    # that is written. constrained_layout re-flows on every draw and needs a few passes to
    # settle on this grid (the first leaves about 1.8 pt too much side margin), so the grid
    # is drawn until no axes moves by more than 0.01 pt and then frozen: the check below and
    # both savefig calls in save() all see these exact positions.
    page_pt = np.tile(fig.get_size_inches() * 72, 2)
    placed = None
    for _ in range(10):
        fig.canvas.draw()
        now = np.array([a.get_position().bounds for a in fig.axes]) * page_pt
        if placed is not None and np.abs(now - placed).max() < 0.01:
            break
        placed = now
    else:
        raise AssertionError("constrained_layout did not settle on the variants grid in 10 "
                             "passes, so the layout frozen here would be arbitrary.")
    fig.set_layout_engine("none")
    note_gaps = assert_text_clears_axes(
        fig, note, {**{f"panel {k}": a for k, a in panel_axes.items()},
                    "the class key": ax_key})
    above = min(note_gaps[f"panel {k}"] for k in "abc")
    below = min(note_gaps[f"panel {k}"] for k in "def")
    assert above < below, (
        f"the euclidean note sits {above:.1f} pt under panels a to c but only {below:.1f} pt "
        "over the cosine titles, so it reads as a heading for d to f. It is hung from panel "
        "b; check what moved it."
    )
    print("  variants: one frame shape per row ("
          + ", ".join(f"{m} {row_asp[m]:.2f}" for m in metrics)
          + "); six panel letters, six frames; fixed dot size (no size encoding, so no "
          f"size key); scale is per panel and the caption says so. MAIN = {main_variant}")
    note_lines = eu_note.count("\n") + 1
    print(f"  euclidean note: {note_lines} line(s) ({note_w * 72:.0f} x {note_h * 72:.1f} pt) "
          "under panels "
          f"a to c, {above:.1f} pt below their frames and {below:.1f} pt above the cosine "
          f"titles (floor {LABEL_CLEARANCE_PT:.1f} pt, measured on the finished layout).")
    main_metric, main_nn = main_variant.split("_nn")
    return save(fig, "wmb_umap_variants", width_key="full_page",
                require_strings=[c.class_name for c in d["classes"].itertuples()]
                # One <text> per line: matplotlib writes each line of a text separately.
                + [f"{main_metric}, {main_nn} neighbours (main)", *eu_note.splitlines()]
                + list(letters))


# --------------------------------------------------------------------------
# captions: written to disk, never only to stdout
# --------------------------------------------------------------------------
def _colour_caveat(conf: dict) -> str:
    """Name every confusable colour pair, and say honestly where position does not help."""
    pairs = conf["pairs"]
    adjacent = [p for p in pairs if p["adjacent"]]
    far = [p for p in pairs if not p["adjacent"]]

    def fmt(ps):
        return "; ".join(f"{p['a']} / {p['b']} (dE {p['delta_e']:.0f}, "
                         f"{p['separation_frac_of_span']:.0%} of the layout span apart)"
                         for p in ps)

    out = (
        "A caveat on the colours, since colour is the only class identification here. "
        f"{len(pairs)} of the 561 class pairs in Allen's palette are closer than a CIE76 "
        f"dE of {DE_CONFUSABLE:.0f}, which is to say hard to tell apart by eye. "
    )
    if far:
        out += (f"For most of them position rescues the reader, because the two classes sit "
                f"far apart in the layout: {fmt(far)}. ")
    if adjacent:
        out += (f"For the rest it does not, and that has to be said plainly: these pairs are "
                f"near-identical in colour and abutting in the layout, so nothing on the "
                f"figure separates them: {fmt(adjacent)}. ")
    out += (
        "The general case of this is the non-neuronal block. 30 Astro-Epen, 32 OEC, 33 "
        "Vascular and 34 Immune are four browns and greys whose closest pair differs by "
        f"only dE {conf['non_neuronal_min_delta_e']:.0f}, and they share one small rim "
        "satellite, so a reader should not attempt to assign them from the key: read that "
        "satellite as non-neuronal and go to the tables for more. This is Allen's own "
        "published palette, which is not repainted here, so the same degeneracy is present "
        "in the published Fig. 1B."
    )
    return out


def _topology_note(s: dict) -> str:
    """The topology sentence, generated from the measured island structure.

    Nothing here is typed from memory of the picture. The previous pass wrote this
    paragraph by eye and asserted a resemblance that the coordinates do not have.
    """
    cos = s["island_structure"]["cosine_nn25"]
    euc = s["island_structure"]["euclidean_nn25"]

    def gap(t, cls):
        for isl in t["islands"]:
            if cls in isl["classes"]:
                return isl["gap_to_continent_frac_of_span"]
        return None

    c01, e01 = gap(cos, "01 IT-ET Glut"), gap(euc, "01 IT-ET Glut")
    cnn, enn = gap(cos, "30 Astro-Epen"), gap(euc, "30 Astro-Epen")
    in01 = cos["classes_mostly_outside_the_continent"]["01 IT-ET Glut"]
    inlsx = cos["classes_mostly_outside_the_continent"]["10 LSX GABA"]

    # The prose below states these numbers. If the embedding ever moves, the prose must
    # not survive it, so the two load-bearing claims are asserted against the data.
    assert in01 < 0.05 and inlsx < 0.05, (
        "01 IT-ET Glut or 10 LSX GABA is now inside the continent, which is the opposite "
        "of what this caption says. Rewrite the caption, do not relax the assertion."
    )
    assert c01 < e01 and cnn < enn, (
        "euclidean no longer flings the islands further out than cosine does, which is the "
        "entire stated reason for choosing cosine"
    )

    return (
        "The metric is our choice, not the paper's. The Yao et al. Methods specify "
        "n_neighbors = 25 and min_dist = 0.4 and never state a distance metric. What the "
        "metric does not change is which groups break away: on centroid data the same "
        "handful (01 IT-ET Glut, 02 NP-CT-L6b Glut, 10 LSX GABA, the non-neuronal block, "
        "31 OPC-Oligo) detaches under both metrics, because a centroid carries no "
        "within-cluster spread and the cells that would bridge those groups to the rest do "
        "not exist here. What it changes is how far out they land. Under cosine one "
        f"continent holds {cos['continent_fraction']:.0%} of the cell types and the islands "
        f"stay proportionate to it (01 IT-ET Glut sits {c01:.0%} of the layout span from "
        f"the continent, the non-neuronal block {cnn:.0%}), so the continent of neuronal "
        "territories remains the dominant structure of the picture, as in Fig. 1B. Under "
        f"euclidean the same two gaps blow out to {e01:.0%} and {enn:.0%} of the span; "
        "those islands then set the layout scale and the neuronal core is compressed into "
        "a blob. That is why cosine is the main embedding, and euclidean is shown beside it "
        "in wmb_umap_variants.svg. The resemblance to Fig. 1B is qualitative, judged by eye "
        "against the printed panel: the published per-cell coordinates were never "
        "downloaded, so no point-for-point comparison was made and none is claimed. It is "
        "also not complete, and the exceptions are worth stating because a reader holding "
        "the two panels side by side will see them at once. In the published figure 01 "
        "IT-ET Glut is a large lobe inside the main mass and 10 LSX GABA is an interior "
        f"territory. Here both are detached: only {in01:.1%} of the 01 centroids and "
        f"{inlsx:.1%} of the 10 LSX centroids fall in the continent. Cosine does not fix "
        "that and no metric can, for the same reason as above: one point per cluster leaves "
        "nothing but the gaps between classes. Cosine is also the better choice on the one "
        "quantitative handle we have (25-NN class purity over cell types: "
        f"{100 * s['umap_cosine_class_purity_unweighted']:.1f}% against "
        f"{100 * s['umap_euclidean_class_purity_unweighted']:.1f}% for euclidean), but that "
        "was found afterwards and is a bonus, not the reason."
    )


def _panel_caption(d, conf: dict) -> str:
    """The manuscript-ready caption for the 89 mm panel.

    THIS IS THE ONLY PLACE THE CENTROID CAVEAT IS NOW WRITTEN FOR THIS PANEL. The on-figure
    footnote is gone, so a caption that omits "cell types, not cells" ships a figure that
    silently claims to be a per-cell UMAP of 4M cells. It is the first sentence for that
    reason.

    Tighter than the diagnostics' captions on purpose: it is meant to be pasted into a
    supplementary legend, so it carries what a reader needs to not be misled (what a point
    is, how the embedding was made, whose choice the metric was, where it does NOT match the
    published panel, and that colour is the only class cue) and sends everything else to the
    tables. Every number in it is read from embedding_summary.json or measured off
    cluster_metadata.csv, and none is typed. The cluster-size range and the PC count used to
    be typed literals; they were right, and they would have gone quietly wrong the first time
    the matrix was rebuilt from a new ABC release, which is the one failure mode this module
    spends all its other assertions preventing.
    """
    s = d["summary"]
    p = s["umap_params"]
    n = len(d["meta"])
    cells = int(d["meta"].n_cells.sum())
    n_lo, n_hi = int(d["meta"].n_cells.min()), int(d["meta"].n_cells.max())
    cos = s["island_structure"]["cosine_nn25"]
    euc = s["island_structure"]["euclidean_nn25"]

    def gap(t, cls):
        for isl in t["islands"]:
            if cls in isl["classes"]:
                return isl["gap_to_continent_frac_of_span"]
        return None

    c01, e01 = gap(cos, "01 IT-ET Glut"), gap(euc, "01 IT-ET Glut")
    in01 = cos["classes_mostly_outside_the_continent"]["01 IT-ET Glut"]
    inlsx = cos["classes_mostly_outside_the_continent"]["10 LSX GABA"]
    assert in01 < 0.05 and inlsx < 0.05, (
        "01 IT-ET Glut or 10 LSX GABA is now inside the continent, and this caption says "
        "the opposite. Rewrite the caption; do not relax the assertion."
    )
    nn_block = ", ".join(c for c in NON_NEURONAL if c != "31 OPC-Oligo")

    return (
        "UMAP of the Allen whole-mouse-brain 10x transcriptomic taxonomy (Yao et al. 2023), "
        "coloured by Allen's 34 cell classes, reconstructing their Fig. 1B. "
        f"Each point is one of the {n:,} transcriptomic cluster centroids, that is one cell "
        f"type, and not one of the {cells:,} cells: a centroid is the mean log2(CPM+1) "
        "profile of its cluster, cell-level expression was never downloaded, and "
        "within-cluster spread is therefore absent by construction, so the territories are "
        "sparser than in the published panel, which draws every cell. Points are one fixed "
        f"size; cluster size ({n_lo:,} to {n_hi:,} cells) is not encoded. "
        f"The embedding is over the top {p['n_pcs']} principal components of "
        f"{p['n_marker_genes']:,} Allen MapMyCells marker genes, with metric = "
        f"{p['metric']}, n_neighbors = {p['n_neighbors']}, min_dist = {p['min_dist']}, "
        f"random_state = {p['random_state']}. n_neighbors and min_dist are the paper's "
        "values; the metric is ours, because their Methods state none, and cosine keeps the "
        f"detached groups proportionate to the continent of neuronal territories, which "
        f"holds {cos['continent_fraction']:.0%} of the cell types (under euclidean the same "
        f"groups land {e01:.0%} of the layout span away and crush that continent into a "
        "blob). "
        "The resemblance to Fig. 1B is qualitative, by eye, and partial: the published "
        "per-cell coordinates were never downloaded, so no point-for-point comparison was "
        "made, and 01 IT-ET Glut, an interior lobe in the published panel, is a detached "
        f"island here ({in01:.1%} of its centroids fall in the continent), as is 10 LSX GABA "
        f"({inlsx:.1%}). One point per cluster leaves nothing but the gaps between classes, "
        "and no metric can put back the cells that would bridge them. "
        # One decimal, and "a class purity of", both deliberately. The continent fraction two
        # sentences up is also 82%, and at 0 decimals the two collide: a referee reading the
        # legend cold takes the second number for a restatement of the first, when one is the
        # share of cell types in the largest component and the other is a k-NN class purity.
        f"Classes occupy coherent territories: "
        f"{100 * s['umap_cosine_class_purity_unweighted']:.1f}% of each centroid's "
        f"{s['purity_k']} nearest neighbouring cell types share its class, against a class "
        f"purity of {100 * s['marker_pca_class_purity_unweighted']:.1f}% in the "
        f"{p['n_pcs']}-PC marker-gene "
        "space the UMAP consumed (a neighbourhood of cell types, never of cells; no "
        "per-cell claim is made). Raw purity has a per-class ceiling, since "
        f"{s['n_classes_with_purity_ceiling_below_1']} of the {len(d['classes'])} classes "
        f"hold {s['purity_k']} or fewer "
        "clusters; rank classes on the ceiling-normalized columns of umap_knn_purity.csv, "
        "never on the raw ones. "
        "Classes are identified by colour alone, in Allen's published palette, which is not "
        f"repainted here: {nn_block} are browns and greys that colour cannot separate (CIE76 "
        f"dE {conf['non_neuronal_min_delta_e']:.0f} at the closest) and that share one small "
        "rim satellite, so read that satellite as non-neuronal. The same degeneracy is in "
        "the published Fig. 1B. "
        "Axes are unlabelled because UMAP coordinates carry no units; the embedding is "
        "rigidly rotated for display, which leaves every distance unchanged."
    )


def captions(d) -> str:
    s = d["summary"]
    n = len(d["meta"])
    cells = d["meta"].n_cells.sum()
    var = d["var"]
    conf = confusable_pairs(d["classes"], d["umap"][["UMAP1", "UMAP2"]].to_numpy(),
                            d["meta"])
    med_cells = int(d["meta"].n_cells.median())
    # Measured, not typed: this fraction is the whole argument for why the 89 mm panel drops
    # the size encoding, and a rebuilt matrix must be able to falsify it.
    frac_small = float((d["meta"].n_cells < 1_000).mean())

    SIZE_KEYED = (
        "Point diameter increases with log10(number of cells in the cluster), "
        "linearly rescaled between a minimum and a maximum (an affine map, not a "
        "proportional one); see the size key, whose swatches bracket the data (the "
        f"median cluster holds {med_cells} cells and {frac_small:.0%} hold under 1,000). "
    )
    SIZE_TOPOLOGY_ONLY = (
        "All dots are one fixed size here: these panels carry no size encoding and no "
        "size key, because the only claim they make is topological. "
    )

    def centroid_note(size_sentence: str) -> str:
        # The centroid caveat. It used to be printed on the face of every figure as well;
        # the footnote is gone, so THIS is now the only place it is written, and every
        # caption below must carry it.
        return (
            f"Each point is one of the {n:,} transcriptomic cluster centroids of the Allen "
            f"whole-mouse-brain 10x atlas (Yao et al. 2023), which together represent "
            f"{cells:,} cells; it is not one point per cell. Cell-level data was never "
            "downloaded, so within-cluster spread is absent by construction and the "
            "territories are sparser than in the published Fig. 1B. "
            + size_sentence
            + "Classes are identified by colour alone, from Allen's official 34-class "
            "palette, exactly as Fig. 1B identifies them, and no class label is drawn "
            f"inside the point cloud. {_colour_caveat(conf)}"
        )

    topo = _topology_note(s)
    return "\n\n".join([
        # The panel goes FIRST: it is the publication artefact, and the three diagnostics
        # below it are not.
        f"wmb_umap_panel.svg | {_panel_caption(d, conf)}",

        "wmb_umap_panel_nokey.svg and wmb_umap_key.svg | The same panel, split: the scatter "
        "alone (points and axis arrows, no key) and the 34-entry class key alone, both on "
        f"the same {FIGURE_WIDTHS['half_page'] * 25.4:.0f} mm canvas. For a compiled figure "
        "in which one key serves several "
        "panels. The scatter is asserted identical to wmb_umap_panel.svg point for point "
        "(coordinates, colours and marker areas) and asserted to be drawn at the same size "
        "on the page (the axes rect is measured on the rendered canvas and compared), so the "
        "caption above applies to this pair unchanged and should be used as it stands. If "
        "the compiled legend already describes the class key once for several panels, the "
        "sentence beginning 'Classes are identified by colour alone' need not be repeated "
        "here; nothing else in that caption may be dropped.",

        "wmb_pca_classes.svg | PCA of cluster centroids over all 32,285 genes "
        "(log2(CPM+1), genes mean-centred, not variance-scaled). PC1 explains "
        f"{100 * s['pc1_variance_ratio']:.1f}% and PC2 {100 * s['pc2_variance_ratio']:.1f}% "
        "of the total variance. The axes are on a common scale, so the visual spread of "
        "PC2 relative to PC1 is the ratio of their standard deviations. PC1 vs PC2 "
        "collapses the subcortical neuronal classes onto one another; that is a property "
        "of a two-dimensional linear projection of a 100-dimensional structure, and it is "
        "shown rather than papered over. In the 100-PC space of this all-gene PCA, "
        f"{100 * s['all_gene_pca_class_purity_unweighted']:.0f}% of each cluster centroid's "
        "25 nearest neighbouring cell types share its class. This all-gene PCA is not the "
        "space the UMAP consumed; the UMAP consumed a separate PCA over the 6,558 marker "
        f"genes. {centroid_note(SIZE_KEYED)}",

        "wmb_pca_scree.svg | Variance explained by each of the first 100 principal "
        "components of the all-gene PCA (bars, left axis), and the cumulative variance "
        "explained by both of the PCAs used in this analysis (lines, right axis). The two "
        "are different spaces over different gene sets and are kept apart on purpose. The "
        "all-gene PCA (32,285 genes) reaches "
        f"{100 * var.loc[var.pc == 100, 'cumulative_all_genes'].item():.1f}% at 100 PCs; it "
        "is the space plotted in wmb_pca_classes.svg and it is not the space the UMAP "
        "consumed. The marker-gene PCA (6,558 genes) reaches "
        f"{100 * var.loc[var.pc == 100, 'cumulative_marker_genes'].item():.1f}% at 100 PCs "
        "and is the space the UMAP consumed. Each percentage is of its own total variance, "
        "over its own gene set, so the two numbers are not comparable to each other and "
        "neither may be quoted as the other's. The dashed rule marks the 100-PC cut, which "
        "is the number of PCs the paper's UMAP takes.",

        "wmb_umap_variants.svg | UMAP over the same 100 marker-gene PCs each time, for two "
        "distance metrics and three neighbourhood sizes. Within a metric the layout is "
        "stable across all three neighbourhood sizes. Across metrics it is not, and that is "
        "the point of the grid: the metric is the parameter the Methods leave free, and it "
        "moves the picture. min_dist, the PC count, the marker gene set and the random seed "
        "are held fixed, so this is not a test of those. The framed panel (cosine, 25 "
        "neighbours, panel e) is the main embedding, shown at publication size in "
        "wmb_umap_panel.svg. "
        f"{topo} Panels a to c look empty for a reason, and the note beneath them says so: "
        "under euclidean the outlying groups land most of a layout span away, so they set "
        "the scale and the neuronal continent is squeezed into a speck. Each panel is "
        "rotated, and reflected where needed, into the orientation of the main panel; both "
        "are isometries and neither changes any distance. The three panels of a row share "
        "one frame shape, which is the shape of that row's embeddings; the two rows do not "
        "share one, because the euclidean embeddings are wide thin streaks and the cosine "
        "ones are nearly square, and forcing a single shape on all six leaves half the "
        "figure white. Where a panel's own layout is a different shape from its row's "
        "(cosine at 50 neighbours is flatter than the other two cosine panels), it is "
        "padded and not stretched, so it does not fill its frame: an anisotropic scale "
        "would distort the embedding, and white space is the cheaper price. Each panel is "
        "scaled independently, so the apparent size or compactness of a cluster is not "
        "comparable between panels: UMAP's output scale is a free parameter of the "
        "optimisation, not a property of the data. Only the topology (which classes sit "
        "together, which form islands) is being compared. "
        "Rescaling all six to a common RMS radius was tried, and it is comparable but "
        "unreadable: the euclidean embeddings place two islands far from a compact core, so "
        "no single linear scale shows both the islands and the core. "
        f"{centroid_note(SIZE_TOPOLOGY_ONLY)}",
    ])


def main():
    d = load_tables()
    s = d["summary"]
    cos = s["island_structure"]["cosine_nn25"]
    print(f"loaded {len(d['meta']):,} cluster centroids, {len(d['classes'])} classes, "
          f"{d['meta'].n_cells.sum():,} cells represented")
    print("points are cell-type centroids, NOT cells: within-cluster spread is absent")
    print(f"main embedding: {d['main_variant']} (the metric is ours; the Methods do not "
          "state one)")
    print(f"PC1 {100 * s['pc1_variance_ratio']:.1f}%  PC2 {100 * s['pc2_variance_ratio']:.1f}%"
          f"  | 25-NN class purity over CELL TYPES: UMAP(cosine) "
          f"{s['umap_cosine_class_purity_unweighted']:.3f},"
          f" UMAP(euclidean) {s['umap_euclidean_class_purity_unweighted']:.3f},"
          f" marker-PC {s['marker_pca_class_purity_unweighted']:.3f},"
          f" all-gene PC {s['all_gene_pca_class_purity_unweighted']:.3f}")
    print("100-PC cumulative variance: all-gene "
          f"{100 * s['all_gene_pca_cumulative_variance_100pc']:.1f}% (plotted in "
          f"wmb_pca_classes.svg), marker-gene "
          f"{100 * s['marker_pca_cumulative_variance_100pc']:.1f}% (consumed by the UMAP). "
          "The scree shows both and attributes neither to the other.")
    print(f"measured topology (not eyeballed): continent {cos['continent_fraction']:.1%} of "
          f"the cell types; 01 IT-ET Glut is an ISLAND "
          f"({cos['classes_mostly_outside_the_continent']['01 IT-ET Glut']:.1%} of it in the "
          "continent), where Fig. 1B has it inside the mass. Every caption says so.")
    print("No class labels are drawn in any point cloud: the 34-entry colour key is the "
          "only identification, as in Fig. 1B.")
    print("The centroid caveat is no longer printed on any figure. It lives in captions.md, "
          "which is rewritten below.\n")
    print("wrote:")

    # THE PANEL first: it is the publication artefact. One layout, three files.
    L = panel_layout(d)
    panel, fp_with_key = fig_umap_panel(d, L, with_key=True)
    nokey, fp_nokey = fig_umap_panel(d, L, with_key=False)
    key = fig_umap_key(d)
    assert_identical_scatter(fp_with_key, fp_nokey, "wmb_umap_panel", "wmb_umap_panel_nokey")
    print(f"  identical-scatter check PASSED: wmb_umap_panel and wmb_umap_panel_nokey plot "
          f"the same {len(fp_with_key['offsets']):,} points, the same "
          f"{len(np.unique(fp_with_key['colors'], axis=0))} colours and the same marker "
          f"area ({float(fp_with_key['sizes'][0]):.2f} pt^2) point for point.")

    written = [panel, nokey, key, fig_pca(d), fig_scree(d), fig_variants(d)]

    text = captions(d)
    (OUTPUTS / "captions.md").write_text(
        "# Figure captions (generated by plot_embeddings.py, do not hand-edit)\n\n"
        + text + "\n"
    )
    print(f"  {OUTPUTS / 'captions.md'}")
    print(f"\nPNG review copies (not in the repo): {PNG_DIR}")
    print("\n--- caption text (also written to results/wmb10x/captions.md) ---\n")
    print(text)
    return written


if __name__ == "__main__":
    main()
