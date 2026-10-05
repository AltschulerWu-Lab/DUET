#!/usr/bin/env python
"""Fig 1b: the pairwise-error-probability (PEP) grid glyph. The values are illustrative.

Copy of the script that drew the paper's glyph (as run 2026-07-08). As run, it wrote
pairwise_error_final.svg (sha256 13b0473c...), which was placed into the Figure 1
layout in Inkscape and edited there by hand.

ILLUSTRATIVE, NOT DATA. Nothing is computed from a noise channel, a decoder or a
candidate pool, and nothing is random (no seed). The grid is the full 3-bit binary
codebook (8 codewords). Each off-diagonal chip is coloured by the Hamming distance d
between the transmitted (row) and decoded (column) codeword, d = 1 darkest. That order
is what a binary symmetric channel with flip rate below 0.5 gives, but the three
colours are set-points picked by eye, not probabilities mapped through a colour scale.
The colour bar is qualitative ("High probability" / "Low probability") and has no
numbers.

The design, as the source describes it:
  * 8x8 grid (full 3-bit binary codebook), codewords ordered by HAMMING WEIGHT
    (000, 001, 010, 100, 011, 101, 110, 111 -- swaps 011 and 100 vs binary order).
  * Rounded terracotta chips floating on white gutters; blank diagonal (correct decode).
  * Colours by Hamming distance d: d=1 #b5473f (terracotta anchor), d=2 #daa39f,
    d=3 #f5e5e4.
  * Row/column ticks are black/white codeword box glyphs.
  * Continuous colour-bar legend at ONE THIRD of the grid height, gradient from the d=3
    colour (bottom, low) to the d=1 colour (top, high).
  Hand-emitted SVG (native <path>/<rect>/<text>/<linearGradient>) so it drops straight
  into Inkscape as editable objects.

Changes from the source script (geometry, colours, order and text are unchanged):
  * imports duet.plotting and calls apply_style() (the repository's rule for figure
    scripts). The SVG is written as text, not through matplotlib, so the stylesheet
    does not reach it: the default output is byte-identical to the paper's
    pairwise_error_final.svg;
  * black comes from duet.plotting.OKABE_ITO["black"] (the same #000000). No shared
    palette holds the terracotta set-points or white, so they stay here;
  * writes into --outdir (run.sh: the config's outdir, results/experiments/
    pep_grid_glyph/), never next to the script;
  * the five corner-radius comparison files pairwise_error_final_rx{6,4,3,2,0}.svg,
    which the source always wrote, are opt-in (--rx-variants).

    python generate_pep_grid.py --outdir ../../results/experiments/pep_grid_glyph
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from duet.plotting import OKABE_ITO, apply_style

apply_style()  # shared figure style; it has no effect on the hand-emitted SVG below

# -- Design parameters (values as in the source) ------------------------------
N_BITS = 3
CELL = 44                       # px per grid cell
CHIP_INSET = 3.0                # white gutter: chip is inset this much on each side
CHIP_RX = 4.0                   # rounded-corner radius (px). 6 == the matplotlib preview
                                # (~16% of the 38px chip); lower for crisper corners, 0 = square.
RX_VARIANTS = [6, 4, 3, 2, 0]   # --rx-variants: also write pairwise_error_final_rx<n>.svg
# Colours by Hamming distance (d): d=1 darkest, d=3 lightest. Illustrative set-points:
# the terracotta anchor #b5473f blended 0, 0.50 and 0.86 of the way to white
# (chosen with a set-point sweep during the figure design). Not in duet.plotting's
# palettes.
D_COLOR = {1: "#b5473f", 2: "#daa39f", 3: "#f5e5e4"}
C_LOW, C_HIGH = D_COLOR[3], D_COLOR[1]   # colour-bar endpoints (low = light, high = dark)
BLACK = OKABE_ITO["black"].lower()       # "#000000", the source's literal
WHITE = "#ffffff"

# Codeword box-glyph geometry.
GBOX_W = 7.0
GBOX_H = 12.0
GBOX_STROKE = 0.6
GLYPH_PAD = 9                   # gap between glyphs and the grid

# Colour-bar geometry.
CBAR_W = 15
CBAR_GAP = 26                   # gap between grid and colour bar
CBAR_LABEL_PAD = 8
CBAR_LABEL_W = 108              # room for 'High probability' / 'Low probability'
CBAR_FRAC = 1.0 / 3.0           # colour-bar height as a fraction of the grid height

TOP = 14
FONT_SIZE = 12                  # SVG user units; Figure 1 re-set the two labels to 8 pt

OUTPUT_NAME = "pairwise_error_final.svg"


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def rounded_rect_path(x: float, y: float, w: float, h: float, r: float) -> str:
    """SVG path for a rounded rectangle with the rounding BAKED INTO the geometry.

    Emitting chips as <path> instead of <rect rx=...> makes them scale uniformly in
    Inkscape: a <rect>'s rx is an absolute length that Inkscape keeps fixed when you
    shrink the shape (so corners over-round), whereas a path's corner arcs scale with
    every other coordinate.
    """
    r = max(0.0, min(r, w / 2, h / 2))
    if r == 0:
        return f"M{x:.2f},{y:.2f} h{w:.2f} v{h:.2f} h{-w:.2f} z"
    return (
        f"M{x + r:.2f},{y:.2f} h{w - 2 * r:.2f} "
        f"a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{r:.2f} v{h - 2 * r:.2f} "
        f"a{r:.2f},{r:.2f} 0 0 1 {-r:.2f},{r:.2f} h{-(w - 2 * r):.2f} "
        f"a{r:.2f},{r:.2f} 0 0 1 {-r:.2f},{-r:.2f} v{-(h - 2 * r):.2f} "
        f"a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{-r:.2f} z"
    )


def hamming_weight_order(n_bits: int):
    """Codeword integers sorted by Hamming weight, ties by value; plus bit-string labels."""
    perm = sorted(range(2 ** n_bits), key=lambda k: (bin(k).count("1"), k))
    labels = [format(k, f"0{n_bits}b") for k in perm]
    return perm, labels


def glyph(bits: str, cx: float, cy: float) -> str:
    """A codeword as a centered row of black(1)/white(0) boxes."""
    n = len(bits)
    total = n * GBOX_W
    x0 = cx - total / 2
    y0 = cy - GBOX_H / 2
    out = []
    for k, b in enumerate(bits):
        fill = BLACK if b == "1" else WHITE
        out.append(
            f'<rect x="{x0 + k * GBOX_W:.2f}" y="{y0:.2f}" width="{GBOX_W}" '
            f'height="{GBOX_H}" fill="{fill}" stroke="{BLACK}" '
            f'stroke-width="{GBOX_STROKE}"/>'
        )
    return "".join(out)


def build_svg(rx: float = CHIP_RX) -> str:
    perm, labels = hamming_weight_order(N_BITS)
    n = len(labels)
    grid = n * CELL

    glyph_w = N_BITS * GBOX_W
    ox = glyph_w + GLYPH_PAD + 6          # grid left origin (room for row glyphs)
    oy = TOP
    cbar_x = ox + grid + CBAR_GAP
    cbar_h = grid * CBAR_FRAC
    cbar_y = oy + (grid - cbar_h) / 2
    svg_w = cbar_x + CBAR_W + CBAR_LABEL_PAD + CBAR_LABEL_W
    svg_h = oy + grid + GLYPH_PAD + GBOX_H + 8

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" '
        f'height="{svg_h:.0f}" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}">',
        f'<style>text {{ font-family: Arial, Helvetica, sans-serif; '
        f'font-size: {FONT_SIZE}px; fill: {BLACK}; }}</style>',
        f'<defs><linearGradient id="cbar" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{C_HIGH}"/>'
        f'<stop offset="100%" stop-color="{C_LOW}"/></linearGradient></defs>',
    ]

    # Chips: off-diagonal only, coloured by Hamming distance.
    for r in range(n):
        for c in range(n):
            if r == c:
                continue
            d = hamming(perm[r], perm[c])
            x = ox + c * CELL + CHIP_INSET
            y = oy + r * CELL + CHIP_INSET
            side = CELL - 2 * CHIP_INSET
            parts.append(
                f'<path d="{rounded_rect_path(x, y, side, side, rx)}" '
                f'fill="{D_COLOR[d]}"/>'
            )

    # Codeword box glyphs: rows (left) and columns (bottom).
    for r in range(n):
        cy = oy + r * CELL + CELL / 2
        parts.append(glyph(labels[r], ox - GLYPH_PAD - glyph_w / 2, cy))
    for c in range(n):
        cx = ox + c * CELL + CELL / 2
        parts.append(glyph(labels[c], cx, oy + grid + GLYPH_PAD + GBOX_H / 2))

    # Continuous colour bar (1/3 grid height) + qualitative labels.
    parts.append(
        f'<path d="{rounded_rect_path(cbar_x, cbar_y, CBAR_W, cbar_h, 3)}" '
        f'fill="url(#cbar)"/>'
    )
    label_x = cbar_x + CBAR_W + CBAR_LABEL_PAD
    parts.append(
        f'<text x="{label_x:.2f}" y="{cbar_y:.2f}" dominant-baseline="hanging">'
        f'High probability</text>'
    )
    parts.append(
        f'<text x="{label_x:.2f}" y="{cbar_y + cbar_h:.2f}" '
        f'dominant-baseline="ideographic">Low probability</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--outdir", required=True, type=Path,
                    help="output folder (run.sh passes the config's outdir)")
    ap.add_argument("--rx-variants", action="store_true",
                    help="also write pairwise_error_final_rx{6,4,3,2,0}.svg "
                         "(the source always did)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / OUTPUT_NAME
    out.write_text(build_svg())
    print(f"Wrote {out}  (rx={CHIP_RX})")
    if args.rx_variants:
        for rx in RX_VARIANTS:
            p = out.with_name(f"pairwise_error_final_rx{rx}.svg")
            p.write_text(build_svg(rx))
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
