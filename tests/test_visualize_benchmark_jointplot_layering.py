"""Draw order and point styling of the DUET-vs-baseline jointplot (Fig. 3b right).

``plot_jointplot`` must draw, bottom to top: baseline points, DUET points,
baseline contour, DUET contour. A single hue-keyed scatter drew the methods in
row order, and DUET rows come first in results.csv, so the baseline buried the
DUET cloud; the contours must also sit above every point layer. Points are small
and faint so they read as a texture under the contours.

Asserted on the SVG output: matplotlib names artists in draw order and SVG
document order is z-order.
"""

from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark"))

from visualize_benchmark import plot_jointplot  # noqa: E402

BASELINE = "Maximum activity"
DUET = "DUET (lambda=0.88)"
COLORS = {BASELINE: "#999999", DUET: "#0072B2"}  # spec_colors order: baseline, DUET


@pytest.fixture(scope="module")
def jointplot_svg(tmp_path_factory) -> str:
    rng = np.random.default_rng(0)
    n = 300
    frames = []
    # DUET rows first, as in results.csv.
    for method, mu in [(DUET, 0.8), (BASELINE, 0.7)]:
        frames.append(
            pd.DataFrame(
                {
                    "Method": method,
                    "Decode accuracy": rng.normal(mu, 0.05, n).clip(0, 1),
                    "Activity score": rng.normal(1.0, 0.2, n),
                    "Trial": 1,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    out = tmp_path_factory.mktemp("jointplot") / "jointplot.svg"
    plot_jointplot(df, COLORS, 1, out)
    return out.read_text()


def _groups(svg: str, prefix: str, attribute: str) -> list[tuple[int, str]]:
    """(document position, first ``attribute`` colour) of each ``<g id="{prefix}_N">``.

    Points are ``<use>`` elements whose fill is the method colour (the marker
    definition before them carries only the white edge stroke); contour paths
    have ``fill: none`` and the method colour as stroke.
    """
    starts = [m.start() for m in re.finditer(rf'id="{prefix}_\d+"', svg)]
    out = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(svg)
        colour = re.search(rf"{attribute}: (#[0-9a-fA-F]{{6}})", svg[start:end])
        assert colour is not None, f"no {attribute} colour in {prefix} group {i}"
        out.append((start, colour.group(1).lower()))
    return out


def test_points_then_contours_baseline_first(jointplot_svg):
    scatter = _groups(jointplot_svg, "PathCollection", "fill")
    contours = _groups(jointplot_svg, "QuadContourSet", "stroke")
    assert len(scatter) == 2, "one scatter layer per method"
    assert len(contours) == 2, "one contour layer per method"

    # Every point layer is below every contour layer.
    assert max(pos for pos, _ in scatter) < min(pos for pos, _ in contours)

    # Within each kind, the baseline is drawn first (bottom) and DUET last (top),
    # regardless of which method's rows come first in the data.
    assert [c for _, c in scatter] == ["#999999", "#0072b2"]
    assert [c for _, c in contours] == ["#999999", "#0072b2"]


def test_points_are_faint_and_contours_opaque(jointplot_svg):
    point = re.search(r'<use [^>]*style="fill: #999999; fill-opacity: ([0-9.]+)', jointplot_svg)
    assert point is not None and float(point.group(1)) == pytest.approx(0.35)

    # Contour paths carry no stroke-opacity, i.e. alpha 1.
    contour_start = jointplot_svg.index('id="QuadContourSet_1"')
    contour = jointplot_svg[contour_start : jointplot_svg.index("</g>", contour_start)]
    assert "stroke-opacity" not in contour
