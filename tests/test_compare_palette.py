"""Tests for the comparison-config palette plumbing.

Covers:
- MethodSource.color: optional dataclass field, defaults to None
- generate_palette: per-method color override wins over auto-assignment
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import sys
from pathlib import Path

# scripts/benchmark/ is not on sys.path by default — comparison.py and
# visualize_comparison.py are scripts, not package modules. Add it here
# so the tests can import them.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

from comparison import MethodSource  # noqa: E402


def test_method_source_color_defaults_to_none():
    src = MethodSource.from_dict({
        "path": "/tmp/results.csv",
        "regex": "DUET.*",
        "label": "DUET (test)",
    })
    assert src.color is None


def test_method_source_color_parsed_when_present():
    src = MethodSource.from_dict({
        "path": "/tmp/results.csv",
        "regex": "DUET.*",
        "label": "DUET (test)",
        "color": "#08306b",
    })
    assert src.color == "#08306b"


def _make_source(label, color=None):
    return MethodSource.from_dict({
        "path": "/tmp/results.csv",
        "regex": "X.*",
        "label": label,
        **({"color": color} if color is not None else {}),
    })


def test_generate_palette_uses_color_override():
    from visualize_comparison import generate_palette  # noqa: E402

    methods = [
        _make_source("DUET (Position-varying, asymmetric error)", color="#08306b"),
        _make_source("DUET (Asymmetric error)",            color="#2171b5"),
    ]
    palette = generate_palette(methods)
    assert palette["DUET (Position-varying, asymmetric error)"] == "#08306b"
    assert palette["DUET (Asymmetric error)"] == "#2171b5"


def test_generate_palette_falls_back_when_no_color():
    """Without color overrides, generate_palette must reproduce the
    pre-existing assignment: BASE_PALETTE substring match wins for known
    baselines, and remaining labels cycle through DUET_COLORS in order."""
    from visualize_comparison import generate_palette, BASE_PALETTE, DUET_COLORS  # noqa: E402

    methods = [
        _make_source("DUET (variant A)"),
        _make_source("Feldman et al. d=1"),
        _make_source("DUET (variant B)"),
    ]
    palette = generate_palette(methods)
    assert palette["DUET (variant A)"]    == DUET_COLORS[0]
    assert palette["Feldman et al. d=1"]  == BASE_PALETTE["Feldman et al."]
    assert palette["DUET (variant B)"]    == DUET_COLORS[1]


def test_generate_palette_mixed_override_and_fallback():
    from visualize_comparison import generate_palette, DUET_COLORS  # noqa: E402

    methods = [
        _make_source("DUET (variant A)", color="#123456"),  # override
        _make_source("DUET (variant B)"),                    # fallback
    ]
    palette = generate_palette(methods)
    assert palette["DUET (variant A)"] == "#123456"
    # Fallback uses DUET_COLORS[0] because variant A consumed the
    # override path, not a DUET_COLORS slot.
    assert palette["DUET (variant B)"] == DUET_COLORS[0]
