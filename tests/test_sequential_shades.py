import matplotlib.colors as mcolors
import pytest
from duet.plotting import sequential_shades


def _luminance(hex_color: str) -> float:
    r, g, b = mcolors.to_rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_returns_n_valid_distinct_hexes():
    shades = sequential_shades("#0072B2", 5)
    assert len(shades) == 5
    assert len(set(shades)) == 5
    for s in shades:
        mcolors.to_rgb(s)  # raises if not a valid color


def test_monotonically_darkens():
    shades = sequential_shades("#0072B2", 5)
    lums = [_luminance(s) for s in shades]
    assert lums == sorted(lums, reverse=True)  # lightest first, darkest last


def test_n_one_returns_base():
    assert mcolors.to_rgb(sequential_shades("#0072B2", 1)[0]) == mcolors.to_rgb("#0072B2")


def test_n_zero_raises():
    with pytest.raises(ValueError):
        sequential_shades("#0072B2", 0)
