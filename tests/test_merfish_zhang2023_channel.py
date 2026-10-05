# tests/test_merfish_zhang2023_channel.py
"""The shipped MERFISH readout channel is present and has the fitted values.

merfish_zhang2023_channel.npy is the 2x2 binary channel the MERFISH experiments
design and evaluate under. Rows are the transmitted bit (0, 1) and columns the
observed bit, so row 1 holds P(1->0) and row 0 holds P(0->1). The rates were
estimated from four samples of Zhang et al. (2023); see the README.txt next to
the file.
"""
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CH = REPO_ROOT / "scripts/benchmark/noise_model_matrices/channels/merfish_zhang2023_channel.npy"


def test_channel_shape():
    assert CH.exists(), f"missing channel matrix {CH}"
    a = np.load(CH)
    assert a.shape == (2, 2) and a.dtype == np.float64
    assert np.allclose(a.sum(axis=1), 1.0), "rows must be probability distributions"


def test_channel_values_are_the_fitted_rates():
    a = np.load(CH)
    assert abs(a[1, 0] - 0.0561) < 1e-4, f"P(1->0) = {a[1, 0]}"
    assert abs(a[0, 1] - 0.0145) < 1e-4, f"P(0->1) = {a[0, 1]}"
