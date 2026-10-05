"""Unit tests for src/duet/benchmark/metrics.py."""
from __future__ import annotations

import numpy as np
import pytest

from duet.benchmark.metrics import compute_hvr
from _optional_deps import benchmark_extra


@benchmark_extra  # compute_hvr uses pymoo
class TestComputeHVR:
    """compute_hvr returns HV(approx) / HV(true) with a shared reference point."""

    def test_perfect_recovery_returns_one(self):
        """If approx == true, HVR is exactly 1.0."""
        true_front = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        approx_front = true_front.copy()
        assert compute_hvr(approx_front, true_front, maximize=True) == pytest.approx(1.0)

    def test_subset_is_less_than_one(self):
        """A strict subset of the true front has smaller HV → HVR < 1."""
        true_front = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        approx_front = np.array([[0.5, 0.5]])  # one interior point
        hvr = compute_hvr(approx_front, true_front, maximize=True)
        assert 0.0 < hvr < 1.0

    def test_explicit_ratio_matches_hand_computation(self):
        """HVR = HV(approx) / HV(true) under the supplied reference point.

        Hand computation (maximize both axes, reference point (0, 0)):
          true_front = {(3, 1), (2, 2), (1, 3)} — all non-dominated.
          The dominated region above (0, 0) decomposes into vertical strips
          (sorted by x ascending):
            x ∈ [0, 1): height = 3 (from (1, 3))   → area 1·3 = 3
            x ∈ [1, 2): height = 2 (from (2, 2))   → area 1·2 = 2
            x ∈ [2, 3): height = 1 (from (3, 1))   → area 1·1 = 1
          HV(true) = 3 + 2 + 1 = 6.

          approx_front = {(2, 2)}.
          HV(approx) = 2 · 2 = 4.

          HVR = 4 / 6 = 2/3.
        """
        true_front = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
        approx_front = np.array([[2.0, 2.0]])
        ref_point = np.array([0.0, 0.0])

        got = compute_hvr(
            approx_front, true_front, ref_point=ref_point, maximize=True,
        )
        assert got == pytest.approx(2.0 / 3.0)

    def test_degenerate_true_front_returns_nan(self):
        """If HV(true_front) == 0, return NaN rather than dividing by zero."""
        # A single point at the reference point itself has zero HV.
        true_front = np.array([[0.0, 0.0]])
        approx_front = np.array([[0.0, 0.0]])
        ref_point = np.array([0.0, 0.0])
        result = compute_hvr(
            approx_front, true_front, ref_point=ref_point, maximize=True,
        )
        assert np.isnan(result)
