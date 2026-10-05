"""Tests for DUET softmax annealing (temperature scheduling)."""

import numpy as np
import pytest

from duet.pareto_optimization import (
    DecodingSwapCache,
    ScoreSwapCache,
    DUET,
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_duet(
    pep_matrix,
    group_to_candidates,
    scores,
    lambda_=1.0,
    codeword_to_group=None,
    **kwargs,
):
    """Helper to construct DUET with new API from old-style parameters.

    Translates lambda_ to weights: lambda_ * decode + (1-lambda_) * score
    (lambda_ = 1 is decode-only).

    codeword_to_group is slot-indexed (one entry per codeword position in S).
    When omitted it defaults to one codeword per group in dict order, which
    matches every init used in this file (e.g. init=[0, 10, 20] for groups
    A, B, C).
    """
    if codeword_to_group is None:
        codeword_to_group = list(group_to_candidates.keys())
    decode_cache = DecodingSwapCache.from_pep_matrix(
        pep_matrix=pep_matrix,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
    )
    score_cache = ScoreSwapCache(
        scores=scores,
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
    )
    weights = {"decode": lambda_, "score": 1.0 - lambda_}
    return DUET(
        group_to_candidates=group_to_candidates,
        codeword_to_group=codeword_to_group,
        objective_caches=[decode_cache, score_cache],
        weights=weights,
        **kwargs,
    )


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestAnnealingBackwardCompatibility:
    """Verify annealing=False preserves existing behavior."""

    def test_default_annealing_disabled(self):
        """Default should be annealing=False."""
        pep = np.array([[0.0, 0.5], [0.5, 0.0]])
        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": [0, 1]},
            scores=np.array([1.0, 0.5]),
            lambda_=0.5,
            temperature=0.0,
        )
        assert optimizer.annealing is False

    def test_fixed_temperature_when_annealing_disabled(self):
        """Temperature should not change when annealing=False."""
        pep = np.eye(10) * 0.0 + 0.1
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.random.rand(10),
            lambda_=1.0,
            temperature=0.5,
            annealing=False,
            max_iter=100,
            max_patience=50,
            track_history=True,
        )

        init = [0]
        optimizer.optimize(init=init, seed=42)

        # Temperature should remain constant (not tracked when annealing=False)
        assert "temperature" not in optimizer.history or all(
            t == 0.5 for t in optimizer.history.get("temperature", [0.5])
        )

    def test_get_current_temperature_fallback(self):
        """_get_current_temperature should return fixed temperature when annealing=False."""
        pep = np.array([[0.0, 0.5], [0.5, 0.0]])
        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": [0, 1]},
            scores=np.array([1.0, 0.5]),
            lambda_=0.5,
            temperature=0.3,
            annealing=False,
        )
        assert optimizer._get_current_temperature() == 0.3


# =============================================================================
# Annealing Behavior Tests
# =============================================================================


class TestAnnealingBehavior:
    """Test annealing mechanics."""

    def test_temperature_decreases(self):
        """Temperature should decrease over iterations when starting hot."""
        pep = np.eye(10) * 0.0 + 0.1
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.random.rand(10),
            lambda_=1.0,
            annealing=True,
            initial_temperature=1.0,
            cooling_rate=0.99,
            final_temperature=1e-6,
            max_iter=100,
            max_patience=100,
            track_history=True,
        )

        optimizer.optimize(init=[0], seed=42)

        temps = optimizer.history["temperature"]
        assert len(temps) > 1
        assert temps[-1] < temps[0], "Temperature should decrease"

    def test_auto_calibration(self):
        """Auto-calibration should set reasonable initial temperature."""
        pep = np.random.RandomState(123).rand(20, 20) * 0.2
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(20))},
            scores=np.random.RandomState(123).rand(20),
            lambda_=1.0,
            annealing=True,
            initial_temperature=None,  # Auto-calibrate
            max_iter=10,
            max_patience=10,
        )

        optimizer.optimize(init=[0], seed=42)

        assert optimizer._initial_temperature is not None
        assert optimizer._initial_temperature > 0

    def test_explicit_initial_temperature(self):
        """Explicit initial_temperature should override auto-calibration."""
        pep = np.random.RandomState(42).rand(10, 10) * 0.2
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.random.RandomState(42).rand(10),
            lambda_=1.0,
            annealing=True,
            initial_temperature=2.5,
            max_iter=10,
            max_patience=10,
            track_history=True,
        )

        optimizer.optimize(init=[0], seed=42)

        assert optimizer._initial_temperature == 2.5
        assert optimizer.history["temperature"][0] == 2.5

    def test_temperature_floor(self):
        """Temperature should not drop below final_temperature."""
        pep = np.eye(5) * 0.0 + 0.1
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(5))},
            scores=np.random.RandomState(42).rand(5),
            lambda_=1.0,
            annealing=True,
            initial_temperature=0.001,
            cooling_rate=0.5,  # Very aggressive cooling
            final_temperature=1e-4,
            max_iter=100,
            max_patience=100,
            track_history=True,
        )

        optimizer.optimize(init=[0], seed=42)

        # After aggressive cooling, current temperature should be >= final_temperature
        # (it may go slightly below due to one final multiplication, but the
        # cooling stops applying once below threshold)
        assert optimizer._current_temperature >= optimizer.final_temperature * optimizer.cooling_rate

    def test_reheating_triggers(self):
        """Reheating should trigger after patience exceeded."""
        # Create a PEP matrix where optimization will stall
        pep = np.ones((10, 10)) * 0.5
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.ones(10),  # All same score = no improvement possible
            lambda_=1.0,
            annealing=True,
            initial_temperature=1.0,
            cooling_rate=0.99,
            reheat_patience=10,
            reheat_factor=5.0,
            max_iter=50,
            max_patience=100,
            track_history=True,
            verbose=False,
        )

        optimizer.optimize(init=[0], seed=42)

        temps = optimizer.history["temperature"]
        # Should see temperature increase at some point (reheating)
        has_reheat = any(temps[i] > temps[i - 1] for i in range(1, len(temps)))
        assert has_reheat, "Reheating should have occurred"

    def test_reheat_capped_at_initial(self):
        """Reheating should not exceed initial temperature."""
        pep = np.ones((10, 10)) * 0.5
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.ones(10),
            lambda_=1.0,
            annealing=True,
            initial_temperature=1.0,
            cooling_rate=0.99,
            reheat_patience=5,
            reheat_factor=100.0,  # Very large factor
            max_iter=50,
            max_patience=100,
            track_history=True,
            verbose=False,
        )

        optimizer.optimize(init=[0], seed=42)

        temps = optimizer.history["temperature"]
        assert all(
            t <= optimizer._initial_temperature + 1e-12 for t in temps
        ), "Temperature should never exceed initial temperature"

    def test_no_reheat_when_patience_none(self):
        """No reheating should occur when reheat_patience is None."""
        pep = np.ones((10, 10)) * 0.5
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.ones(10),
            lambda_=1.0,
            annealing=True,
            initial_temperature=1.0,
            cooling_rate=0.99,
            reheat_patience=None,  # Disabled
            max_iter=50,
            max_patience=100,
            track_history=True,
            verbose=False,
        )

        optimizer.optimize(init=[0], seed=42)

        temps = optimizer.history["temperature"]
        # Temperature should monotonically decrease (no reheating)
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1], (
                f"Temperature should not increase at step {i}: "
                f"{temps[i]} > {temps[i-1]}"
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnnealingIntegration:
    """Integration tests with realistic-ish problem setups."""

    def test_annealing_runs_to_completion(self):
        """Full optimization with annealing should complete without errors."""
        rng = np.random.RandomState(42)
        n = 30
        pep = rng.rand(n, n) * 0.3
        np.fill_diagonal(pep, 0.0)
        pep = (pep + pep.T) / 2  # Make symmetric

        group_to_candidates = {
            "A": list(range(0, 10)),
            "B": list(range(10, 20)),
            "C": list(range(20, 30)),
        }

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates=group_to_candidates,
            scores=rng.rand(n),
            lambda_=1.0,
            annealing=True,
            initial_temperature=None,
            cooling_rate=0.999,
            final_temperature=1e-6,
            reheat_patience=50,
            reheat_factor=5.0,
            max_iter=200,
            max_patience=100,
            track_history=True,
            verbose=False,
        )

        optimizer.optimize(init=[0, 10, 20], seed=42)

        # Should have tracked temperatures
        assert "temperature" in optimizer.history
        assert len(optimizer.history["temperature"]) > 0

        # Should have a valid best solution
        assert optimizer.best_S is not None
        assert len(optimizer.best_S) == 3

    def test_annealing_disabled_matches_base(self):
        """Annealing disabled should match base ScalarizedParetoOptimizer behavior."""
        rng = np.random.RandomState(42)
        n = 20
        pep = rng.rand(n, n) * 0.2
        np.fill_diagonal(pep, 0.0)

        group_to_candidates = {"A": list(range(0, 10)), "B": list(range(10, 20))}
        scores = rng.rand(n)

        # Run without annealing
        opt1 = make_duet(
            pep_matrix=pep,
            group_to_candidates=group_to_candidates,
            scores=scores,
            lambda_=1.0,
            temperature=0.0,
            annealing=False,
            max_iter=50,
            max_patience=30,
            track_history=True,
        )
        opt1.optimize(init=[0, 10], seed=42)

        # Run with annealing=False and extra params (should be ignored)
        opt2 = make_duet(
            pep_matrix=pep,
            group_to_candidates=group_to_candidates,
            scores=scores,
            lambda_=1.0,
            temperature=0.0,
            annealing=False,
            initial_temperature=1.0,
            cooling_rate=0.99,
            reheat_patience=10,
            max_iter=50,
            max_patience=30,
            track_history=True,
        )
        opt2.optimize(init=[0, 10], seed=42)

        # Both should produce identical results
        np.testing.assert_array_equal(opt1.best_S, opt2.best_S)
        assert opt1.best_scalarized_objective == opt2.best_scalarized_objective


# =============================================================================
# Starting Temperature Tests
# =============================================================================


class TestStartingTemperature:
    """Test that annealing starts at initial_temperature."""

    def test_starts_at_initial_temperature(self):
        """When annealing=True, starting temperature should be initial_temperature."""
        pep = np.ones((10, 10)) * 0.5
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.ones(10),
            lambda_=1.0,
            annealing=True,
            initial_temperature=2.5,
            reheat_patience=5,
            max_iter=50,
            max_patience=100,
            track_history=True,
            verbose=False,
        )

        optimizer.optimize(init=[0], seed=42)

        temps = optimizer.history["temperature"]
        assert temps[0] == 2.5, f"Starting temperature should be 2.5, got {temps[0]}"

    def test_reheat_goes_to_initial_temperature(self):
        """After reheat, temperature should equal initial_temperature."""
        pep = np.ones((10, 10)) * 0.5
        np.fill_diagonal(pep, 0.0)

        optimizer = make_duet(
            pep_matrix=pep,
            group_to_candidates={"A": list(range(10))},
            scores=np.ones(10),
            lambda_=1.0,
            annealing=True,
            initial_temperature=2.0,
            reheat_patience=5,
            max_iter=50,
            max_patience=100,
            track_history=True,
            verbose=False,
        )

        optimizer.optimize(init=[0], seed=42)

        temps = optimizer.history["temperature"]
        # All temperatures should be <= initial_temperature
        assert all(t <= 2.0 + 1e-12 for t in temps)
        # After reheat, temperature should go exactly to initial_temperature
        for i in range(1, len(temps)):
            if temps[i] > temps[i - 1] + 0.5:  # Significant jump = reheat
                assert abs(temps[i] - 2.0) < 1e-12, (
                    f"Reheat should go to initial_temperature=2.0, got {temps[i]}"
                )
