# tests/test_duet_core.py
"""Unit tests for _run_duet_core with mocked DUET + PEP infrastructure."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from duet.runner.core import (
    DuetOptimizerConfig,
    ObjectiveSpec,
    _run_duet_core,
)


def _tiny_candidates():
    """Minimal CandidatePool-like stub with attributes the core reads."""
    return SimpleNamespace(
        sequences=["aaa", "aac", "aag", "aat", "acc"],
        scores=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        groups=["A", "A", "B", "B", "C"],
        group_to_candidates={"A": [0, 1], "B": [2, 3], "C": [4]},
        pool_size=5,
        total_selections=3,
        quotas={"A": 1, "B": 1, "C": 1},
        num_groups=3,
        # New attributes consumed by _run_single_lambda / _run_duet_core.
        # Stub with identity dedup (no duplicates): unique_sequences == sequences.
        codeword_to_group=["A", "B", "C"],
        candidate_to_sequence_idx=None,
        unique_sequences=["aaa", "aac", "aag", "aat", "acc"],
    )


def _stub_score_factory(c, *, codeword_to_group, candidate_to_sequence_idx=None):
    # Module-level for picklability.
    cache = MagicMock()
    return cache


def _stub_weights(lambda_):
    return {"decode": lambda_, "score": 1.0 - lambda_}


class TestRunDuetCore:
    @patch("duet.runner.core._compute_pep_matrix")
    @patch("duet.runner.core.DUET")
    @patch("duet.runner.core.DecodingSwapCache")
    def test_invokes_weight_schedule_per_lambda(self, mock_decode_cls, mock_DUET, mock_pep):
        """_run_duet_core calls weight_schedule for each lambda and returns sorted results."""
        mock_pep.return_value = (np.zeros((5, 5), dtype=np.uint16), 1000)
        mock_optimizer = MagicMock()
        mock_optimizer.best_S = np.array([0, 2, 4])
        mock_optimizer.history = {}
        mock_DUET.return_value = mock_optimizer

        # Use module-level _stub_weights (picklable). Verify lambda values via result
        # structure rather than tracking internal calls (local functions aren't picklable).
        spec = ObjectiveSpec(
            cache_factories={"score": _stub_score_factory},
            weight_schedule=_stub_weights,
        )
        cfg = DuetOptimizerConfig(lambda_=[0.0, 0.5, 1.0], num_cpus=1)

        result = _run_duet_core(
            candidates=_tiny_candidates(),
            pep_config=MagicMock(num_cpus=1),
            optimizer_config=cfg,
            init=np.array([0, 2, 4]),
            alphabet_size=4,
            objective_spec=spec,
            seed=1,
            cache_dir=None,
        )
        # One result per lambda, returned in sorted order.
        assert result["lambdas"] == [0.0, 0.5, 1.0]
        assert len(result["best_indices"]) == 3

    @patch("duet.runner.core._compute_pep_matrix")
    @patch("duet.runner.core.DUET")
    @patch("duet.runner.core.DecodingSwapCache")
    def test_results_sorted_by_lambda(self, mock_decode_cls, mock_DUET, mock_pep):
        mock_pep.return_value = (np.zeros((5, 5), dtype=np.uint16), 1000)
        mock_DUET.return_value = MagicMock(
            best_S=np.array([0, 2, 4]), history={},
        )
        spec = ObjectiveSpec(
            cache_factories={"score": _stub_score_factory},
            weight_schedule=_stub_weights,
        )
        cfg = DuetOptimizerConfig(lambda_=[1.0, 0.0, 0.5], num_cpus=1)
        result = _run_duet_core(
            candidates=_tiny_candidates(),
            pep_config=MagicMock(num_cpus=1),
            optimizer_config=cfg,
            init=np.array([0, 2, 4]),
            alphabet_size=4,
            objective_spec=spec,
            seed=1,
        )
        assert result["lambdas"] == [0.0, 0.5, 1.0]
