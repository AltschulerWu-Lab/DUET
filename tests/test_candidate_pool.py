"""Tests for CandidatePool optional scores."""
import numpy as np
import pytest

from duet.candidate_pool import CandidatePool


class TestOptionalScores:
    """Test that CandidatePool works with scores=None."""

    def test_scores_none_creates_ones(self):
        """When scores=None, internal scores should be all ones."""
        pool = CandidatePool(
            sequences=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int8),
            group_to_candidates={"g0": [0, 1]},
            quotas={"g0": 1},
            scores=None,
        )
        np.testing.assert_array_equal(pool.scores, np.ones(2))

    def test_scores_none_validation_passes(self):
        """Pool with scores=None should pass all validation."""
        pool = CandidatePool(
            sequences=np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int8),
            group_to_candidates={"g0": [0, 1], "g1": [2]},
            quotas={"g0": 1, "g1": 1},
            scores=None,
        )
        assert pool.pool_size == 3
        assert pool.num_groups == 2

    def test_explicit_scores_still_work(self):
        """Existing behavior with explicit scores should be unchanged."""
        scores = np.array([0.9, 0.1])
        pool = CandidatePool(
            sequences=np.array([[0, 1], [2, 3]], dtype=np.int8),
            group_to_candidates={"g0": [0, 1]},
            quotas={"g0": 1},
            scores=scores,
        )
        np.testing.assert_array_equal(pool.scores, scores)

    def test_sample_initial_selection_with_none_scores(self):
        """sample_initial_selection should work with scores=None."""
        pool = CandidatePool(
            sequences=np.array([[0, 1], [2, 3], [1, 0], [3, 2]], dtype=np.int8),
            group_to_candidates={"g0": [0, 1], "g1": [2, 3]},
            quotas={"g0": 1, "g1": 1},
            scores=None,
        )
        selection = pool.sample_initial_selection(strategy="random", seed=42)
        assert len(selection) == 2

    def test_from_dataframe_still_works(self):
        """from_dataframe should still work (it always provides scores)."""
        import pandas as pd
        df = pd.DataFrame({
            "Group": ["g0", "g0", "g1"],
            "Sequence": ["AT", "CG", "TA"],
            "Score": [0.9, 0.8, 0.7],
            "Quota": [1, 1, 1],
        })
        pool = CandidatePool.from_dataframe(df)
        assert pool.pool_size == 3
