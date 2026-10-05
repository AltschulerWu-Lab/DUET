"""Tests for InitializationStrategy protocol and shared strategies."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from duet.initialization import (
    InitializationStrategy,
    InitMetadata,
    RandomInit,
    BestScoreInit,
    CodebookWarmStart,
)


@pytest.fixture
def tiny_pool():
    """Minimal CandidatePool-like stub with the attributes strategies read."""
    # sequences: 12 candidates across 4 groups of 3 each.
    sequences = [f"seq{i:02d}" for i in range(12)]
    scores = np.arange(12, dtype=float)
    groups = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]
    group_to_candidates = {
        "A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8], "D": [9, 10, 11],
    }

    def sample_initial_selection(strategy, seed):
        rng = np.random.default_rng(seed)
        if strategy == "random":
            return np.array([rng.choice(g) for g in group_to_candidates.values()])
        if strategy == "best_score":
            # one best-score candidate per group
            return np.array([g[-1] for g in group_to_candidates.values()])
        raise ValueError(strategy)

    return SimpleNamespace(
        sequences=sequences,
        scores=scores,
        groups=groups,
        group_to_candidates=group_to_candidates,
        pool_size=12,
        sample_initial_selection=sample_initial_selection,
    )


class TestRandomInit:
    def test_returns_one_index_per_group(self, tiny_pool):
        idx, meta = RandomInit().get_initial_indices(tiny_pool, seed=7)
        assert len(idx) == 4
        assert meta == InitMetadata(description="random", num_swaps=0)

    def test_reproducible(self, tiny_pool):
        a, _ = RandomInit().get_initial_indices(tiny_pool, seed=7)
        b, _ = RandomInit().get_initial_indices(tiny_pool, seed=7)
        np.testing.assert_array_equal(a, b)


class TestBestScoreInit:
    def test_picks_highest_score_per_group(self, tiny_pool):
        idx, meta = BestScoreInit().get_initial_indices(tiny_pool, seed=0)
        # Per the stub, best_score picks the last index of each group.
        np.testing.assert_array_equal(idx, [2, 5, 8, 11])
        assert meta.description == "best_score"


class TestCodebookWarmStart:
    def test_exact_lookup(self, tmp_path, tiny_pool):
        csv = tmp_path / "codebook.csv"
        pd.DataFrame({
            "Gene": ["g0", "g1"],
            "Sequence": ["seq03", "seq06"],
        }).to_csv(csv, index=False)
        strat = CodebookWarmStart(codebook_path=csv, description_tag="codebook_1")
        idx, meta = strat.get_initial_indices(tiny_pool, seed=0)
        np.testing.assert_array_equal(idx, [3, 6])
        assert meta.description == "codebook_1"
        assert meta.num_swaps == 0

    def test_with_noise_swaps_some(self, tmp_path, tiny_pool):
        csv = tmp_path / "codebook.csv"
        pd.DataFrame({
            "Gene": [f"g{i}" for i in range(10)],
            "Sequence": [f"seq{i:02d}" for i in range(10)],  # 10 of 12
        }).to_csv(csv, index=False)
        strat = CodebookWarmStart(
            codebook_path=csv, noise_percent=20.0, description_tag="codebook_1",
        )
        idx, meta = strat.get_initial_indices(tiny_pool, seed=42)
        assert len(idx) == 10
        assert meta.num_swaps == 2  # 20% of 10
        assert "noise" in meta.description
