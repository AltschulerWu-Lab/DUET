# src/duet/ops_benchmark/init.py
"""OPS-specific initialization strategies and YAML parser.

Strategies that require `baseline_results` (Sivanandan/Feldman warm starts)
live here rather than in duet/initialization.py because they are inherently
benchmark-context-only — they need a baseline to already have run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from duet.candidate_pool import CandidatePool
from duet.initialization import (
    BestScoreInit,
    InitializationStrategy,
    InitMetadata,
    RandomInit,
)


@dataclass
class SivanandanWarmStart:
    """Warm-start from a Sivanandan baseline run at a given edit distance.

    Requires context key: `baseline_results: dict[str, dict[int, np.ndarray]]`.
    """
    edit_distance: int

    def get_initial_indices(self, candidates, seed, *, baseline_results, **_ctx):
        idx = baseline_results["sivanandan"][self.edit_distance]
        idx = fill_incomplete_codebook(idx, candidates, seed)
        return idx, InitMetadata(description=f"sivanandan(ED={self.edit_distance})")


@dataclass
class FeldmanWarmStart:
    """Warm-start from a Feldman baseline run at a given edit distance.

    Requires context key: `baseline_results: dict[str, dict[int, np.ndarray]]`.
    """
    edit_distance: int

    def get_initial_indices(self, candidates, seed, *, baseline_results, **_ctx):
        idx = baseline_results["feldman"][self.edit_distance]
        idx = fill_incomplete_codebook(idx, candidates, seed)
        return idx, InitMetadata(description=f"feldman(ED={self.edit_distance})")


def fill_incomplete_codebook(
    indices: np.ndarray,
    candidates: CandidatePool,
    seed: int,
) -> np.ndarray:
    """Fill incomplete codebook with random selections for missing groups."""
    if len(indices) >= candidates.total_selections:
        return indices

    indices_set = set(indices.tolist())
    candidate_to_group = {
        c: g
        for g, cs in candidates.group_to_candidates.items()
        for c in cs
    }
    group_counts: Dict[str, int] = {}
    for i in indices:
        group = candidate_to_group[int(i)]
        group_counts[group] = group_counts.get(group, 0) + 1

    rng = np.random.default_rng(seed)
    filled = list(indices)

    for group, group_indices in candidates.group_to_candidates.items():
        num_needed = candidates.quotas[group]
        num_have = group_counts.get(group, 0)
        num_missing = num_needed - num_have
        if num_missing > 0:
            available = [i for i in group_indices if i not in indices_set]
            chosen = rng.choice(available, size=num_missing, replace=False)
            filled.extend(chosen.tolist())
            indices_set.update(chosen.tolist())

    print(
        f"WARNING: Baseline returned {len(indices)} indices, "
        f"expected {candidates.total_selections}. "
        f"Filled {len(filled) - len(indices)} with random selections."
    )
    return np.array(filled, dtype=int)


def parse_init(d: Dict[str, Any] | None) -> InitializationStrategy:
    """Build an InitializationStrategy from the YAML dict.

    CLI overrides are applied by the caller to `d` BEFORE invoking this —
    strategies stay immutable after construction.
    """
    if d is None:
        return RandomInit()
    t = d.get("type", "random")
    if t == "random":
        return RandomInit()
    if t in ("best_score", "best_activity"):
        return BestScoreInit()
    if t == "sivanandan":
        if "edit_distance" not in d:
            raise ValueError("sivanandan warm-start requires 'edit_distance'")
        return SivanandanWarmStart(edit_distance=d["edit_distance"])
    if t == "feldman":
        if "edit_distance" not in d:
            raise ValueError("feldman warm-start requires 'edit_distance'")
        return FeldmanWarmStart(edit_distance=d["edit_distance"])
    raise ValueError(f"Unknown OPS init type: {t}")
