# src/duet/initialization.py
"""Initialization strategies for DUET runs.

Protocol + production-accessible strategies. Benchmark-only warm-starts
(Sivanandan / Feldman) live in duet/ops_benchmark/init.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Protocol

import numpy as np
import pandas as pd

from duet.candidate_pool import CandidatePool


@dataclass
class InitMetadata:
    """Summary info about an initialization; used by runners for logging."""
    description: str
    num_swaps: int = 0


class InitializationStrategy(Protocol):
    """Pluggable initialization returning (indices, metadata)."""

    def get_initial_indices(
        self,
        candidates: CandidatePool,
        seed: int,
        **context: Any,
    ) -> tuple[np.ndarray, InitMetadata]: ...


@dataclass
class RandomInit:
    """Random selection from each group."""

    def get_initial_indices(self, candidates, seed, **_ctx):
        idx = candidates.sample_initial_selection(strategy="random", seed=seed)
        return idx, InitMetadata(description="random")


@dataclass
class BestScoreInit:
    """Top-scoring candidate per group."""

    def get_initial_indices(self, candidates, seed, **_ctx):
        idx = candidates.sample_initial_selection(strategy="best_score", seed=seed)
        return idx, InitMetadata(description="best_score")


@dataclass
class CodebookWarmStart:
    """Warm-start from a precomputed codebook (e.g. Chen 2015 MERFISH codebooks).

    Looks up the codebook's sequences in the candidate library and returns
    the resulting indices, optionally with a fraction randomly swapped.
    """

    codebook_path: Path
    noise_percent: float = 0.0
    description_tag: str = "codebook"
    sequence_col: str = "Sequence"

    def get_initial_indices(self, candidates, seed, **_ctx):
        seqs = pd.read_csv(self.codebook_path, dtype={self.sequence_col: str})[self.sequence_col].tolist()
        idx = _lookup_indices(seqs, candidates.sequences)
        num_swaps = 0
        desc = self.description_tag
        if self.noise_percent > 0:
            idx, num_swaps = _apply_noise(
                idx, candidates.pool_size, self.noise_percent, seed,
            )
            desc = f"{self.description_tag}+{self.noise_percent:.1f}%noise"
        return idx, InitMetadata(description=desc, num_swaps=num_swaps)


# Module-private helpers ---------------------------------------------------

def _lookup_indices(
    sequences: List[str], library: List[str] | np.ndarray,
) -> np.ndarray:
    """Map sequence strings to indices in the candidate library.

    Raises ValueError if any sequence is missing.
    """
    library_list = library.tolist() if isinstance(library, np.ndarray) else library
    seq_to_idx = {seq: i for i, seq in enumerate(library_list)}
    indices = []
    for seq in sequences:
        if seq not in seq_to_idx:
            raise ValueError(f"Sequence not found in candidate library: {seq}")
        indices.append(seq_to_idx[seq])
    return np.array(indices, dtype=np.int64)


def _apply_noise(
    init: np.ndarray,
    pool_size: int,
    noise_percent: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Randomly swap `noise_percent` of `init` with other candidates.

    Returns (modified_indices, num_swaps).
    """
    if noise_percent <= 0:
        return init.copy(), 0

    rng = np.random.default_rng(seed)
    n = len(init)
    n_to_swap = max(1, int(round(n * noise_percent / 100.0)))

    init_set = set(init.tolist())
    available = [i for i in range(pool_size) if i not in init_set]
    if len(available) < n_to_swap:
        raise ValueError(
            f"Not enough candidates for noise: need {n_to_swap}, "
            f"have {len(available)}"
        )

    positions = rng.choice(n, size=n_to_swap, replace=False)
    replacements = rng.choice(available, size=n_to_swap, replace=False)

    noisy = init.copy()
    for pos, rep in zip(positions, replacements):
        noisy[pos] = rep
    return noisy, n_to_swap
