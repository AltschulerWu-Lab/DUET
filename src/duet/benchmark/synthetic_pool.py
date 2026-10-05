"""Synthetic candidate pool generation for benchmarking."""
from __future__ import annotations

import numpy as np

from duet.candidate_pool import CandidatePool


def create_synthetic_pool(
    num_groups: int,
    candidates_per_group: int,
    seq_length: int,
    alphabet_size: int,
    quota: int,
    seed: int | np.random.SeedSequence,
) -> CandidatePool:
    """Generate a random synthetic candidate pool.

    Creates random integer-encoded sequences grouped into num_groups,
    each with candidates_per_group candidates. Scores default to
    uniform ones (set by CandidatePool when scores=None).

    Args:
        num_groups: Number of groups (N).
        candidates_per_group: Candidates per group (k).
        seq_length: Length of each sequence (L).
        alphabet_size: Number of distinct symbols (q).
        quota: Number of candidates to select per group (m).
        seed: Random seed for reproducibility. Accepts int or
            np.random.SeedSequence; np.random.default_rng(seed) handles
            both at runtime.

    Returns:
        CandidatePool with integer-encoded sequences and no scores.
    """
    rng = np.random.default_rng(seed)

    pool_size = num_groups * candidates_per_group
    sequences = rng.integers(0, alphabet_size, size=(pool_size, seq_length), dtype=np.int8)

    group_to_candidates = {
        f"group_{i}": list(
            range(i * candidates_per_group, (i + 1) * candidates_per_group)
        )
        for i in range(num_groups)
    }

    quotas = {f"group_{i}": quota for i in range(num_groups)}

    return CandidatePool(
        sequences=sequences,
        group_to_candidates=group_to_candidates,
        quotas=quotas,
        scores=None,
    )


def create_2d_synthetic_pool(
    num_groups: int,
    candidates_per_group: int,
    seq_length: int,
    alphabet_size: int,
    quota: int,
    seed: int,
) -> CandidatePool:
    """Generate a synthetic pool with iid U(0, 1) scores attached.

    Sequence and score draws use independent child SeedSequences spawned
    from `seed`, so trials seeded as `base + t` for t = 0, 1, ... have
    pairwise-independent score and sequence streams (no cross-trial
    coupling).

    Args:
        num_groups: Number of groups (N).
        candidates_per_group: Candidates per group (k).
        seq_length: Length of each sequence (L).
        alphabet_size: Number of distinct symbols (q).
        quota: Number of candidates to select per group (m).
        seed: Top-level seed; spawn(2) derives separate sequence and
            score sub-streams.

    Returns:
        CandidatePool with integer-encoded sequences and iid U(0, 1) scores.
    """
    seq_seed, score_seed = np.random.SeedSequence(seed).spawn(2)
    pool = create_synthetic_pool(
        num_groups=num_groups,
        candidates_per_group=candidates_per_group,
        seq_length=seq_length,
        alphabet_size=alphabet_size,
        quota=quota,
        seed=seq_seed,
    )
    score_rng = np.random.default_rng(score_seed)
    scores = score_rng.uniform(0.0, 1.0, size=pool.pool_size)
    return CandidatePool(
        sequences=pool.sequences,
        group_to_candidates=pool.group_to_candidates,
        quotas=pool.quotas,
        scores=scores,
    )
