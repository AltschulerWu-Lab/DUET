"""Exhaustive codebook enumeration for optimality benchmarks."""
from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def enumerate_all_codebooks(
    group_to_candidates: Dict[str, List[int]],
    quotas: Dict[str, int],
) -> List[np.ndarray]:
    """Enumerate all possible codebooks respecting per-group quotas.

    For each group, chooses quota candidates from its pool.
    Total codebooks = product over groups of C(|candidates_g|, quota_g).

    Args:
        group_to_candidates: Mapping from group name to candidate indices.
        quotas: Number of candidates to select per group.

    Returns:
        List of codebook index arrays, each of shape (total_selections,).
    """
    groups = sorted(group_to_candidates.keys())

    per_group_selections = []
    for group in groups:
        candidates = group_to_candidates[group]
        num_select = quotas[group]
        selections = list(combinations(candidates, num_select))
        per_group_selections.append(selections)

    all_codebooks = []
    for combo in product(*per_group_selections):
        indices = np.array([idx for selection in combo for idx in selection])
        all_codebooks.append(indices)

    return all_codebooks


def compute_total_codebooks(
    group_to_candidates: Dict[str, List[int]],
    quotas: Dict[str, int],
) -> int:
    """Compute total number of possible codebooks without enumerating.

    Args:
        group_to_candidates: Mapping from group name to candidate indices.
        quotas: Number of candidates to select per group.

    Returns:
        Total number of possible codebooks.
    """
    total = 1
    for group, candidates in group_to_candidates.items():
        num_select = quotas[group]
        total *= comb(len(candidates), num_select)
    return total


def find_optimal_codebook(
    codebooks: List[np.ndarray],
    evaluator,
) -> Tuple[np.ndarray, float]:
    """Find the codebook with highest mean decode accuracy.

    Iterates over all codebooks, evaluating each with the evaluator,
    and tracks the running best without storing all results.

    Args:
        codebooks: List of codebook index arrays.
        evaluator: Object with get_accuracy(indices) -> float method.

    Returns:
        Tuple of (best_indices, best_accuracy).
    """
    best_indices = None
    best_accuracy = -1.0

    for codebook in tqdm(codebooks, desc="Finding optimal codebook"):
        accuracy = evaluator.get_accuracy(codebook)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_indices = codebook

    return best_indices, best_accuracy
