"""Swap-neighborhood delta statistics for surrogate-objective fidelity.

The DUET-vs-baseline-objectives benchmark asks a *global* question: across all
enumerated codebooks, does a surrogate objective rank them the way true decode
accuracy does? This module asks the *local* question that local search actually
faces: standing at one codebook, over the one-swap moves available from it, does
the surrogate's change point the same way accuracy's change does?

A surrogate can be an excellent global ranker and still be a poor search guide.
The two failure modes this module separates are:

  * **wrong gradient** -- the surrogate orders the neighborhood, but orders it
    differently from accuracy (`local_rho`, `sign_agreement`); and
  * **no gradient** -- the surrogate assigns many moves the identical value, so
    it cannot order them at all (`tie_fraction`, `is_flat`).

These are reported separately because averaging them together would credit a
surrogate that says nothing with the same score as one that is merely noisy.

The neighbor table is a pure function of pool geometry, so it is built once and
reused across every (trial, noise channel, surrogate) combination.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import rankdata

from duet.benchmark.baseline_objectives import round_for_ranking

__all__ = [
    "build_swap_neighbors",
    "codebook_index_order",
    "neighborhood_stats",
    "STAT_COLUMNS",
]


# Absolute tolerance for treating two *accuracy* deltas as tied when locating
# the accuracy-optimal swap. Accuracy is a mean over the codebook's members of a
# mean over `num_samples` per-read credits, so its granularity is of order
# 1 / (num_samples * |S|) -- about 2e-5 for the shipped 10,000 samples and
# |S| = 5. Genuinely equal values are equal up to float summation error, and
# 1e-12 separates that from the smallest real gap by many orders of magnitude.
# Surrogate deltas use `round_for_ranking` instead -- see `neighborhood_stats`.
_ACC_TIE_TOL = 1e-12


# Columns emitted by `neighborhood_stats`, in reporting order.
STAT_COLUMNS: List[str] = [
    "local_rho",
    "tie_fraction",
    "sign_agreement",
    "best_swap_regret",
    "best_swap_hit",
    "is_flat",
]


def codebook_index_order(num_candidates: int, quota: int) -> List[Tuple[int, ...]]:
    """The codebook enumeration order, as sorted candidate-index tuples.

    `duet.benchmark.exhaustive.enumerate_all_codebooks` emits single-group
    codebooks in `itertools.combinations` order, and the benchmark parquets
    store that position as `codebook_index`. This analysis indexes those
    parquets positionally, so the assumption is stated here in one place and
    asserted in the tests rather than left implicit at each call site.
    """
    return list(itertools.combinations(range(num_candidates), quota))


def build_swap_neighbors(num_candidates: int, quota: int) -> np.ndarray:
    """Map each codebook to the codebooks reachable by one swap.

    A swap removes one selected candidate and adds one unselected candidate, so
    every codebook has exactly ``quota * (num_candidates - quota)`` neighbors.
    With a single group and a flat quota, every such neighbor is itself a member
    of the enumeration -- the neighborhood never leaves the table.

    Args:
        num_candidates: Size of the candidate pool.
        quota: Codebook size.

    Returns:
        Integer array of shape ``(C(num_candidates, quota), quota *
        (num_candidates - quota))``. Row ``a`` holds the codebook indices
        reachable from codebook ``a`` by a single swap.
    """
    combos = codebook_index_order(num_candidates, quota)
    rank = {combo: i for i, combo in enumerate(combos)}

    n_neighbors = quota * (num_candidates - quota)
    neighbors = np.empty((len(combos), n_neighbors), dtype=np.int32)

    for a, combo in enumerate(combos):
        members = set(combo)
        row = []
        for removed in combo:
            kept = members - {removed}
            for added in range(num_candidates):
                if added in members:
                    continue
                row.append(rank[tuple(sorted(kept | {added}))])
        neighbors[a] = row

    return neighbors


def neighborhood_stats(
    accuracy: np.ndarray,
    surrogate: np.ndarray,
    neighbors: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-neighborhood agreement between surrogate deltas and accuracy deltas.

    Row ``i`` of `neighbors` is the neighborhood of codebook ``i``; deltas are
    measured from codebook ``i``'s own values, so `accuracy` and `surrogate`
    must be indexed so that position ``i`` is that codebook. Deltas are
    *directed* (from the incumbent outward), which is what a local search sees:
    it stands at one codebook and scores the moves leaving it.

    Surrogate deltas pass through `round_for_ranking` before any rank or
    equality test. This matters more here than for raw objective values: a
    delta is a difference of two float sums, so two moves whose surrogate change
    is mathematically identical routinely differ in the last bits, and both
    `rankdata` and the argmax tie set would treat them as distinct. Accuracy
    deltas are deliberately left unrounded, matching the convention documented
    on `round_for_ranking`.

    Args:
        accuracy: True decode accuracy per codebook, shape ``(n_codebooks,)``.
        surrogate: Surrogate objective per codebook, same shape and indexing.
            Must be max-direction (larger = better), as all five entries of
            `duet.benchmark.baseline_objectives.OBJECTIVES` are.
        neighbors: Neighbor table from `build_swap_neighbors`, shape
            ``(n_rows, n_neighbors)``. ``n_rows`` may be fewer than
            ``n_codebooks``; row ``i`` is always keyed to codebook ``i``.

    Returns:
        Dict of `STAT_COLUMNS` to arrays of shape ``(n_rows,)``:

        - ``local_rho``: Spearman correlation between the surrogate deltas and
          the accuracy deltas over this neighborhood. NaN when either side is
          constant (see ``is_flat``).
        - ``tie_fraction``: fraction of moves the surrogate scores as no change.
        - ``sign_agreement``: among moves the surrogate does *not* tie, the
          fraction whose direction matches accuracy's. NaN if all moves tie.
        - ``best_swap_regret``: accuracy left on the table by following the
          surrogate -- the best available accuracy delta minus the mean accuracy
          delta over the surrogate's tied-argmax set (expected regret under a
          uniform tiebreak, mirroring `compute_argmax_regret`'s convention).
        - ``best_swap_hit``: fraction of that tied-argmax set which is in fact
          accuracy-optimal.
        - ``is_flat``: True where the surrogate takes one value over the entire
          neighborhood and therefore cannot order any move.
    """
    accuracy = np.asarray(accuracy, dtype=np.float64)
    surrogate = np.asarray(surrogate, dtype=np.float64)
    neighbors = np.asarray(neighbors)

    n_rows = neighbors.shape[0]
    self_idx = np.arange(n_rows)

    delta_acc = accuracy[neighbors] - accuracy[self_idx][:, None]
    delta_surr = round_for_ranking(
        surrogate[neighbors] - surrogate[self_idx][:, None]
    )

    is_flat = delta_surr.max(axis=1) == delta_surr.min(axis=1)
    tie_fraction = (delta_surr == 0.0).mean(axis=1)

    # --- Spearman, vectorized over rows -----------------------------------
    # `rankdata(..., axis=1)` average-ranks ties, matching `scipy.stats.
    # spearmanr`'s tie handling; a per-row `spearmanr` loop would give the same
    # numbers but takes minutes over the ~1.2M neighborhoods this is run on.
    rank_surr = rankdata(delta_surr, axis=1)
    rank_acc = rankdata(delta_acc, axis=1)
    cs = rank_surr - rank_surr.mean(axis=1, keepdims=True)
    ca = rank_acc - rank_acc.mean(axis=1, keepdims=True)
    denom = np.sqrt((cs * cs).sum(axis=1) * (ca * ca).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        # denom == 0 exactly when one side is constant; NaN is the correct
        # answer there and matches spearmanr's ConstantInputWarning result.
        local_rho = np.where(denom > 0, (cs * ca).sum(axis=1) / denom, np.nan)

    # --- Directional agreement on the moves the surrogate can order --------
    ordered = delta_surr != 0.0
    agree = (np.sign(delta_surr) == np.sign(delta_acc)) & ordered
    n_ordered = ordered.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sign_agreement = np.where(
            n_ordered > 0, agree.sum(axis=1) / n_ordered, np.nan
        )

    # --- What a greedy step following this surrogate would cost ------------
    best_surr = delta_surr.max(axis=1, keepdims=True)
    chosen = delta_surr == best_surr            # the surrogate's tied argmax set
    n_chosen = chosen.sum(axis=1)

    best_acc = delta_acc.max(axis=1)
    mean_acc_chosen = (delta_acc * chosen).sum(axis=1) / n_chosen
    best_swap_regret = best_acc - mean_acc_chosen

    optimal = delta_acc >= (best_acc[:, None] - _ACC_TIE_TOL)
    best_swap_hit = (chosen & optimal).sum(axis=1) / n_chosen

    return {
        "local_rho": local_rho,
        "tie_fraction": tie_fraction,
        "sign_agreement": sign_agreement,
        "best_swap_regret": best_swap_regret,
        "best_swap_hit": best_swap_hit,
        "is_flat": is_flat,
    }
