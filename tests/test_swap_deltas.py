"""Unit tests for the swap-neighborhood delta core (`duet.benchmark.swap_deltas`).

The neighbor table is where a silent bug would corrupt every downstream number
without producing an obvious error, so its structural invariants are asserted
directly rather than inferred from the figures. The tie-aware metrics are
tested against hand-computed cases with deliberate ties, because the surrogate
whose behaviour this experiment is designed to expose (minimum Hamming
distance) is tied across most of its neighborhood.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import itertools
from math import comb

import numpy as np
import pytest

from duet.benchmark.swap_deltas import (
    build_swap_neighbors,
    codebook_index_order,
    neighborhood_stats,
)


# --------------------------------------------------------------------------
# Neighbor table
# --------------------------------------------------------------------------

def test_codebook_index_order_matches_itertools_combinations():
    """Row i of the enumeration is the i-th `itertools.combinations` tuple.

    The whole analysis indexes the source parquet positionally by
    `codebook_index`; if this ordering assumption is wrong, every delta pairs
    the wrong codebooks and nothing downstream would signal it.
    """
    assert codebook_index_order(6, 3) == list(itertools.combinations(range(6), 3))


@pytest.mark.parametrize("n,k", [(6, 3), (8, 2), (15, 5)])
def test_neighbor_table_shape(n, k):
    nbr = build_swap_neighbors(n, k)
    assert nbr.shape == (comb(n, k), k * (n - k))


@pytest.mark.parametrize("n,k", [(6, 3), (8, 2)])
def test_every_neighbor_differs_by_exactly_one_member(n, k):
    combos = codebook_index_order(n, k)
    nbr = build_swap_neighbors(n, k)
    for a, cb in enumerate(combos):
        for b in nbr[a]:
            shared = set(cb) & set(combos[b])
            assert len(shared) == k - 1, (
                f"codebook {cb} -> {combos[b]} differs by "
                f"{k - len(shared)} members, expected exactly 1"
            )


@pytest.mark.parametrize("n,k", [(6, 3), (8, 2)])
def test_neighbor_relation_is_symmetric(n, k):
    """If B is reachable from A by one swap, A is reachable from B."""
    nbr = build_swap_neighbors(n, k)
    for a in range(nbr.shape[0]):
        for b in nbr[a]:
            assert a in nbr[b], f"{a}->{b} present but {b}->{a} missing"


@pytest.mark.parametrize("n,k", [(6, 3), (8, 2)])
def test_neighbors_are_distinct_and_exclude_self(n, k):
    nbr = build_swap_neighbors(n, k)
    for a in range(nbr.shape[0]):
        row = nbr[a]
        assert a not in row, "a codebook is listed as its own neighbor"
        assert len(set(row.tolist())) == len(row), "duplicate neighbor entries"


# --------------------------------------------------------------------------
# Neighborhood statistics
# --------------------------------------------------------------------------

def _single_neighborhood(acc_neighbors, surr_neighbors, acc_self=0.0, surr_self=0.0):
    """Build a 1-codebook problem whose neighborhood has the given deltas.

    Returns (accuracy, surrogate, neighbors) laid out so that codebook 0's
    neighbors are codebooks 1..m with exactly the requested delta values.
    """
    accuracy = np.array([acc_self] + [acc_self + d for d in acc_neighbors])
    surrogate = np.array([surr_self] + [surr_self + d for d in surr_neighbors])
    neighbors = np.arange(1, len(acc_neighbors) + 1)[None, :]
    return accuracy, surrogate, neighbors


def test_perfect_surrogate_gives_rho_one():
    acc = [0.1, 0.2, 0.3, -0.1]
    accuracy, surrogate, nbr = _single_neighborhood(acc, [2 * d for d in acc])
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    assert stats["local_rho"][0] == pytest.approx(1.0)


def test_sign_reversed_surrogate_gives_rho_minus_one():
    acc = [0.1, 0.2, 0.3, -0.1]
    accuracy, surrogate, nbr = _single_neighborhood(acc, [-d for d in acc])
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    assert stats["local_rho"][0] == pytest.approx(-1.0)


def test_flat_surrogate_is_flagged_and_rho_is_nan():
    """A surrogate constant over the whole neighborhood has no local gradient.

    This must be reported as NaN + is_flat, never silently dropped: 'the
    surrogate cannot distinguish any move here' is the finding, and averaging
    it away would overstate that surrogate's fidelity.
    """
    accuracy, surrogate, nbr = _single_neighborhood([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    assert stats["is_flat"][0]
    assert np.isnan(stats["local_rho"][0])
    assert stats["tie_fraction"][0] == pytest.approx(1.0)


def test_tie_fraction_counts_zero_deltas():
    accuracy, surrogate, nbr = _single_neighborhood(
        [0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 1.0, 2.0],
    )
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    assert stats["tie_fraction"][0] == pytest.approx(0.5)


def test_pseudo_ties_below_rounding_threshold_count_as_ties():
    """Deltas of float sums that are mathematically equal must tie exactly.

    A delta of two summed floats is more prone to last-bit disagreement than
    the sums themselves, so without rounding these would be ranked as distinct.
    """
    accuracy, surrogate, nbr = _single_neighborhood(
        [0.1, 0.2, 0.3], [1.0, 1.0 + 1e-14, 2.0],
    )
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    # First two neighbours tie with each other but not with the third; none of
    # them equals the self value, so tie_fraction (vs zero) is 0.
    assert stats["tie_fraction"][0] == pytest.approx(0.0)
    # The pseudo-tie must not create a spurious strict ordering.
    assert stats["local_rho"][0] == pytest.approx(
        neighborhood_stats(
            *_single_neighborhood([0.1, 0.2, 0.3], [1.0, 1.0, 2.0])
        )["local_rho"][0]
    )


def test_sign_agreement_ignores_tied_swaps():
    """Swaps the surrogate cannot order are excluded from the sign rate.

    Counting them would conflate 'points the wrong way' with 'says nothing',
    which are different failure modes and are reported separately.
    """
    accuracy, surrogate, nbr = _single_neighborhood(
        [0.1, -0.2, 0.3, 0.4], [1.0, -1.0, -1.0, 0.0],
    )
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    # Non-tied swaps: (+,+) agree, (-,-) agree, (+,-) disagree -> 2/3.
    assert stats["sign_agreement"][0] == pytest.approx(2.0 / 3.0)


def test_best_swap_regret_averages_over_surrogate_ties():
    """With a tied surrogate argmax, regret is the mean over the tied set.

    `argmax`'s first-index rule would report an arbitrary member of the tie,
    which for a heavily-tied surrogate is not a measurement of anything.
    """
    # Surrogate ranks swaps 0 and 1 equal-best; their true accuracy deltas are
    # 0.3 and 0.1. Best available delta is 0.4 (swap 3).
    accuracy, surrogate, nbr = _single_neighborhood(
        [0.3, 0.1, 0.0, 0.4], [5.0, 5.0, 1.0, 2.0],
    )
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    # Expected regret under uniform tiebreak = 0.4 - mean(0.3, 0.1) = 0.2
    assert stats["best_swap_regret"][0] == pytest.approx(0.2)
    # Hit rate = fraction of the tied set that is actually accuracy-optimal = 0
    assert stats["best_swap_hit"][0] == pytest.approx(0.0)


def test_best_swap_regret_zero_when_surrogate_picks_optimum():
    accuracy, surrogate, nbr = _single_neighborhood(
        [0.3, 0.1, 0.0, 0.4], [1.0, 2.0, 3.0, 9.0],
    )
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    assert stats["best_swap_regret"][0] == pytest.approx(0.0)
    assert stats["best_swap_hit"][0] == pytest.approx(1.0)


def test_best_swap_hit_counts_accuracy_ties_as_optimal():
    """Two swaps tied at the accuracy optimum both count as hits."""
    accuracy, surrogate, nbr = _single_neighborhood(
        [0.4, 0.4, 0.1], [9.0, 1.0, 2.0],
    )
    stats = neighborhood_stats(accuracy, surrogate, nbr)
    assert stats["best_swap_hit"][0] == pytest.approx(1.0)
    assert stats["best_swap_regret"][0] == pytest.approx(0.0)


def test_stats_are_computed_per_row_independently():
    """Two codebooks with different neighborhoods get their own statistics."""
    accuracy = np.array([0.0, 0.1, 0.2, 0.5, 0.6])
    surrogate = np.array([0.0, 1.0, 2.0, 0.0, 0.0])
    neighbors = np.array([[1, 2], [3, 4]])
    stats = neighborhood_stats(accuracy, surrogate, neighbors)
    assert stats["local_rho"].shape == (2,)
    assert stats["local_rho"][0] == pytest.approx(1.0)   # perfectly ordered
    assert stats["is_flat"][1]                            # both deltas equal


def test_real_pool_geometry_has_fifty_neighbors():
    """The shipped experiment's geometry: 15 candidates choose 5."""
    nbr = build_swap_neighbors(15, 5)
    assert nbr.shape == (3003, 50)
