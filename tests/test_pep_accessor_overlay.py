"""Logical-overlay duplicate-offset tests for PEPAccessor."""
import numpy as np
import pytest

from duet.pep_accessor import InMemoryPEPAccessor


def test_get_diagonal_overlays_offset():
    X = np.array([[2.0, 0.3], [0.3, 2.0]], dtype=np.float32)
    acc = InMemoryPEPAccessor(X, n_samples=None, duplicate_offset=50.0)
    assert acc.get_diagonal().tolist() == [52.0, 52.0]


def test_get_row_overlays_offset_only_at_diagonal_slot():
    X = np.array([[2.0, 0.3], [0.3, 2.0]], dtype=np.float32)
    acc = InMemoryPEPAccessor(X, n_samples=None, duplicate_offset=50.0)
    row0 = acc.get_row(0)
    assert row0[0] == 52.0
    np.testing.assert_allclose(row0[1], 0.3, rtol=1e-6)


def test_get_rows_overlays_offset_at_each_rows_diagonal_slot():
    X = np.array([[2.0, 0.3, 0.1],
                  [0.3, 2.0, 0.2],
                  [0.1, 0.2, 2.0]], dtype=np.float32)
    acc = InMemoryPEPAccessor(X, n_samples=None, duplicate_offset=50.0)
    rows = acc.get_rows(np.array([0, 2]))
    assert rows[0, 0] == 52.0
    assert rows[1, 2] == 52.0
    np.testing.assert_allclose(rows[0, 2], 0.1, rtol=1e-6)
    np.testing.assert_allclose(rows[1, 0], 0.1, rtol=1e-6)


def test_stored_matrix_unchanged_by_offset_change():
    X = np.array([[2.0, 0.3], [0.3, 2.0]], dtype=np.float32).copy()
    acc = InMemoryPEPAccessor(X, n_samples=None, duplicate_offset=50.0)
    assert acc._X[0, 0] == 2.0
    assert acc._X[1, 1] == 2.0


def test_offset_default_is_zero():
    X = np.array([[2.0, 0.3], [0.3, 2.0]], dtype=np.float32)
    acc = InMemoryPEPAccessor(X, n_samples=None)
    assert acc.get_diagonal().tolist() == [2.0, 2.0]
    assert acc.get_row(0)[0] == 2.0


def test_get_row_by_candidate_translates_via_candidate_to_sequence_idx():
    # U=3 unique rows; pool_size=4 candidates with one duplicate (c=0 and c=2 share seq u=0)
    X = np.array([[2.0, 0.3, 0.1],
                  [0.3, 2.0, 0.2],
                  [0.1, 0.2, 2.0]], dtype=np.float32)
    c_to_u = np.array([0, 1, 0, 2], dtype=np.int32)
    acc = InMemoryPEPAccessor(
        X, n_samples=None, duplicate_offset=0.0,
        candidate_to_sequence_idx=c_to_u,
    )
    np.testing.assert_array_equal(acc.get_row_by_candidate(0), acc.get_row_by_candidate(2))
    rows = acc.get_rows_by_candidate(np.array([0, 1, 2]))
    np.testing.assert_array_equal(rows[0], rows[2])
    np.testing.assert_array_equal(rows[1], acc.get_row(1))


def test_get_row_by_candidate_without_c_to_u_falls_through():
    """When candidate_to_sequence_idx is None, candidate-indexed methods
    delegate to the plain sequence-indexed methods (no translation needed)."""
    X = np.array([[2.0, 0.3], [0.3, 2.0]], dtype=np.float32)
    acc = InMemoryPEPAccessor(X, n_samples=None)
    np.testing.assert_array_equal(acc.get_row_by_candidate(0), acc.get_row(0))
