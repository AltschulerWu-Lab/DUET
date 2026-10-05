"""
Unit tests for Hamming distance computations
"""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import numpy as np
from duet.hamming import DenseHammingMatrix, SparseHammingMatrix
from duet.codebook_evaluator import DNAEncoder


# Helper to encode DNA sequences
def encode_dna(seqs):
    """Encode a list of DNA strings to numpy array."""
    encoder = DNAEncoder()
    return encoder.encode(seqs)


def evaluate_collisions(seqs, expected_min_hamming, expected_collisions, threshold=1):
    encoded = encode_dna(seqs)
    dense_mat = DenseHammingMatrix(encoded)
    sparse_mat = SparseHammingMatrix(encoded, threshold)

    for hmat in [dense_mat, sparse_mat]:
        collisions, min_hamming = hmat.get_collisions()
        assert min_hamming == hmat.min()
        assert min_hamming == expected_min_hamming
        assert collisions == expected_collisions


def test_small():
    seq_list = ['AAAA', 'AAAG', 'GGGG']
    evaluate_collisions(seq_list, 1, {0, 1})


def test_min_hamming_zero():
    seq_list = ['AAAA', 'AAAG', 'AAAA']
    evaluate_collisions(seq_list, 0, {0, 2})


def test_min_hamming_all_zero():
    seqs = ['AAAA', 'AAAA', 'AAAA']
    evaluate_collisions(seqs, 0, {0, 1, 2})


def test_sparse_data_repr():
    seqs = ['AAAA', 'AAAA', 'AAAG', 'AAGG']
    encoded = encode_dna(seqs)
    hmat = SparseHammingMatrix(encoded, threshold=1)

    expected_data = np.array(
        [[0., 0., 1., np.inf],
         [0., 0., 1., np.inf],
         [1., 1., 0., 1.],
         [np.inf, np.inf, 1., 0.]]
    )
    assert np.array_equal(expected_data, hmat.get_data())

    sparse_repr = np.array(
        [[0., 1., 2., 0.],
         [1., 0., 2., 0.],
         [2., 2., 0., 2.],
         [0., 0., 2., 0.]]
    )
    assert np.array_equal(sparse_repr, hmat.data.todense())


def test_dense_sparse_equality():
    seqs = ['AAAA', 'AAAA', 'AAAG', 'AAGG']
    encoded = encode_dna(seqs)
    assert np.array_equal(
        SparseHammingMatrix(encoded, threshold=np.inf).get_data(),
        DenseHammingMatrix(encoded).get_data()
    )


def test_collisions_beyond_sparse_threshold():
    seqs = ['AAAAAAAAAA', 'AAAAAAAAAA', 'AAAGGGGGGG']
    encoded = encode_dna(seqs)
    hamming_mat = SparseHammingMatrix(encoded, threshold=3)
    collisions, min_hamming = hamming_mat.get_collisions({1, 2})
    assert min_hamming == np.inf
    assert collisions == {1, 2}


def test_lower_threshold():
    seqs = ['AAAA', 'AAGG', 'AGGG']
    encoded = encode_dna(seqs)
    hamming_mat = SparseHammingMatrix(encoded, threshold=3)
    expected_data = np.array([
        [0., 2., 3.],
        [2., 0., 1.],
        [3., 1., 0.]
    ])
    assert np.array_equal(expected_data, hamming_mat.get_data())

    hamming_mat2 = hamming_mat.lower_threshold(2)
    expected_data = np.array([
        [0., 2., np.inf],
        [2., 0., 1.],
        [np.inf, 1., 0.]
    ])
    assert np.array_equal(expected_data, hamming_mat2.get_data())

    hamming_mat3 = SparseHammingMatrix(encoded, threshold=2)
    assert np.array_equal(hamming_mat2.get_data(), hamming_mat3.get_data())
    assert np.array_equal(hamming_mat2.data.todense(), hamming_mat3.data.todense())
