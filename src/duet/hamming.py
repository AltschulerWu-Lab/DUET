"""Functions for hamming distance computation and visualization.

This module provides HammingMatrix classes for computing pairwise Hamming distances
between sequences. Classes accept pre-encoded numpy arrays (encoding is the caller's
responsibility).
"""
import warnings
from collections import Counter
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
import seaborn as sns


###############
# Computation #
###############

class HammingMatrix:
    """Abstract class for computing and storing a matrix of pairwise Hamming distances.

    Accepts pre-encoded integer arrays. Encoding is the caller's responsibility.

    Args:
        encoded_sequences: Integer-encoded sequences as np.ndarray of shape (n_sequences, seq_length).
                          Values should be in [0, alphabet_size).
        n_jobs: Number of parallel jobs for computation
        metric: Distance metric (default 'hamming')
        construct_matrix: Whether to construct the matrix on init
    """
    def __init__(
        self,
        encoded_sequences: np.ndarray,
        n_jobs: int | None = None,
        metric: str = 'hamming',
        construct_matrix: bool = True
    ):
        # Validate input is numpy array
        if not isinstance(encoded_sequences, np.ndarray):
            raise TypeError(
                f"encoded_sequences must be np.ndarray, got {type(encoded_sequences).__name__}. "
                f"Encoding is the caller's responsibility - use an encoder first."
            )
        if encoded_sequences.ndim != 2:
            raise ValueError(
                f"encoded_sequences must be 2D array of shape (n_sequences, seq_length), "
                f"got {encoded_sequences.ndim}D array with shape {encoded_sequences.shape}"
            )

        self.encoded_sequences = encoded_sequences
        self.n_jobs = n_jobs
        self.metric = metric

        # Derived properties
        self.num_sequences = encoded_sequences.shape[0]
        self.seq_len = encoded_sequences.shape[1]

        if self.seq_len == 0:
            raise ValueError("Sequences must be non-empty (seq_length > 0)")

        # construct pairwise hamming matrix
        self.data = self._construct_hamming_matrix() if construct_matrix else None

    def _construct_hamming_matrix(self) -> np.ndarray:
        """Constructs a matrix of pairwise Hamming distances between sequences.

        Returns:
            np.ndarray: Matrix of pairwise Hamming distances
        """
        raise NotImplementedError

    def min(self, index_set: Set[int] | None = None) -> float:
        """Returns the minimum Hamming distance, subsetted based on `index_set`.

        If `index_set` is None, then the entire matrix is used.
        """
        raise NotImplementedError

    def get_collisions(self, index_set: Set[int] | None = None) -> Tuple[Set[int], float]:
        """Returns (1) the set of indices from `index_set` which have the minimum Hamming distance
        and (2) the minimum Hamming distance.

        If `index_set` is None, then the entire matrix is used.
        """
        raise NotImplementedError

    def get_data(self) -> np.ndarray:
        """Returns the matrix of pairwise Hamming distances."""
        raise NotImplementedError

    def subset(self, index_set: Set[int]) -> 'HammingMatrix':
        """Returns a new HammingMatrix object with the data subsetted to the indices in `index_set`."""
        raise NotImplementedError

    @staticmethod
    def encode_seq_dna(seq: str) -> np.ndarray:
        """Legacy static method for DNA encoding.

        Args:
            seq: DNA sequence string

        Returns:
            Integer-encoded array (A=0, C=1, G=2, T=3)

        Note:
            This uses a different encoding than DNAEncoder (which uses A=0, T=1, C=2, G=3).
            Kept for backward compatibility with existing code that depends on this ordering.
        """
        base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        return np.array(list(map(base_to_num.get, seq)))


class DenseHammingMatrix(HammingMatrix):
    """Construct a dense matrix of pairwise Hamming distances between sequences.

    Internally, diagonal (self-self) entries are set to np.inf.

    Accepts pre-encoded integer arrays. Encoding is the caller's responsibility.

    Args:
        encoded_sequences: Integer-encoded sequences as np.ndarray of shape (n_sequences, seq_length)
        n_jobs: Number of parallel jobs
        metric: Distance metric (default 'hamming')
        construct_matrix: Whether to construct matrix on init
    """
    def _construct_hamming_matrix(self) -> np.ndarray:
        with warnings.catch_warnings():
            # Suppress sklearn 1.8.0 internal threading bug (issue #32631)
            warnings.filterwarnings(
                "ignore",
                message=r".*sklearn\.utils\.parallel\.delayed.*",
                category=UserWarning,
                module=r"sklearn\.utils\.parallel",
            )
            mat = self.seq_len * pairwise_distances(
                self.encoded_sequences,
                metric=self.metric,
                n_jobs=self.n_jobs
            )
        # we assign diagonal elements to np.inf
        # this makes computation of minimum non-diagonal Hamming distance more efficient
        # than alternative approaches (e.g. masking --> finding min)
        np.fill_diagonal(mat, np.inf)
        return mat

    def min(self, index_set: Set[int] | None = None) -> float:
        index_list = list(range(self.data.shape[0])) if index_set is None else list(index_set)
        return np.min(self.data[index_list, :][:, index_list])

    def get_collisions(self, index_set: Set[int] | None = None) -> Tuple[Set[int], float]:
        index_list = list(range(self.data.shape[0]) if index_set is None else list(index_set))
        mat = self.data[index_list, :][:, index_list]
        min_val = np.min(mat)
        x, _ = np.where(mat == min_val)
        return set(index_list[i] for i in x), min_val

    def get_data(self) -> np.ndarray:
        mat = self.data.copy()
        np.fill_diagonal(mat, 0)
        return mat

    def subset(self, index_set: Set[int]) -> 'DenseHammingMatrix':
        index_list = sorted(index_set)
        hamming_mat = DenseHammingMatrix(
            self.encoded_sequences[index_list],
            n_jobs=self.n_jobs,
            metric=self.metric,
            construct_matrix=False
        )
        hamming_mat.data = self.data[index_list, :][:, index_list]
        return hamming_mat


class SparseHammingMatrix(HammingMatrix):
    """Construct a sparse matrix of pairwise Hamming distances between sequences.

    Intended for large datasets. All distances values greater than `threshold` are not
    stored in the matrix. Diagonal (self-self) entries are also not stored.

    Accepts pre-encoded integer arrays. Encoding is the caller's responsibility.

    Note: Hamming distances reported from class methods are designed to be consistent with
    those reported by the dense implementation. However, internally, the sparse implementation
    stores Hamming distance values exactly one higher than the dense implementation.

    This is ultimately a quirk of using scipy.sparse matrices as the underlying data structure.
    scipy.sparse matrices use 0 as the default value for missing entries. Since missing entries
    correspond to Hamming distances greater than `threshold`, these become indistinguishable
    from a Hamming distance of 0.

    Therefore, to construct the sparse distance matrix, we apply the following procedure:
    1. All distances greater than `threshold` are mapped to np.inf
    2. All entries are incremented by 1
    3. All np.inf entries are mapped to 0
    4. All diagonal entries are mapped to 0
    5. Construct a sparse matrix from the resultant matrix

    Args:
        encoded_sequences: Integer-encoded sequences as np.ndarray of shape (n_sequences, seq_length)
        threshold: Maximum Hamming distance to store
        n_jobs: Number of parallel jobs
        working_memory: Memory budget in MB for chunked computation
        construct_matrix: Whether to construct matrix on init
    """
    def __init__(
        self,
        encoded_sequences: np.ndarray,
        threshold: int,
        n_jobs: int | None = None,
        working_memory: int | None = None,
        construct_matrix: bool = True
    ):
        self.threshold = threshold
        self.working_memory = working_memory
        super().__init__(
            encoded_sequences,
            n_jobs=n_jobs,
            construct_matrix=construct_matrix
        )

    def _construct_hamming_matrix(self) -> scipy.sparse.csr_matrix:
        # compute pairwise distances
        with warnings.catch_warnings():
            # Suppress sklearn 1.8.0 internal threading bug (issue #32631)
            warnings.filterwarnings(
                "ignore",
                message=r".*sklearn\.utils\.parallel\.delayed.*",
                category=UserWarning,
                module=r"sklearn\.utils\.parallel",
            )
            gen = pairwise_distances_chunked(
                self.encoded_sequences,
                metric='hamming',
                n_jobs=self.n_jobs,
                working_memory=self.working_memory
            )
            mat_list = []
            for x in gen:
                x = self.seq_len * x
                x[x > self.threshold] = np.inf
                x += 1
                x[np.isinf(x)] = 0
                mat_list.append(scipy.sparse.lil_matrix(x))
        mat = scipy.sparse.vstack(mat_list)
        mat.setdiag(0)
        return mat.tocsr()

    def min(self, index_set: Set[int] | None = None) -> float:
        index_list = list(range(self.data.shape[0]) if index_set is None else list(index_set))
        mat = self.data[index_list, :][:, index_list]
        nonzero_idx = mat.nonzero()
        if len(nonzero_idx[0]) == 0:
            # if there are no nonzero entries, then all entries are np.inf
            # by definition
            return np.inf
        return mat[nonzero_idx].min() - 1

    def get_collisions(self, index_set: Set[int] | None = None) -> Tuple[Set[int], float]:
        index_list = list(range(self.data.shape[0]) if index_set is None else list(index_set))
        mat = self.data[index_list, :][:, index_list]
        nonzero_idx = mat.nonzero()
        if len(nonzero_idx[0]) == 0:
            # if there are no nonzero entries, then all entries are np.inf
            # by definition
            return set(index_list), np.inf
        min_val = mat[nonzero_idx].min()
        x, _, _ = scipy.sparse.find(mat == min_val)
        return set(index_list[i] for i in x), min_val - 1

    def get_data(self) -> np.ndarray:
        mat = self.data.todense()
        mat[mat == 0] = np.inf
        mat -= 1
        np.fill_diagonal(mat, 0)
        return mat

    def subset(self, index_set: Set[int]) -> 'SparseHammingMatrix':
        index_list = sorted(index_set)
        hamming_mat = SparseHammingMatrix(
            self.encoded_sequences[index_list],
            self.threshold,
            n_jobs=self.n_jobs,
            working_memory=self.working_memory,
            construct_matrix=False
        )
        hamming_mat.data = self.data[index_list, :][:, index_list]
        return hamming_mat

    def lower_threshold(self, threshold: int) -> 'SparseHammingMatrix':
        """Returns a sparse matrix with a lower threshold.

        Args:
            threshold: The new threshold (must be <= current threshold)
        """
        if threshold > self.threshold:
            raise ValueError('New threshold cannot be greater than the current threshold')
        hamming_mat = SparseHammingMatrix(
            self.encoded_sequences,
            threshold,
            n_jobs=self.n_jobs,
            working_memory=self.working_memory,
            construct_matrix=False
        )
        hamming_mat.data = self.data.copy()
        # remove elements greater than the new threshold
        # compare against threshold + 1 because internal representation
        # of data stores all entries as one higher
        hamming_mat.data[hamming_mat.data > threshold + 1] = 0
        hamming_mat.data.eliminate_zeros()
        return hamming_mat


#################
# Visualization #
#################

def plot_hamming_matrix(hamming_mat: HammingMatrix):
    """Plot a heatmap of the Hamming distance matrix.

    Args:
        hamming_mat: HammingMatrix instance

    Returns:
        Tuple of (figure, axes)
    """
    def discrete_cmap_params(cmap, min_val, max_val):
        ticks = np.arange(min_val, max_val + 1)
        boundaries = np.arange(min_val - .5, max_val + 1.5)
        cmap = plt.get_cmap(cmap, max_val - min_val + 1)
        return cmap, ticks, boundaries

    mat = hamming_mat.get_data()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # make discrete color map
    cmap, ticks, boundaries = discrete_cmap_params(
        sns.color_palette('mako', as_cmap=True),
        mat[~mask].min(),
        mat[~mask].max()
    )
    sns.heatmap(
        mat, mask=mask, cmap=cmap.reversed(),
        linewidths=.5,
        cbar_kws={'ticks': ticks, 'boundaries': boundaries},
        ax=ax
    )
    ax.set_title('Guide mismatch scores', weight='bold')
    fig.tight_layout()
    return fig, ax


def construct_distance_to_count(
    encoded_sequences: np.ndarray,
    n_jobs: int | None = None,
    working_memory: int | None = None
) -> Counter:
    """Compute histogram of pairwise Hamming distances.

    Args:
        encoded_sequences: Integer-encoded sequences as np.ndarray of shape (n_sequences, seq_length)
        n_jobs: Number of parallel jobs
        working_memory: Memory budget in MB

    Returns:
        Counter mapping distance -> count
    """
    if not isinstance(encoded_sequences, np.ndarray):
        raise TypeError(
            f"encoded_sequences must be np.ndarray, got {type(encoded_sequences).__name__}. "
            f"Encoding is the caller's responsibility - use an encoder first."
        )
    if encoded_sequences.ndim != 2:
        raise ValueError(
            f"encoded_sequences must be 2D, got {encoded_sequences.ndim}D"
        )

    num_sequences = encoded_sequences.shape[0]
    seq_len = encoded_sequences.shape[1]

    gen = pairwise_distances_chunked(
        encoded_sequences,
        metric='hamming',
        n_jobs=n_jobs,
        working_memory=working_memory
    )
    distance_to_count = Counter()
    for x in gen:
        # collect pairwise distances
        distances = (seq_len * x.flatten()).astype(int)
        distance_to_count.update(Counter(distances))
    # correct for double counting
    for k in list(distance_to_count.keys()):
        if k == 0:
            # diagonal entries are not double counted, so we remove them
            assert (distance_to_count[k] - num_sequences) % 2 == 0
            distance_to_count[k] = (distance_to_count[k] - num_sequences) / 2 + num_sequences
        else:
            assert distance_to_count[k] % 2 == 0
            distance_to_count[k] /= 2
    return distance_to_count


def plot_hamming_distribution(
    X,
    n_jobs: int | None = None,
    working_memory: int | None = None,
    ignore_diagonal: bool = True
):
    """Plot the distribution of pairwise Hamming distances between sequences.

    Args:
        X: Integer-encoded sequences (np.ndarray) or HammingMatrix instance
        n_jobs: Number of parallel jobs
        working_memory: Memory budget in MB
        ignore_diagonal: Whether to ignore diagonal entries (default True)

    Returns:
        Tuple of (figure, axes)
    """
    if isinstance(X, HammingMatrix):
        X_mat = X.get_data()
        mask = np.zeros_like(X_mat, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        distance_to_count = Counter(X_mat[mask])
        if ignore_diagonal:
            distance_to_count[0] -= X_mat.shape[0]
        num_sequences = X_mat.shape[0]
    elif isinstance(X, np.ndarray):
        distance_to_count = construct_distance_to_count(
            X,
            n_jobs=n_jobs,
            working_memory=working_memory
        )
        if ignore_diagonal:
            distance_to_count[0] -= X.shape[0]
    else:
        raise TypeError(
            f"X must be np.ndarray or HammingMatrix, got {type(X).__name__}. "
            f"Encoding is the caller's responsibility - use an encoder first."
        )

    fig, ax = plt.subplots()
    plot_df = pd.DataFrame(distance_to_count.items(), columns=['Hamming distance', 'Count'])
    sns.barplot(
        data=plot_df,
        x='Hamming distance',
        y='Count',
        width=0.4,
        color='lightblue',
        ax=ax
    )
    ax.set_title('Guide sequence Hamming distances', weight='bold')
    return fig, ax