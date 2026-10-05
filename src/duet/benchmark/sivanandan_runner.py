from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np

from duet.greedy_optimizers import SivanandanOptimizer
from duet.hamming import SparseHammingMatrix


@dataclass
class SivanandanParams:
    """Parameters for Sivanandan et al. baseline.

    Attributes:
        target_min_hamming: Target minimum Hamming distance between selected guides.
        control_groups: Set of group names that are controls. Controls are
            processed after all targeting genes.
    """
    target_min_hamming: int = 3
    control_groups: Set[str] | None = None


def run_sivanandan(
    encoded_sequences: np.ndarray,
    gene_to_indices: Dict[str, List[int]],
    gene_to_N: Dict[str, int],
    scores: np.ndarray,
    params: SivanandanParams,
    num_cpus: int = 1,
) -> np.ndarray:
    """Run Sivanandan et al. greedy Hamming distance optimization.

    Args:
        encoded_sequences: Pre-encoded sequences as np.ndarray of shape (n_sequences, seq_length).
                          For dual-guide screens, this should be in **observed space**
                          (not transmitted hex space) to correctly compute observation-based
                          Hamming distances.
        gene_to_indices: Mapping from gene name to list of sequence indices.
        gene_to_N: Mapping from gene name to number of guides to select.
        scores: Array of candidate scores indexed by candidate index.
            Candidates within each group are tried in descending score order.
        params: Sivanandan optimization parameters.
        num_cpus: Number of CPUs for parallel computation.

    Returns:
        Array of selected sequence indices.

    Note:
        For dual-guide screens, the caller is responsible for mapping sequences
        from transmitted space (16 symbols) to observed space (4/7/10 symbols
        depending on chemistry) before calling this function.
    """
    # Build sparse Hamming matrix over pre-encoded sequences
    mat = SparseHammingMatrix(
        encoded_sequences,
        threshold=params.target_min_hamming,
        n_jobs=num_cpus,
    )
    opt = SivanandanOptimizer()
    S = opt.optimize(
        mat,
        gene_to_indices,
        gene_to_N,
        target_min_hamming=params.target_min_hamming,
        scores=scores,
        control_groups=params.control_groups,
    )
    return np.array(list(S), dtype=int)
