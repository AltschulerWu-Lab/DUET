"""Test-suite-wide configuration."""
import matplotlib

# Force a headless backend before any test imports trigger matplotlib's
# default backend selection. matplotlib.use() must happen before the
# first Figure is created.
matplotlib.use("Agg")


# =============================================================================
# Shared fixtures (moved from src/tests/conftest.py)
# =============================================================================

import pytest
import numpy as np
from typing import Dict, List

# Import DUET components
from duet.codebook_evaluator import (
    SymmetricEpsilon,
    PositionVaryingEpsilon,
    AsymmetricChannel,
    HammingDistance,
    WeightedHammingDistance,
    SymmetricNLL,
    PositionVaryingNLL,
    UniqueMinimum,
    MarginDecoding,
    PairwisePosteriorThreshold,
    CodebookEvaluator,
    DNAEncoder,
    HexEncoder,
    get_encoder,
    create_evaluator_for_dna,
)
from duet.pareto_optimization import (
    DecodingSwapCache,
    ScoreSwapCache,
    DUET,
)


# =============================================================================
# Basic Data Fixtures
# =============================================================================


@pytest.fixture
def simple_dna_library() -> List[str]:
    """Small DNA library for basic tests."""
    return ["ATCG", "ATGG", "GCTA", "GCTG"]


@pytest.fixture
def medium_dna_library() -> List[str]:
    """Medium-sized DNA library with more variety."""
    return [
        "ATCGATCG",
        "ATCGATGG",
        "ATCGGTCG",
        "GCTAGCTA",
        "GCTAGCTG",
        "GCTGGCTA",
        "TTAATTAA",
        "TTAATTAG",
        "TTAATTGA",
    ]


@pytest.fixture
def identical_sequences_library() -> List[str]:
    """Library with all identical sequences (edge case)."""
    return ["ATCG", "ATCG", "ATCG", "ATCG"]


@pytest.fixture
def maximally_distant_library() -> List[str]:
    """Library with maximally distant sequences."""
    return ["AAAA", "TTTT", "CCCC", "GGGG"]


# =============================================================================
# Group Structure Fixtures
# =============================================================================


@pytest.fixture
def simple_group_to_candidates() -> Dict[str, List[int]]:
    """2 groups, 2 candidates each (4 candidates total)."""
    return {
        "groupA": [0, 1],
        "groupB": [2, 3],
    }


@pytest.fixture
def medium_group_to_candidates() -> Dict[str, List[int]]:
    """3 groups, 3 candidates each (9 candidates total)."""
    return {
        "groupA": [0, 1, 2],
        "groupB": [3, 4, 5],
        "groupC": [6, 7, 8],
    }


@pytest.fixture
def unbalanced_group_to_candidates() -> Dict[str, List[int]]:
    """Unbalanced groups: 1, 2, and 4 candidates."""
    return {
        "groupA": [0],
        "groupB": [1, 2],
        "groupC": [3, 4, 5, 6],
    }


@pytest.fixture
def single_group_to_candidates() -> Dict[str, List[int]]:
    """Single group with multiple candidates."""
    return {
        "groupA": [0, 1, 2, 3],
    }


# =============================================================================
# Score Fixtures
# =============================================================================


@pytest.fixture
def simple_scores() -> np.ndarray:
    """Scores for 4 candidates."""
    return np.array([0.9, 0.7, 0.8, 0.6])


@pytest.fixture
def medium_scores() -> np.ndarray:
    """Scores for 9 candidates."""
    return np.array([0.9, 0.7, 0.5, 0.8, 0.6, 0.4, 0.85, 0.75, 0.65])


@pytest.fixture
def uniform_scores() -> np.ndarray:
    """All equal scores."""
    return np.array([0.5, 0.5, 0.5, 0.5])


@pytest.fixture
def zero_scores() -> np.ndarray:
    """All zero scores."""
    return np.array([0.0, 0.0, 0.0, 0.0])


# =============================================================================
# PEP Matrix Fixtures
# =============================================================================


@pytest.fixture
def simple_pep_matrix() -> np.ndarray:
    """
    4x4 PEP matrix with known values.

    Structure:
    - Diagonal is 1.0 (self always wins)
    - Symmetric off-diagonal values representing confusion probabilities
    """
    return np.array([
        [1.0, 0.3, 0.1, 0.15],
        [0.3, 1.0, 0.15, 0.1],
        [0.1, 0.15, 1.0, 0.25],
        [0.15, 0.1, 0.25, 1.0],
    ])


@pytest.fixture
def asymmetric_pep_matrix() -> np.ndarray:
    """4x4 PEP matrix where M[i,j] != M[j,i] for most pairs.

    Used to test that X = M + M^T is computed correctly (symmetric
    fixtures mask bugs where M and M^T are accidentally conflated).
    """
    return np.array([
        [1.0, 0.3, 0.1, 0.15],
        [0.25, 1.0, 0.2, 0.1],
        [0.12, 0.18, 1.0, 0.3],
        [0.1, 0.05, 0.22, 1.0],
    ])


@pytest.fixture
def asymmetric_count_matrix():
    """uint16 count matrix + n_samples, equivalent to asymmetric_pep_matrix * 1000."""
    n_samples = 1000
    M_float = np.array([
        [1.0, 0.3, 0.1, 0.15],
        [0.25, 1.0, 0.2, 0.1],
        [0.12, 0.18, 1.0, 0.3],
        [0.1, 0.05, 0.22, 1.0],
    ])
    return (M_float * n_samples).astype(np.uint16), n_samples


@pytest.fixture
def identity_pep_matrix() -> np.ndarray:
    """PEP matrix with no confusion (perfect discrimination)."""
    return np.eye(4)


@pytest.fixture
def high_confusion_pep_matrix() -> np.ndarray:
    """PEP matrix with high confusion (poor discrimination)."""
    return np.array([
        [1.0, 0.8, 0.7, 0.75],
        [0.8, 1.0, 0.75, 0.7],
        [0.7, 0.75, 1.0, 0.8],
        [0.75, 0.7, 0.8, 1.0],
    ])


@pytest.fixture
def pep_matrix_with_duplicates() -> np.ndarray:
    """
    PEP matrix where guides 0 and 1 are identical sequences.
    PEP[0,1] = PEP[1,0] = 1.0
    """
    return np.array([
        [1.0, 1.0, 0.1, 0.15],
        [1.0, 1.0, 0.15, 0.1],
        [0.1, 0.15, 1.0, 0.25],
        [0.15, 0.1, 0.25, 1.0],
    ])


# =============================================================================
# Noise Channel Fixtures
# =============================================================================


@pytest.fixture
def symmetric_noise_low() -> SymmetricEpsilon:
    """Low error rate symmetric noise channel."""
    return SymmetricEpsilon(epsilon=0.05, alphabet_size=4)


@pytest.fixture
def symmetric_noise_medium() -> SymmetricEpsilon:
    """Medium error rate symmetric noise channel."""
    return SymmetricEpsilon(epsilon=0.1, alphabet_size=4)


@pytest.fixture
def symmetric_noise_high() -> SymmetricEpsilon:
    """High error rate symmetric noise channel."""
    return SymmetricEpsilon(epsilon=0.25, alphabet_size=4)


@pytest.fixture
def position_varying_noise() -> PositionVaryingEpsilon:
    """Position-varying noise channel for length-4 sequences."""
    return PositionVaryingEpsilon(
        epsilons=np.array([0.05, 0.1, 0.15, 0.2]),
        alphabet_size=4
    )


# =============================================================================
# Decoding Metric Fixtures
# =============================================================================


@pytest.fixture
def hamming_distance() -> HammingDistance:
    """Standard Hamming distance."""
    return HammingDistance()


@pytest.fixture
def weighted_hamming() -> WeightedHammingDistance:
    """Weighted Hamming distance for length-4 sequences."""
    return WeightedHammingDistance(weights=np.array([1.0, 2.0, 1.5, 0.5]))


@pytest.fixture
def symmetric_nll() -> SymmetricNLL:
    """Symmetric NLL decoding metric."""
    return SymmetricNLL(epsilon=0.1, alphabet_size=4)


# =============================================================================
# Decoding Rule Fixtures
# =============================================================================


@pytest.fixture
def unique_minimum_rule() -> UniqueMinimum:
    """Unique minimum decoding rule."""
    return UniqueMinimum()


@pytest.fixture
def margin_decoding_rule() -> MarginDecoding:
    """Margin decoding rule with k=1."""
    return MarginDecoding(k=1.0)


@pytest.fixture
def posterior_threshold_rule() -> PairwisePosteriorThreshold:
    """Pairwise posterior threshold rule."""
    return PairwisePosteriorThreshold(threshold=0.5)


# =============================================================================
# Encoder Fixtures
# =============================================================================


@pytest.fixture
def dna_encoder() -> DNAEncoder:
    """DNA encoder instance."""
    return DNAEncoder()


@pytest.fixture
def hex_encoder() -> HexEncoder:
    """Hex encoder instance."""
    return HexEncoder()


# =============================================================================
# Initial Selection Fixtures
# =============================================================================


@pytest.fixture
def simple_init_selection() -> np.ndarray:
    """Initial selection: one guide per gene (indices 0, 2)."""
    return np.array([0, 2])


@pytest.fixture
def medium_init_selection() -> np.ndarray:
    """Initial selection for 3 genes: indices 0, 3, 6."""
    return np.array([0, 3, 6])


# =============================================================================
# Helper Functions
# =============================================================================


def build_candidate_to_group(group_to_candidates: Dict[str, List[int]]) -> Dict[int, str]:
    """Build inverse mapping from candidate index to group name."""
    return {
        candidate: group
        for group, candidates in group_to_candidates.items()
        for candidate in candidates
    }


@pytest.fixture
def simple_candidate_to_group(simple_group_to_candidates) -> Dict[int, str]:
    """Inverse mapping for simple group structure."""
    return build_candidate_to_group(simple_group_to_candidates)


@pytest.fixture
def medium_candidate_to_group(medium_group_to_candidates) -> Dict[int, str]:
    """Inverse mapping for medium group structure."""
    return build_candidate_to_group(medium_group_to_candidates)


# =============================================================================
# Composite Fixtures for Full Setup
# =============================================================================


@pytest.fixture
def simple_swap_cache_setup(
    simple_pep_matrix,
    simple_scores,
    simple_group_to_candidates,
    simple_candidate_to_group,
):
    """Complete setup for DecodingSwapCache and ScoreSwapCache testing."""
    return {
        "pep_matrix": simple_pep_matrix,
        "scores": simple_scores,
        "group_to_candidates": simple_group_to_candidates,
        "candidate_to_group": simple_candidate_to_group,
        "init": np.array([0, 2]),  # One from each group
    }


@pytest.fixture
def simple_duet_setup(
    simple_pep_matrix,
    simple_scores,
    simple_group_to_candidates,
    simple_dna_library,
):
    """Complete setup for DUET testing."""
    return {
        "pep_matrix": simple_pep_matrix,
        "scores": simple_scores,
        "group_to_candidates": simple_group_to_candidates,
        "sequences": simple_dna_library,
        "init": np.array([0, 2]),
    }


@pytest.fixture
def multi_rep_group_to_candidates() -> Dict[str, List[int]]:
    """2 groups, 4 candidates each, for multi-representative testing."""
    return {
        "groupA": [0, 1, 2, 3],
        "groupB": [4, 5, 6, 7],
    }


@pytest.fixture
def multi_rep_swap_cache_setup(multi_rep_group_to_candidates):
    """Setup for multi-representative swap cache testing (2 reps per group).

    Groups: A=[0,1,2,3], B=[4,5,6,7]
    Initial selection: [0, 1, 4, 5] (2 reps from each group)
    Uses a random asymmetric PEP matrix to exercise X = M + M^T.
    """
    np.random.seed(2026)
    N = 8
    M = np.random.uniform(0.05, 0.3, size=(N, N))
    np.fill_diagonal(M, 1.0)

    scores = np.random.uniform(0.3, 0.9, size=N)

    return {
        "pep_matrix": M,
        "scores": scores,
        "group_to_candidates": multi_rep_group_to_candidates,
        "init": np.array([0, 1, 4, 5]),
    }


@pytest.fixture
def simple_evaluator_setup(
    simple_dna_library,
    symmetric_noise_medium,
    hamming_distance,
    unique_minimum_rule,
):
    """Complete setup for CodebookEvaluator testing."""
    return {
        "library": simple_dna_library,
        "noise_channel": symmetric_noise_medium,
        "decoding_metric": hamming_distance,
        "decoding_rule": unique_minimum_rule,
        "n_samples": 1000,
        "seed": 42,
    }
