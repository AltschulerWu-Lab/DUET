"""
Refactored codebook evaluation system using composition-based architecture
with Strategy Pattern for modular noise, decoding metric, and decoding logic.

Architecture:
- CodebookEvaluator: Engine class that orchestrates evaluation using injected strategies
- NoiseChannel: Strategy for generating noisy samples from transmitted sequences
- DecodingMetric: Strategy for computing decoding metric (cost) between sequences
- DecodingRule: Strategy for determining decoding success/failure
- SequenceEncoder: Adapter for converting domain inputs (DNA strings) to numpy arrays
- ConfigurableMixin: Mixin providing get_config() for fingerprinting/serialization
"""

import atexit
import inspect
import os
import shutil
import tempfile
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Dict, Optional, Type
from multiprocessing import Pool, cpu_count
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm
import logging

from duet.utils import _worker_init_blas

logger = logging.getLogger(__name__)

MAX_SAMPLES_UINT16 = 32767  # 2 * n_samples must fit in uint16

# Sentinel value replacing +inf in BLAS cost matrices to avoid 0 * inf = NaN.
# Finite NLL costs are bounded by ~709 (-log of smallest float64), so 1e30 is
# safely above any realistic cost while well below float64 max (~1.8e308).
_INF_SENTINEL = 1e30


def one_hot_encode(seqs: np.ndarray, q: int) -> np.ndarray:
    """Encode integer sequences as one-hot vectors.

    Args:
        seqs: Integer array of shape (M, L) with values in [0, q).
        q: Alphabet size (number of distinct symbols).

    Returns:
        One-hot encoded array of shape (M, L*q), dtype float32.
    """
    M, L = seqs.shape
    phi = np.zeros((M, L * q), dtype=np.float32)
    rows = np.repeat(np.arange(M), L)
    cols = (np.tile(np.arange(L), M) * q + seqs.ravel()).astype(int)
    phi[rows, cols] = 1.0
    return phi


@dataclass(frozen=True)
class _BLASPrecomputed:
    """Opaque container for precomputed BLAS codebook state."""
    psi_tx: np.ndarray   # (N, L*q_obs) cost-weighted feature matrix, float32
    q_obs: int           # observed alphabet size (needed for one-hot encoding)


# =============================================================================
# Type Aliases for Clarity (TODO: Remove these)
# =============================================================================

# Sequences in the transmitted alphabet (e.g., DNA bases 0-3, or dual-guide pairs 0-15)
TransmittedSequence = np.ndarray  # Shape (L,), values in [0, transmitted_alphabet_size)

# Sequences in the observed alphabet (may differ from transmitted for lossy channels)
ObservedSequence = np.ndarray  # Shape (L,), values in [0, observed_alphabet_size)


# =============================================================================
# Error Correction Metrics Dataclass
# =============================================================================

@dataclass
class ErrorCorrectionMetrics:
    """
    Container for error correction metrics at codebook and codeword levels.

    Categorizes decoding outcomes into three mutually exclusive categories:
    - No error: Sample received with no channel errors (observed == transmitted)
    - Corrected: Sample had channel errors but decoded correctly
    - Failed: Sample had channel errors and decoding failed

    Attributes:
        no_error_rate: Codebook-level % of samples with no channel errors
        corrected_rate: Codebook-level % of samples with errors that decoded correctly
        failed_rate: Codebook-level % of samples that failed to decode

        codeword_no_error: Per-codeword array of no-error rates, shape (n_codewords,)
        codeword_corrected: Per-codeword array of corrected rates, shape (n_codewords,)
        codeword_failed: Per-codeword array of failed rates, shape (n_codewords,)
    """

    # Codebook-level aggregate rates (percentages as fractions 0-1)
    no_error_rate: float
    corrected_rate: float
    failed_rate: float

    # Per-codeword arrays (rates as fractions 0-1)
    codeword_no_error: np.ndarray
    codeword_corrected: np.ndarray
    codeword_failed: np.ndarray

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "no_error_rate": self.no_error_rate,
            "corrected_rate": self.corrected_rate,
            "failed_rate": self.failed_rate,
            "codeword_no_error": self.codeword_no_error.tolist(),
            "codeword_corrected": self.codeword_corrected.tolist(),
            "codeword_failed": self.codeword_failed.tolist(),
        }


# =============================================================================
# ConfigurableMixin for Fingerprinting/Serialization
# =============================================================================

class ConfigurableMixin:
    """
    Mixin providing get_config() for strategy classes.

    Enables fingerprinting and serialization by introspecting __init__ parameters.
    Convention: attribute names must match __init__ parameter names.
    """

    def get_config(self) -> dict:
        """
        Return configuration dict for fingerprinting/serialization.

        Default implementation introspects __init__ parameters.
        Subclasses can override for custom behavior.

        Returns:
            Dict with 'type' key (class name) and all __init__ parameters
        """
        sig = inspect.signature(self.__init__)
        config = {'type': type(self).__name__}
        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            value = getattr(self, param_name)
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                value = value.tolist()
            config[param_name] = value
        return config


# =============================================================================
# Shared Helper Functions for Building Channel Matrices
# =============================================================================

def build_channel_matrix_from_symbol_epsilons(epsilons: np.ndarray) -> np.ndarray:
    """
    Build a single channel matrix from per-symbol error probabilities.

    Creates a q×q row-stochastic channel matrix where each symbol a has
    its own error probability epsilon[a]:
        T[a, a] = 1 - epsilon[a]  (correct transmission)
        T[a, b] = epsilon[a] / (q-1) for b ≠ a  (uniform errors to other symbols)

    This models a channel where different transmitted symbols may have
    different reliability, but errors are uniformly distributed among
    the remaining symbols.

    Args:
        epsilons: 1D array of shape (q,) where epsilons[a] is the error
                  probability for transmitted symbol a. All values must
                  be in [0, 1]. Alphabet size q is inferred from array length.

    Returns:
        Channel matrix of shape (q, q), row-stochastic.

    Raises:
        ValueError: If epsilons is not 1D, has length < 2, or contains
                   values outside [0, 1].

    Example:
        >>> epsilons = np.array([0.1, 0.2, 0.15, 0.1])  # Per-symbol error rates for ATCG
        >>> T = build_channel_matrix_from_symbol_epsilons(epsilons)
        >>> T[0, 0]  # P(observe A | transmit A) = 1 - 0.1 = 0.9
        0.9
        >>> T[1, 0]  # P(observe A | transmit T) = 0.2 / 3 ≈ 0.067
        0.0666...
    """
    epsilons = np.asarray(epsilons, dtype=np.float64)

    if epsilons.ndim != 1:
        raise ValueError(f"epsilons must be 1D array, got {epsilons.ndim}D")

    q = len(epsilons)
    if q < 2:
        raise ValueError(f"alphabet_size must be at least 2, got {q}")

    if np.any(epsilons < 0) or np.any(epsilons > 1):
        raise ValueError("All epsilon values must be in [0, 1]")

    # Build matrix: each row a gets off-diagonal value epsilons[a]/(q-1)
    off_diag = epsilons / (q - 1)  # (q,)
    channel_matrix = np.broadcast_to(off_diag[:, None], (q, q)).copy()
    np.fill_diagonal(channel_matrix, 1.0 - epsilons)

    return channel_matrix


def build_positional_channel_matrices_from_epsilons(epsilons: np.ndarray) -> np.ndarray:
    """
    Build position-specific channel matrices from per-position per-symbol error probabilities.

    Creates an (L, q, q) array of row-stochastic channel matrices where
    position i and symbol a have error probability epsilons[i, a]:
        T[i][a, a] = 1 - epsilons[i, a]  (correct transmission at position i)
        T[i][a, b] = epsilons[i, a] / (q-1) for b ≠ a  (uniform errors)

    This models a channel where noise characteristics vary both by position
    (e.g., sequencing cycle) and by transmitted symbol, but errors at each
    position are uniformly distributed among the remaining symbols.

    Args:
        epsilons: 2D array of shape (L, q) where epsilons[i, a] is the error
                  probability at position i for transmitted symbol a. All values
                  must be in [0, 1]. Sequence length L and alphabet size q are
                  inferred from array shape.

    Returns:
        Channel matrices of shape (L, q, q), each slice row-stochastic.

    Raises:
        ValueError: If epsilons is not 2D, has shape[1] < 2, or contains
                   values outside [0, 1].

    Example:
        >>> # 3 positions, 4 symbols (DNA)
        >>> epsilons = np.array([[0.1, 0.1, 0.1, 0.1],   # Position 0: uniform 10%
        ...                      [0.2, 0.15, 0.15, 0.2], # Position 1: A/G more error-prone
        ...                      [0.05, 0.05, 0.05, 0.05]])  # Position 2: low error
        >>> T = build_positional_channel_matrices_from_epsilons(epsilons)
        >>> T.shape
        (3, 4, 4)
    """
    epsilons = np.asarray(epsilons, dtype=np.float64)

    if epsilons.ndim != 2:
        raise ValueError(f"epsilons must be 2D array, got {epsilons.ndim}D")

    L, q = epsilons.shape
    if q < 2:
        raise ValueError(f"alphabet_size must be at least 2, got {q}")

    if np.any(epsilons < 0) or np.any(epsilons > 1):
        raise ValueError("All epsilon values must be in [0, 1]")

    # Build matrices: off_diag[i, a] = epsilons[i, a] / (q-1)
    off_diag = epsilons / (q - 1)  # (L, q)

    # Broadcast to (L, q, q): each position i, row a gets value off_diag[i, a]
    channel_matrices = np.broadcast_to(off_diag[:, :, None], (L, q, q)).copy()

    # Set diagonals: T[i, a, a] = 1 - epsilons[i, a] for all i, a
    diag_indices = np.arange(q)
    channel_matrices[:, diag_indices, diag_indices] = 1.0 - epsilons

    return channel_matrices


# =============================================================================
# Encoder ABC and Implementations (Adapter Pattern)
# =============================================================================

class SequenceEncoder(ABC):
    """
    Abstract base class for encoding/decoding sequences to/from numpy arrays.

    Encoders provide a bidirectional mapping between string representations
    and integer-encoded numpy arrays. Each encoder defines:
    - alphabet_size: Number of distinct symbols
    - alphabet: String of valid characters in order (index = encoded value)

    The design supports arbitrary encoding schemes (binary, DNA, hex, etc.)
    through a registry pattern. Use get_encoder() to obtain encoders by
    alphabet size, or register_encoder() to add custom encoders.

    Example:
        >>> encoder = get_encoder(alphabet_size=4)  # DNAEncoder
        >>> encoded = encoder.encode(["ATCG", "GCTA"])
        >>> encoded.shape
        (2, 4)
        >>> encoder.decode(encoded)
        ['ATCG', 'GCTA']
    """

    @property
    @abstractmethod
    def alphabet_size(self) -> int:
        """Size of the encoded alphabet (number of distinct symbols)."""
        pass

    @property
    @abstractmethod
    def alphabet(self) -> str:
        """
        Valid characters in this encoding scheme, ordered by index.

        The i-th character in the string encodes to integer i.
        For example, DNAEncoder.alphabet = "ATCG" means A→0, T→1, C→2, G→3.
        """
        pass

    @abstractmethod
    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Encode string sequences to integer array.

        Args:
            sequences: List of string sequences. All must have the same length
                      and contain only characters from self.alphabet.

        Returns:
            Array of shape (n_sequences, seq_length) with dtype int8,
            values in [0, alphabet_size).

        Raises:
            ValueError: If sequences have different lengths or contain
                       invalid characters.
        """
        pass

    @abstractmethod
    def decode(self, encoded: np.ndarray) -> List[str]:
        """
        Decode integer array back to string sequences.

        Args:
            encoded: Array of shape (n_sequences, seq_length) with integer
                    values in [0, alphabet_size).

        Returns:
            List of string sequences.
        """
        pass

    def validate(self, sequence: str) -> bool:
        """
        Check if sequence contains only valid characters.

        Args:
            sequence: String to validate.

        Returns:
            True if all characters are in self.alphabet.
        """
        return all(c in self.alphabet for c in sequence)


class DNAEncoder(SequenceEncoder):
    """
    Encoder for DNA sequences mapping A,T,C,G → 0,1,2,3.

    This is the default encoder for single-guide CRISPR screens and
    standard DNA sequence analysis.

    Mapping:
        A → 0, T → 1, C → 2, G → 3

    Example:
        >>> encoder = DNAEncoder()
        >>> encoder.encode(["ATCG"])
        array([[0, 1, 2, 3]], dtype=int8)
    """

    alphabet_size = 4
    alphabet = "ATCG"
    _char_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    _idx_to_char = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Convert list of DNA strings to (N, seq_length) int8 array.

        Args:
            sequences: List of DNA sequences (strings containing only A, T, C, G)

        Returns:
            numpy array of shape (len(sequences), seq_length) with dtype int8
        """
        if not sequences:
            raise ValueError("Cannot encode empty sequence list")

        seq_length = len(sequences[0])
        if not all(len(seq) == seq_length for seq in sequences):
            raise ValueError("All sequences must have the same length")

        result = np.empty((len(sequences), seq_length), dtype=np.int8)
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq):
                if base not in self._char_to_idx:
                    raise ValueError(f"Invalid base '{base}' at position {j} in sequence {i}")
                result[i, j] = self._char_to_idx[base]
        return result

    def decode(self, encoded: np.ndarray) -> List[str]:
        """
        Convert (N, seq_length) int array to list of DNA strings.

        Args:
            encoded: numpy array of encoded sequences

        Returns:
            List of DNA strings
        """
        return [''.join(self._idx_to_char[int(base)] for base in seq) for seq in encoded]


class HexEncoder(SequenceEncoder):
    """
    Encoder for 16-symbol hexadecimal sequences (0-9, A-F).

    Used for dual-guide CRISPR screens where each position encodes
    a pair of bases from two guides. The encoding is:
        symbol = 4 * base1_idx + base2_idx

    Mapping:
        0→0, 1→1, ..., 9→9, A→10, B→11, C→12, D→13, E→14, F→15

    Note: Input is case-insensitive (both 'a' and 'A' map to 10).

    Example:
        >>> encoder = HexEncoder()
        >>> encoder.encode(["0F", "A5"])
        array([[ 0, 15],
               [10,  5]], dtype=int8)
    """

    alphabet_size = 16
    alphabet = "0123456789ABCDEF"
    _char_to_idx = {c: i for i, c in enumerate(alphabet)}
    _idx_to_char = {i: c for i, c in enumerate(alphabet)}

    # Add lowercase mappings for case-insensitive encoding
    _char_to_idx.update({c.lower(): i for i, c in enumerate(alphabet) if c.isalpha()})

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Convert list of hex strings to (N, seq_length) int8 array.

        Args:
            sequences: List of hexadecimal sequences (case-insensitive)

        Returns:
            numpy array of shape (len(sequences), seq_length) with dtype int8
        """
        if not sequences:
            raise ValueError("Cannot encode empty sequence list")

        seq_length = len(sequences[0])
        if not all(len(seq) == seq_length for seq in sequences):
            raise ValueError("All sequences must have the same length")

        result = np.empty((len(sequences), seq_length), dtype=np.int8)
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                if char not in self._char_to_idx:
                    raise ValueError(f"Invalid character '{char}' at position {j} in sequence {i}")
                result[i, j] = self._char_to_idx[char]
        return result

    def decode(self, encoded: np.ndarray) -> List[str]:
        """
        Convert (N, seq_length) int array to list of hex strings.

        Args:
            encoded: numpy array of encoded sequences

        Returns:
            List of hexadecimal strings (uppercase)
        """
        return [''.join(self._idx_to_char[int(val)] for val in row) for row in encoded]


class BinaryEncoder(SequenceEncoder):
    """
    Encoder for binary sequences (0, 1).

    Useful for simple two-state systems or testing.

    Mapping:
        '0' → 0, '1' → 1

    Example:
        >>> encoder = BinaryEncoder()
        >>> encoder.encode(["0110", "1001"])
        array([[0, 1, 1, 0],
               [1, 0, 0, 1]], dtype=int8)
    """

    alphabet_size = 2
    alphabet = "01"

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Convert list of binary strings to (N, seq_length) int8 array.

        Args:
            sequences: List of binary sequences (strings of '0' and '1')

        Returns:
            numpy array of shape (len(sequences), seq_length) with dtype int8
        """
        if not sequences:
            raise ValueError("Cannot encode empty sequence list")

        seq_length = len(sequences[0])
        if not all(len(seq) == seq_length for seq in sequences):
            raise ValueError("All sequences must have the same length")

        result = np.empty((len(sequences), seq_length), dtype=np.int8)
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                if char not in ('0', '1'):
                    raise ValueError(f"Invalid character '{char}' at position {j} in sequence {i}")
                result[i, j] = int(char)
        return result

    def decode(self, encoded: np.ndarray) -> List[str]:
        """
        Convert (N, seq_length) int array to list of binary strings.

        Args:
            encoded: numpy array of encoded sequences

        Returns:
            List of binary strings
        """
        return [''.join(str(int(val)) for val in row) for row in encoded]


# =============================================================================
# Encoder Registry (Extensible Factory Pattern)
# =============================================================================

_ENCODER_REGISTRY: Dict[int, Type[SequenceEncoder]] = {
    2: BinaryEncoder,
    4: DNAEncoder,
    16: HexEncoder,
}


def register_encoder(alphabet_size: int, encoder_cls: Type[SequenceEncoder]) -> None:
    """
    Register a custom encoder for a given alphabet size.

    This allows extending the framework with new encoding schemes without
    modifying the core codebase.

    Args:
        alphabet_size: The alphabet size this encoder handles.
        encoder_cls: The encoder class (must be a subclass of SequenceEncoder).

    Raises:
        TypeError: If encoder_cls is not a subclass of SequenceEncoder.

    Example:
        >>> class TernaryEncoder(SequenceEncoder):
        ...     alphabet_size = 3
        ...     alphabet = "012"
        ...     # ... implement encode/decode ...
        >>> register_encoder(3, TernaryEncoder)
    """
    if not issubclass(encoder_cls, SequenceEncoder):
        raise TypeError(f"encoder_cls must be a subclass of SequenceEncoder, got {encoder_cls}")
    _ENCODER_REGISTRY[alphabet_size] = encoder_cls


def get_encoder(alphabet_size: int = 4) -> SequenceEncoder:
    """
    Get encoder instance for given alphabet size.

    Default is DNA encoding (alphabet_size=4).

    Args:
        alphabet_size: Size of the alphabet. Supported values:
            - 2: BinaryEncoder ('0', '1')
            - 4: DNAEncoder ('A', 'T', 'C', 'G') [default]
            - 16: HexEncoder ('0'-'9', 'A'-'F')

    Returns:
        SequenceEncoder instance.

    Raises:
        ValueError: If no encoder is registered for the given alphabet size.

    Example:
        >>> encoder = get_encoder(4)  # DNAEncoder
        >>> encoder = get_encoder(16)  # HexEncoder
        >>> encoder = get_encoder()  # DNAEncoder (default)
    """
    if alphabet_size not in _ENCODER_REGISTRY:
        raise ValueError(
            f"No encoder registered for alphabet_size={alphabet_size}. "
            f"Available: {sorted(_ENCODER_REGISTRY.keys())}. "
            f"Use register_encoder() to add custom encoders."
        )
    return _ENCODER_REGISTRY[alphabet_size]()


# =============================================================================
# Strategy Abstract Base Classes
# =============================================================================

class NoiseChannel(ConfigurableMixin, ABC):
    """
    Abstract base class for noise generation models.

    Implementations must support vectorized numpy operations for efficiency.
    """

    @abstractmethod
    def generate(self, sequence: TransmittedSequence, n_samples: int, rng: np.random.Generator) -> ObservedSequence:
        """
        Generate noisy samples from a transmitted sequence.

        Args:
            sequence: Original transmitted sequence of shape (seq_length,)
            n_samples: Number of noisy samples to generate
            rng: numpy random Generator for reproducibility

        Returns:
            Array of observed (noisy) sequences with shape (n_samples, seq_length).
            Values are in the observed alphabet, which may differ from the
            transmitted alphabet for lossy channels.
        """
        pass


class DecodingMetric(ConfigurableMixin, ABC):
    """
    Abstract base class for decoding metric (cost) computation.

    A decoding metric function d(x, y) measures how different two sequences are.
    Unlike a distance metric, decoding metric functions do not require:
    - d(x, x) = 0 (identity of indiscernibles)
    - d(x, y) = d(y, x) (symmetry)
    - d(x, z) ≤ d(x, y) + d(y, z) (triangle inequality)

    This generality allows modeling channels where transmitted and observed
    alphabets differ, and where the "cost" of observing y given x was
    transmitted is simply -log P(y | x).

    Implementations must support vectorized numpy operations.
    """

    @abstractmethod
    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Compute decoding metric matrix between observed and transmitted sequences.

        Args:
            observed: Array of observed sequences with shape (M, seq_length).
                     Values are in the observed alphabet.
            transmitted: Array of transmitted sequences with shape (N, seq_length).
                        Values are in the transmitted alphabet.

        Returns:
            Decoding-metric matrix of shape (M, N) where entry [i,j] is the
            cost/decoding metric of observed[i] given transmitted[j].
        """
        pass

    @abstractmethod
    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Compute element-wise decoding metrics between observed and transmitted sequences.

        Args:
            observed: Array of observed sequences with shape (M, seq_length)
            transmitted: Array of corresponding transmitted sequences with shape (M, seq_length)

        Returns:
            Array of decoding metrics with shape (M,)
        """
        pass

    def precompute_transmitted(self, codebook: np.ndarray) -> Any:
        """Precompute opaque codebook state for BLAS-accelerated cost computation.

        Returns precomputed state, or None if BLAS path is not supported.
        """
        return None

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: Any
    ) -> np.ndarray:
        """Compute cost matrix using precomputed codebook state.

        Returns shape (K, N). Each subclass handles its own encoding internally.
        """
        raise NotImplementedError


class DecodingRule(ConfigurableMixin, ABC):
    """
    Abstract base class for decoding rules.

    Determines which codewords "compete" with the transmitted codeword
    for each received sequence.
    """

    @abstractmethod
    def identify_competitors(
        self,
        cost_matrix: np.ndarray,
        transmitted_costs: np.ndarray
    ) -> csr_matrix:
        """
        Identify which codewords compete with the transmitted codeword.

        A competitor is a codeword that could cause a decoding error or ambiguity.

        Args:
            cost_matrix: Costs from observed sequences to all codewords,
                        shape (M, N) where M is number of samples, N is codebook size
            transmitted_costs: Costs from observed sequences to their
                              transmitted codewords, shape (M,)

        Returns:
            Sparse boolean matrix of shape (M, N) where True indicates
            that codeword j competes with the transmitted codeword for sample i
        """
        pass

    @abstractmethod
    def competitor_margin(self) -> float:
        """Return the margin value for competitor identification.

        All decoding rules reduce to: competitors = (cost <= tx_cost + margin).
        This method exposes that margin for use by alternative computation
        backends (e.g., GPU) without requiring isinstance checks.

        Returns:
            The margin float. Zero for UniqueMinimum, k for MarginDecoding,
            logit(threshold) for posterior threshold rules.
        """
        pass


# =============================================================================
# Noise Channel Implementations
# =============================================================================

class SymmetricEpsilon(NoiseChannel):
    """
    Symmetric error probability noise channel (q-ary symmetric channel).

    Each position independently has probability epsilon of being mutated
    to one of the (q-1) other symbols (each with probability epsilon/(q-1)).
    """

    def __init__(self, epsilon: float, alphabet_size: int = 4):
        """
        Initialize with uniform error probability.

        Args:
            epsilon: Probability that any position is incorrect, must be in [0, 1]
            alphabet_size: Size of the alphabet (default: 4 for DNA/quaternary)
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if alphabet_size < 2:
            raise ValueError(f"alphabet_size must be at least 2, got {alphabet_size}")
        self.epsilon = epsilon
        self.alphabet_size = alphabet_size

    def generate(self, sequence: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        seq_length = len(sequence)

        # Generate mutation mask: which positions get mutated
        mutation_mask = rng.random((n_samples, seq_length)) < self.epsilon

        # Generate random offsets in {1, 2, ..., q-1} for mutations
        offsets = rng.integers(1, self.alphabet_size, size=(n_samples, seq_length))

        # Replicate base sequence for all samples
        base_tiled = np.tile(sequence, (n_samples, 1))

        # Apply mutations using modular arithmetic
        mutated = (base_tiled + offsets) % self.alphabet_size

        # Select mutated or original based on mask
        return np.where(mutation_mask, mutated, base_tiled).astype(sequence.dtype)


class PositionVaryingEpsilon(NoiseChannel):
    """
    Position-varying error probability noise channel.

    Each position has its own error probability, allowing modeling of
    position-dependent noise (e.g., sequencing errors that vary by cycle).
    """

    def __init__(self, epsilons: np.ndarray, alphabet_size: int = 4):
        """
        Initialize with per-position error probabilities.

        Args:
            epsilons: 1D array of shape (seq_length,) with error probability
                     for each position, all values must be in [0, 1]
            alphabet_size: Size of the alphabet (default: 4 for DNA/quaternary)
        """
        epsilons = np.asarray(epsilons)
        if epsilons.ndim != 1:
            raise ValueError(f"epsilons must be 1D array, got shape {epsilons.shape}")
        if not (np.all(epsilons >= 0) and np.all(epsilons <= 1)):
            raise ValueError("All epsilon values must be in [0, 1]")
        if alphabet_size < 2:
            raise ValueError(f"alphabet_size must be at least 2, got {alphabet_size}")
        self.epsilons = epsilons.astype(np.float64)
        self.alphabet_size = alphabet_size

    def generate(self, sequence: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        seq_length = len(sequence)

        if len(self.epsilons) != seq_length:
            raise ValueError(
                f"epsilons length {len(self.epsilons)} must match sequence length {seq_length}"
            )

        # Generate mutation mask with position-specific probabilities
        # Broadcasting: (n_samples, seq_length) < (seq_length,) -> (n_samples, seq_length)
        mutation_mask = rng.random((n_samples, seq_length)) < self.epsilons

        # Generate random offsets in {1, 2, ..., q-1}
        offsets = rng.integers(1, self.alphabet_size, size=(n_samples, seq_length))

        # Replicate base sequence
        base_tiled = np.tile(sequence, (n_samples, 1))

        # Apply mutations
        mutated = (base_tiled + offsets) % self.alphabet_size

        return np.where(mutation_mask, mutated, base_tiled).astype(sequence.dtype)


class AsymmetricChannel(NoiseChannel):
    """
    Asymmetric noise channel.

    Each position uses the same channel matrix T where T[a, b] is the
    probability of observing symbol b given that symbol a was transmitted.

    Supports non-square channel matrices where the transmitted and observed
    alphabets have different sizes. This is useful for modeling information-lossy
    channels such as dual-guide optical pooled screens.

    For square matrices, this is a generalization of the symmetric channel that
    allows for asymmetric substitution probabilities (e.g., A→C may differ from C→A).
    """

    def __init__(self, channel_matrix: np.ndarray):
        """
        Initialize with channel matrix.

        Args:
            channel_matrix: Row-stochastic matrix of shape (tx_size, obs_size) where
                           T[a, b] = P(observe b | transmit a).
                           Rows must sum to 1, all entries must be in [0, 1].
                           Transmitted alphabet size tx_size and observed alphabet
                           size obs_size are inferred from the matrix shape.
        """
        channel_matrix = np.asarray(channel_matrix, dtype=np.float64)

        if channel_matrix.ndim != 2:
            raise ValueError(f"channel_matrix must be 2D, got {channel_matrix.ndim}D")

        transmitted_alphabet_size = channel_matrix.shape[0]
        observed_alphabet_size = channel_matrix.shape[1]

        if transmitted_alphabet_size < 2:
            raise ValueError(f"transmitted_alphabet_size must be at least 2, got {transmitted_alphabet_size}")
        if observed_alphabet_size < 1:
            raise ValueError(f"observed_alphabet_size must be at least 1, got {observed_alphabet_size}")

        if np.any(channel_matrix < 0) or np.any(channel_matrix > 1):
            raise ValueError("All channel matrix entries must be in [0, 1]")

        row_sums = channel_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Channel matrix rows must sum to 1, got row sums: {row_sums}")

        self.channel_matrix = channel_matrix
        self.transmitted_alphabet_size = transmitted_alphabet_size
        self.observed_alphabet_size = observed_alphabet_size

    def generate(self, sequence: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        seq_length = len(sequence)

        # Get probability rows for each position: T[sequence[i], :] for all i
        # Shape: (seq_length, observed_alphabet_size)
        prob_rows = self.channel_matrix[sequence, :]

        # Compute cumulative probabilities for inverse transform sampling
        # Shape: (seq_length, observed_alphabet_size)
        cumsum = np.cumsum(prob_rows, axis=1)

        # Generate uniform random values
        # Shape: (n_samples, seq_length)
        u = rng.random((n_samples, seq_length))

        # Vectorized inverse transform sampling:
        # For each (sample, position), find smallest symbol index where u < cumsum
        # Expand dims: u -> (n_samples, seq_length, 1), cumsum -> (1, seq_length, observed_alphabet_size)
        # Compare: (n_samples, seq_length, observed_alphabet_size) boolean array
        # argmax finds first True along axis=2
        result = (u[:, :, None] < cumsum[None, :, :]).argmax(axis=2)

        return result.astype(np.int8)

    @classmethod
    def from_symbol_epsilons(cls, epsilons: np.ndarray) -> 'AsymmetricChannel':
        """
        Factory method to create noise channel from per-symbol error probabilities.

        Creates a square channel matrix where each symbol a has error probability epsilons[a]:
            T[a, a] = 1 - epsilons[a] (correct transmission)
            T[a, b] = epsilons[a] / (q-1) for a ≠ b (uniform error to other symbols)

        This models a channel where different symbols have different reliability
        (e.g., some bases are more prone to sequencing errors than others).

        Args:
            epsilons: 1D array of shape (q,) with error probability for each symbol.
                     Alphabet size q is inferred from array length.

        Returns:
            AsymmetricChannel instance.

        Example:
            >>> # Different error rates for A, T, C, G
            >>> epsilons = np.array([0.08, 0.12, 0.10, 0.09])
            >>> noise = AsymmetricChannel.from_symbol_epsilons(epsilons)
        """
        channel_matrix = build_channel_matrix_from_symbol_epsilons(epsilons)
        return cls(channel_matrix)


class PositionVaryingAsymmetricChannel(NoiseChannel):
    """
    Position-varying, asymmetric noise channel.

    Each position has its own channel matrix, allowing for position-dependent
    noise characteristics. Supports non-square channel matrices where the
    transmitted and observed alphabets have different sizes.

    This is the most general discrete memoryless channel model for
    sequences with position-dependent noise.
    """

    def __init__(self, channel_matrices: np.ndarray):
        """
        Initialize with per-position channel matrices.

        Args:
            channel_matrices: Array of shape (seq_length, tx_size, obs_size) where
                             channel_matrices[i, a, b] = P(observe b | transmit a) at position i.
                             Each (tx_size, obs_size) slice must be row-stochastic.
                             Alphabet sizes are inferred from the matrix shape.
        """
        channel_matrices = np.asarray(channel_matrices, dtype=np.float64)

        if channel_matrices.ndim != 3:
            raise ValueError(
                f"channel_matrices must be 3D, got {channel_matrices.ndim}D"
            )

        transmitted_alphabet_size = channel_matrices.shape[1]
        observed_alphabet_size = channel_matrices.shape[2]

        if transmitted_alphabet_size < 2:
            raise ValueError(f"transmitted_alphabet_size must be at least 2, got {transmitted_alphabet_size}")
        if observed_alphabet_size < 1:
            raise ValueError(f"observed_alphabet_size must be at least 1, got {observed_alphabet_size}")

        if np.any(channel_matrices < 0) or np.any(channel_matrices > 1):
            raise ValueError("All channel matrix entries must be in [0, 1]")

        # Check row-stochasticity for each position
        row_sums = channel_matrices.sum(axis=2)  # Shape: (seq_length, transmitted_alphabet_size)
        if not np.allclose(row_sums, 1.0):
            bad_positions = np.where(~np.isclose(row_sums, 1.0).all(axis=1))[0]
            raise ValueError(
                f"Channel matrix rows must sum to 1. "
                f"Violations at positions: {bad_positions.tolist()}"
            )

        self.channel_matrices = channel_matrices
        self.transmitted_alphabet_size = transmitted_alphabet_size
        self.observed_alphabet_size = observed_alphabet_size

    @property
    def seq_length(self) -> int:
        """Expected sequence length based on number of channel matrices."""
        return self.channel_matrices.shape[0]

    def generate(self, sequence: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        seq_length = len(sequence)

        if seq_length != self.seq_length:
            raise ValueError(
                f"Sequence length {seq_length} must match number of "
                f"channel matrices {self.seq_length}"
            )

        # Get probability row for each position using advanced indexing:
        # channel_matrices[i, sequence[i], :] for all i
        # Shape: (seq_length, observed_alphabet_size)
        pos_indices = np.arange(seq_length)
        prob_rows = self.channel_matrices[pos_indices, sequence, :]

        # Compute cumulative probabilities for inverse transform sampling
        # Shape: (seq_length, observed_alphabet_size)
        cumsum = np.cumsum(prob_rows, axis=1)

        # Generate uniform random values
        # Shape: (n_samples, seq_length)
        u = rng.random((n_samples, seq_length))

        # Vectorized inverse transform sampling:
        # Expand dims: u -> (n_samples, seq_length, 1), cumsum -> (1, seq_length, observed_alphabet_size)
        # argmax finds first True along axis=2
        result = (u[:, :, None] < cumsum[None, :, :]).argmax(axis=2)

        return result.astype(np.int8)

    @classmethod
    def from_positional_symbol_epsilons(cls, epsilons: np.ndarray) -> 'PositionVaryingAsymmetricChannel':
        """
        Factory method to create noise channel from per-position per-symbol error probabilities.

        Creates position-specific square channel matrices where position i and symbol a
        have error probability epsilons[i, a]:
            T[i][a, a] = 1 - epsilons[i, a] (correct transmission at position i)
            T[i][a, b] = epsilons[i, a] / (q-1) for a ≠ b (uniform errors)

        This models a channel where noise characteristics vary both by position
        (e.g., sequencing cycle) and by transmitted symbol.

        Args:
            epsilons: 2D array of shape (L, q) where epsilons[i, a] is the error
                     probability at position i for transmitted symbol a.
                     Sequence length L and alphabet size q are inferred from shape.

        Returns:
            PositionVaryingAsymmetricChannel instance.

        Example:
            >>> # 5 positions, 4 symbols (DNA)
            >>> epsilons = np.array([
            ...     [0.05, 0.05, 0.05, 0.05],  # Position 0: uniform low error
            ...     [0.10, 0.08, 0.08, 0.10],  # Position 1: A/G slightly higher
            ...     [0.15, 0.15, 0.15, 0.15],  # Position 2: uniform medium error
            ...     [0.12, 0.10, 0.10, 0.12],  # Position 3
            ...     [0.20, 0.20, 0.20, 0.20],  # Position 4: higher error (end of read)
            ... ])
            >>> noise = PositionVaryingAsymmetricChannel.from_positional_symbol_epsilons(epsilons)
        """
        channel_matrices = build_positional_channel_matrices_from_epsilons(epsilons)
        return cls(channel_matrices)


# =============================================================================
# Decoding Metric Implementations
# =============================================================================

class HammingDistance(DecodingMetric):
    """
    Standard Hamming distance.

    Counts the number of positions where two sequences differ.
    Requires observed and transmitted alphabets to be the same.
    """

    def __init__(self, alphabet_size=None):
        """Initialize Hamming distance.

        Args:
            alphabet_size: If set, use this alphabet size for BLAS precomputation
                instead of inferring from the codebook. Needed when the codebook
                doesn't span the full alphabet.
        """
        self.alphabet_size = alphabet_size

    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of all pairwise Hamming distances.

        Uses broadcasting: (M, 1, L) != (1, N, L) -> (M, N, L) -> sum -> (M, N)
        """
        # observed: (M, L), transmitted: (N, L)
        # Expand dims for broadcasting
        return np.sum(observed[:, None, :] != transmitted[None, :, :], axis=2)

    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """Element-wise Hamming distance computation."""
        return np.sum(observed != transmitted, axis=1)

    def precompute_transmitted(self, codebook: np.ndarray) -> _BLASPrecomputed:
        q = self.alphabet_size if self.alphabet_size is not None else int(codebook.max()) + 1
        phi_tx = one_hot_encode(codebook, q)
        psi_tx = np.float32(1.0) - phi_tx
        assert psi_tx.dtype == np.float32, f"psi_tx should be float32, got {psi_tx.dtype}"
        return _BLASPrecomputed(psi_tx=psi_tx, q_obs=q)

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: _BLASPrecomputed
    ) -> np.ndarray:
        phi_obs = one_hot_encode(observed, precomputed.q_obs)
        return phi_obs @ precomputed.psi_tx.T


class WeightedHammingDistance(DecodingMetric):
    """
    Weighted Hamming distance.

    Computes d = Σ w_i · 𝟙(x_i ≠ y_i) where weights are static per column.
    Useful when some positions are more important for discrimination.
    Requires observed and transmitted alphabets to be the same.
    """

    def __init__(self, weights: np.ndarray, alphabet_size=None):
        """
        Initialize with per-position weights.

        Args:
            weights: 1D array of shape (seq_length,) with non-negative weights
            alphabet_size: If set, use this alphabet size for BLAS precomputation
                instead of inferring from the codebook. Needed when the codebook
                doesn't span the full alphabet.
        """
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1D array, got shape {weights.shape}")
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative")
        self.weights = weights.astype(np.float64)
        self.alphabet_size = alphabet_size

    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Vectorized weighted Hamming distance computation.

        Broadcasting: (M, N, L) * (L,) -> (M, N, L) -> sum -> (M, N)
        """
        # Mismatch indicator: (M, N, L)
        mismatches = (observed[:, None, :] != transmitted[None, :, :]).astype(np.float64)
        # Weighted sum over positions: (M, N)
        return np.sum(mismatches * self.weights, axis=2)

    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """Element-wise weighted Hamming distance."""
        mismatches = (observed != transmitted).astype(np.float64)
        return np.sum(mismatches * self.weights, axis=1)

    def precompute_transmitted(self, codebook: np.ndarray) -> _BLASPrecomputed:
        q = self.alphabet_size if self.alphabet_size is not None else int(codebook.max()) + 1
        phi_tx = one_hot_encode(codebook, q)
        weights_f32 = self.weights.astype(np.float32)
        psi_tx = (np.float32(1.0) - phi_tx) * np.repeat(weights_f32, q)
        assert psi_tx.dtype == np.float32, f"psi_tx should be float32, got {psi_tx.dtype}"
        return _BLASPrecomputed(psi_tx=psi_tx, q_obs=q)

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: _BLASPrecomputed
    ) -> np.ndarray:
        phi_obs = one_hot_encode(observed, precomputed.q_obs)
        return phi_obs @ precomputed.psi_tx.T


class SymmetricNLL(DecodingMetric):
    """
    Symmetric negative log-likelihood decoding metric for q-ary symmetric channels.

    Computes d(obs, tx) = w · Σᵢ 𝟙(obsᵢ ≠ txᵢ) where the weight is derived from
    a uniform error probability for a q-ary symmetric channel:

        w = log((q-1)(1 - ε) / ε)

    This is the negative log-likelihood ratio for a mismatch at any position.
    Minimizing this decoding metric is equivalent to maximum likelihood decoding.

    This is a simplified version of PositionVaryingNLL where all positions share the
    same error probability. When ε is the same at all positions, the NLL
    decoding metric is proportional to Hamming distance with scale factor w.

    Weight interpretation:
        - Reliable channel (ε ≪ 0.5): Large positive weight (errors are "expensive")
        - Unreliable channel (ε ≈ 0.5): Weight ≈ log(q-1) (channel still informative)
        - ε = 0: Weight = +∞ (mismatches impossible; handled cleanly via np.where)
        - ε > 0.5: Weight may become small or negative (warning issued)

    Note: Unlike distance metrics, this decoding metric is not guaranteed to
    satisfy the metric axioms: it is a likelihood-based cost, not a true metric.
    In particular, the weight may be negative (when ε > 0.5), in which case it
    can violate non-negativity and the triangle inequality.

    Requires observed and transmitted alphabets to be the same.
    """

    def __init__(self, epsilon: float, alphabet_size: int = 4):
        """
        Initialize with uniform error probability.

        Args:
            epsilon: Error probability, same for all positions. Should be in
                    [0, 0.5] for well-defined non-negative weights. Zero is
                    handled correctly (infinite weight, but no numerical issues
                    since mismatches at zero-error channels should never occur).
            alphabet_size: Size of the alphabet (default: 4 for DNA/quaternary)
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if alphabet_size < 2:
            raise ValueError(f"alphabet_size must be at least 2, got {alphabet_size}")

        # Warn about edge cases that affect weight interpretation
        if epsilon > 0.5:
            logger.warning(
                f"epsilon={epsilon} > 0.5; weight may be small or negative"
            )

        self.epsilon = epsilon
        self.alphabet_size = alphabet_size

        # Precompute weight: w = log((q-1) * (1 - ε) / ε)
        # When ε = 0: weight = +inf (any mismatch is infinitely costly)
        # When ε = 1: weight = -inf (degenerate case)
        if epsilon == 0.0:
            self.weight = np.inf
        else:
            self.weight = np.log((alphabet_size - 1) * (1.0 - epsilon) / epsilon)

    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Vectorized NLL decoding metric computation.

        Uses np.where to avoid 0 * inf = nan when epsilon = 0.
        Broadcasting: (M, 1, L) != (1, N, L) -> (M, N, L) -> sum -> (M, N)
        """
        # Count mismatches: (M, N)
        mismatch_counts = np.sum(observed[:, None, :] != transmitted[None, :, :], axis=2)
        # Multiply by weight, handling inf * 0 case
        return np.where(mismatch_counts > 0, mismatch_counts * self.weight, 0.0)

    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """Element-wise NLL decoding metric."""
        # Count mismatches: (M,)
        mismatch_counts = np.sum(observed != transmitted, axis=1)
        # Multiply by weight, handling inf * 0 case
        return np.where(mismatch_counts > 0, mismatch_counts * self.weight, 0.0)

    def precompute_transmitted(self, codebook: np.ndarray) -> _BLASPrecomputed:
        q = self.alphabet_size
        phi_tx = one_hot_encode(codebook, q)
        w_safe = np.float32(min(self.weight, _INF_SENTINEL))
        psi_tx = (np.float32(1.0) - phi_tx) * w_safe
        assert psi_tx.dtype == np.float32, f"psi_tx should be float32, got {psi_tx.dtype}"
        return _BLASPrecomputed(psi_tx=psi_tx, q_obs=q)

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: _BLASPrecomputed
    ) -> np.ndarray:
        phi_obs = one_hot_encode(observed, precomputed.q_obs)
        return phi_obs @ precomputed.psi_tx.T


class PositionVaryingNLL(DecodingMetric):
    """
    Position-varying negative log-likelihood decoding metric for q-ary symmetric channels.

    Computes d(obs, tx) = Σᵢ wᵢ · 𝟙(obsᵢ ≠ txᵢ) where weights are derived from
    position-specific error probabilities for a q-ary symmetric channel:

        wᵢ = log((q-1)(1 - εᵢ) / εᵢ)

    This is the negative log-likelihood ratio for a mismatch at position i.
    Minimizing this decoding metric is equivalent to maximum likelihood decoding.

    Weight interpretation:
        - Reliable positions (εᵢ ≪ 0.5): Large positive weight (errors are "expensive")
        - Unreliable positions (εᵢ ≈ 0.5): Weight ≈ log(q-1) (position still contributes)
        - εᵢ = 0: Weight = +∞ (mismatches impossible; handled cleanly via np.where)
        - εᵢ > 0.5: Weight may become small or negative (warning issued)

    Note: Unlike distance metrics, this decoding metric is not guaranteed to
    satisfy the metric axioms: it is a likelihood-based cost, not a true metric.
    In particular, weights may be negative (when εᵢ > 0.5), in which case it
    can violate non-negativity and the triangle inequality.

    Requires observed and transmitted alphabets to be the same.
    """

    def __init__(self, epsilons: np.ndarray, alphabet_size: int = 4):
        """
        Initialize with per-position error probabilities.

        Args:
            epsilons: 1D array of shape (seq_length,) with error probability
                     for each position. Values should be in [0, 0.5] for
                     well-defined non-negative weights. Zero values are handled
                     correctly (infinite weight, but no numerical issues since
                     mismatches at zero-error positions should never occur).
            alphabet_size: Size of the alphabet (default: 4 for DNA/quaternary)
        """
        epsilons = np.asarray(epsilons)
        if epsilons.ndim != 1:
            raise ValueError(f"epsilons must be 1D array, got shape {epsilons.shape}")
        if np.any(epsilons < 0) or np.any(epsilons > 1):
            raise ValueError("All epsilon values must be in [0, 1]")
        if alphabet_size < 2:
            raise ValueError(f"alphabet_size must be at least 2, got {alphabet_size}")

        # Warn about edge cases that affect weight interpretation
        if np.any(epsilons > 0.5):
            logger.warning(
                "epsilons contains values > 0.5; weights may be small or negative at these positions"
            )

        self.epsilons = epsilons.astype(np.float64)
        self.alphabet_size = alphabet_size

        # Precompute weights: w_i = log((q-1) * (1 - ε_i) / ε_i)
        # When ε_i = 0: weight = +inf (correctly handled via np.where in compute methods)
        # When ε_i = 1: weight = -inf (degenerate case)
        with np.errstate(divide='ignore'):  # Suppress warning for log(inf) when ε=0
            self.weights = np.log((alphabet_size - 1) * (1.0 - self.epsilons) / self.epsilons)

    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Vectorized NLL decoding metric computation.

        Uses np.where to avoid 0 * inf = nan when epsilon = 0 at some positions.
        Broadcasting: where((M, N, L), (L,), scalar) -> (M, N, L) -> sum -> (M, N)
        """
        mismatches = observed[:, None, :] != transmitted[None, :, :]
        # np.where correctly returns 0.0 when condition is False, even if weights contain inf
        costs = np.where(mismatches, self.weights, 0.0)
        return np.sum(costs, axis=2)

    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """Element-wise NLL decoding metric."""
        mismatches = observed != transmitted
        costs = np.where(mismatches, self.weights, 0.0)
        return np.sum(costs, axis=1)

    def precompute_transmitted(self, codebook: np.ndarray) -> _BLASPrecomputed:
        q = self.alphabet_size
        phi_tx = one_hot_encode(codebook, q)
        weights_safe = np.minimum(self.weights, _INF_SENTINEL).astype(np.float32)
        psi_tx = (np.float32(1.0) - phi_tx) * np.repeat(weights_safe, q)
        assert psi_tx.dtype == np.float32, f"psi_tx should be float32, got {psi_tx.dtype}"
        return _BLASPrecomputed(psi_tx=psi_tx, q_obs=q)

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: _BLASPrecomputed
    ) -> np.ndarray:
        phi_obs = one_hot_encode(observed, precomputed.q_obs)
        return phi_obs @ precomputed.psi_tx.T


class AsymmetricNLL(DecodingMetric):
    """
    Negative log-likelihood decoding metric for a channel matrix.

    For a channel with matrix T where T[a, b] = P(observe b | transmit a),
    this computes the negative log-likelihood:

        d(obs, tx) = Σᵢ -log(T[txᵢ, obsᵢ])

    This is exactly the negative log-probability of the observation given
    the transmitted sequence. Minimizing this decoding metric is equivalent
    to maximum likelihood decoding.

    Supports non-square channel matrices where the transmitted and observed
    alphabets have different sizes. This is useful for modeling information-lossy
    channels such as dual-guide optical pooled screens.

    Note: Unlike distance metrics, this decoding metric:
    - May have d(x, x) > 0 (self-decoding metric is the entropy of the channel)
    - Is not symmetric: d(x, y) ≠ d(y, x) in general
    - Does not satisfy the triangle inequality
    """

    def __init__(self, channel_matrix: np.ndarray):
        """
        Initialize with channel matrix.

        Args:
            channel_matrix: Row-stochastic matrix of shape (tx_size, obs_size) where
                           T[a, b] = P(observe b | transmit a).
                           Rows must sum to 1, all entries must be in [0, 1].
        """
        channel_matrix = np.asarray(channel_matrix, dtype=np.float64)

        if channel_matrix.ndim != 2:
            raise ValueError(f"channel_matrix must be 2D, got {channel_matrix.ndim}D")

        transmitted_alphabet_size = channel_matrix.shape[0]
        observed_alphabet_size = channel_matrix.shape[1]

        if transmitted_alphabet_size < 2:
            raise ValueError(f"transmitted_alphabet_size must be at least 2, got {transmitted_alphabet_size}")
        if observed_alphabet_size < 1:
            raise ValueError(f"observed_alphabet_size must be at least 1, got {observed_alphabet_size}")

        if np.any(channel_matrix < 0) or np.any(channel_matrix > 1):
            raise ValueError("All channel matrix entries must be in [0, 1]")

        row_sums = channel_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Channel matrix rows must sum to 1, got row sums: {row_sums}")

        # Warn about zero entries
        if np.any(channel_matrix == 0):
            logger.warning(
                "Channel matrix contains zero entries; cost will be +inf for these transitions"
            )

        self.channel_matrix = channel_matrix
        self.transmitted_alphabet_size = transmitted_alphabet_size
        self.observed_alphabet_size = observed_alphabet_size

        # Precompute cost matrix: C[a, b] = -log(T[a, b])
        with np.errstate(divide='ignore'):
            self.cost_matrix = -np.log(channel_matrix)

    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of NLL decoding metrics.

        For each pair (observed[m], transmitted[n]):
        d = Σᵢ C[transmitted[n,i], observed[m,i]]

        Args:
            observed: Observed sequences, shape (M, L) - values in [0, obs_alphabet_size)
            transmitted: Transmitted sequences, shape (N, L) - values in [0, tx_alphabet_size)

        Returns:
            Decoding-metric matrix of shape (M, N)
        """
        M, L = observed.shape
        N = transmitted.shape[0]

        # Expand for broadcasting: transmitted (N, L) and observed (M, L)
        # transmitted_expanded: (1, N, L), observed_expanded: (M, 1, L)
        transmitted_expanded = transmitted[None, :, :]  # (1, N, L)
        observed_expanded = observed[:, None, :]  # (M, 1, L)

        # Costs: C[transmitted, observed] for each position
        # Use advanced indexing: cost_matrix[transmitted, observed] for each position
        costs = self.cost_matrix[transmitted_expanded, observed_expanded]  # (M, N, L)

        # Sum over positions
        return np.sum(costs, axis=2)  # (M, N)

    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Element-wise NLL decoding metric computation.

        Args:
            observed: Observed sequences, shape (M, L)
            transmitted: Transmitted sequences, shape (M, L)

        Returns:
            Array of decoding metrics, shape (M,)
        """
        # Costs: C[transmitted, observed] for each position
        costs = self.cost_matrix[transmitted, observed]  # (M, L)

        # Sum over positions
        return np.sum(costs, axis=1)

    @classmethod
    def from_symbol_epsilons(cls, epsilons: np.ndarray) -> 'AsymmetricNLL':
        """
        Factory method to create decoding metric from per-symbol error probabilities.

        Creates a square channel matrix where each symbol a has error probability epsilons[a]:
            T[a, a] = 1 - epsilons[a] (correct transmission)
            T[a, b] = epsilons[a] / (q-1) for a ≠ b (uniform error to other symbols)

        This models a channel where different symbols have different reliability.
        Alphabet size q is inferred from the array length.

        Args:
            epsilons: 1D array of shape (q,) with error probability for each symbol.
                     All values must be in [0, 1].

        Returns:
            AsymmetricNLL instance.

        Example:
            >>> # Different error rates for A, T, C, G
            >>> epsilons = np.array([0.08, 0.12, 0.10, 0.09])
            >>> decoding_metric = AsymmetricNLL.from_symbol_epsilons(epsilons)
        """
        channel_matrix = build_channel_matrix_from_symbol_epsilons(epsilons)
        return cls(channel_matrix)

    def precompute_transmitted(self, codebook: np.ndarray) -> _BLASPrecomputed:
        q_obs = self.observed_alphabet_size
        N = codebook.shape[0]
        cost_safe = np.minimum(self.cost_matrix, _INF_SENTINEL).astype(np.float32)
        # cost_safe shape (q_tx, q_obs), codebook shape (N, L)
        # cost_safe[codebook] -> (N, L, q_obs)
        psi_tx = cost_safe[codebook].reshape(N, -1)
        assert psi_tx.dtype == np.float32, f"psi_tx should be float32, got {psi_tx.dtype}"
        return _BLASPrecomputed(psi_tx=psi_tx, q_obs=q_obs)

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: _BLASPrecomputed
    ) -> np.ndarray:
        phi_obs = one_hot_encode(observed, precomputed.q_obs)
        return phi_obs @ precomputed.psi_tx.T


class PositionVaryingAsymmetricNLL(DecodingMetric):
    """
    Negative log-likelihood decoding metric for position-specific channel matrices.

    For a channel where position i has matrix T^(i):
        T^(i)[a, b] = P(observe b at position i | transmit a at position i)

    This computes the negative log-likelihood:

        d(obs, tx) = Σᵢ -log(T^(i)[txᵢ, obsᵢ])

    Supports non-square channel matrices where the transmitted and observed
    alphabets have different sizes.

    Note: Unlike distance metrics, this decoding metric:
    - May have d(x, x) > 0
    - Is not symmetric
    - Does not satisfy the triangle inequality
    """

    def __init__(self, channel_matrices: np.ndarray):
        """
        Initialize with per-position channel matrices.

        Args:
            channel_matrices: Array of shape (seq_length, tx_size, obs_size) where
                             channel_matrices[i, a, b] = P(observe b | transmit a) at position i.
                             Each (tx_size, obs_size) slice must be row-stochastic.
        """
        channel_matrices = np.asarray(channel_matrices, dtype=np.float64)

        if channel_matrices.ndim != 3:
            raise ValueError(
                f"channel_matrices must be 3D, got {channel_matrices.ndim}D"
            )

        transmitted_alphabet_size = channel_matrices.shape[1]
        observed_alphabet_size = channel_matrices.shape[2]

        if transmitted_alphabet_size < 2:
            raise ValueError(f"transmitted_alphabet_size must be at least 2, got {transmitted_alphabet_size}")
        if observed_alphabet_size < 1:
            raise ValueError(f"observed_alphabet_size must be at least 1, got {observed_alphabet_size}")

        if np.any(channel_matrices < 0) or np.any(channel_matrices > 1):
            raise ValueError("All channel matrix entries must be in [0, 1]")

        # Check row-stochasticity for each position
        row_sums = channel_matrices.sum(axis=2)  # Shape: (seq_length, transmitted_alphabet_size)
        if not np.allclose(row_sums, 1.0):
            bad_positions = np.where(~np.isclose(row_sums, 1.0).all(axis=1))[0]
            raise ValueError(
                f"Channel matrix rows must sum to 1. "
                f"Violations at positions: {bad_positions.tolist()}"
            )

        # Warn about zero entries
        if np.any(channel_matrices == 0):
            logger.warning(
                "Channel matrices contain zero entries; cost will be +inf for these transitions"
            )

        self.channel_matrices = channel_matrices
        self.transmitted_alphabet_size = transmitted_alphabet_size
        self.observed_alphabet_size = observed_alphabet_size

        # Precompute cost matrices: C^(i)[a, b] = -log(T^(i)[a, b])
        with np.errstate(divide='ignore'):
            self.cost_matrices = -np.log(channel_matrices)  # (L, tx_size, obs_size)

    @property
    def seq_length(self) -> int:
        """Expected sequence length."""
        return self.channel_matrices.shape[0]

    def compute(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of NLL decoding metrics.

        Args:
            observed: Observed sequences, shape (M, L)
            transmitted: Transmitted sequences, shape (N, L)

        Returns:
            Decoding-metric matrix of shape (M, N)
        """
        M, L = observed.shape
        N = transmitted.shape[0]

        if L != self.seq_length:
            raise ValueError(
                f"Sequence length {L} must match number of "
                f"channel matrices {self.seq_length}"
            )

        # Fully vectorized 3D indexing using broadcasting:
        # We want cost_matrices[i, transmitted[n, i], observed[m, i]] for all (m, n, i)

        # Position indices: (1, 1, L) broadcasts to (M, N, L)
        pos_idx = np.arange(L)[None, None, :]

        # Transmitted indices: (1, N, L) broadcasts to (M, N, L)
        tx_idx = transmitted[None, :, :]

        # Observed indices: (M, 1, L) broadcasts to (M, N, L)
        obs_idx = observed[:, None, :]

        # Costs via advanced indexing: (M, N, L)
        costs = self.cost_matrices[pos_idx, tx_idx, obs_idx]

        # Sum over positions
        return np.sum(costs, axis=2)

    def compute_pairwise(self, observed: np.ndarray, transmitted: np.ndarray) -> np.ndarray:
        """
        Element-wise NLL decoding metric computation.

        Args:
            observed: Observed sequences, shape (M, L)
            transmitted: Transmitted sequences, shape (M, L)

        Returns:
            Array of decoding metrics, shape (M,)
        """
        M, L = observed.shape

        if L != self.seq_length:
            raise ValueError(
                f"Sequence length {L} must match number of "
                f"channel matrices {self.seq_length}"
            )

        # Fully vectorized using 2D advanced indexing:
        # costs[m, i] = cost_matrices[i, transmitted[m, i], observed[m, i]]

        # Position indices: (1, L) broadcasts to (M, L)
        pos_idx = np.arange(L)[None, :]

        # Costs: (M, L)
        costs = self.cost_matrices[pos_idx, transmitted, observed]

        # Sum over positions
        return np.sum(costs, axis=1)

    @classmethod
    def from_positional_symbol_epsilons(cls, epsilons: np.ndarray) -> 'PositionVaryingAsymmetricNLL':
        """
        Factory method to create decoding metric from per-position per-symbol error probabilities.

        Creates position-specific square channel matrices where position i and symbol a
        have error probability epsilons[i, a]:
            T[i][a, a] = 1 - epsilons[i, a] (correct transmission at position i)
            T[i][a, b] = epsilons[i, a] / (q-1) for a ≠ b (uniform errors)

        This models a channel where noise characteristics vary both by position
        and by transmitted symbol. Sequence length L and alphabet size q are
        inferred from the array shape.

        Args:
            epsilons: 2D array of shape (L, q) where epsilons[i, a] is the error
                     probability at position i for transmitted symbol a.
                     All values must be in [0, 1].

        Returns:
            PositionVaryingAsymmetricNLL instance.

        Example:
            >>> # 5 positions, 4 symbols (DNA)
            >>> epsilons = np.array([
            ...     [0.05, 0.05, 0.05, 0.05],  # Position 0
            ...     [0.10, 0.08, 0.08, 0.10],  # Position 1
            ...     [0.15, 0.15, 0.15, 0.15],  # Position 2
            ...     [0.12, 0.10, 0.10, 0.12],  # Position 3
            ...     [0.20, 0.20, 0.20, 0.20],  # Position 4
            ... ])
            >>> decoding_metric = PositionVaryingAsymmetricNLL.from_positional_symbol_epsilons(epsilons)
        """
        channel_matrices = build_positional_channel_matrices_from_epsilons(epsilons)
        return cls(channel_matrices)

    def precompute_transmitted(self, codebook: np.ndarray) -> _BLASPrecomputed:
        q_obs = self.observed_alphabet_size
        N = codebook.shape[0]
        L = codebook.shape[1]
        cost_safe = np.minimum(self.cost_matrices, _INF_SENTINEL).astype(np.float32)
        # cost_safe shape (L, q_tx, q_obs), codebook shape (N, L)
        # Advanced indexing: cost_safe[arange(L), codebook] -> (N, L, q_obs)
        psi_tx = cost_safe[np.arange(L), codebook].reshape(N, L * q_obs)
        assert psi_tx.dtype == np.float32, f"psi_tx should be float32, got {psi_tx.dtype}"
        return _BLASPrecomputed(psi_tx=psi_tx, q_obs=q_obs)

    def compute_with_precomputed(
        self, observed: np.ndarray, precomputed: _BLASPrecomputed
    ) -> np.ndarray:
        phi_obs = one_hot_encode(observed, precomputed.q_obs)
        return phi_obs @ precomputed.psi_tx.T


# =============================================================================
# Decoding Rule Implementations
# =============================================================================

class UniqueMinimum(DecodingRule):
    """
    Unique minimum decoding rule.

    A codeword s competes with the transmitted codeword s* if cost(r, s) ≤ cost(r, s*).
    Correct decoding occurs when exactly one codeword achieves the minimum cost.

    Mathematical equivalence:
        This is equivalent to PairwisePosteriorThreshold with threshold=0.5,
        or MarginDecoding with k=0. All three formulations identify competitors
        as codewords with cost ≤ transmitted cost.

    The default decoding rule of the public API (``rule=`` of
    :func:`duet.design_ops`, :func:`duet.design_merfish`, :func:`duet.evaluate`).

    Examples
    --------
    >>> import duet
    >>> rule = duet.UniqueMinimum()
    >>> rule.competitor_margin()
    0.0
    """

    def __init__(self):
        """Initialize unique minimum decoding rule (no parameters)."""
        pass

    def competitor_margin(self) -> float:
        return 0.0

    def identify_competitors(
        self,
        cost_matrix: np.ndarray,
        transmitted_costs: np.ndarray
    ) -> csr_matrix:
        """
        Returns True where cost(r, s) ≤ cost(r, s*).
        """
        # cost_matrix: (M, N), transmitted_costs: (M,)
        # Broadcasting: (M, N) <= (M, 1) -> (M, N)
        competitors = cost_matrix <= transmitted_costs[:, None]
        return csr_matrix(competitors, dtype=bool)


class MarginDecoding(DecodingRule):
    """
    Margin-based decoding rule.

    A codeword s competes with the transmitted codeword s* if cost(r, s) ≤ cost(r, s*) + k.
    The margin k allows for some tolerance in the decoding decision.

    When k=0, this is equivalent to UniqueMinimum.
    When k>0, it's more conservative (more competitors).
    When k<0, it's more permissive (fewer competitors).

    Mathematical equivalence:
        This is equivalent to PairwisePosteriorThreshold with:
            threshold = σ(k) = 1 / (1 + exp(-k))

        where σ is the logistic (sigmoid) function.

        Common correspondences:
            k = 0    ↔  threshold = 0.500
            k = 1    ↔  threshold = 0.731
            k = 2    ↔  threshold = 0.881
            k = 2.94 ↔  threshold = 0.950

    The margin decoder of the Methods. The public API accepts it as ``rule=``
    with k >= 0.

    Examples
    --------
    >>> import duet
    >>> rule = duet.MarginDecoding(k=2.0)
    >>> rule.competitor_margin()
    2.0
    """

    def __init__(self, k: float):
        """
        Initialize with margin parameter.

        Args:
            k: Margin threshold. Competitors satisfy cost(r, s) ≤ cost(r, s*) + k
        """
        self.k = k

    def competitor_margin(self) -> float:
        return float(self.k)

    def identify_competitors(
        self,
        cost_matrix: np.ndarray,
        transmitted_costs: np.ndarray
    ) -> csr_matrix:
        """
        Returns True where cost(r, s) ≤ cost(r, s*) + k.
        """
        competitors = cost_matrix <= (transmitted_costs[:, None] + self.k)
        return csr_matrix(competitors, dtype=bool)


class PairwisePosteriorThreshold(DecodingRule):
    """
    Pairwise posterior threshold decoding rule.

    Decodes observation r to codeword s* if the pairwise posterior
    π_{s*,j}(r) ≥ threshold for ALL competitors s_j.

    The pairwise posterior is:
        π_{ij}(r) = 1 / (1 + exp(-Δ_ij(r)))

    where Δ_ij(r) = d(r, s_j) - d(r, s_i) is the gap.

    This is mathematically equivalent to MarginDecoding with:
        k = logit(threshold) = log(threshold / (1 - threshold))

    Common correspondences:
        threshold = 0.500 ↔  k = 0    (UniqueMinimum)
        threshold = 0.731 ↔  k = 1
        threshold = 0.881 ↔  k = 2
        threshold = 0.900 ↔  k ≈ 2.20
        threshold = 0.950 ↔  k ≈ 2.94
        threshold = 0.990 ↔  k ≈ 4.60
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize with pairwise posterior threshold.

        Args:
            threshold: Minimum pairwise posterior probability required.
                       Must be in (0, 1). Default 0.5 is equivalent to
                       unique minimum decoding.
        """
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")

        self.threshold = threshold
        # Convert to margin for implementation: k = logit(τ)
        self._margin = np.log(threshold / (1 - threshold))

    def competitor_margin(self) -> float:
        return float(self._margin)

    def identify_competitors(
        self,
        cost_matrix: np.ndarray,
        transmitted_costs: np.ndarray
    ) -> csr_matrix:
        """
        Identify competitors based on pairwise posterior threshold.

        A codeword s_j competes with transmitted s* if π_{s*,j}(r) ≤ threshold,
        which is equivalent to: gap = d(r, s_j) - d(r, s*) ≤ margin.

        Returns True where cost(r, s) ≤ cost(r, s*) + margin.
        """
        competitors = cost_matrix <= (transmitted_costs[:, None] + self._margin)
        return csr_matrix(competitors, dtype=bool)


class ApproximatePosteriorThreshold(DecodingRule):
    """
    Approximate full posterior threshold via pairwise thresholds.

    WARNING: This is an APPROXIMATION. The pairwise error probability (PEP) matrix
    cannot exactly capture full posterior threshold decoding because the posterior
    P(s* | r, S) depends on the entire codebook S, which varies during optimization.
    The PEP matrix is precomputed over the full library G, not the current codebook S.

    This class provides a conservative approximation by converting a target full
    posterior threshold τ to a pairwise threshold τ_pair such that:

        If all pairwise posteriors satisfy π_{s*,j}(r) ≥ τ_pair for j ∈ S,
        then P(s* | r, S) ≥ τ for a codebook with n_eff effective competitors.

    The conversion formula:
        τ_pair = (τ × n_eff) / (1 + τ × (n_eff - 1))

    where n_eff is the effective number of competitors (typically |S| - 1).

    Common conversions for n_eff = 3:
        ┌─────────────┬────────────┬──────────────────────────┐
        │ τ (target)  │  τ_pair    │  Interpretation          │
        ├─────────────┼────────────┼──────────────────────────┤
        │    0.50     │   0.750    │  Majority confidence     │
        │    0.75     │   0.900    │  Good confidence         │
        │    0.90     │   0.964    │  High confidence         │
        │    0.95     │   0.983    │  Very high confidence    │
        │    0.99     │   0.997    │  Near certainty          │
        └─────────────┴────────────┴──────────────────────────┘

    Notes:
        - For large n_eff, τ_pair → 1 (very conservative)
        - For n_eff = 1, τ_pair = τ (exact for binary comparison)
        - This approximation is loose; actual posterior often exceeds the bound
        - For exact posterior threshold decoding, use direct Monte Carlo evaluation
    """

    def __init__(self, threshold: float = 0.5, n_eff: int = 3):
        """
        Initialize with target posterior threshold and effective competitor count.

        Args:
            threshold: Target full posterior threshold τ ∈ (0, 1).
                      This is the desired minimum P(s* | r, S).
            n_eff: Effective number of competitors. Typically set to expected
                  codebook size minus 1, or a representative value. Must be ≥ 1.
                  Default is 3.

        Raises:
            ValueError: If threshold not in (0, 1) or n_eff < 1.
        """
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if n_eff < 1:
            raise ValueError(f"n_eff must be >= 1, got {n_eff}")

        self.threshold = threshold
        self.n_eff = n_eff

        # Convert to pairwise threshold: τ_pair = (τ × n_eff) / (1 + τ × (n_eff - 1))
        self.threshold_pair = (threshold * n_eff) / (1 + threshold * (n_eff - 1))

        # Convert to margin for implementation: k = logit(τ_pair)
        self._margin = np.log(self.threshold_pair / (1 - self.threshold_pair))

    def competitor_margin(self) -> float:
        return float(self._margin)

    def identify_competitors(
        self,
        cost_matrix: np.ndarray,
        transmitted_costs: np.ndarray
    ) -> csr_matrix:
        """
        Identify competitors based on converted pairwise threshold.

        Returns True where cost(r, s) ≤ cost(r, s*) + margin.
        """
        competitors = cost_matrix <= (transmitted_costs[:, None] + self._margin)
        return csr_matrix(competitors, dtype=bool)


# =============================================================================
# Memory-Mapped Array Helpers
# =============================================================================

def _mmap_array(arr: np.ndarray, mmap_dir: str) -> np.memmap:
    """Write a numpy array to a temp file and return a read-only memmap."""
    path = os.path.join(mmap_dir, f"mmap_{id(arr)}.dat")
    fp = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    fp[:] = arr
    fp.flush()
    del fp  # close write-mode memmap before opening read-only
    return np.memmap(path, dtype=arr.dtype, mode='r', shape=arr.shape)


def _mmap_csr_matrix(mat: csr_matrix, mmap_dir: str) -> csr_matrix:
    """Replace CSR internal arrays with read-only memmaps. Returns new CSR matrix."""
    data = _mmap_array(mat.data, mmap_dir)
    indices = _mmap_array(mat.indices, mmap_dir)
    indptr = _mmap_array(mat.indptr, mmap_dir)
    return csr_matrix((data, indices, indptr), shape=mat.shape)


# =============================================================================
# Worker Functions for Multiprocessing
# =============================================================================

def _worker_initialize_cache(args: Tuple) -> Tuple[int, np.ndarray, csr_matrix, np.ndarray]:
    """
    Worker function for parallel cache initialization.

    Generates samples, computes costs, and identifies competitors for one codeword.
    Returns (codeword_idx, observed, competitors, no_error) where no_error is a
    boolean array indicating samples with no channel errors.
    """
    (codeword_idx, codeword, codebook, noise_channel,
     decoding_metric, decoding_rule, n_samples, seed) = args

    # Create worker-specific RNG from seed
    rng = np.random.default_rng(seed)

    # 1. Generate noisy samples (observations)
    observed = noise_channel.generate(codeword, n_samples, rng)

    # 2. Compute costs to all codewords
    all_costs = decoding_metric.compute(observed, codebook)

    # 3. Extract costs to transmitted codeword from the all-pairs matrix
    transmitted_costs = all_costs[:, codeword_idx]

    # 4. Identify competitors
    competitors = decoding_rule.identify_competitors(all_costs, transmitted_costs)

    # 5. Precompute no-error flags (observed == transmitted for all positions)
    no_error = np.all(observed == codeword, axis=1)  # (K,) bool

    return codeword_idx, observed, competitors, no_error


def _worker_pep_batch(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker function for streaming PEP matrix computation.

    Processes a batch of codewords and immediately collapses to counts,
    avoiding memory issues from storing the full boolean matrix.

    Uses BLAS-accelerated decoding metric computation when supported by the
    decoding metric function, precomputing codebook state once per batch.

    When sample_batch_size is set, processes Monte Carlo samples in
    mini-batches to reduce peak memory from K*N*9 bytes to
    sample_batch_size*N*9 bytes per codeword.
    """
    (batch_indices, codebook, noise_channel, decoding_metric,
     decoding_rule, n_samples, seed, sample_batch_size) = args

    rng = np.random.default_rng(seed)
    batch_size = len(batch_indices)
    num_codewords = len(codebook)

    # Result: count rows for this batch (Batch, N_Total_Codewords)
    # uint16 accumulation safety: each batch contributes at most
    # sample_batch_size counts per position. Accumulating K/sample_batch_size
    # batches can produce at most K total counts. Since K <= MAX_SAMPLES_UINT16
    # (32,767) and uint16 max is 65,535, overflow is impossible.
    count_rows = np.zeros((batch_size, num_codewords), dtype=np.uint16)

    # Precompute codebook state once per worker (BLAS path)
    precomputed = decoding_metric.precompute_transmitted(codebook)

    for local_idx, codeword_idx in enumerate(batch_indices):
        codeword = codebook[codeword_idx]

        # Generate all K samples at once (preserves RNG consistency)
        observed = noise_channel.generate(codeword, n_samples, rng)

        if sample_batch_size is None or sample_batch_size >= n_samples:
            # No sample batching — original path
            if precomputed is not None:
                all_costs = decoding_metric.compute_with_precomputed(observed, precomputed)
            else:
                all_costs = decoding_metric.compute(observed, codebook)

            transmitted_costs = all_costs[:, codeword_idx]
            competitors = decoding_rule.identify_competitors(all_costs, transmitted_costs)

            if hasattr(competitors, 'toarray'):
                competitors_dense = competitors.toarray()
            else:
                competitors_dense = competitors

            count_rows[local_idx, :] = competitors_dense.sum(axis=0).astype(np.uint16)
        else:
            # Sample-batched path: process mini-batches to reduce peak memory
            for start in range(0, n_samples, sample_batch_size):
                end = min(start + sample_batch_size, n_samples)
                obs_batch = observed[start:end]  # numpy view, no copy

                if precomputed is not None:
                    costs_batch = decoding_metric.compute_with_precomputed(obs_batch, precomputed)
                else:
                    costs_batch = decoding_metric.compute(obs_batch, codebook)

                transmitted_costs = costs_batch[:, codeword_idx]
                competitors = decoding_rule.identify_competitors(costs_batch, transmitted_costs)

                if hasattr(competitors, 'toarray'):
                    competitors_dense = competitors.toarray()
                else:
                    competitors_dense = competitors

                count_rows[local_idx, :] += competitors_dense.sum(axis=0).astype(np.uint16)

    return batch_indices, count_rows


# =============================================================================
# CodebookEvaluator - The Engine Class
# =============================================================================

class CodebookEvaluator:
    """
    Engine class for codebook evaluation using composition-based architecture.

    This class:
    - Accepts ONLY pre-encoded numpy integer arrays (no string parsing in __init__)
    - Holds state (codebook and sparse matrix)
    - Executes logic using injected strategies (NoiseChannel, DecodingMetric, DecodingRule)

    Use the `from_dna_list` or `from_sequence_list` factory methods to create from strings.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        noise_channel: NoiseChannel,
        decoding_metric: DecodingMetric,
        decoding_rule: DecodingRule,
        n_samples: int,
        seed: Optional[int] = None,
        store_observed: bool = False,
        mmap_cache: bool = True,
        scratch_dir: Optional[str] = None,
    ):
        """
        Initialize CodebookEvaluator with pre-encoded codebook and strategies.

        Args:
            codebook: Pre-encoded numpy array of shape (num_codewords, seq_length).
                     Must be integer dtype (int8, int16, int32, int64).
            noise_channel: Strategy for generating noisy samples
            decoding_metric: Strategy for computing costs between sequences
            decoding_rule: Strategy for identifying competitors
            n_samples: Number of samples to generate per codeword for Monte Carlo estimation
            seed: Random seed for reproducibility
            store_observed: If True, retain _observed_sequences in memory after
                cache initialization. Default False saves 22-36 GB at genome scale.
            mmap_cache: If True, back the CSR sparse matrix and no-error flags
                with memory-mapped temp files after cache init. Default True
                saves ~4.5 GB at genome scale with negligible runtime cost.
            scratch_dir: Directory for temporary mmap files. Default None uses
                the system temp directory (/tmp). Set to a path on a filesystem
                with sufficient space for large-scale runs.

        Raises:
            TypeError: If codebook is not integer dtype
        """
        # Validate codebook dtype - must be integer
        if not np.issubdtype(codebook.dtype, np.integer):
            raise TypeError(
                f"codebook must be integer dtype (int8/int16/int32/int64), got {codebook.dtype}"
            )

        if codebook.ndim != 2:
            raise ValueError(f"codebook must be 2D array, got {codebook.ndim}D")

        self.codebook = codebook.copy()
        self.noise_channel = noise_channel
        self.decoding_metric = decoding_metric
        self.decoding_rule = decoding_rule
        self.n_samples = n_samples
        self.seed = seed
        self.store_observed = store_observed
        self.mmap_cache = mmap_cache
        self.scratch_dir = scratch_dir

        # Internal state - populated by initialize_cache
        self._sparse_matrix: Optional[csr_matrix] = None
        self._observed_sequences: Optional[np.ndarray] = None
        self._no_error_flags: Optional[np.ndarray] = None
        self._codeword_row_indices: Optional[List[np.ndarray]] = None
        self._mmap_dir: Optional[str] = None

    def _cleanup_mmap(self):
        """Remove the temporary mmap directory if it exists."""
        if self._mmap_dir is not None and os.path.isdir(self._mmap_dir):
            shutil.rmtree(self._mmap_dir, ignore_errors=True)
            self._mmap_dir = None

    def close(self):
        """Explicitly clean up memory-mapped temp files."""
        self._cleanup_mmap()

    @classmethod
    def from_sequence_list(
        cls,
        sequence_list: List[str],
        encoder: SequenceEncoder,
        noise_channel: NoiseChannel,
        decoding_metric: DecodingMetric,
        decoding_rule: DecodingRule,
        n_samples: int,
        seed: Optional[int] = None,
        store_observed: bool = False,
        mmap_cache: bool = True,
        scratch_dir: Optional[str] = None,
    ) -> 'CodebookEvaluator':
        """
        Factory method to create CodebookEvaluator from string sequences using any encoder.

        This is the generic factory method for arbitrary alphabets.

        Args:
            sequence_list: List of string sequences
            encoder: SequenceEncoder instance for encoding
            noise_channel: Strategy for generating noisy samples
            decoding_metric: Strategy for computing costs between sequences
            decoding_rule: Strategy for identifying competitors
            n_samples: Number of samples to generate per codeword for Monte Carlo estimation
            seed: Random seed for reproducibility
            store_observed: If True, retain _observed_sequences in memory after
                cache initialization. Default False saves memory at genome scale.
            mmap_cache: If True, back the CSR sparse matrix and no-error flags
                with memory-mapped temp files after cache init. Default True.
            scratch_dir: Directory for temporary mmap files. Default None uses
                the system temp directory.

        Returns:
            CodebookEvaluator instance with encoded codebook
        """
        codebook = encoder.encode(sequence_list)
        return cls(
            codebook, noise_channel, decoding_metric, decoding_rule, n_samples,
            seed, store_observed=store_observed, mmap_cache=mmap_cache,
            scratch_dir=scratch_dir,
        )

    @classmethod
    def from_dna_list(
        cls,
        dna_list: List[str],
        noise_channel: NoiseChannel,
        decoding_metric: DecodingMetric,
        decoding_rule: DecodingRule,
        n_samples: int,
        seed: Optional[int] = None,
        store_observed: bool = False,
        mmap_cache: bool = True,
        scratch_dir: Optional[str] = None,
    ) -> 'CodebookEvaluator':
        """
        Factory method to create CodebookEvaluator from DNA string list.

        This is a convenience wrapper around from_sequence_list with DNAEncoder.

        Args:
            dna_list: List of DNA sequences (strings containing A, T, C, G)
            noise_channel: Strategy for generating noisy samples
            decoding_metric: Strategy for computing costs between sequences
            decoding_rule: Strategy for identifying competitors
            n_samples: Number of samples to generate per codeword for Monte Carlo estimation
            seed: Random seed for reproducibility
            store_observed: If True, retain _observed_sequences in memory after
                cache initialization. Default False saves memory at genome scale.
            mmap_cache: If True, back the CSR sparse matrix and no-error flags
                with memory-mapped temp files after cache init. Default True.
            scratch_dir: Directory for temporary mmap files. Default None uses
                the system temp directory.

        Returns:
            CodebookEvaluator instance with encoded codebook
        """
        encoder = DNAEncoder()
        return cls.from_sequence_list(
            dna_list, encoder, noise_channel, decoding_metric, decoding_rule,
            n_samples, seed, store_observed=store_observed, mmap_cache=mmap_cache,
            scratch_dir=scratch_dir,
        )

    @property
    def num_codewords(self) -> int:
        """Number of codewords in the codebook."""
        return len(self.codebook)

    @property
    def seq_length(self) -> int:
        """Length of each codeword sequence."""
        return self.codebook.shape[1]

    @property
    def is_initialized(self) -> bool:
        """Whether the cache has been initialized."""
        return self._sparse_matrix is not None

    def initialize_cache(
        self,
        n_jobs: Optional[int] = None,
        device: str = "cpu",
    ) -> Dict:
        """
        Populate the sparse competitor matrix using the injected strategies.

        This method:
        1. Generates noisy samples using noise_channel.generate()
        2. Computes costs using decoding_metric.compute() and compute_pairwise()
        3. Identifies competitors using decoding_rule.identify_competitors()

        The resulting sparse matrix is stored in CSR format for efficient row access.
        Uses n_samples and seed from instance attributes set during __init__.

        Args:
            n_jobs: Number of parallel processes (default: CPU count)
            device: Computation device. ``"cpu"`` (default) uses the
                multiprocessing pipeline. GPU options (require CuPy):
                ``"gpu"`` — single GPU (device 0),
                ``"gpu:all"`` — all available GPUs,
                ``"gpu:0,2"`` — specific GPU IDs.

        Returns:
            Dictionary with initialization statistics
        """
        if device.startswith("gpu"):
            from duet.gpu_utils import parse_device, require_cupy
            require_cupy(device)
            from duet.gpu_cache import initialize_cache_gpu
            gpu_ids = parse_device(device)
            return initialize_cache_gpu(self, gpu_ids=gpu_ids)

        if n_jobs is None:
            n_jobs = cpu_count()

        n_samples = self.n_samples
        seed = self.seed

        # Generate reproducible seeds for each worker using SeedSequence
        root_ss = np.random.SeedSequence(seed)
        child_seeds = [ss.generate_state(1)[0] for ss in root_ss.spawn(self.num_codewords)]

        # Prepare worker arguments
        worker_args = [
            (i, self.codebook[i], self.codebook, self.noise_channel,
             self.decoding_metric, self.decoding_rule, n_samples, child_seeds[i])
            for i in range(self.num_codewords)
        ]

        logger.info(f"Initializing cache: {self.num_codewords} codewords × {n_samples} samples")
        logger.info(f"Using {n_jobs} processes")

        # Execute workers
        if n_jobs > 1 and self.num_codewords > 1:
            with Pool(n_jobs, initializer=_worker_init_blas) as pool:
                results = pool.map(_worker_initialize_cache, worker_args)
        else:
            results = [_worker_initialize_cache(args) for args in worker_args]

        # Sort results by codeword index and collect
        results = sorted(results, key=lambda x: x[0])

        sparse_matrices = []
        observed_list = []
        no_error_list = []
        codeword_row_indices = []
        current_row = 0

        for codeword_idx, observed, sparse_mat, no_error in results:
            observed_list.append(observed)
            sparse_matrices.append(sparse_mat)
            no_error_list.append(no_error)

            # Track row indices for this codeword
            num_rows = observed.shape[0]
            codeword_row_indices.append(np.arange(current_row, current_row + num_rows))
            current_row += num_rows

        # Stack sparse matrices vertically - result is CSR format
        self._sparse_matrix = vstack(sparse_matrices, format='csr')
        self._no_error_flags = np.concatenate(no_error_list)
        if self.store_observed:
            self._observed_sequences = np.vstack(observed_list)
        self._codeword_row_indices = codeword_row_indices

        # Optionally back arrays with memory-mapped temp files
        if self.mmap_cache:
            self._cleanup_mmap()  # clean up any previous mmap dir on re-init
            self._mmap_dir = tempfile.mkdtemp(prefix="duet_cache_", dir=self.scratch_dir)
            self._sparse_matrix = _mmap_csr_matrix(self._sparse_matrix, self._mmap_dir)
            self._no_error_flags = _mmap_array(self._no_error_flags, self._mmap_dir)
            atexit.register(self._cleanup_mmap)

        stats = {
            'num_codewords': self.num_codewords,
            'seq_length': self.seq_length,
            'n_samples': n_samples,
            'total_rows': self._sparse_matrix.shape[0],
            'matrix_shape': self._sparse_matrix.shape,
            'nnz': self._sparse_matrix.nnz,
            'density': self._sparse_matrix.nnz / np.prod(self._sparse_matrix.shape),
            'memory_bytes': (
                self._sparse_matrix.data.nbytes +
                self._sparse_matrix.indices.nbytes +
                self._sparse_matrix.indptr.nbytes
            ),
            'no_error_flags_bytes': self._no_error_flags.nbytes,
        }

        logger.info(f"Cache initialized: {stats['matrix_shape']}, nnz={stats['nnz']}")

        return stats

    def compute_pep_matrix(
        self,
        batch_size: int = 100,
        n_jobs: Optional[int] = None,
        sample_batch_size: int | None = None,
        output: np.ndarray | None = None,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Pairwise Error Probability matrix using streaming (map-reduce) computation.

        This method computes the PEP matrix WITHOUT materializing the full boolean
        cache in memory. It processes codewords in batches, immediately summing
        over samples to produce count rows (uint16).

        PEP[i,j] = number of simulated reads of codeword i in which codeword j
        competes with i under the decoding rule (for unique-minimum decoding,
        cost(read, j) <= cost(read, i)); the diagonal counts i against itself.
        Divide by n_samples to get probabilities.

        Uses n_samples and seed from instance attributes set during __init__.

        Args:
            batch_size: Number of codewords to process per batch
            n_jobs: Number of parallel processes (default: CPU count)
            sample_batch_size: Number of Monte Carlo samples to process per
                mini-batch within each codeword. Reduces peak memory from
                K*N*9 bytes to sample_batch_size*N*9 bytes per worker.
                None (default) means auto: uses min(K, max(1, 2GB // (N*9))).
                Must be > 0 if explicitly provided.
            output: Optional pre-allocated buffer to write results into.
                Must be uint16 with shape (num_codewords, num_codewords).
                Can be a numpy array or a memory-mapped file (np.memmap).
                When None (default), an in-memory zeros array is allocated.
            device: Computation device. ``"cpu"`` (default) uses the
                multiprocessing pipeline. GPU options (require CuPy):
                ``"gpu"`` — single GPU (device 0),
                ``"gpu:all"`` — all available GPUs,
                ``"gpu:0,2"`` — specific GPU IDs.

        Returns:
            Tuple of (count_matrix, n_samples) where count_matrix is uint16
            of shape (num_codewords, num_codewords).
        """
        n_samples = self.n_samples
        seed = self.seed

        if n_samples > MAX_SAMPLES_UINT16:
            raise NotImplementedError(
                f"n_samples={n_samples} exceeds maximum {MAX_SAMPLES_UINT16} for uint16 "
                f"storage (2 * n_samples must fit in uint16). "
                f"Future: use uint32 storage for large sample counts."
            )

        # Resolve sample_batch_size
        if sample_batch_size is not None and sample_batch_size <= 0:
            raise ValueError(
                f"sample_batch_size must be a positive integer, got {sample_batch_size}"
            )

        if device.startswith("gpu"):
            from duet.gpu_utils import parse_device, require_cupy
            require_cupy(device)
            from duet.gpu_pep import compute_pep_matrix_gpu
            gpu_ids = parse_device(device)
            result, n_samples_gpu = compute_pep_matrix_gpu(
                self,
                batch_size=batch_size,
                sample_batch_size=sample_batch_size,
                output=output,
                gpu_ids=gpu_ids,
            )
            if hasattr(result, 'flush'):
                result.flush()
            return result, n_samples_gpu

        if sample_batch_size is None:
            # Auto heuristic: target 2 GB peak per worker
            target_bytes = 2 * 1024**3
            N = self.num_codewords
            sample_batch_size = min(n_samples, max(1, target_bytes // (N * 9)))

        if n_jobs is None:
            n_jobs = cpu_count()

        # Create batches of codeword indices
        indices = np.arange(self.num_codewords)
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        # Generate seeds for each batch
        root_ss = np.random.SeedSequence(seed)
        batch_seeds = [ss.generate_state(1)[0] for ss in root_ss.spawn(len(batches))]

        # Prepare worker arguments
        worker_args = [
            (batch, self.codebook, self.noise_channel, self.decoding_metric,
             self.decoding_rule, n_samples, batch_seeds[i], sample_batch_size)
            for i, batch in enumerate(batches)
        ]

        logger.info(f"Computing PEP matrix: {self.num_codewords} codewords in {len(batches)} batches")
        logger.info(f"Samples per codeword: {n_samples}, batch size: {batch_size}")

        # Allocate or validate output buffer
        if output is None:
            output = np.zeros((self.num_codewords, self.num_codewords), dtype=np.uint16)
        else:
            if output.shape != (self.num_codewords, self.num_codewords):
                raise ValueError(
                    f"output shape {output.shape} != expected "
                    f"({self.num_codewords}, {self.num_codewords})"
                )
            if output.dtype != np.uint16:
                raise ValueError(f"output dtype {output.dtype} != expected uint16")

        # Map + reduce in a single streaming pass.
        # Each batch result is written directly into the output buffer and
        # then discardable. The parallel/serial branch is the ONLY branch point.
        # The Pool context manager MUST remain open for the full duration of
        # lazy iteration — do not separate the iterator from the `with` block.
        desc = f"PEP matrix ({self.num_codewords} codewords)"
        if n_jobs > 1 and len(batches) > 1:
            with Pool(n_jobs, initializer=_worker_init_blas) as pool:
                for batch_indices, count_rows in tqdm(
                    pool.imap_unordered(_worker_pep_batch, worker_args),
                    total=len(batches), desc=desc,
                ):
                    output[batch_indices, :] = count_rows
        else:
            for args in tqdm(worker_args, desc=desc):
                batch_indices, count_rows = _worker_pep_batch(args)
                output[batch_indices, :] = count_rows

        # Flush if output is a memory-mapped file
        if hasattr(output, 'flush'):
            output.flush()

        logger.info("PEP matrix computation complete")
        return output, self.n_samples

    def _estimate_codeword_batch_size(
        self,
        num_selected: int,
        mem_budget_bytes: int,
    ) -> int:
        """Estimate how many codewords fit in one batch given a memory budget.

        The dominant cost of ``sparse_matrix[rows, :][:, cols]`` is the
        intermediate from the row extraction.  For B codewords with K
        samples each and avg_nnz nonzeros per row, the intermediate has
        roughly ``B * K * avg_nnz`` nonzeros, each costing ~12 bytes
        (int64 index + bool data + indptr amortized).
        """
        total_rows, total_cols = self._sparse_matrix.shape
        if total_rows == 0:
            return num_selected
        avg_nnz_per_row = self._sparse_matrix.nnz / total_rows
        bytes_per_codeword = self.n_samples * avg_nnz_per_row * 12
        if bytes_per_codeword <= 0:
            return num_selected
        batch = int(mem_budget_bytes / bytes_per_codeword)
        return max(1, min(batch, num_selected))

    def get_codeword_accuracy(
        self,
        indices: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        mem_budget_gb: float = 20.0,
    ) -> np.ndarray:
        """
        Compute decoding accuracy for each codeword in the specified subset.

        When indices is provided, this simulates having a codebook containing only
        the specified codewords. Decoding is performed against this subset, and
        accuracy is computed for each codeword in the subset.

        Accuracy for codeword i = (1/K) Σ_k 𝟙[decode(r_k) = s_i]
        where r_k are the K samples from codeword i, and decode(r) finds the
        unique minimum cost codeword among the selected subset.

        Args:
            indices: Optional array of codeword indices specifying the active codebook.
                     If None, uses all codewords. When provided, BOTH:
                     - Only samples from these codewords are evaluated
                     - Only these codewords are considered as potential matches
            batch_size: Number of codewords to process per batch.  If None,
                        computed from ``mem_budget_gb``.
            mem_budget_gb: Target peak anonymous memory per batch in GB
                           (default 20).  Ignored when ``batch_size`` is set.

        Returns:
            Array of accuracies with shape (len(indices),) or (num_codewords,)
        """
        if not self.is_initialized:
            raise RuntimeError("Cache must be initialized first. Call initialize_cache().")

        if indices is None:
            indices = np.arange(self.num_codewords)

        indices = np.asarray(indices)

        if len(indices) == 0:
            return np.array([], dtype=np.float64)

        if batch_size is None:
            batch_size = self._estimate_codeword_batch_size(
                len(indices), int(mem_budget_gb * 1e9),
            )

        if len(indices) <= batch_size:
            row_indices = np.concatenate(
                [self._codeword_row_indices[idx] for idx in indices]
            )
            submat = self._sparse_matrix[row_indices, :][:, indices]
            row_sums = np.asarray(submat.sum(axis=1)).flatten()
            correct = (row_sums == 1).astype(np.float64)
            return correct.reshape(len(indices), self.n_samples).mean(axis=1)

        # Batch by codewords to limit peak anonymous memory.
        # Column set is always the full `indices` (decoding considers all
        # selected codewords), but rows are processed in batches.
        accuracies = np.empty(len(indices), dtype=np.float64)
        for batch_start in range(0, len(indices), batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_cw = indices[batch_start:batch_end]
            batch_rows = np.concatenate(
                [self._codeword_row_indices[idx] for idx in batch_cw]
            )
            submat = self._sparse_matrix[batch_rows, :][:, indices]
            row_sums = np.asarray(submat.sum(axis=1)).flatten()
            correct = (row_sums == 1).astype(np.float64)
            accuracies[batch_start:batch_end] = correct.reshape(
                len(batch_cw), self.n_samples,
            ).mean(axis=1)

        return accuracies

    def get_accuracy(
        self,
        indices: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute overall decoding accuracy (averaged over codewords).

        P(correct) = (1/|S|) Σ_{s∈S} P(correct | S = s)

        Args:
            indices: Optional subset of codeword indices defining the active codebook.

        Returns:
            Mean accuracy across codewords in the subset.
        """
        return float(np.mean(self.get_codeword_accuracy(
            indices=indices,
        )))

    def get_pairwise_error_from_cache(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute PEP matrix from the initialized cache.

        This uses the already-computed sparse matrix rather than
        regenerating samples. Useful when cache is already initialized.

        Args:
            indices: Optional subset of codeword indices. If provided, returns
                     PEP matrix only for those codewords (both as transmitters
                     and as potential confusion targets).

        Returns:
            PEP matrix of shape (len(indices), len(indices)) or (num_codewords, num_codewords)
        """
        if not self.is_initialized:
            raise RuntimeError("Cache must be initialized first. Call initialize_cache().")

        if indices is None:
            indices = np.arange(self.num_codewords)

        indices = np.asarray(indices)
        num_selected = len(indices)

        if num_selected == 0:
            return np.array([], dtype=np.float64).reshape(0, 0)

        # Collect row indices for selected codewords
        row_indices = np.concatenate([self._codeword_row_indices[idx] for idx in indices])

        # Subset both rows and columns
        submat = self._sparse_matrix[row_indices, :][:, indices]

        # Convert to dense, reshape to (num_selected, n_samples, num_selected), average over samples
        # submat has shape (num_selected * n_samples, num_selected)
        pep_matrix = np.asarray(submat.toarray()).reshape(num_selected, self.n_samples, num_selected).mean(axis=1)

        return pep_matrix

    def get_stats(self) -> Dict:
        """Get statistics about the current state."""
        stats = {
            'num_codewords': self.num_codewords,
            'seq_length': self.seq_length,
            'noise_channel': type(self.noise_channel).__name__,
            'decoding_metric': type(self.decoding_metric).__name__,
            'decoding_rule': type(self.decoding_rule).__name__,
            'cache_initialized': self.is_initialized,
            'n_samples': self.n_samples,
            'seed': self.seed,
        }

        if self.is_initialized:
            stats.update({
                'matrix_shape': self._sparse_matrix.shape,
                'matrix_nnz': self._sparse_matrix.nnz,
                'matrix_density': self._sparse_matrix.nnz / np.prod(self._sparse_matrix.shape),
                'memory_bytes': (
                    self._sparse_matrix.data.nbytes +
                    self._sparse_matrix.indices.nbytes +
                    self._sparse_matrix.indptr.nbytes
                )
            })

        return stats

    def get_error_correction_metrics(
        self,
        indices: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        mem_budget_gb: float = 20.0,
    ) -> ErrorCorrectionMetrics:
        """
        Compute error correction metrics for each codeword in the specified subset.

        Categorizes each sample into three mutually exclusive categories:
        - No error: observed == transmitted (no channel errors occurred)
        - Corrected: observed != transmitted but decoded correctly
        - Failed: observed != transmitted and decoding failed

        Args:
            indices: Optional array of codeword indices specifying the active codebook.
                     If None, uses all codewords. When provided:
                     - Only samples from these codewords are evaluated
                     - Only these codewords are considered as potential matches
            batch_size: Number of codewords to process per batch.  If None,
                        computed from ``mem_budget_gb``.
            mem_budget_gb: Target peak anonymous memory per batch in GB
                           (default 20).  Ignored when ``batch_size`` is set.

        Returns:
            ErrorCorrectionMetrics with codebook-level and codeword-level rates.

        Raises:
            RuntimeError: If cache has not been initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache must be initialized first. Call initialize_cache().")

        if indices is None:
            indices = np.arange(self.num_codewords)

        indices = np.asarray(indices)
        num_selected = len(indices)

        if num_selected == 0:
            return ErrorCorrectionMetrics(
                no_error_rate=0.0,
                corrected_rate=0.0,
                failed_rate=0.0,
                codeword_no_error=np.array([], dtype=np.float64),
                codeword_corrected=np.array([], dtype=np.float64),
                codeword_failed=np.array([], dtype=np.float64),
            )

        if batch_size is None:
            batch_size = self._estimate_codeword_batch_size(
                num_selected, int(mem_budget_gb * 1e9),
            )

        codeword_no_error = np.empty(num_selected, dtype=np.float64)
        codeword_corrected = np.empty(num_selected, dtype=np.float64)
        codeword_failed = np.empty(num_selected, dtype=np.float64)
        total_no_error = 0
        total_corrected = 0
        total_failed = 0
        total_samples = 0

        for batch_start in range(0, num_selected, batch_size):
            batch_end = min(batch_start + batch_size, num_selected)
            batch_cw = indices[batch_start:batch_end]
            n_batch = batch_end - batch_start

            batch_rows = np.concatenate(
                [self._codeword_row_indices[idx] for idx in batch_cw]
            )

            submat = self._sparse_matrix[batch_rows, :][:, indices]
            row_sums = np.asarray(submat.sum(axis=1)).flatten()
            correct_decoding = (row_sums == 1)

            no_error_batch = np.concatenate([
                self._no_error_flags[self._codeword_row_indices[idx]]
                for idx in batch_cw
            ])

            has_errors = ~no_error_batch
            corrected_batch = has_errors & correct_decoding
            failed_batch = has_errors & ~correct_decoding

            codeword_no_error[batch_start:batch_end] = (
                no_error_batch.reshape(n_batch, self.n_samples).mean(axis=1)
            )
            codeword_corrected[batch_start:batch_end] = (
                corrected_batch.reshape(n_batch, self.n_samples).mean(axis=1)
            )
            codeword_failed[batch_start:batch_end] = (
                failed_batch.reshape(n_batch, self.n_samples).mean(axis=1)
            )

            total_no_error += int(no_error_batch.sum())
            total_corrected += int(corrected_batch.sum())
            total_failed += int(failed_batch.sum())
            total_samples += len(batch_rows)

        return ErrorCorrectionMetrics(
            no_error_rate=total_no_error / total_samples,
            corrected_rate=total_corrected / total_samples,
            failed_rate=total_failed / total_samples,
            codeword_no_error=codeword_no_error,
            codeword_corrected=codeword_corrected,
            codeword_failed=codeword_failed,
        )

    # =========================================================================
    # Batch evaluation methods (sparse-dense matmul)
    # =========================================================================

    def _batch_matmul_core(
        self,
        codebooks: Sequence[np.ndarray],
        mem_budget_gb: float,
        compute_error_metrics: bool,
    ) -> Tuple[List[np.ndarray], Optional[List[ErrorCorrectionMetrics]]]:
        """Evaluate multiple codebooks simultaneously via sparse-dense matmul.

        Single-pass algorithm that reads each CSR row exactly once and
        broadcasts results across all codebooks via masks_matrix multiplication.

        Args:
            codebooks: Sequence of index arrays, each defining a codebook.
                Codebooks may have different lengths; all must be non-empty.
            mem_budget_gb: Memory budget in GB for batch size estimation.
            compute_error_metrics: If True, also compute ErrorCorrectionMetrics.

        Returns:
            Tuple of (codeword_accuracy, error_metrics_or_none) where
            codeword_accuracy is a list of 1D arrays, one per codebook.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache must be initialized first. Call initialize_cache().")

        if len(codebooks) == 0:
            return [], ([] if compute_error_metrics else None)

        if any(len(cb) == 0 for cb in codebooks):
            raise ValueError("codebooks must be non-empty")

        sparse_matrix = self._sparse_matrix
        no_error_flags = self._no_error_flags
        N = self.num_codewords
        K = self.n_samples
        S = len(codebooks)

        # Build masks_matrix: (N, S) int16
        masks_matrix = np.zeros((N, S), dtype=np.int16)
        codeword_membership: List[List[Tuple[int, int]]] = [[] for _ in range(N)]

        for s_idx, cb in enumerate(codebooks):
            for out_pos, idx in enumerate(cb):
                masks_matrix[int(idx), s_idx] = 1
                codeword_membership[int(idx)].append((s_idx, out_pos))

        # Pre-load indptr into RAM for sequential access
        indptr = np.array(sparse_matrix.indptr)

        # Batch size estimation
        avg_nnz = sparse_matrix.nnz / sparse_matrix.shape[0] if sparse_matrix.shape[0] > 0 else 0
        mem_per_codeword = (
            K * S * 2          # matmul output (int16)
            + K * S            # correct bool array
            + K * avg_nnz * 8  # CSR indices in page cache
        )
        if mem_per_codeword > 0:
            B = max(1, int(mem_budget_gb * 1e9 / mem_per_codeword))
            B = min(B, N)
        else:
            B = N

        # Pre-allocate per-codebook output arrays (ragged: one 1D array per codebook)
        sol_acc = [np.empty(len(cb), dtype=np.float64) for cb in codebooks]
        if compute_error_metrics:
            sol_no_error = [np.empty(len(cb), dtype=np.float64) for cb in codebooks]
            sol_corrected = [np.empty(len(cb), dtype=np.float64) for cb in codebooks]
            sol_failed = [np.empty(len(cb), dtype=np.float64) for cb in codebooks]

        # Single-pass CSR loop
        for batch_start in range(0, N, B):
            batch_end = min(batch_start + B, N)
            n_batch = batch_end - batch_start

            # Extract contiguous CSR block via indptr arithmetic
            row_start = batch_start * K
            row_end = batch_end * K
            ip_start = int(indptr[row_start])
            ip_end = int(indptr[row_end])

            sub_indptr = indptr[row_start: row_end + 1] - ip_start
            sub_indices = sparse_matrix.indices[ip_start:ip_end]
            sub_data = sparse_matrix.data[ip_start:ip_end]

            block = csr_matrix(
                (sub_data, sub_indices, sub_indptr),
                shape=(row_end - row_start, N),
            )

            # Sparse-dense matmul: (n_batch*K, N) @ (N, S) -> (n_batch*K, S)
            row_sums = block @ masks_matrix

            # Classify: correct decoding = exactly 1 competitor in codebook
            correct = (row_sums == 1)
            correct_3d = correct.reshape(n_batch, K, S)
            batch_acc = correct_3d.mean(axis=1)  # (n_batch, S)

            # Error correction metrics (optional)
            if compute_error_metrics:
                ne_flags = no_error_flags[row_start:row_end].reshape(n_batch, K, 1)
                has_error = ~ne_flags
                corrected_3d = has_error & correct_3d
                failed_3d = has_error & ~correct_3d

                batch_no_error = ne_flags[..., 0].mean(axis=1)      # (n_batch,)
                batch_corrected = corrected_3d.mean(axis=1)          # (n_batch, S)
                batch_failed = failed_3d.mean(axis=1)                # (n_batch, S)

            # Scatter results to per-codebook output arrays
            for local_i in range(n_batch):
                cw_idx = batch_start + local_i
                for s_idx, out_pos in codeword_membership[cw_idx]:
                    sol_acc[s_idx][out_pos] = batch_acc[local_i, s_idx]
                    if compute_error_metrics:
                        sol_no_error[s_idx][out_pos] = batch_no_error[local_i]
                        sol_corrected[s_idx][out_pos] = batch_corrected[local_i, s_idx]
                        sol_failed[s_idx][out_pos] = batch_failed[local_i, s_idx]

        # Assemble error metrics
        metrics_list = None
        if compute_error_metrics:
            metrics_list = []
            for s_idx in range(S):
                metrics_list.append(ErrorCorrectionMetrics(
                    no_error_rate=float(sol_no_error[s_idx].mean()),
                    corrected_rate=float(sol_corrected[s_idx].mean()),
                    failed_rate=float(sol_failed[s_idx].mean()),
                    codeword_no_error=sol_no_error[s_idx],
                    codeword_corrected=sol_corrected[s_idx],
                    codeword_failed=sol_failed[s_idx],
                ))

        return sol_acc, metrics_list

    def batch_get_accuracy(
        self,
        codebooks: Sequence[np.ndarray],
        *,
        mem_budget_gb: float = 20.0,
    ) -> np.ndarray:
        """Mean decode accuracy per codebook. Shape: (num_codebooks,).

        Batch analog of get_accuracy(). Evaluates all codebooks simultaneously
        using a single-pass sparse-dense matmul over the CSR cache.

        Args:
            codebooks: Sequence of index arrays, each defining a codebook.
                Codebooks may have different lengths; all must be non-empty.
            mem_budget_gb: Memory budget in GB for intermediate arrays.

        Returns:
            1D array of mean decode accuracies, one per codebook.
        """
        cw_acc, _ = self._batch_matmul_core(
            codebooks, mem_budget_gb, compute_error_metrics=False,
        )
        return np.array([a.mean() for a in cw_acc], dtype=np.float64)

    def batch_get_codeword_accuracy(
        self,
        codebooks: Sequence[np.ndarray],
        *,
        mem_budget_gb: float = 20.0,
    ) -> List[np.ndarray]:
        """Per-codeword accuracy per codebook.
        Returns a list of 1D arrays, one per codebook.

        Batch analog of get_codeword_accuracy(). Evaluates all codebooks
        simultaneously using a single-pass sparse-dense matmul over the CSR
        cache.

        Args:
            codebooks: Sequence of index arrays, each defining a codebook.
                Codebooks may have different lengths; all must be non-empty.
            mem_budget_gb: Memory budget in GB for intermediate arrays.

        Returns:
            List of 1D arrays, one per codebook, each containing per-codeword accuracies.
        """
        cw_acc, _ = self._batch_matmul_core(
            codebooks, mem_budget_gb, compute_error_metrics=False,
        )
        return cw_acc

    def batch_get_error_correction_metrics(
        self,
        codebooks: Sequence[np.ndarray],
        *,
        mem_budget_gb: float = 20.0,
    ) -> List[ErrorCorrectionMetrics]:
        """Error correction metrics per codebook.

        Batch analog of get_error_correction_metrics(). Evaluates all
        codebooks simultaneously using a single-pass sparse-dense matmul.

        Args:
            codebooks: Sequence of index arrays, each defining a codebook.
                Codebooks may have different lengths; all must be non-empty.
            mem_budget_gb: Memory budget in GB for intermediate arrays.

        Returns:
            List of ErrorCorrectionMetrics, one per codebook.
        """
        _, metrics = self._batch_matmul_core(
            codebooks, mem_budget_gb, compute_error_metrics=True,
        )
        return metrics

    def batch_get_accuracy_with_error_metrics(
        self,
        codebooks: Sequence[np.ndarray],
        *,
        mem_budget_gb: float = 20.0,
    ) -> Tuple[List[np.ndarray], List[ErrorCorrectionMetrics]]:
        """Per-codeword accuracy and error correction metrics in a single pass.

        Returns both codeword_accuracy and error correction metrics from one
        matmul pass. Use this instead of calling batch_get_codeword_accuracy
        and batch_get_error_correction_metrics separately when both are needed,
        to avoid running the matmul twice.

        Accuracy is computed directly from the matmul result, not derived from
        the error metrics.

        Args:
            codebooks: Sequence of index arrays, each defining a codebook.
                Codebooks may have different lengths; all must be non-empty.
            mem_budget_gb: Memory budget in GB for intermediate arrays.

        Returns:
            Tuple of (codeword_accuracy, error_metrics) where
            codeword_accuracy is a list of 1D arrays, one per codebook,
            and error_metrics is a list of ErrorCorrectionMetrics, one per
            codebook.
        """
        return self._batch_matmul_core(
            codebooks, mem_budget_gb, compute_error_metrics=True,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_evaluator_for_dna(
    dna_list: List[str],
    epsilon: float,
    n_samples: int,
    seed: Optional[int] = None,
    store_observed: bool = False,
    mmap_cache: bool = True,
    scratch_dir: Optional[str] = None,
) -> CodebookEvaluator:
    """
    Convenience function to create a CodebookEvaluator with simple defaults.

    Args:
        dna_list: List of DNA sequences
        epsilon: Uniform error probability
        n_samples: Number of samples per codeword for Monte Carlo estimation
        seed: Random seed for reproducibility
        store_observed: If True, retain _observed_sequences in memory after
            cache initialization. Default False saves memory at genome scale.
        mmap_cache: If True, back the CSR sparse matrix and no-error flags
            with memory-mapped temp files after cache init. Default True.
        scratch_dir: Directory for temporary mmap files. Default None uses
            the system temp directory.

    Returns:
        Configured CodebookEvaluator instance
    """
    noise_channel = SymmetricEpsilon(epsilon)
    decoding_metric = HammingDistance(alphabet_size=4)
    decoding_rule = UniqueMinimum()

    return CodebookEvaluator.from_dna_list(
        dna_list, noise_channel, decoding_metric, decoding_rule, n_samples,
        seed, store_observed=store_observed, mmap_cache=mmap_cache,
        scratch_dir=scratch_dir,
    )
