"""
Dual-guide factory classes for creating paired candidates.

This module provides infrastructure for dual-guide optical pooled screens where
two candidates target the same group. It includes:

- DualGuideEncoder: Encodes pairs of DNA sequences into a 16-symbol alphabet
- PairingStrategy: Abstract base class for candidate pairing strategies
- SameGroupPairingStrategy: Pairs candidates targeting the same group (n-choose-2)
- DualGuideWeissmanLibraryFactory: Base factory for dual-guide Weissman libraries
- DualGuideWeissmanCRISPRiFactory: Convenience subclass for CRISPRi
- DualGuideWeissmanCRISPRaFactory: Convenience subclass for CRISPRa

The 16-symbol alphabet encodes (candidate1_base, candidate2_base) pairs at each position,
enabling compatibility with chemistry-aware channel models that capture
the physics of dual-guide observation (e.g., four-color, two-color, three-color).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from duet.codebook_evaluator import HexEncoder
from duet.candidate_pool import CandidatePool
from duet.candidate_pool_factory import WeissmanLibraryFactory
from duet.data_paths import default_data_file


# =============================================================================
# Encoder
# =============================================================================


class DualGuideEncoder(HexEncoder):
    """
    Encodes pairs of DNA sequences into a 16-symbol alphabet.

    Inherits from HexEncoder, adding dual-guide-specific utilities for
    converting between DNA base pairs and hex symbols.

    Each position encodes the (guide1_base, guide2_base) pair as:
        symbol = 4 * base1_idx + base2_idx

    Base indices: A=0, T=1, C=2, G=3

    The 16 symbols represent all possible base pairs:
        0=(A,A), 1=(A,T), 2=(A,C), 3=(A,G),
        4=(T,A), 5=(T,T), 6=(T,C), 7=(T,G),
        8=(C,A), 9=(C,T), A=(C,C), B=(C,G),
        C=(G,A), D=(G,T), E=(G,C), F=(G,G)

    String representation uses hexadecimal characters: 0-9, A-F

    Example:
        >>> DualGuideEncoder.encode_pair("ATCG", "GCTA")
        '3695'
        >>> DualGuideEncoder.decode_pair("3695")
        ('ATCG', 'GCTA')
    """

    # DNA base mappings for pair encoding/decoding
    _BASE_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3}
    _IDX_TO_BASE = {0: "A", 1: "T", 2: "C", 3: "G"}

    @classmethod
    def encode_pair(cls, seq1: str, seq2: str) -> str:
        """
        Encode two DNA sequences into a dual-guide hex string.

        Args:
            seq1: First DNA sequence (guide 1).
            seq2: Second DNA sequence (guide 2).

        Returns:
            Encoded string using hexadecimal characters (0-9, A-F).

        Raises:
            ValueError: If sequences have different lengths.
            KeyError: If sequences contain invalid bases.

        Example:
            >>> DualGuideEncoder.encode_pair("ATCG", "GCTA")
            '3695'
        """
        if len(seq1) != len(seq2):
            raise ValueError(f"Sequence length mismatch: {len(seq1)} vs {len(seq2)}")

        result = []
        for b1, b2 in zip(seq1, seq2):
            pair_idx = 4 * cls._BASE_TO_IDX[b1] + cls._BASE_TO_IDX[b2]
            result.append(format(pair_idx, "X"))
        return "".join(result)

    @classmethod
    def decode_pair(cls, encoded: str) -> Tuple[str, str]:
        """
        Decode a dual-guide hex string back to two DNA sequences.

        Args:
            encoded: Encoded dual-guide string.

        Returns:
            Tuple of (guide1_sequence, guide2_sequence).

        Example:
            >>> DualGuideEncoder.decode_pair("3695")
            ('ATCG', 'GCTA')
        """
        seq1, seq2 = [], []
        for char in encoded:
            pair_idx = int(char, 16)
            seq1.append(cls._IDX_TO_BASE[pair_idx // 4])
            seq2.append(cls._IDX_TO_BASE[pair_idx % 4])
        return "".join(seq1), "".join(seq2)

    @classmethod
    def pair_to_symbol(cls, base1: str, base2: str) -> str:
        """
        Convert a single (guide1_base, guide2_base) pair to hex symbol.

        Args:
            base1: Base from guide 1 (A, T, C, or G).
            base2: Base from guide 2 (A, T, C, or G).

        Returns:
            Single hex character (0-9, A-F).

        Example:
            >>> DualGuideEncoder.pair_to_symbol("A", "G")
            '3'
        """
        idx = 4 * cls._BASE_TO_IDX[base1] + cls._BASE_TO_IDX[base2]
        return format(idx, "X")

    @classmethod
    def symbol_to_pair(cls, symbol: str) -> Tuple[str, str]:
        """
        Convert hex symbol back to (guide1_base, guide2_base) pair.

        Args:
            symbol: Single hex character (0-9, A-F).

        Returns:
            Tuple of (guide1_base, guide2_base).

        Example:
            >>> DualGuideEncoder.symbol_to_pair("3")
            ('A', 'G')
        """
        idx = int(symbol, 16)
        return cls._IDX_TO_BASE[idx // 4], cls._IDX_TO_BASE[idx % 4]

    @classmethod
    def to_integer_array(cls, sequences: List[str]) -> np.ndarray:
        """
        Convert list of encoded hex strings to integer array.

        Useful for passing to CodebookEvaluator which expects integer-encoded
        sequences for channel matrix indexing.

        Args:
            sequences: List of encoded dual-guide hex strings.

        Returns:
            Array of shape (n_sequences, seq_length) with values in [0, 15].

        Raises:
            ValueError: If sequences list is empty.

        Note:
            This is equivalent to calling the inherited encode() method,
            but provided for backward compatibility.
        """
        return cls().encode(sequences)

    @classmethod
    def from_integer_array(cls, array: np.ndarray) -> List[str]:
        """
        Convert integer array back to list of encoded hex strings.

        Args:
            array: Array of shape (n_sequences, seq_length) with values in [0, 15].

        Returns:
            List of encoded dual-guide hex strings.

        Note:
            This is equivalent to calling the inherited decode() method,
            but provided for backward compatibility.
        """
        return cls().decode(array)


# =============================================================================
# Chemistry-Specific DualGuideEncoder Subclasses
# =============================================================================


class TwoColorDualGuideEncoder(DualGuideEncoder):
    """
    Encoder for dual-guide sequences with two-color chemistry observation mapping.

    Inherits all functionality from DualGuideEncoder for transmitted hex space
    encoding/decoding. Adds encode_to_observed() for mapping to observation space.

    Two-color chemistry (standard Illumina):
        Base encoding: A=(1,1), T=(0,1), C=(1,0), G=(0,0)
        Observation is OR of the two base encodings:
            (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    The observation alphabet has 4 symbols.
    """

    observed_alphabet_size: int = 4
    chemistry: str = "two_color"

    def __init__(self):
        super().__init__()
        self._phi, _ = get_observation_mapping("two_color_or")

    def encode_to_observed(self, hex_array: np.ndarray) -> np.ndarray:
        """
        Map integer array from transmitted hex space to observed space.

        Args:
            hex_array: Shape (n_sequences, seq_length), values in [0, 15]

        Returns:
            Shape (n_sequences, seq_length), values in [0, 3]
        """
        b1 = hex_array // 4
        b2 = hex_array % 4
        return np.vectorize(self._phi)(b1, b2).astype(np.int8)


class ThreeColorDualGuideEncoder(DualGuideEncoder):
    """
    Encoder for dual-guide sequences with three-color chemistry observation mapping.

    Inherits all functionality from DualGuideEncoder for transmitted hex space
    encoding/decoding. Adds encode_to_observed() for mapping to observation space.

    Three-color chemistry:
        A, T, C have colors 0, 1, 2; G is dark.
        Observation is the set of colors present:
            0: {} (both G)
            1: {0} (A present, no T/C)
            2: {1} (T present, no A/C)
            3: {2} (C present, no A/T)
            4: {0,1} (A and T)
            5: {0,2} (A and C)
            6: {1,2} (T and C)

    The observation alphabet has 7 symbols.
    """

    observed_alphabet_size: int = 7
    chemistry: str = "three_color"

    def __init__(self):
        super().__init__()
        self._phi, _ = get_observation_mapping("three_color")

    def encode_to_observed(self, hex_array: np.ndarray) -> np.ndarray:
        """
        Map integer array from transmitted hex space to three-color observed space.

        Args:
            hex_array: Shape (n_sequences, seq_length), values in [0, 15]

        Returns:
            Shape (n_sequences, seq_length), values in [0, 6]
        """
        b1 = hex_array // 4
        b2 = hex_array % 4
        return np.vectorize(self._phi)(b1, b2).astype(np.int8)


class FourColorDualGuideEncoder(DualGuideEncoder):
    """
    Encoder for dual-guide sequences with four-color chemistry observation mapping.

    Inherits all functionality from DualGuideEncoder for transmitted hex space
    encoding/decoding. Adds encode_to_observed() for mapping to observation space.

    Four-color chemistry:
        Each base has a unique color. Observation is the unordered set of colors
        (bases) present:
            - Same-base: obs 0-3 for (A,A), (T,T), (C,C), (G,G)
            - Different-base: obs 4-9 for unordered pairs
                (A,T)->4, (A,C)->5, (A,G)->6, (T,C)->7, (T,G)->8, (C,G)->9

    The observation alphabet has 10 symbols.
    """

    observed_alphabet_size: int = 10
    chemistry: str = "four_color"

    def __init__(self):
        super().__init__()
        self._phi, _ = get_observation_mapping("four_color")

    def encode_to_observed(self, hex_array: np.ndarray) -> np.ndarray:
        """
        Map integer array from transmitted hex space to four-color observed space.

        Args:
            hex_array: Shape (n_sequences, seq_length), values in [0, 15]

        Returns:
            Shape (n_sequences, seq_length), values in [0, 9]
        """
        b1 = hex_array // 4
        b2 = hex_array % 4
        return np.vectorize(self._phi)(b1, b2).astype(np.int8)


def get_dual_guide_encoder(chemistry: str) -> DualGuideEncoder:
    """
    Factory to get appropriate encoder for chemistry.

    Args:
        chemistry: One of 'two_color', 'three_color', 'four_color'.

    Returns:
        DualGuideEncoder subclass instance with encode_to_observed() method.

    Raises:
        ValueError: If chemistry is not recognized.

    Example:
        >>> encoder = get_dual_guide_encoder('four_color')
        >>> hex_array = encoder.encode(["0F", "A5"])  # Transmitted space
        >>> obs_array = encoder.encode_to_observed(hex_array)  # Observed space
    """
    encoders = {
        "two_color": TwoColorDualGuideEncoder,
        "three_color": ThreeColorDualGuideEncoder,
        "four_color": FourColorDualGuideEncoder,
    }
    if chemistry not in encoders:
        raise ValueError(
            f"Unknown chemistry: '{chemistry}'. "
            f"Expected one of {list(encoders.keys())}"
        )
    return encoders[chemistry]()


def get_alphabet_sizes(chemistry: str) -> Tuple[int, int]:
    """
    Return (transmitted_alphabet_size, observed_alphabet_size) for a chemistry.

    Args:
        chemistry: One of 'dna', 'two_color', 'three_color', 'four_color'.

    Returns:
        Tuple of (transmitted_alphabet_size, observed_alphabet_size).

    Raises:
        ValueError: If chemistry is not recognized.

    Example:
        >>> get_alphabet_sizes('dna')
        (4, 4)
        >>> get_alphabet_sizes('four_color')
        (16, 10)
    """
    sizes = {
        "dna": (4, 4),
        "binary": (2, 2),
        "two_color": (16, 4),
        "three_color": (16, 7),
        "four_color": (16, 10),
    }
    if chemistry not in sizes:
        raise ValueError(
            f"Unknown chemistry: '{chemistry}'. "
            f"Expected one of {list(sizes.keys())}"
        )
    return sizes[chemistry]


def validate_channel_matrix(matrix: np.ndarray, chemistry: str) -> None:
    """
    Validate channel matrix dimensions match chemistry.

    Args:
        matrix: Channel matrix of shape (transmitted_size, observed_size).
        chemistry: One of 'dna', 'two_color', 'three_color', 'four_color'.

    Raises:
        ValueError: If matrix dimensions don't match expected sizes for chemistry.

    Example:
        >>> matrix = np.random.rand(16, 10)
        >>> validate_channel_matrix(matrix, 'four_color')  # OK
        >>> validate_channel_matrix(matrix, 'dna')  # Raises ValueError
    """
    transmitted_size, observed_size = get_alphabet_sizes(chemistry)
    expected_shape = (transmitted_size, observed_size)

    if matrix.ndim != 2:
        raise ValueError(
            f"Channel matrix must be 2D, got {matrix.ndim}D array with shape {matrix.shape}"
        )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Channel matrix shape {matrix.shape} incompatible with chemistry '{chemistry}'. "
            f"Expected {expected_shape} (transmitted={transmitted_size}, observed={observed_size})."
        )


def validate_positional_channel_matrices(
    matrices: np.ndarray,
    chemistry: str,
    seq_length: int | None = None
) -> None:
    """
    Validate positional channel matrices dimensions match chemistry.

    Args:
        matrices: Positional channel matrices of shape (L, transmitted_size, observed_size).
        chemistry: One of 'dna', 'two_color', 'three_color', 'four_color'.
        seq_length: If provided, verify first dimension matches this value.

    Raises:
        ValueError: If matrix dimensions don't match expected sizes for chemistry.

    Example:
        >>> matrices = np.random.rand(8, 16, 10)
        >>> validate_positional_channel_matrices(matrices, 'four_color')  # OK
        >>> validate_positional_channel_matrices(matrices, 'four_color', seq_length=8)  # OK
        >>> validate_positional_channel_matrices(matrices, 'dna')  # Raises ValueError
    """
    transmitted_size, observed_size = get_alphabet_sizes(chemistry)

    if matrices.ndim != 3:
        raise ValueError(
            f"Positional channel matrices must be 3D, got {matrices.ndim}D array "
            f"with shape {matrices.shape}"
        )

    if seq_length is not None and matrices.shape[0] != seq_length:
        raise ValueError(
            f"Positional channel matrices has {matrices.shape[0]} positions, "
            f"expected {seq_length}"
        )

    expected_shape_suffix = (transmitted_size, observed_size)
    if matrices.shape[1:] != expected_shape_suffix:
        raise ValueError(
            f"Positional channel matrices shape {matrices.shape} incompatible with "
            f"chemistry '{chemistry}'. Expected (L, {transmitted_size}, {observed_size}) "
            f"where L is sequence length."
        )


# =============================================================================
# Pairing Strategies
# =============================================================================


class PairingStrategy(ABC):
    """Abstract base class for candidate pairing strategies.

    Subclasses implement different ways of forming candidate pairs from
    single-candidate libraries (e.g., same-group pairs, cross-group pairs).
    """

    @abstractmethod
    def create_pairs(
        self,
        df: pd.DataFrame,
        seq_length: int,
        control_group_name: str,
        max_control_pairs: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Create paired candidate entries from single-candidate DataFrame.

        Args:
            df: Single-candidate DataFrame with Gene, Sequence, Activity score columns.
            seq_length: Truncate sequences to this length.
            control_group_name: Group name for controls.
            max_control_pairs: Maximum number of control pairs to generate.
            rng: Random number generator for sampling.

        Returns:
            DataFrame with columns:
                - Gene: Group name (same for both candidates in pair)
                - Sequence: Encoded dual-guide sequence
                - Activity score: Combined score
                - Candidate1_Sequence: Original sequence of first candidate
                - Candidate2_Sequence: Original sequence of second candidate
                - Candidate1_Score: Score of first candidate
                - Candidate2_Score: Score of second candidate
        """
        pass


class SameGroupPairingStrategy(PairingStrategy):
    """
    Pairs candidates targeting the same group using n-choose-2 combinations.

    For each group with n candidates, generates all (n choose 2) pairs.
    Groups with only one candidate are skipped (cannot form pairs).
    Control pairs are sampled if they exceed max_control_pairs.

    The combined score is the mean of individual candidate scores.

    Example:
        >>> strategy = SameGroupPairingStrategy()
        >>> df_pairs = strategy.create_pairs(df, seq_length=8, ...)
    """

    def create_pairs(
        self,
        df: pd.DataFrame,
        seq_length: int,
        control_group_name: str,
        max_control_pairs: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Create all n-choose-2 pairs within each group."""
        rows = []

        for group_name, group in df.groupby("Gene"):
            group = group.reset_index(drop=True)
            n_candidates = len(group)

            if n_candidates < 2:
                continue  # Can't form pairs with single candidate

            # Generate all n-choose-2 pairs
            all_pairs = list(combinations(range(n_candidates), 2))

            # Sample control pairs if too many
            if group_name == control_group_name and len(all_pairs) > max_control_pairs:
                sampled_idx = rng.choice(
                    len(all_pairs), size=max_control_pairs, replace=False
                )
                all_pairs = [all_pairs[i] for i in sampled_idx]

            # Create entries for each pair
            for idx1, idx2 in all_pairs:
                seq1 = group.loc[idx1, "Sequence"][:seq_length]
                seq2 = group.loc[idx2, "Sequence"][:seq_length]
                score1 = group.loc[idx1, "Activity score"]
                score2 = group.loc[idx2, "Activity score"]

                rows.append(
                    {
                        "Gene": group_name,
                        "Sequence": DualGuideEncoder.encode_pair(seq1, seq2),
                        "Activity score": (score1 + score2) / 2,
                        "Candidate1_Sequence": seq1,
                        "Candidate2_Sequence": seq2,
                        "Candidate1_Score": score1,
                        "Candidate2_Score": score2,
                    }
                )

        return pd.DataFrame(rows)


# =============================================================================
# Factory Classes
# =============================================================================


def _create_pairing_strategy(strategy_name: str) -> PairingStrategy:
    """Create a pairing strategy from its name.

    Args:
        strategy_name: Name of the strategy (e.g., "same_group").

    Returns:
        PairingStrategy instance.

    Raises:
        ValueError: If strategy name is not recognized.
    """
    strategies = {
        "same_group": SameGroupPairingStrategy,
        "same_gene": SameGroupPairingStrategy,  # Alias for backward compatibility
    }

    if strategy_name not in strategies:
        valid = ", ".join(sorted(set(strategies.keys())))
        raise ValueError(
            f"Unknown pairing strategy: '{strategy_name}'. Valid options: {valid}"
        )

    return strategies[strategy_name]()


class DualGuideWeissmanLibraryFactory(WeissmanLibraryFactory):
    """
    Factory for dual-guide candidates from Weissman libraries.

    Extends WeissmanLibraryFactory to generate paired candidate constructs
    using a configurable pairing strategy. The factory:

    1. Loads and filters single-candidate data (inherited from parent)
    2. Applies pairing strategy to create candidate pairs
    3. Encodes pairs using 16-symbol alphabet
    4. Returns CandidatePool with dual-guide metadata

    The resulting CandidatePool can be used with channel-based noise channels
    that capture the physics of dual-guide observation.

    Example:
        >>> strategy = SameGroupPairingStrategy()
        >>> factory = DualGuideWeissmanLibraryFactory(
        ...     csv_path=Path("data/CRISPRi_v2_1.csv"),
        ...     pairing_strategy=strategy,
        ...     seq_rounds=8,
        ...     quota=2,
        ...     num_controls=30,
        ... )
        >>> candidates = factory.create(num_groups=100, seed=42)
    """

    def __init__(
        self,
        csv_path: Path,
        pairing_strategy: PairingStrategy | str,
        seq_rounds: int,
        quota: int,
        num_controls: int,
        min_rank: int = 10,
        control_group_name: str = "negative_control",
        max_control_pairs: int = 1000,
    ):
        """Initialize dual-guide factory.

        Args:
            csv_path: Path to CSV file.
            pairing_strategy: PairingStrategy instance or strategy name string.
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidate PAIRS to select per group.
            num_controls: Number of control PAIRS to select.
            min_rank: Maximum rank to include (filters out low-ranked candidates).
            control_group_name: Group name used for controls in the CSV.
            max_control_pairs: Maximum control pairs to generate before sampling.
        """
        super().__init__(
            csv_path=csv_path,
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
        )

        # Handle string or PairingStrategy instance
        if isinstance(pairing_strategy, str):
            self.pairing_strategy = _create_pairing_strategy(pairing_strategy)
        else:
            self.pairing_strategy = pairing_strategy

        self.max_control_pairs = max_control_pairs

    def create(
        self,
        num_groups: int | None = None,
        seed: int | None = None,
    ) -> CandidatePool:
        """Create dual-guide CandidatePool.

        Args:
            num_groups: If provided, randomly sample this many groups
                (excluding controls from the count).
            seed: Random seed for reproducibility.

        Returns:
            CandidatePool instance with dual-guide encoded sequences.
        """
        rng = np.random.default_rng(seed)

        # Step 1-3: Load, filter by rank, sample groups (reuse parent methods)
        df = self._load_csv()
        df = self._filter_by_rank(df)
        if num_groups is not None:
            df = self._sample_groups(df, num_groups, seed)

        # Store full single-candidate DataFrame for reference
        df_single_candidate = df.copy()

        # Step 4: Create pairs using strategy
        df_pairs = self.pairing_strategy.create_pairs(
            df=df,
            seq_length=self.seq_rounds,
            control_group_name=self.control_group_name,
            max_control_pairs=self.max_control_pairs,
            rng=rng,
        )

        if len(df_pairs) == 0:
            raise ValueError(
                "No candidate pairs could be generated. Ensure groups have at least 2 candidates."
            )

        # Step 5: Add 'Quota' column
        df_pairs = self._add_quotas(df_pairs)

        # Step 6: Fill NaN scores for controls
        df_pairs = self._fill_control_scores(df_pairs)

        # Step 7: Rename columns from CSV format to CandidatePool format
        df_pairs = df_pairs.rename(columns={
            "Gene": "Group",
            "Activity score": "Score",
        })

        # Step 8: Construct CandidatePool
        candidates = CandidatePool.from_dataframe(df_pairs)

        # Add metadata
        candidates.metadata.update(
            {
                "source_file": str(self.csv_path),
                "source_dataframe": df_single_candidate,
                "seq_rounds": self.seq_rounds,
                "quota": self.quota,
                "num_controls": self.num_controls,
                "min_rank": self.min_rank,
                "control_group_name": self.control_group_name,
                "control_groups": {self.control_group_name} if self.num_controls > 0 else set(),
                "num_groups_sampled": num_groups,
                "seed": seed,
                "dual_guide": True,
                "pairing_strategy": type(self.pairing_strategy).__name__,
                "max_control_pairs": self.max_control_pairs,
                "alphabet_size": 16,
                "encoder": "DualGuideEncoder",
            }
        )

        return candidates


class DualGuideWeissmanCRISPRiFactory(DualGuideWeissmanLibraryFactory):
    """Factory with default path for Weissman CRISPRi dual-guide library.

    This is a convenience subclass that provides sensible defaults for
    dual-guide constructs from the CRISPRi library (Horlbeck et al. 2016).

    Example:
        >>> factory = DualGuideWeissmanCRISPRiFactory(
        ...     seq_rounds=8,
        ...     quota=2,
        ...     num_controls=30,
        ... )
        >>> candidates = factory.create(num_groups=100, seed=42)
    """

    DEFAULT_CSV_PATH = (
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "Horlbeck_2016"
        / "CRISPRi_v2_1.csv"
    )
    DEFAULT_DATA_RELPATH = "processed/Horlbeck_2016/CRISPRi_v2_1.csv"

    def __init__(
        self,
        csv_path: Path | None = None,
        pairing_strategy: PairingStrategy | str | None = None,
        seq_rounds: int = 8,
        quota: int = 2,
        num_controls: int = 30,
        min_rank: int = 10,
        control_group_name: str = "negative_control",
        max_control_pairs: int = 1000,
    ):
        """Initialize CRISPRi dual-guide factory.

        Args:
            csv_path: Path to CSV file. If None, uses DEFAULT_CSV_PATH, or
                $DUET_DATA_DIR/DEFAULT_DATA_RELPATH when DUET_DATA_DIR is set.
            pairing_strategy: PairingStrategy instance or name. Default: "same_group".
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidate PAIRS to select per group.
            num_controls: Number of control PAIRS to select.
            min_rank: Maximum rank to include.
            control_group_name: Group name used for controls.
            max_control_pairs: Maximum control pairs to generate before sampling.
        """
        super().__init__(
            csv_path=csv_path or default_data_file(
                self.DEFAULT_DATA_RELPATH, self.DEFAULT_CSV_PATH
            ),
            pairing_strategy=pairing_strategy or SameGroupPairingStrategy(),
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
            max_control_pairs=max_control_pairs,
        )


class DualGuideWeissmanCRISPRaFactory(DualGuideWeissmanLibraryFactory):
    """Factory with default path for Weissman CRISPRa dual-guide library.

    This is a convenience subclass that provides sensible defaults for
    dual-guide constructs from the CRISPRa library (Horlbeck et al. 2016).

    The default table (hCRISPRa-v2, Supplementary Table S5 of Horlbeck et al.
    2016) is not shipped with DUET. Pass csv_path, or build
    data/processed/Horlbeck_2016/CRISPRa.csv yourself with the columns that
    WeissmanLibraryFactory expects.

    Example:
        >>> factory = DualGuideWeissmanCRISPRaFactory(
        ...     seq_rounds=8,
        ...     quota=2,
        ...     num_controls=30,
        ... )
        >>> candidates = factory.create(num_groups=100, seed=42)
    """

    DEFAULT_CSV_PATH = (
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "Horlbeck_2016"
        / "CRISPRa.csv"
    )
    DEFAULT_DATA_RELPATH = "processed/Horlbeck_2016/CRISPRa.csv"

    def __init__(
        self,
        csv_path: Path | None = None,
        pairing_strategy: PairingStrategy | str | None = None,
        seq_rounds: int = 8,
        quota: int = 2,
        num_controls: int = 30,
        min_rank: int = 10,
        control_group_name: str = "negative_control",
        max_control_pairs: int = 1000,
    ):
        """Initialize CRISPRa dual-guide factory.

        Args:
            csv_path: Path to CSV file. If None, uses DEFAULT_CSV_PATH, or
                $DUET_DATA_DIR/DEFAULT_DATA_RELPATH when DUET_DATA_DIR is set.
            pairing_strategy: PairingStrategy instance or name. Default: "same_group".
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidate PAIRS to select per group.
            num_controls: Number of control PAIRS to select.
            min_rank: Maximum rank to include.
            control_group_name: Group name used for controls.
            max_control_pairs: Maximum control pairs to generate before sampling.
        """
        super().__init__(
            csv_path=csv_path or default_data_file(
                self.DEFAULT_DATA_RELPATH, self.DEFAULT_CSV_PATH
            ),
            pairing_strategy=pairing_strategy or SameGroupPairingStrategy(),
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
            max_control_pairs=max_control_pairs,
        )


# =============================================================================
# Single-Guide Channel Matrix Builders
# =============================================================================


def build_single_base_channel(epsilon: float) -> np.ndarray:
    """
    Build a standard 4×4 DNA channel matrix with uniform error probability.

    The channel models independent, identically distributed errors where each
    base has probability epsilon of being misread as a uniformly random base.

    Channel matrix C[b, o] = P(observe o | transmit b):
        - Diagonal: 1 - epsilon + epsilon/4 = 1 - 3*epsilon/4
        - Off-diagonal: epsilon/4

    Args:
        epsilon: Per-base error probability in [0, 1].

    Returns:
        Channel matrix of shape (4, 4) where rows sum to 1.
        Base ordering: A=0, T=1, C=2, G=3.

    Example:
        >>> C = build_single_base_channel(0.04)
        >>> C.shape
        (4, 4)
        >>> np.allclose(C.sum(axis=1), 1.0)
        True
    """
    channel = np.full((4, 4), epsilon / 4)
    np.fill_diagonal(channel, 1 - epsilon + epsilon / 4)
    return channel


def build_positional_single_base_channels(epsilons: np.ndarray) -> np.ndarray:
    """
    Build positional single-base channel matrices from per-position epsilons.

    Args:
        epsilons: Array of shape (L,) with per-position error probabilities.

    Returns:
        Array of shape (L, 4, 4) with per-position channel matrices.

    Example:
        >>> epsilons = np.array([0.01, 0.02, 0.03, 0.04])
        >>> Cs = build_positional_single_base_channels(epsilons)
        >>> Cs.shape
        (4, 4, 4)
    """
    epsilons = np.asarray(epsilons)
    L = len(epsilons)
    channels = np.empty((L, 4, 4))

    for i, eps in enumerate(epsilons):
        channels[i] = build_single_base_channel(eps)

    return channels


# =============================================================================
# Observation Mappings for Dual-Guide Chemistries
# =============================================================================


def get_observation_mapping(
    chemistry: str,
) -> Tuple[callable, int]:
    """
    Get the observation mapping function and alphabet size for a chemistry.

    The observation mapping φ(o1, o2) -> obs_idx maps the pair of observed
    bases (after noise) to the dual-guide observation index.

    Args:
        chemistry: One of "four_color", "two_color_or", "three_color".

    Returns:
        Tuple of (mapping_function, num_observations) where:
            - mapping_function(o1, o2) returns the observation index
            - num_observations is the size of the observation alphabet

    Raises:
        ValueError: If chemistry is not recognized.

    Example:
        >>> phi, n_obs = get_observation_mapping("four_color")
        >>> n_obs
        10
        >>> phi(0, 1)  # A and T observed -> unordered pair
        4
    """
    if chemistry == "four_color":
        return _four_color_observation_mapping(), 10
    elif chemistry == "two_color_or":
        return _two_color_or_observation_mapping(), 4
    elif chemistry == "three_color":
        return _three_color_observation_mapping(), 7
    else:
        valid = ["four_color", "two_color_or", "three_color"]
        raise ValueError(f"Unknown chemistry: '{chemistry}'. Valid options: {valid}")


def _four_color_observation_mapping() -> callable:
    """
    Four-color chemistry: each base has a unique color.

    Observation is the unordered set of colors (bases) present:
        - Same-base: obs 0-3 for (A,A), (T,T), (C,C), (G,G)
        - Different-base: obs 4-9 for unordered pairs
            (A,T)->4, (A,C)->5, (A,G)->6, (T,C)->7, (T,G)->8, (C,G)->9
    """

    def phi(o1: int, o2: int) -> int:
        if o1 == o2:
            return o1
        lo, hi = min(o1, o2), max(o1, o2)
        if lo == 0:
            return 4 + hi - 1
        elif lo == 1:
            return 7 + hi - 2
        else:
            return 9

    return phi


def _two_color_or_observation_mapping() -> callable:
    """
    Two-color OR chemistry (standard Illumina).

    Base encoding: A=(1,1), T=(0,1), C=(1,0), G=(0,0)
    Observation is OR of the two base encodings:
        (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    """
    base_encoding = {
        0: (1, 1),  # A
        1: (0, 1),  # T
        2: (1, 0),  # C
        3: (0, 0),  # G
    }

    def phi(o1: int, o2: int) -> int:
        enc1, enc2 = base_encoding[o1], base_encoding[o2]
        or_ch1 = enc1[0] | enc2[0]
        or_ch2 = enc1[1] | enc2[1]
        return 2 * or_ch1 + or_ch2

    return phi


def _three_color_observation_mapping() -> callable:
    """
    Three-color chemistry: A, T, C have colors 0, 1, 2; G is dark.

    Observation is the set of colors present:
        0: {} (both G)
        1: {0} (A present, no T/C)
        2: {1} (T present, no A/C)
        3: {2} (C present, no A/T)
        4: {0,1} (A and T)
        5: {0,2} (A and C)
        6: {1,2} (T and C)
    """
    base_to_color = {0: 0, 1: 1, 2: 2, 3: None}

    def phi(o1: int, o2: int) -> int:
        c1, c2 = base_to_color[o1], base_to_color[o2]
        colors = set()
        if c1 is not None:
            colors.add(c1)
        if c2 is not None:
            colors.add(c2)

        if len(colors) == 0:
            return 0
        elif len(colors) == 1:
            return 1 + list(colors)[0]
        else:
            colors_sorted = sorted(colors)
            if colors_sorted == [0, 1]:
                return 4
            elif colors_sorted == [0, 2]:
                return 5
            else:
                return 6

    return phi


# =============================================================================
# Dual-Guide Channel Matrix Builders from Single-Guide Channels
# =============================================================================


def build_dual_guide_channel_from_single(
    single_channel: np.ndarray,
    chemistry: str = "four_color",
) -> np.ndarray:
    """
    Build dual-guide channel matrix from a single-guide channel matrix.

    This is the core transformation that computes:
        D[t, φ(o1, o2)] = Σ C[b1, o1] * C[b2, o2]

    where t = 4*b1 + b2 is the transmitted pair encoding, and the sum is over
    all (o1, o2) pairs that map to the same observation under φ.

    Args:
        single_channel: Single-guide channel matrix of shape (4, 4) where
            C[b, o] = P(observe o | transmit b).
        chemistry: Observation chemistry. One of:
            - "four_color": 10 observations (unordered base pairs)
            - "two_color_or": 4 observations (OR of Illumina encoding)
            - "three_color": 7 observations (A/T/C colored, G dark)

    Returns:
        Dual-guide channel matrix of shape (16, num_observations) where
        rows sum to 1.

    Raises:
        ValueError: If single_channel is not shape (4, 4) or chemistry unknown.

    Example:
        >>> C = build_single_base_channel(0.05)
        >>> D = build_dual_guide_channel_from_single(C, "four_color")
        >>> D.shape
        (16, 10)
        >>> np.allclose(D.sum(axis=1), 1.0)
        True
    """
    single_channel = np.asarray(single_channel)
    if single_channel.shape != (4, 4):
        raise ValueError(
            f"single_channel must have shape (4, 4), got {single_channel.shape}"
        )

    phi, num_obs = get_observation_mapping(chemistry)

    # Build 16 x num_obs channel matrix
    dual_channel = np.zeros((16, num_obs))

    for tx_pair in range(16):
        b1, b2 = tx_pair // 4, tx_pair % 4

        for o1 in range(4):
            for o2 in range(4):
                prob = single_channel[b1, o1] * single_channel[b2, o2]
                obs_idx = phi(o1, o2)
                dual_channel[tx_pair, obs_idx] += prob

    return dual_channel


def build_positional_dual_guide_channels_from_single(
    single_channels: np.ndarray,
    chemistry: str = "four_color",
) -> np.ndarray:
    """
    Build positional dual-guide channel matrices from single-guide channels.

    Applies build_dual_guide_channel_from_single at each position.

    Args:
        single_channels: Array of shape (L, 4, 4) with per-position
            single-guide channel matrices.
        chemistry: Observation chemistry (see build_dual_guide_channel_from_single).

    Returns:
        Array of shape (L, 16, num_observations) with per-position
        dual-guide channel matrices.

    Example:
        >>> epsilons = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        >>> Cs = build_positional_single_base_channels(epsilons)
        >>> Ds = build_positional_dual_guide_channels_from_single(Cs, "four_color")
        >>> Ds.shape
        (8, 16, 10)
    """
    single_channels = np.asarray(single_channels)
    if single_channels.ndim != 3 or single_channels.shape[1:] != (4, 4):
        raise ValueError(
            f"single_channels must have shape (L, 4, 4), got {single_channels.shape}"
        )

    L = single_channels.shape[0]
    _, num_obs = get_observation_mapping(chemistry)

    dual_channels = np.empty((L, 16, num_obs))
    for i in range(L):
        dual_channels[i] = build_dual_guide_channel_from_single(
            single_channels[i], chemistry
        )

    return dual_channels


# =============================================================================
# Convenience Functions (Backward Compatible)
# =============================================================================


def build_dual_guide_channel_matrix(
    epsilon: float,
    chemistry: str = "four_color",
) -> np.ndarray:
    """
    Build dual-guide channel matrix from uniform error probability.

    This is a convenience function that combines:
        1. build_single_base_channel(epsilon)
        2. build_dual_guide_channel_from_single(C, chemistry)

    Args:
        epsilon: Per-base error probability. Each base independently has
            probability epsilon of being misread as a uniform random base.
        chemistry: One of "four_color", "two_color_or", "three_color".

    Returns:
        Channel matrix of shape (16, obs_alphabet_size) where rows sum to 1.

    Raises:
        ValueError: If chemistry is not recognized.

    Example:
        >>> D = build_dual_guide_channel_matrix(0.05, "four_color")
        >>> D.shape
        (16, 10)
    """
    single_channel = build_single_base_channel(epsilon)
    return build_dual_guide_channel_from_single(single_channel, chemistry)


def build_positional_dual_guide_channel_matrices(
    epsilons: np.ndarray,
    chemistry: str = "four_color",
) -> np.ndarray:
    """
    Build positional dual-guide channel matrices from per-position epsilons.

    This is a convenience function that combines:
        1. build_positional_single_base_channels(epsilons)
        2. build_positional_dual_guide_channels_from_single(Cs, chemistry)

    Args:
        epsilons: Array of shape (L,) with per-position error probabilities.
        chemistry: One of "four_color", "two_color_or", "three_color".

    Returns:
        Array of shape (L, 16, num_observations) with per-position
        dual-guide channel matrices.

    Example:
        >>> epsilons = np.linspace(0.01, 0.08, 8)
        >>> Ds = build_positional_dual_guide_channel_matrices(epsilons, "four_color")
        >>> Ds.shape
        (8, 16, 10)
    """
    single_channels = build_positional_single_base_channels(epsilons)
    return build_positional_dual_guide_channels_from_single(single_channels, chemistry)