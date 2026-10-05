"""
CandidatePoolFactory: Abstract factory and concrete implementations for
creating CandidatePool from various data sources.

This module provides:
- CandidatePoolFactory: Abstract base class for factories
- WeissmanLibraryFactory: Factory for Weissman lab CRISPRi/CRISPRa libraries
- WeissmanCRISPRiFactory: Convenience subclass with default CRISPRi path
- WeissmanCRISPRaFactory: Convenience subclass with default CRISPRa path
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from duet.benchmark.synthetic_pool import create_synthetic_pool
from duet.candidate_pool import CandidatePool
from duet.data_paths import default_data_file, missing_data_message


class CandidatePoolFactory(ABC):
    """Abstract factory for creating CandidatePool from data sources.

    Factories handle dataset-specific loading, filtering, and preprocessing
    to produce a standardized CandidatePool object.

    Subclasses must implement the create() method to handle their specific
    data format and preprocessing requirements.
    """

    @abstractmethod
    def create(
        self,
        num_groups: int | None = None,
        seed: int | None = None,
    ) -> CandidatePool:
        """Create a CandidatePool instance.

        Args:
            num_groups: If provided, randomly sample this many groups.
                If None, use all groups.
            seed: Random seed for group sampling.

        Returns:
            CandidatePool instance.
        """
        pass


class WeissmanLibraryFactory(CandidatePoolFactory):
    """Factory for Weissman lab CRISPRi/CRISPRa libraries.

    Handles loading from CSV, filtering by candidate rank, truncating sequences,
    and computing per-group selection quotas.

    Expected CSV columns:
        - Gene: Group name (will be mapped to Group internally)
        - Sequence: Full DNA sequence (will be truncated to seq_rounds)
        - Activity score: Candidate activity score (NaN for controls)
        - Rank: Candidate rank within group (for filtering)

    Example:
        >>> factory = WeissmanLibraryFactory(
        ...     csv_path=Path("data/CRISPRi_v2_1.csv"),
        ...     seq_rounds=10,
        ...     quota=2,
        ...     num_controls=200,
        ... )
        >>> candidates = factory.create(num_groups=1000, seed=42)
    """

    def __init__(
        self,
        csv_path: Path,
        seq_rounds: int,
        quota: int,
        num_controls: int,
        min_rank: int = 10,
        control_group_name: str = "negative_control",
    ):
        """Initialize factory.

        Args:
            csv_path: Path to CSV file.
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidates to select per group.
            num_controls: Number of control candidates to select.
            min_rank: Maximum rank to include (filters out low-ranked candidates).
            control_group_name: Group name used for controls in the CSV.
        """
        self.csv_path = Path(csv_path)
        self.seq_rounds = seq_rounds
        self.quota = quota
        self.num_controls = num_controls
        self.min_rank = min_rank
        self.control_group_name = control_group_name

    def create(
        self,
        num_groups: int | None = None,
        seed: int | None = None,
    ) -> CandidatePool:
        """Create CandidatePool from Weissman library CSV.

        Steps:
            1. Load CSV
            2. Filter by min_rank (keeps controls regardless of rank)
            3. Sample groups if num_groups specified
            4. Truncate sequences to seq_rounds
            5. Add 'Quota' column
            6. Fill NaN scores for controls
            7. Construct CandidatePool

        Args:
            num_groups: If provided, randomly sample this many groups
                (excluding controls from the count).
            seed: Random seed for group sampling.

        Returns:
            CandidatePool instance.

        Raises:
            ValueError: If CSV is missing required columns or data is invalid.
            FileNotFoundError: If csv_path does not exist.
        """
        # Step 1: Load CSV
        df = self._load_csv()

        # Step 2: Filter by rank
        df = self._filter_by_rank(df)

        # Step 3: Sample groups if requested
        if num_groups is not None:
            df = self._sample_groups(df, num_groups, seed)

        # Store pre-truncation DataFrame for baselines that need full sequences + Rank
        df_full = df.copy()

        # Step 4: Truncate sequences
        df = self._truncate_sequences(df)

        # Step 5: Add 'Quota' column
        df = self._add_quotas(df)

        # Step 6: Fill NaN scores for controls
        df = self._fill_control_scores(df)

        # Step 7: Rename columns from CSV format to CandidatePool format
        df = df.rename(columns={
            "Gene": "Group",
            "Activity score": "Score",
        })

        # Step 8: Construct CandidatePool
        candidates = CandidatePool.from_dataframe(df)

        # Add metadata about factory configuration
        candidates.metadata.update({
            "source_file": str(self.csv_path),
            "source_dataframe": df_full,  # Full sequences + Rank for baselines
            "seq_rounds": self.seq_rounds,
            "quota": self.quota,
            "num_controls": self.num_controls,
            "min_rank": self.min_rank,
            "control_group_name": self.control_group_name,
            "control_groups": {self.control_group_name} if self.num_controls > 0 else set(),
            "num_groups_sampled": num_groups,
            "seed": seed,
        })

        return candidates

    def _load_csv(self) -> pd.DataFrame:
        """Load and validate CSV structure.

        Returns:
            DataFrame with required columns.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            ValueError: If required columns are missing.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(missing_data_message(self.csv_path))

        df = pd.read_csv(self.csv_path, low_memory=False)

        # Validate required columns for raw CSV (different from CandidatePool requirements)
        required = {"Gene", "Sequence", "Rank", "Activity score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        return df

    def _filter_by_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter candidates by rank, keeping all controls if self.num_controls > 0.

        Args:
            df: DataFrame with Gene and Rank columns.

        Returns:
            Filtered DataFrame.
        """
        is_control = df["Gene"] == self.control_group_name
        is_good_rank = df["Rank"] <= self.min_rank

        return df.loc[(is_control & (self.num_controls > 0)) | is_good_rank].reset_index(drop=True)

    def _sample_groups(
        self,
        df: pd.DataFrame,
        num_groups: int,
        seed: int | None,
    ) -> pd.DataFrame:
        """Randomly sample groups (excludes controls from sampling).

        Controls are always included regardless of sampling.

        Args:
            df: DataFrame with Gene column.
            num_groups: Number of groups to sample (not counting controls).
            seed: Random seed for reproducibility.

        Returns:
            DataFrame containing only sampled groups and controls.

        Raises:
            ValueError: If num_groups exceeds available groups.
        """
        rng = np.random.default_rng(seed)

        # Get all non-control groups
        all_groups = [g for g in df["Gene"].unique() if g != self.control_group_name]

        if num_groups > len(all_groups):
            raise ValueError(
                f"num_groups={num_groups} exceeds available groups ({len(all_groups)})"
            )

        # Sample groups
        sampled_groups = set(rng.choice(all_groups, size=num_groups, replace=False))

        # Keep sampled groups and controls
        keep_mask = df["Gene"].isin(sampled_groups) | (df["Gene"] == self.control_group_name)

        return df.loc[keep_mask].reset_index(drop=True)

    def _truncate_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Truncate sequences to seq_rounds length.

        Args:
            df: DataFrame with Sequence column.

        Returns:
            DataFrame with truncated sequences.
        """
        df = df.copy()
        df["Sequence"] = df["Sequence"].str[: self.seq_rounds]
        return df

    def _add_quotas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'Quota' column based on quota and num_controls.

        Args:
            df: DataFrame with Gene column.

        Returns:
            DataFrame with added 'Quota' column.
        """
        df = df.copy()

        def get_quota(group: str) -> int:
            if group == self.control_group_name:
                return self.num_controls
            return self.quota

        df["Quota"] = df["Gene"].apply(get_quota)
        return df

    def _fill_control_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN scores for controls with 1.0.

        Args:
            df: DataFrame with Gene and Activity score columns.

        Returns:
            DataFrame with filled scores.
        """
        df = df.copy()
        is_control = df["Gene"] == self.control_group_name
        df.loc[is_control, "Activity score"] = df.loc[is_control, "Activity score"].fillna(1.0)
        return df


class WeissmanCRISPRiFactory(WeissmanLibraryFactory):
    """Factory with default path for Weissman CRISPRi v2.1 library.

    This is a convenience subclass that provides sensible defaults for the
    CRISPRi library from Horlbeck et al. 2016.

    Example:
        >>> factory = WeissmanCRISPRiFactory(seq_rounds=10, quota=2, num_controls=200)
        >>> candidates = factory.create(num_groups=1000, seed=42)
    """

    DEFAULT_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "Horlbeck_2016" / "CRISPRi_v2_1.csv"
    DEFAULT_DATA_RELPATH = "processed/Horlbeck_2016/CRISPRi_v2_1.csv"

    def __init__(
        self,
        csv_path: Path | None = None,
        seq_rounds: int = 10,
        quota: int = 2,
        num_controls: int = 200,
        min_rank: int = 10,
        control_group_name: str = "negative_control",
    ):
        """Initialize CRISPRi factory.

        Args:
            csv_path: Path to CSV file. If None, uses DEFAULT_CSV_PATH, or
                $DUET_DATA_DIR/DEFAULT_DATA_RELPATH when DUET_DATA_DIR is set.
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidates to select per group.
            num_controls: Number of control candidates to select.
            min_rank: Maximum rank to include.
            control_group_name: Group name used for controls.
        """
        super().__init__(
            csv_path=csv_path or default_data_file(
                self.DEFAULT_DATA_RELPATH, self.DEFAULT_CSV_PATH
            ),
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
        )


class WeissmanCRISPRaFactory(WeissmanLibraryFactory):
    """Factory with default path for Weissman CRISPRa library.

    This is a convenience subclass that provides sensible defaults for the
    CRISPRa library from Horlbeck et al. 2016.

    The default table (hCRISPRa-v2, Supplementary Table S5 of Horlbeck et al.
    2016) is not shipped with DUET. Pass csv_path, or build
    data/processed/Horlbeck_2016/CRISPRa.csv yourself with the columns that
    WeissmanLibraryFactory expects.

    Example:
        >>> factory = WeissmanCRISPRaFactory(seq_rounds=10, quota=2, num_controls=200)
        >>> candidates = factory.create(num_groups=1000, seed=42)
    """

    DEFAULT_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "Horlbeck_2016" / "CRISPRa.csv"
    DEFAULT_DATA_RELPATH = "processed/Horlbeck_2016/CRISPRa.csv"

    def __init__(
        self,
        csv_path: Path | None = None,
        seq_rounds: int = 10,
        quota: int = 2,
        num_controls: int = 200,
        min_rank: int = 10,
        control_group_name: str = "negative_control",
    ):
        """Initialize CRISPRa factory.

        Args:
            csv_path: Path to CSV file. If None, uses DEFAULT_CSV_PATH, or
                $DUET_DATA_DIR/DEFAULT_DATA_RELPATH when DUET_DATA_DIR is set.
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidates to select per group.
            num_controls: Number of control candidates to select.
            min_rank: Maximum rank to include.
            control_group_name: Group name used for controls.
        """
        super().__init__(
            csv_path=csv_path or default_data_file(
                self.DEFAULT_DATA_RELPATH, self.DEFAULT_CSV_PATH
            ),
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
        )


# =============================================================================
# Convenience Factory Function
# =============================================================================

_INT_TO_DNA = np.array(list("ATCG"))


def _int_array_to_dna_strings(sequences: np.ndarray) -> list[str]:
    """Convert integer-encoded sequences to DNA strings.

    Mapping matches DNAEncoder.alphabet = "ATCG" (A→0, T→1, C→2, G→3).
    """
    return ["".join(_INT_TO_DNA[seq]) for seq in sequences]


def create_pool_from_source(
    source: str,
    seq_rounds: int,
    quota: int,
    num_controls: int,
    min_rank: int = 10,
    num_groups: int | None = None,
    seed: int | None = None,
    control_group_name: str = "negative_control",
    # Dual-guide specific parameters
    pairing_strategy: str | None = None,
    max_control_pairs: int = 1000,
    # CRISPick-specific parameters
    csv_path: str | Path | None = None,
    control_quotas: Dict[str, int] | None = None,
    score_method: str | None = None,
    # Synthetic-specific parameters
    candidates_per_group: int | None = None,
    alphabet_size: int = 4,
) -> CandidatePool:
    """Create CandidatePool from source specification.

    This is a convenience function that creates a CandidatePool instance
    from a source string, which can be either a keyword for a built-in
    library or a path to a CSV file.

    Args:
        source: One of:
            - "WeissmanCRISPRi": Use default CRISPRi library (Horlbeck 2016)
            - "WeissmanCRISPRa": Use default CRISPRa library (Horlbeck 2016)
            - "DualGuideWeissmanCRISPRi": Dual-guide pairs from CRISPRi
            - "DualGuideWeissmanCRISPRa": Dual-guide pairs from CRISPRa
            - "CRISPick": Genome-scale CRISPick library
            - "synthetic": Randomly generated synthetic pool (for benchmarking)
            - Path to CSV file with required columns
        seq_rounds: Number of sequencing rounds (sequence prefix length).
        quota: Number of candidates (or candidate PAIRS for dual-guide) to select per group.
        num_controls: Number of control candidates (or PAIRS) to select.
        min_rank: Maximum rank to include (default: 10).
        num_groups: If provided, randomly sample this many groups.
            If None, use all groups.
        seed: Random seed for group sampling.
        control_group_name: Group name used for controls (default: "negative_control").
        pairing_strategy: For dual-guide sources, the pairing strategy name
            (default: "same_gene"). Options: "same_gene".
        max_control_pairs: For dual-guide sources, maximum control pairs to generate
            before sampling (default: 1000).
        csv_path: Optional override for CSV file location (used by CRISPick source).
        control_quotas: Per-control-group quotas for CRISPick, e.g.
            {"NO_SITE": 100, "ONE_SITE_INTERGENIC": 900}.
        score_method: Scoring method for CRISPick source. One of "picking_round"
            (default) or "normalized_rank". If None, uses factory default.
        candidates_per_group: Number of candidates per group (required for
            source="synthetic").
        alphabet_size: Alphabet size for synthetic sequences (default: 4).

    Returns:
        CandidatePool instance.

    Raises:
        ValueError: If source is not recognized and path does not exist.
        FileNotFoundError: If source is a path that does not exist.

    Example:
        >>> # Using built-in single-guide library
        >>> candidates = create_pool_from_source(
        ...     source="WeissmanCRISPRi",
        ...     seq_rounds=10,
        ...     quota=2,
        ...     num_controls=200,
        ...     num_groups=1000,
        ...     seed=42,
        ... )

        >>> # Using dual-guide library
        >>> candidates = create_pool_from_source(
        ...     source="DualGuideWeissmanCRISPRi",
        ...     seq_rounds=8,
        ...     quota=2,
        ...     num_controls=30,
        ...     num_groups=100,
        ...     pairing_strategy="same_gene",
        ...     seed=42,
        ... )

        >>> # Using custom CSV
        >>> candidates = create_pool_from_source(
        ...     source="/path/to/custom_library.csv",
        ...     seq_rounds=5,
        ...     quota=3,
        ...     num_controls=100,
        ... )
    """
    # Single-guide sources
    if source == "WeissmanCRISPRi":
        factory = WeissmanCRISPRiFactory(
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
        )
        return factory.create(num_groups=num_groups, seed=seed)

    elif source == "WeissmanCRISPRa":
        factory = WeissmanCRISPRaFactory(
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
        )
        return factory.create(num_groups=num_groups, seed=seed)

    # Dual-guide sources
    elif source == "DualGuideWeissmanCRISPRi":
        from duet.dual_guide_factory import DualGuideWeissmanCRISPRiFactory

        factory = DualGuideWeissmanCRISPRiFactory(
            pairing_strategy=pairing_strategy or "same_gene",
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
            max_control_pairs=max_control_pairs,
        )
        return factory.create(num_groups=num_groups, seed=seed)

    elif source == "DualGuideWeissmanCRISPRa":
        from duet.dual_guide_factory import DualGuideWeissmanCRISPRaFactory

        factory = DualGuideWeissmanCRISPRaFactory(
            pairing_strategy=pairing_strategy or "same_gene",
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
            max_control_pairs=max_control_pairs,
        )
        return factory.create(num_groups=num_groups, seed=seed)

    elif source == "CRISPick":
        from duet.crispick_factory import CRISPickFactory

        kwargs = dict(
            csv_path=csv_path,
            seq_rounds=seq_rounds,
            quota=quota,
            min_rank=min_rank,
            control_quotas=control_quotas,
        )
        if score_method is not None:
            kwargs["score_method"] = score_method
        factory = CRISPickFactory(**kwargs)
        return factory.create(num_groups=num_groups, seed=seed)

    elif source == "synthetic":
        if candidates_per_group is None:
            raise ValueError(
                "source='synthetic' requires candidates_per_group to be specified"
            )
        if num_groups is None:
            raise ValueError(
                "source='synthetic' requires num_groups to be specified"
            )
        pool = create_synthetic_pool(
            num_groups=num_groups,
            candidates_per_group=candidates_per_group,
            seq_length=seq_rounds,
            alphabet_size=alphabet_size,
            quota=quota,
            seed=seed,
        )
        dna_sequences = _int_array_to_dna_strings(pool.sequences)
        return CandidatePool(
            sequences=dna_sequences,
            group_to_candidates=pool.group_to_candidates,
            quotas=pool.quotas,
            scores=pool.scores,
            metadata=pool.metadata,
        )

    else:
        # Treat as path to CSV
        csv_path_resolved = Path(source)
        if not csv_path_resolved.exists():
            raise FileNotFoundError(
                f"Source '{source}' is not a recognized keyword "
                f"(WeissmanCRISPRi, WeissmanCRISPRa, DualGuideWeissmanCRISPRi, "
                f"DualGuideWeissmanCRISPRa, CRISPick, synthetic) and file does not exist"
            )
        factory = WeissmanLibraryFactory(
            csv_path=csv_path_resolved,
            seq_rounds=seq_rounds,
            quota=quota,
            num_controls=num_controls,
            min_rank=min_rank,
            control_group_name=control_group_name,
        )
        return factory.create(num_groups=num_groups, seed=seed)
