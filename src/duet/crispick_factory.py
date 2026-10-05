"""
CRISPickFactory: Factory for creating CandidatePool from CRISPick libraries.

CRISPick is a genome-scale CRISPR guide design tool. This factory handles
its CSV format, which differs from Weissman libraries in column names,
score semantics, and control group structure.

Key differences from Weissman:
- Column names: Target Gene ID / sgRNA Sequence / Pick Order / Picking Round
- Score: No activity score; uses Picking Round as proxy via linear inversion
- Controls: Two groups (NO_SITE, ONE_SITE_INTERGENIC) with separate quotas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

from duet.candidate_pool import CandidatePool
from duet.candidate_pool_factory import CandidatePoolFactory
from duet.data_paths import SOURCE_DATA_DIR, default_data_file, missing_data_message

logger = logging.getLogger(__name__)

# The default table is not shipped with DUET (Broad Institute GPP terms of
# use). Build it with scripts/data_processing/build_crispick_candidates.py.
DEFAULT_DATA_RELPATH = "processed/CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv"
DEFAULT_CSV_PATH = SOURCE_DATA_DIR / DEFAULT_DATA_RELPATH


class CRISPickFactory(CandidatePoolFactory):
    """Factory for CRISPick genome-scale guide libraries.

    Expected CSV columns:
        - Target Gene ID: Ensembl gene ID (mapped to Gene internally)
        - sgRNA Sequence: Guide sequence (mapped to Sequence)
        - Pick Order: Candidate rank within gene (mapped to Rank)
        - Picking Round: Round in which guide was picked (used to derive score)

    Control rows (NO_SITE, ONE_SITE_INTERGENIC) have NaN for Pick Order and
    Picking Round. They are kept as separate groups in the CandidatePool,
    each with their own quota from control_quotas.

    Example:
        >>> factory = CRISPickFactory(seq_rounds=10, quota=4, min_rank=10)
        >>> candidates = factory.create(num_groups=1000, seed=42)
    """

    CRISPICK_COLUMNS = {
        "Target Gene ID",
        "sgRNA Sequence",
        "Pick Order",
        "Picking Round",
    }

    VALID_SCORE_METHODS = ("picking_round", "normalized_rank")

    def __init__(
        self,
        csv_path: Path | str | None = None,
        seq_rounds: int = 10,
        quota: int = 4,
        min_rank: int = 10,
        score_method: str = "picking_round",
        control_quotas: Dict[str, int] | None = None,
    ):
        """Initialize CRISPick factory.

        Args:
            csv_path: Path to CRISPick CSV. If None, uses DEFAULT_CSV_PATH
                (data/processed/CRISPick/... in a source checkout), or
                $DUET_DATA_DIR/DEFAULT_DATA_RELPATH when DUET_DATA_DIR is set.
                The default table is not shipped with DUET; build it with
                scripts/data_processing/build_crispick_candidates.py.
            seq_rounds: Number of sequencing rounds (sequence prefix length).
            quota: Number of candidates to select per gene group.
            min_rank: Maximum Pick Order to include (filters low-ranked guides).
            score_method: Method to compute activity scores. One of
                'picking_round' (linear inversion of picking round) or
                'normalized_rank' (linear inversion of pick order rank).
            control_quotas: Per-control-group quotas, e.g.
                {"NO_SITE": 100, "ONE_SITE_INTERGENIC": 900}.
                Defaults to {"NO_SITE": 100, "ONE_SITE_INTERGENIC": 900}.
        """
        self.csv_path = (
            Path(csv_path)
            if csv_path
            else default_data_file(DEFAULT_DATA_RELPATH, DEFAULT_CSV_PATH)
        )
        self.seq_rounds = seq_rounds
        self.quota = quota
        self.min_rank = min_rank
        self.control_quotas = control_quotas or {
            "NO_SITE": 100,
            "ONE_SITE_INTERGENIC": 900,
        }
        if score_method not in self.VALID_SCORE_METHODS:
            raise ValueError(
                f"score_method must be one of {self.VALID_SCORE_METHODS}, "
                f"got '{score_method}'"
            )
        self.score_method = score_method

    @property
    def _control_group_names(self) -> Set[str]:
        return set(self.control_quotas.keys())

    def create(
        self,
        num_groups: int | None = None,
        seed: int | None = None,
    ) -> CandidatePool:
        """Create CandidatePool from CRISPick CSV.

        Steps:
            1. Load CSV, validate columns, rename to internal standard
            2. Convert Picking Round to activity score via linear inversion
            3. Filter by rank (keep all controls)
            4. Drop genes with too few candidates after filtering
            5. Sample groups if num_groups specified (controls always included)
            6. Store df_full for Feldman compatibility
            7. Truncate sequences to seq_rounds
            8. Add per-group quotas
            9. Fill NaN scores for controls
            10. Rename to CandidatePool columns and build

        Args:
            num_groups: If provided, randomly sample this many gene groups
                (controls are always included, not counted toward this number).
            seed: Random seed for group sampling.

        Returns:
            CandidatePool instance.
        """
        # Step 1: Load and rename
        logger.info(f"Loading CRISPick CSV: {self.csv_path}")
        df = self._load_csv()
        control_mask = df["Gene"].isin(self._control_group_names)
        n_genes = df.loc[~control_mask, "Gene"].nunique()
        n_ctrl = df.loc[control_mask, "Gene"].nunique()
        logger.info(f"Loaded {len(df)} rows, {n_genes} genes + {n_ctrl} control groups")

        # Step 2: Compute score from Picking Round
        df = self._compute_scores(df)

        # Step 3: Filter by rank
        df = self._filter_by_rank(df)
        logger.info(f"After rank filter (Rank <= {self.min_rank}): {len(df)} rows")

        # Step 4: Drop genes with fewer candidates than quota
        df = self._drop_small_groups(df)

        # Step 5: Sample groups if requested
        if num_groups is not None:
            df = self._sample_groups(df, num_groups, seed)
            logger.info(f"Sampled {num_groups} genes: {len(df)} rows remaining")

        # Step 6: Store pre-truncation copy for Feldman compatibility
        df_full = self._make_feldman_df(df)

        # Step 7: Truncate sequences
        df = self._truncate_sequences(df)

        # Step 8: Add quotas
        df = self._add_quotas(df)

        # Step 9: Fill NaN scores for controls
        df = self._fill_control_scores(df)

        # Step 10: Rename to CandidatePool format and build
        df = df.rename(columns={
            "Gene": "Group",
            "Activity score": "Score",
        })

        candidates = CandidatePool.from_dataframe(df)

        candidates.metadata.update({
            "source_file": str(self.csv_path),
            "source_dataframe": df_full,
            "seq_rounds": self.seq_rounds,
            "quota": self.quota,
            "min_rank": self.min_rank,
            "score_method": self.score_method,
            "control_quotas": self.control_quotas,
            "control_groups": set(self.control_quotas.keys()) if self.control_quotas else set(),
            "num_groups_sampled": num_groups,
            "seed": seed,
        })

        return candidates

    def _load_csv(self) -> pd.DataFrame:
        """Load CSV, validate CRISPick columns, and rename to internal standard."""
        if not self.csv_path.exists():
            raise FileNotFoundError(missing_data_message(self.csv_path))

        df = pd.read_csv(self.csv_path, low_memory=False)

        missing = self.CRISPICK_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required CRISPick columns: {sorted(missing)}"
            )

        df = df.rename(columns={
            "Target Gene ID": "Gene",
            "sgRNA Sequence": "Sequence",
            "Pick Order": "Rank",
        })

        return df

    def _compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute activity score based on configured score_method.

        For 'picking_round': score = (max_round + 1 - picking_round) / max_round
        For 'normalized_rank': score = (min_rank - rank) / (min_rank - 1)

        Controls (NaN values) get NaN here; filled later by _fill_control_scores.
        """
        df = df.copy()
        if self.score_method == "picking_round":
            max_round = df["Picking Round"].max()
            df["Activity score"] = (max_round + 1 - df["Picking Round"]) / max_round
        else:  # normalized_rank
            if self.min_rank <= 1:
                df["Activity score"] = 1.0
            else:
                df["Activity score"] = (self.min_rank - df["Rank"]) / (self.min_rank - 1)
        return df

    def _filter_by_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by Pick Order rank, keeping all controls."""
        is_control = df["Gene"].isin(self._control_group_names)
        is_good_rank = df["Rank"] <= self.min_rank
        return df.loc[is_control | is_good_rank].reset_index(drop=True)

    def _drop_small_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop gene groups with fewer candidates than quota after filtering."""
        control_names = self._control_group_names
        gene_mask = ~df["Gene"].isin(control_names)
        group_sizes = df.loc[gene_mask].groupby("Gene").size()
        small_groups = group_sizes[group_sizes < self.quota].index
        if len(small_groups) > 0:
            logger.warning(
                f"Dropping {len(small_groups)} genes with fewer than "
                f"{self.quota} candidates after rank filtering"
            )
            df = df.loc[~df["Gene"].isin(small_groups)].reset_index(drop=True)
        return df

    def _sample_groups(
        self,
        df: pd.DataFrame,
        num_groups: int,
        seed: int | None,
    ) -> pd.DataFrame:
        """Randomly sample gene groups (controls always included)."""
        rng = np.random.default_rng(seed)

        all_genes = [
            g for g in df["Gene"].unique()
            if g not in self._control_group_names
        ]

        if num_groups > len(all_genes):
            raise ValueError(
                f"num_groups={num_groups} exceeds available genes ({len(all_genes)})"
            )

        sampled = set(rng.choice(all_genes, size=num_groups, replace=False))
        keep_mask = df["Gene"].isin(sampled) | df["Gene"].isin(self._control_group_names)
        return df.loc[keep_mask].reset_index(drop=True)

    def _make_feldman_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pre-truncation DataFrame for Feldman compatibility.

        Returns DataFrame with Gene, Sequence, Rank columns. Control groups
        keep their names (NO_SITE, ONE_SITE_INTERGENIC), so Feldman et al.
        gets each group's own quota from the pool's quotas. CRISPick gives
        controls no Pick Order; their Rank is set to 1. NaN would sort after
        every gene guide in Feldman's ED=1 prefix de-duplication, so controls
        would lose every prefix tie; 1 matches the controls' activity score
        of 1.0 in the pool.
        """
        df_full = df[["Gene", "Sequence", "Rank"]].copy()
        is_control = df_full["Gene"].isin(self._control_group_names)
        df_full.loc[is_control, "Rank"] = df_full.loc[is_control, "Rank"].fillna(1)
        return df_full

    def _truncate_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Sequence"] = df["Sequence"].str[: self.seq_rounds]
        return df

    def _add_quotas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        control_names = self._control_group_names

        def get_quota(gene: str) -> int:
            if gene in control_names:
                return self.control_quotas[gene]
            return self.quota

        df["Quota"] = df["Gene"].apply(get_quota)
        return df

    def _fill_control_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        is_control = df["Gene"].isin(self._control_group_names)
        df.loc[is_control, "Activity score"] = df.loc[
            is_control, "Activity score"
        ].fillna(1.0)
        return df
