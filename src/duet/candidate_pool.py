"""
CandidatePool: Container for candidate data ready for optimization.

This module provides the CandidatePool class which holds preprocessed candidate
data with all necessary structures for running DUET optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd


# Type alias for sequences: can be list of strings or numpy array
SequenceList = Union[List[str], np.ndarray]


@dataclass
class CandidatePool:
    """Container for candidate data ready for optimization.

    Build one from a candidate table with :meth:`from_table` (public API) and
    summarize it with :meth:`describe`; the other members are the advanced
    layer (docs/adr/0003-public-api.md).

    This class holds preprocessed candidate data with all necessary structures
    for running DUET optimization. It expects data that has already been
    filtered, truncated, and annotated with selection counts.

    Attributes:
        sequences: List of sequences (DNA strings or encoded strings) or
            numpy array of integer-encoded sequences. For dual-guide mode,
            sequences are encoded using DualGuideEncoder (16-symbol alphabet).
        group_to_candidates: Mapping from group name to list of candidate
            indices that belong to that group. A candidate may appear in
            more than one group (multi-group membership).
        quotas: How many candidates to select per group.
        scores: Activity scores indexed by candidate index. Optional;
            defaults to all-ones when None is passed.
        metadata: Optional metadata (source file, parameters, etc.).

    Example:
        >>> df = pd.DataFrame({
        ...     "Group": ["GroupA", "GroupA", "GroupB", "GroupB"],
        ...     "Sequence": ["ATCG", "GCTA", "TTAA", "CCGG"],
        ...     "Score": [0.9, 0.8, 0.95, 0.85],
        ...     "Quota": [1, 1, 1, 1],
        ... })
        >>> candidates = CandidatePool.from_dataframe(df)
        >>> init = candidates.sample_initial_selection(strategy="random", seed=42)
    """

    sequences: SequenceList
    group_to_candidates: Dict[str, List[int]]
    quotas: Dict[str, int]
    scores: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate internal consistency after initialization."""
        if self.scores is None:
            self.scores = np.ones(len(self.sequences))
        self._validate()

    def _validate(self):
        """Validate internal consistency of the data structures.

        Raises:
            ValueError: If any validation check fails.
        """
        # Check sequences is non-empty
        n_seqs = len(self.sequences)
        if n_seqs == 0:
            raise ValueError("sequences cannot be empty")

        # Check sequences have consistent length
        if isinstance(self.sequences, np.ndarray):
            # For numpy arrays, shape[1] is the sequence length
            if self.sequences.ndim != 2:
                raise ValueError(
                    f"sequences array must be 2D, got {self.sequences.ndim}D"
                )
            seq_lengths = {self.sequences.shape[1]}
        else:
            # For list of strings
            seq_lengths = {len(s) for s in self.sequences}

        if len(seq_lengths) > 1:
            raise ValueError(
                f"All sequences must have the same length. Found lengths: {sorted(seq_lengths)}"
            )

        # Check scores length matches sequences
        if len(self.scores) != n_seqs:
            raise ValueError(
                f"scores length ({len(self.scores)}) must match "
                f"sequences length ({n_seqs})"
            )

        # group_to_candidates must cover every quota key and vice versa
        quota_keys = set(self.quotas.keys())
        gtc_keys = set(self.group_to_candidates.keys())
        if quota_keys != gtc_keys:
            raise ValueError(
                f"quotas keys {sorted(quota_keys)} must match "
                f"group_to_candidates keys {sorted(gtc_keys)}"
            )

        # Every candidate index must be valid; every candidate must be reachable
        seen = set()
        for group, candidates in self.group_to_candidates.items():
            for c in candidates:
                if not (0 <= c < n_seqs):
                    raise ValueError(
                        f"group_to_candidates[{group!r}] contains out-of-range index {c} "
                        f"(pool_size={n_seqs})"
                    )
                seen.add(c)
        unreachable = set(range(n_seqs)) - seen
        if unreachable:
            raise ValueError(
                f"candidates {sorted(unreachable)[:5]}... are in no group (pool_size={n_seqs})"
            )

        # Check quota <= available candidates per group, and quota >= 0
        for group, quota in self.quotas.items():
            available = len(self.group_to_candidates[group])
            if quota > available:
                raise ValueError(
                    f"Group '{group}' requires {quota} selections but only has {available} candidates"
                )
            if quota < 0:
                raise ValueError(f"Group '{group}' has negative quota: {quota}")

        # Check scores has no NaN values
        if np.isnan(self.scores).any():
            nan_indices = np.where(np.isnan(self.scores))[0]
            raise ValueError(
                f"scores contains NaN values at indices: {nan_indices.tolist()}"
            )

    @property
    def seq_length(self) -> int:
        """Length of candidate sequences."""
        if isinstance(self.sequences, np.ndarray):
            if self.sequences.size == 0:
                return 0
            return self.sequences.shape[1]
        else:
            if not self.sequences:
                return 0
            return len(self.sequences[0])

    @property
    def pool_size(self) -> int:
        """Total number of candidates in the pool."""
        return len(self.sequences)

    @property
    def num_groups(self) -> int:
        """Number of groups (including controls if present)."""
        return len(self.group_to_candidates)

    @cached_property
    def candidate_to_groups(self) -> Dict[int, frozenset[str]]:
        """Inverse map: candidate index -> frozenset of groups it belongs to.

        Lazy/debug-only: at MERFISH scale (~100K candidates x ~11 groups each)
        this is ~150-250 MB of Python objects that the hot path never reads.
        Use group_to_candidates[group] for enumeration; use codeword_to_group[i]
        for the position's group identity.
        """
        result: Dict[int, set] = {}
        for group, cands in self.group_to_candidates.items():
            for c in cands:
                result.setdefault(c, set()).add(group)
        return {c: frozenset(gs) for c, gs in result.items()}

    @cached_property
    def codeword_to_group(self) -> List[str]:
        """Length-sum(quotas.values()) list: codeword position -> group name.

        Derived from quotas.items() in insertion order; each group replicated
        quotas[g] times. Fixed for the run; only S[i] changes during optimization.
        """
        out: List[str] = []
        for group, quota in self.quotas.items():
            out.extend([group] * quota)
        return out

    @cached_property
    def unique_sequences(self) -> SequenceList:
        """De-duplicated sequences in first-occurrence order, type-preserving.

        For sequence-based PEP dedup: the PEP matrix is indexed by these
        U <= pool_size rows. Returns the same type as ``self.sequences`` (an
        ndarray of unique rows for ndarray input, a list[str] for list input),
        so the PEP producer's library type and cache key are unchanged when
        there are no duplicates.
        """
        if isinstance(self.sequences, np.ndarray):
            seen: Dict[tuple, int] = {}
            first_rows: List[int] = []
            for i, row in enumerate(self.sequences):
                key = tuple(int(x) for x in row)
                if key not in seen:
                    seen[key] = len(first_rows)
                    first_rows.append(i)
            return self.sequences[first_rows]
        seen_strs: Dict[str, int] = {}
        out_strs: List[str] = []
        for s in self.sequences:
            if s not in seen_strs:
                seen_strs[s] = len(out_strs)
                out_strs.append(s)
        return out_strs

    @cached_property
    def candidate_to_sequence_idx(self) -> np.ndarray:
        """Shape (pool_size,), int32. candidate_to_sequence_idx[c] = u
        where unique_sequences[u] == sequences[c].

        First-occurrence ordering matches ``unique_sequences`` by construction
        (both iterate ``self.sequences`` in order).
        """
        if isinstance(self.sequences, np.ndarray):
            key_to_u: Dict[tuple, int] = {}
            out = np.empty(len(self.sequences), dtype=np.int32)
            for c, row in enumerate(self.sequences):
                key = tuple(int(x) for x in row)
                if key not in key_to_u:
                    key_to_u[key] = len(key_to_u)
                out[c] = key_to_u[key]
            return out
        str_to_u = {s: u for u, s in enumerate(self.unique_sequences)}
        return np.array(
            [str_to_u[s] for s in self.sequences], dtype=np.int32,
        )

    @property
    def total_selections(self) -> int:
        """Total number of candidates to select (sum of quotas)."""
        return sum(self.quotas.values())

    def sample_initial_selection(
        self,
        strategy: Literal["random", "best_score"] = "random",
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate initial candidate indices for optimization.

        Walks codewords in codeword_to_group order; picks one candidate from the
        codeword's group, preferring candidates not yet used. Falls back to the
        full group if duplicate-free choice is unavailable (invariants 1-2 still
        hold; only the soft duplicate-free preference is violated, and the
        diagonal duplicate penalty handles it at the objective level).

        Args:
            strategy: "random" or "best_score".
            seed: Random seed (used by "random").

        Returns:
            Array of candidate indices with length = total_selections.
        """
        if strategy not in ("random", "best_score"):
            raise ValueError(f"Unknown strategy: {strategy}")

        rng = np.random.default_rng(seed)
        S = np.empty(len(self.codeword_to_group), dtype=np.int64)
        used: set = set()

        for i, group in enumerate(self.codeword_to_group):
            candidates = self.group_to_candidates[group]
            eligible = [c for c in candidates if c not in used]
            if not eligible:
                eligible = list(candidates)

            if strategy == "random":
                S[i] = int(rng.choice(eligible))
            else:  # best_score
                best = max(eligible, key=lambda c: self.scores[c])
                S[i] = int(best)

            used.add(int(S[i]))

        return S

    def to_optimizer_kwargs(self) -> Dict[str, Any]:
        """Return kwargs suitable for DUET-family optimizers.

        Returns:
            Dict with keys: group_to_candidates, scores.
        """
        return {
            "group_to_candidates": self.group_to_candidates,
            "scores": self.scores,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert back to DataFrame representation.

        Useful for saving/inspection. Includes Group, Sequence,
        Score, Quota columns.

        Returns:
            DataFrame with the same structure as expected by from_dataframe.
        """
        rows = []

        # Handle both list and numpy array sequences
        if isinstance(self.sequences, np.ndarray):
            # Convert numpy array rows to strings for DataFrame
            sequences_list = ["".join(str(x) for x in row) for row in self.sequences]
        else:
            sequences_list = self.sequences

        pairs = [
            (c, g)
            for g, cs in self.group_to_candidates.items()
            for c in cs
        ]
        for idx, group in pairs:
            rows.append(
                {
                    "Group": group,
                    "Sequence": sequences_list[idx],
                    "Score": self.scores[idx],
                    "Quota": self.quotas[group],
                }
            )
        return pd.DataFrame(rows)

    def get_sequences_as_array(self, alphabet_size: int = 4) -> np.ndarray:
        """Get sequences as integer-encoded numpy array.

        Useful for passing to CodebookEvaluator or other numerical operations.

        For DNA sequences (alphabet_size=4), uses A=0, T=1, C=2, G=3.
        For dual-guide encoded sequences (alphabet_size=16), uses hex decoding.

        Args:
            alphabet_size: Size of the alphabet. 4 for DNA, 16 for dual-guide.

        Returns:
            Array of shape (pool_size, seq_length) with integer-encoded sequences.
        """
        if isinstance(self.sequences, np.ndarray):
            return self.sequences

        if alphabet_size == 4:
            # DNA encoding
            base_to_idx = {"A": 0, "T": 1, "C": 2, "G": 3}
            result = np.empty((self.pool_size, self.seq_length), dtype=np.int8)
            for i, seq in enumerate(self.sequences):
                for j, base in enumerate(seq):
                    result[i, j] = base_to_idx[base]
            return result
        elif alphabet_size == 2:
            # Binary encoding: '0' -> 0, '1' -> 1
            result = np.empty((self.pool_size, self.seq_length), dtype=np.int8)
            for i, seq in enumerate(self.sequences):
                for j, char in enumerate(seq):
                    result[i, j] = int(char)
            return result
        elif alphabet_size == 16:
            # Dual-guide hex encoding
            char_to_idx = {format(i, "X"): i for i in range(16)}
            result = np.empty((self.pool_size, self.seq_length), dtype=np.int8)
            for i, seq in enumerate(self.sequences):
                for j, char in enumerate(seq):
                    result[i, j] = char_to_idx[char]
            return result
        else:
            raise ValueError(f"Unsupported alphabet_size: {alphabet_size}")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        seq_length: int | None = None,
    ) -> "CandidatePool":
        """Construct from DataFrame with required columns.

        Required columns:
            - Group: Group name (str)
            - Sequence: DNA sequence (str)
            - Score: Numeric score (float)
            - Quota: Candidates to select for this group (int)

        Args:
            df: DataFrame with required columns.
            seq_length: If provided, truncate sequences to this length.
                If None, use sequences as-is.

        Returns:
            CandidatePool instance.

        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        # Validate required columns
        required_columns = {"Group", "Sequence", "Score", "Quota"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

        # Make a copy to avoid modifying the original
        df = df.copy().reset_index(drop=True)

        # Truncate sequences if seq_length is specified
        if seq_length is not None:
            df["Sequence"] = df["Sequence"].str[:seq_length]

        # One candidate per DataFrame row (row order == candidate index, since
        # df was reset_index above). Two rows whose sequences coincide — after
        # truncation, or a multi-targeting reagent listed once per gene — become
        # DISTINCT candidates that happen to share a codeword. Sequence-level PEP
        # dedup is handled downstream (unique_sequences /
        # candidate_to_sequence_idx); a reagent targeting several genes simply
        # appears as one candidate per group. Each row keeps its own Score, so
        # differing scores for a shared sequence are not a conflict.
        sequences: List[str] = df["Sequence"].tolist()
        scores = df["Score"].to_numpy(dtype=float)
        group_to_candidates: Dict[str, List[int]] = {}
        for idx, group in enumerate(df["Group"].tolist()):
            group_to_candidates.setdefault(group, []).append(idx)

        # Build quotas (one entry per Group; reject within-group mismatches).
        quotas: Dict[str, int] = {}
        for group in group_to_candidates:
            group_quotas = df.loc[df["Group"] == group, "Quota"].unique()
            if len(group_quotas) > 1:
                raise ValueError(
                    f"Group {group!r} has inconsistent 'Quota' values: "
                    f"{group_quotas.tolist()}"
                )
            quotas[group] = int(group_quotas[0])

        metadata = {
            "seq_length": seq_length,
            "source_shape": df.shape,
        }
        return cls(
            sequences=sequences,
            group_to_candidates=group_to_candidates,
            scores=scores,
            quotas=quotas,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Public constructor and summary (duet public API, docs/adr/0003)
    # ------------------------------------------------------------------

    @classmethod
    def from_table(
        cls,
        df: pd.DataFrame,
        *,
        group: str,
        sequence: str,
        score: str | None = None,
        quota: "int | Dict[Any, int] | str" = 1,
        seq_length: int | None = None,
        alphabet: str = "ACGT",
    ) -> "CandidatePool":
        """Build a pool from a candidate table, one row per candidate.

        Each group is a candidate set C_i (a gene, another target, or a set
        of controls) from which ``quota`` k_i candidates are selected.

        Row order is kept and it seeds the Monte Carlo draws of the PEP:
        re-sorting the table redraws them and can change the codebooks.

        Parameters
        ----------
        df : DataFrame
            One row per candidate.
        group : str
            Column naming each candidate's group.
        sequence : str
            Column with the codeword (barcode) sequence.
        score : str, optional
            Column with a per-candidate score (for example predicted
            activity), the secondary objective of :func:`duet.design_ops`
            (ignored by :func:`duet.design`).
            Every candidate needs a score, controls included; the paper's
            pool gives non-targeting controls a score of 1.0. Without a
            score column every candidate scores 1.
        quota : int, dict or str, default 1
            Candidates to select per group: one int for every group, a
            {group: k} mapping, or the name of a column (constant within
            each group).
        seq_length : int, optional
            Keep the first ``seq_length`` symbols of every sequence (the
            number of sequencing rounds).
        alphabet : str, default "ACGT"
            Symbols the sequences may use: the DNA bases (any order) or
            ``"01"`` for binary barcodes. Checked against every sequence.

        Returns
        -------
        CandidatePool

        Examples
        --------
        >>> import pandas as pd, duet
        >>> df = pd.DataFrame({"gene": ["A", "A", "B", "B"],
        ...                    "barcode": ["ACGT", "AGGT", "TTCA", "TGCA"],
        ...                    "activity": [0.9, 0.5, 0.7, 0.8]})
        >>> pool = duet.CandidatePool.from_table(df, group="gene", sequence="barcode",
        ...                                      score="activity", quota=1)
        >>> pool.pool_size, pool.total_selections
        (4, 2)
        """
        from duet.channels import check_alphabet

        check_alphabet(alphabet)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame with one row per candidate")
        for col in (group, sequence) + ((score,) if score is not None else ()):
            if col not in df.columns:
                raise ValueError(f"the table has no column {col!r}; columns: {list(df.columns)}")
        if len(df) == 0:
            raise ValueError("the table is empty")
        table = df.reset_index(drop=True)
        if table[group].isna().any():
            rows = table.index[table[group].isna()].tolist()[:5]
            raise ValueError(f"group column {group!r} is empty at rows {rows}")
        seqs = table[sequence]
        if seqs.isna().any():
            rows = table.index[seqs.isna()].tolist()[:5]
            raise ValueError(f"sequence column {sequence!r} is empty at rows {rows}")
        seqs = seqs.astype(str)
        input_sequences = seqs.tolist()
        if seq_length is not None:
            if isinstance(seq_length, bool) or not isinstance(seq_length, (int, np.integer)) or seq_length < 1:
                raise ValueError(f"seq_length must be a positive integer, got {seq_length!r}")
            short = seqs.str.len() < seq_length
            if short.any():
                rows = table.index[short].tolist()[:5]
                raise ValueError(
                    f"{int(short.sum())} sequences are shorter than seq_length={seq_length} "
                    f"(rows {rows})"
                )
            seqs = seqs.str[:seq_length]
        lengths = seqs.str.len().value_counts()
        if len(lengths) > 1:
            raise ValueError(
                f"sequences have different lengths {sorted(lengths.index.tolist())} "
                f"(counts {lengths.sort_index().tolist()}); pass seq_length= to "
                "truncate them to a common number of rounds"
            )
        allowed = set(alphabet)
        bad = seqs[[not set(s) <= allowed for s in seqs]]
        if len(bad):
            symbols = sorted(set("".join(bad)) - allowed)
            raise ValueError(
                f"{len(bad)} sequences use symbols {symbols} outside the alphabet "
                f"{alphabet!r}, e.g. row {bad.index[0]}: {bad.iloc[0]!r}"
            )
        if score is None:
            scores = pd.Series(1.0, index=table.index)
        else:
            scores = pd.to_numeric(table[score], errors="coerce")
            scores = scores.where(np.isfinite(scores.astype(float)))  # +-inf -> NaN
            if scores.isna().any():
                rows = table.index[scores.isna()].tolist()
                groups = sorted({str(g) for g in table.loc[rows, group]})[:5]
                raise ValueError(
                    f"score column {score!r} is missing, infinite or not numeric for {len(rows)} "
                    f"rows (groups {groups}). Every candidate needs a score, controls "
                    "included: the paper's pool gives non-targeting controls a score "
                    "of 1.0."
                )
        sizes = table.groupby(group, sort=False).size()
        if isinstance(quota, str):
            if quota not in table.columns:
                raise ValueError(f"quota column {quota!r} is not in the table")
            per_group = table.groupby(group, sort=False)[quota].nunique()
            if (per_group > 1).any():
                g = per_group.index[per_group > 1][0]
                raise ValueError(f"quota column {quota!r} varies within group {g!r}")
            quotas = table.groupby(group, sort=False)[quota].first()
        elif isinstance(quota, dict):
            missing = [g for g in sizes.index if g not in quota]
            if missing:
                raise ValueError(f"quota has no entry for groups {missing[:5]}")
            quotas = pd.Series({g: quota[g] for g in sizes.index})
        else:
            quotas = pd.Series(quota, index=sizes.index)
        for g, q in quotas.items():
            if isinstance(q, bool) or not float(q).is_integer() or q < 0:
                raise ValueError(f"quota for group {g!r} must be a non-negative integer, got {q!r}")
        over = [(g, int(quotas[g]), int(sizes[g])) for g in sizes.index if quotas[g] > sizes[g]]
        if over:
            g, q, n = over[0]
            raise ValueError(
                f"{len(over)} groups ask for more candidates than they have, e.g. "
                f"group {g!r}: quota {q} but {n} candidates"
            )
        canonical = pd.DataFrame({
            "Group": table[group],
            "Sequence": seqs,
            "Score": scores.astype(float),
            "Quota": table[group].map(quotas).astype(int),
        })
        pool = cls.from_dataframe(canonical)
        pool.metadata.update({
            "alphabet": alphabet,
            "seq_length": seq_length,
            "columns": {"group": group, "sequence": sequence, "score": score},
            "input_sequences": input_sequences,  # before truncation
            "source_shape": df.shape,
        })
        return pool

    def describe(self, num_samples: int = 10_000) -> "PoolDescription":
        """Sizes of the design problem, before any compute.

        Parameters
        ----------
        num_samples : int, default 10,000
            PEP reads per unique codeword; sets the number of simulated reads.

        Returns
        -------
        PoolDescription
            Groups, candidates, unique codewords U, duplicate candidates,
            codebook size, swaps per optimizer iteration, and the PEP's memory
            (2U² bytes, a dense U x U uint16 matrix) and disk footprint when
            memory-mapped and cached (4U²: a raw and a symmetric copy; 6U²
            while it is built). A PEP small enough to be held in memory is
            cached as a single 2U² copy.

        Examples
        --------
        >>> pool.describe(num_samples=10_000)  # doctest: +SKIP
        """
        from duet.api import MAX_SAMPLES

        if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)) \
                or not 1 <= num_samples <= MAX_SAMPLES:
            raise ValueError(f"num_samples must be an integer from 1 to {MAX_SAMPLES:,}, got {num_samples!r}")
        U = len(self.unique_sequences)
        swaps = sum(len(self.group_to_candidates[g]) - 1 for g in self.codeword_to_group)
        return PoolDescription(
            groups=self.num_groups,
            candidates=self.pool_size,
            codebook_size=self.total_selections,
            unique_codewords=U,
            duplicate_candidates=self.pool_size - U,
            seq_length=self.seq_length,
            swaps_per_iteration=int(swaps),
            num_samples=int(num_samples),
            pep_reads=int(U) * int(num_samples),
            pep_memory_bytes=2 * U * U,
            pep_disk_bytes=4 * U * U,
        )


@dataclass(frozen=True)
class PoolDescription:
    """Sizes of a design problem, from :meth:`CandidatePool.describe`.

    Attributes
    ----------
    groups, candidates, codebook_size : int
        Number of groups, of candidates and of codewords to select.
    unique_codewords : int
        U, the number of distinct sequences; the PEP is U x U.
    duplicate_candidates : int
        Candidates that share their codeword with an earlier candidate.
    seq_length : int
        Codeword length.
    swaps_per_iteration : int
        Within-group swaps the optimizer scores at each iteration.
    num_samples, pep_reads : int
        PEP reads per unique codeword, and in total.
    pep_memory_bytes : int
        The dense U x U uint16 PEP in memory: 2U² bytes.
    pep_disk_bytes : int
        Its footprint when cached: 4U² (a raw and a symmetric copy).

    Examples
    --------
    >>> import pandas as pd, duet
    >>> df = pd.DataFrame({"gene": ["A", "A", "B", "B"], "barcode": ["ACGT", "AGGT", "TTCA", "TGCA"]})
    >>> d = duet.CandidatePool.from_table(df, group="gene", sequence="barcode").describe()
    >>> d.unique_codewords, d.pep_memory_bytes
    (4, 32)
    """

    groups: int
    candidates: int
    codebook_size: int
    unique_codewords: int
    duplicate_candidates: int
    seq_length: int
    swaps_per_iteration: int
    num_samples: int
    pep_reads: int
    pep_memory_bytes: int
    pep_disk_bytes: int

    def to_dict(self) -> Dict[str, int]:
        """The fields as a dict."""
        return dict(self.__dict__)

    def __str__(self) -> str:
        def size(n: int) -> str:
            for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
                if n >= scale:
                    return f"{n / scale:.1f} {unit}"
            return f"{n} B"

        return "\n".join([
            "Candidate pool",
            f"  groups                  {self.groups:,}",
            f"  candidates              {self.candidates:,}",
            f"  codebook size           {self.codebook_size:,}",
            f"  unique codewords (U)    {self.unique_codewords:,}",
            f"  duplicate candidates    {self.duplicate_candidates:,} (share a codeword with an earlier candidate)",
            f"  sequence length         {self.seq_length}",
            f"  swaps per iteration     {self.swaps_per_iteration:,}",
            f"  PEP reads               {self.pep_reads:,} ({self.num_samples:,} per unique codeword)",
            f"  PEP in memory           {size(self.pep_memory_bytes)} (U x U uint16)",
            f"  PEP on disk if cached   {size(self.pep_disk_bytes)} (raw and symmetric copies)",
        ])

    __repr__ = __str__
