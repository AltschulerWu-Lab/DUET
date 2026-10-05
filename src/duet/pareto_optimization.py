"""
Pareto optimization of matching error and candidate score.

This module provides multi-objective Pareto optimization using pluggable
ObjectiveSwapCache instances. Each cache computes marginal gains for a
single objective, and the optimizer combines them via scalarization weights.

Key classes:
    ObjectiveSwapCache (ABC): Per-objective swap cache interface
    DecodingSwapCache: Decode accuracy via PEP union bound
    ScoreSwapCache: Activity score (simple delta)
    ParetoOptimizer (ABC): Base optimizer with groups, stopping, history
    ScalarizedParetoOptimizer (ABC): N-objective scalarization via weights dict
    DUET: Concrete optimizer with annealing and softmax selection
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
import random

import numpy as np

logger = logging.getLogger(__name__)


def enumerate_within_group_swaps(
    S: np.ndarray,
    codeword_to_group,
    group_to_candidates: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all valid within-group swaps.

    For each codeword position i in S, enumerates every candidate in
    group_to_candidates[codeword_to_group[i]] except S[i] itself. Duplicates
    in S are allowed at this enumeration layer, so the exclusion is just
    `c != S[i]`, not `c not in S_set`. They are filtered out downstream:
    for disjoint-group pools `DUET.step()` applies a structural within-group
    duplicate guard that hard-blocks adding a candidate already selected in
    its group, and the diagonal duplicate-penalty overlay additionally
    discourages duplicates via the objective.

    Args:
        S: Current codebook selection (1D array of candidate indices).
        codeword_to_group: Sequence (list or array-like) of length len(S);
            codeword_to_group[i] is the group that codeword position i belongs to.
            Bound to the position, not to S[i].
        group_to_candidates: Mapping from group name to list of candidate indices.

    Returns:
        codebook_index_to_remove: int32 array (position in S to remove)
        pool_index_to_add: int32 array (pool index to add)
    """
    remove_parts = []
    add_parts = []
    for i in range(len(S)):
        cands = np.asarray(
            group_to_candidates[codeword_to_group[i]], dtype=np.int32,
        )
        cands = cands[cands != S[i]]
        remove_parts.append(np.full(len(cands), i, dtype=np.int32))
        add_parts.append(cands)
    if not remove_parts:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    return np.concatenate(remove_parts), np.concatenate(add_parts)


# =============================================================================
# SharedDecodeState — pre-computed shared state for parallel DUET workers
# =============================================================================


@dataclass(frozen=True)
class SharedDecodeState:
    """Pre-computed shared state for DecodingSwapCache initialization.

    Contains everything needed to construct a valid DecodingSwapCache
    without calling build_cache (which reads |S| rows of X).
    Shared across all parallel DUET workers starting from the same init.

    IMPORTANT: Workers must COPY these arrays, as update_after_swap
    mutates pep_sum_pool, deltas, pool_index_to_add, S_indices, and S_seq
    in place.
    """
    S_indices: np.ndarray                # (|S|,) int32
    S_seq: np.ndarray                    # (|S|,) int32, sequence-indexed S
    pep_sum_pool: np.ndarray             # (U,) float64
    diag_X: np.ndarray                   # (U,) float64
    diag_const: float                    # X(s, s) = 2.0 + duplicate_offset
    codebook_index_to_remove: np.ndarray # (num_swaps,) int32
    pool_index_to_add: np.ndarray        # (num_swaps,) int32
    deltas: np.ndarray                   # (num_swaps,) float64


def build_shared_decode_state(
    S: np.ndarray,
    accessor: "PEPAccessor",
    group_to_candidates: dict[str, list[int]],
    codeword_to_group: list[str],
    candidate_to_sequence_idx: np.ndarray | None = None,
    batch_size: int = 256,
) -> SharedDecodeState:
    """Build shared DecodingSwapCache state using streaming row reads.

    Pre-computes pep_sum_pool (length U), deltas, and swap arrays once in
    the parent process, streaming through S in batches to avoid
    materializing the full (|S|, U) matrix. Peak memory is
    batch_size * U * 8 bytes. The accessor applies the duplicate-offset
    overlay logically on the diagonal, so no sparse correction loop is
    needed here.

    Args:
        S: Current codebook selection (1D array of candidate indices).
        accessor: PEPAccessor for reading rows of X in sequence space.
        group_to_candidates: Mapping from group to candidate indices.
        codeword_to_group: Sequence of length |S|; codeword_to_group[i] is
            the group bound to codeword position i.
        candidate_to_sequence_idx: Optional dedup map from candidate index
            (length pool_size) to sequence index (length U). When None,
            candidates are interpreted directly as sequence indices.
        batch_size: Number of codebook positions to read per batch.

    Returns:
        SharedDecodeState containing all arrays needed by from_shared_state.
    """
    S_seq = (
        candidate_to_sequence_idx[S].astype(np.int32, copy=True)
        if candidate_to_sequence_idx is not None
        else S.astype(np.int32, copy=True)
    )
    diag_const = float(accessor.get_diagonal()[0])

    codebook_index_to_remove, pool_index_to_add = enumerate_within_group_swaps(
        S, codeword_to_group, group_to_candidates,
    )

    num_swaps = len(codebook_index_to_remove)
    U = accessor.N
    S_len = len(S)

    # Build index: for each codebook position p, which swap indices have
    # codebook_index_to_remove == p?
    pos_to_swap_indices: dict[int, list[int]] = defaultdict(list)
    for swap_idx in range(num_swaps):
        pos = int(codebook_index_to_remove[swap_idx])
        pos_to_swap_indices[pos].append(swap_idx)

    # Phase A: Streaming accumulation in sequence space.
    pep_sum_pool = np.zeros(U, dtype=np.float64)
    x_old_new = np.empty(num_swaps, dtype=np.float64)

    for start in range(0, S_len, batch_size):
        end = min(start + batch_size, S_len)
        X_batch = accessor.get_rows(S_seq[start:end])  # offset overlay applied
        pep_sum_pool += X_batch.sum(axis=0)
        for pos in range(start, end):
            swap_indices = pos_to_swap_indices.get(pos)
            if swap_indices is not None:
                for swap_idx in swap_indices:
                    add_cand = pool_index_to_add[swap_idx]
                    u_add = (
                        int(candidate_to_sequence_idx[add_cand])
                        if candidate_to_sequence_idx is not None
                        else int(add_cand)
                    )
                    x_old_new[swap_idx] = X_batch[pos - start, u_add]

    # Phase B: Compute deltas (requires completed pep_sum_pool, sequence-indexed).
    if num_swaps > 0:
        pool_indices_old = S[codebook_index_to_remove]
        u_old = (
            candidate_to_sequence_idx[pool_indices_old]
            if candidate_to_sequence_idx is not None
            else pool_indices_old
        )
        u_new = (
            candidate_to_sequence_idx[pool_index_to_add]
            if candidate_to_sequence_idx is not None
            else pool_index_to_add
        )
        deltas = pep_sum_pool[u_old] - pep_sum_pool[u_new] + x_old_new - diag_const
    else:
        deltas = np.array([], dtype=np.float64)

    return SharedDecodeState(
        S_indices=S.astype(np.int32, copy=True),
        S_seq=S_seq,
        pep_sum_pool=pep_sum_pool,
        diag_X=accessor.get_diagonal(),
        diag_const=diag_const,
        codebook_index_to_remove=codebook_index_to_remove,
        pool_index_to_add=pool_index_to_add,
        deltas=deltas,
    )


# =============================================================================
# ObjectiveSwapCache hierarchy
# =============================================================================


class ObjectiveSwapCache(ABC):
    """Per-objective swap cache. Computes marginal gains for a single objective.

    Each subclass owns the computation of marginal deltas for one objective
    (e.g., decoding accuracy, activity score, optical crowding). The optimizer
    combines deltas from multiple caches via scalarization weights.

    Delta convention (sum-form):
        ``cache.deltas[k]`` is |S| times the per-swap change in
        ``compute_objective(S)``, NOT the change itself. Equivalently, the
        running update ``running_objective += delta / |S|`` (in
        ``ScalarizedParetoOptimizer.step``) recovers ``compute_objective``.
        Subclasses MUST follow this convention so cross-objective weighting
        stays scale-consistent. Two reasons:

        1. Cross-objective balance. If one cache emits mean-form deltas and
           another emits sum-form, the sum-form one dominates argmax by a
           factor of |S| regardless of the user's lambda weighting.
        2. Numerical stability. Per-swap changes in a mean-form objective
           shrink as O(1/|S|); for large |S| they sink toward float epsilon
           and the argmax over ~|S|*(N-|S|) candidate deltas becomes
           ill-conditioned. Sum-form deltas stay O(1) per swap regardless
           of |S|.
    """

    def __init__(
        self,
        name: str,
        group_to_candidates: dict[str, list[int]],
    ):
        self.name = name
        self.group_to_candidates = group_to_candidates
        self._is_valid = False
        self.deltas = np.array([])
        self._codebook_index_to_remove = np.array([], dtype=np.int32)
        self._pool_index_to_add = np.array([], dtype=np.int32)

    @abstractmethod
    def build_cache(self, S: np.ndarray) -> None:
        """Build cache for all possible swaps from current state S."""
        ...

    @abstractmethod
    def update_after_swap(
        self, remove_idx: int, old_candidate: int, new_candidate: int
    ) -> None:
        """Update cache incrementally after a swap is performed."""
        ...

    @abstractmethod
    def compute_objective(self, S: np.ndarray) -> float:
        """Compute absolute objective value (higher = better)."""
        ...

    def get_deltas(self) -> np.ndarray:
        """Return the current marginal gain deltas."""
        if not self._is_valid:
            raise ValueError("Cache is not valid. Call build_cache() first.")
        return self.deltas

    def get_swaps(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (codebook_index_to_remove, pool_index_to_add) arrays."""
        if not self._is_valid:
            raise ValueError("Cache is not valid. Call build_cache() first.")
        return self._codebook_index_to_remove, self._pool_index_to_add


class DecodingSwapCache(ObjectiveSwapCache):
    """Swap cache for decode accuracy via PEP union bound.

    Uses a PEPAccessor to read the symmetrized PEP matrix X = M + M^T,
    and maintains running state (pep_sum_pool, deltas) for O(N) incremental
    updates after each swap.

    PEP-side state is sequence-indexed (length U <= pool_size after dedup).
    S_seq = candidate_to_sequence_idx[S] is cached once at build_cache; all
    inner-loop translations from candidate index -> sequence index happen at
    the call-site boundary, never in the hot path. The diagonal duplicate
    penalty is applied logically by the accessor (overlay on row[i]) and
    captured as a scalar `_diag_const = X(s, s) = 2.0 + offset` at build time.
    """

    def __init__(
        self,
        pep_accessor: "PEPAccessor",
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        candidate_to_sequence_idx: np.ndarray | None = None,
        name: str = "decode",
    ):
        super().__init__(name, group_to_candidates)
        self._accessor = pep_accessor
        self._codeword_to_group = codeword_to_group
        self._c_to_u = candidate_to_sequence_idx
        # Pool size in candidate space (== U when no dedup; > U with dedup).
        if candidate_to_sequence_idx is not None:
            self._pool_size_total = len(candidate_to_sequence_idx)
        else:
            self._pool_size_total = pep_accessor.N
        self._U = pep_accessor.N
        self._diag_const = None
        self._pep_sum_pool = None
        self._S_indices = None
        self._S_seq = None
        self._diag_X = None

    @classmethod
    def from_pep_matrix(
        cls,
        pep_matrix: np.ndarray,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        candidate_to_sequence_idx: np.ndarray | None = None,
        name: str = "decode",
        duplicate_offset: float = 0.0,
        n_samples: int | None = None,
    ):
        """Construct from a raw asymmetric PEP matrix.

        Creates an InMemoryPEPAccessor with the supplied duplicate_offset
        overlay and candidate_to_sequence_idx; no separate sparse penalty.
        """
        from duet.pep_accessor import InMemoryPEPAccessor
        accessor = InMemoryPEPAccessor.from_pep_matrix(
            pep_matrix,
            n_samples=n_samples,
            duplicate_offset=duplicate_offset,
            candidate_to_sequence_idx=candidate_to_sequence_idx,
        )
        return cls(
            pep_accessor=accessor,
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
            candidate_to_sequence_idx=candidate_to_sequence_idx,
            name=name,
        )

    @classmethod
    def from_shared_state(
        cls,
        pep_accessor: "PEPAccessor",
        state: SharedDecodeState,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        candidate_to_sequence_idx: np.ndarray | None = None,
    ) -> "DecodingSwapCache":
        """Construct from pre-built shared state (skips build_cache).

        Each array from state is copied because update_after_swap mutates
        pep_sum_pool, deltas, pool_index_to_add, S_indices, and S_seq in place.
        """
        instance = cls(
            pep_accessor=pep_accessor,
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
            candidate_to_sequence_idx=candidate_to_sequence_idx,
        )
        instance._S_indices = state.S_indices.copy()
        instance._S_seq = state.S_seq.copy()
        instance._pep_sum_pool = state.pep_sum_pool.copy()
        instance._diag_X = state.diag_X.copy()
        instance._diag_const = float(state.diag_const)
        instance._codebook_index_to_remove = state.codebook_index_to_remove.copy()
        instance._pool_index_to_add = state.pool_index_to_add.copy()
        instance.deltas = state.deltas.copy()
        instance._is_valid = True
        return instance

    def _to_seq(self, candidates):
        """Translate candidate indices to sequence indices (no-op if no dedup)."""
        if self._c_to_u is None:
            return candidates
        return self._c_to_u[candidates]

    def build_cache(self, S: np.ndarray) -> None:
        """Build cache using union bound approximation.

        Reads |S| rows of X in sequence space (the accessor overlays the
        diagonal offset). Computes pep_sum_pool (length U) and initial deltas.
        """
        if self._is_valid:
            return  # Already initialized (e.g., via from_shared_state)

        self._S_indices = S.astype(np.int32, copy=True)
        self._S_seq = self._to_seq(S).astype(np.int32, copy=True)

        # Capture the offset-aware diagonal constant once.
        self._diag_const = float(self._accessor.get_diagonal()[0])

        # Enumerate swaps using the codeword's group; allows duplicates in S.
        self._codebook_index_to_remove, self._pool_index_to_add = (
            enumerate_within_group_swaps(
                S, self._codeword_to_group, self.group_to_candidates,
            )
        )

        # Read |S| rows of X (sequence-indexed); accessor applies overlay.
        X_S = self._accessor.get_rows(self._S_seq)  # shape (|S|, U), float64

        self._pep_sum_pool = X_S.sum(axis=0)        # length U
        self._diag_X = self._accessor.get_diagonal()  # length U

        self._compute_initial_deltas(X_S)
        self._is_valid = True

    def _compute_initial_deltas(self, X_S: np.ndarray) -> None:
        """Compute initial deltas using cached codebook rows."""
        if len(self._codebook_index_to_remove) == 0:
            self.deltas = np.array([], dtype=np.float64)
            return

        pool_indices_old = self._S_indices[self._codebook_index_to_remove]
        u_old = self._to_seq(pool_indices_old)
        u_new = self._to_seq(self._pool_index_to_add)

        pep_sum_old = self._pep_sum_pool[u_old]
        pep_sum_new = self._pep_sum_pool[u_new]

        # X(s_old, s_new) from cached rows. positions_old indexes into X_S
        # (the (|S|, U) row block); column is u_new.
        candidate_to_position = np.empty(self._pool_size_total, dtype=np.intp)
        candidate_to_position[self._S_indices] = np.arange(len(self._S_indices))
        positions_old = candidate_to_position[pool_indices_old]
        x_old_new = X_S[positions_old, u_new]

        self.deltas = pep_sum_old - pep_sum_new + x_old_new - self._diag_const

    def update_after_swap(
        self, remove_idx: int, old_candidate: int, new_candidate: int
    ) -> None:
        """Update cache incrementally after a swap.

        Under the new "duplicates-in-S allowed; relabel scoped to remove_idx"
        rule, the sibling-recompute branch from the previous design is gone:
        all entries at codewords != remove_idx keep their (codeword, c_new)
        identity, and the uniform Category C correction applies to all of them.
        """
        if not self._is_valid:
            raise ValueError("Cache is not valid. Cannot update invalid cache.")

        u_old = int(self._c_to_u[old_candidate]) if self._c_to_u is not None else int(old_candidate)
        u_new = int(self._c_to_u[new_candidate]) if self._c_to_u is not None else int(new_candidate)

        # --- Step 1: Read 2 rows from X (accessor applies offset overlay) ---
        x_row_old = self._accessor.get_row(u_old)   # length U
        x_row_new = self._accessor.get_row(u_new)   # length U

        # --- Step 2: Compute delta_alpha (length U) and update pep_sum_pool ---
        delta_alpha = x_row_new - x_row_old
        self._pep_sum_pool += delta_alpha

        # --- Step 3: Category C — uniform correction at all OTHER codewords ---
        mask_other = (self._codebook_index_to_remove != remove_idx)
        if np.any(mask_other):
            old_cands = self._S_indices[self._codebook_index_to_remove[mask_other]]
            new_cands = self._pool_index_to_add[mask_other]
            u_old_others = self._to_seq(old_cands)
            u_new_others = self._to_seq(new_cands)
            self.deltas[mask_other] += delta_alpha[u_old_others] - delta_alpha[u_new_others]

        # --- Step 4: Bookkeeping — scoped relabel ---
        mask_at_i = (self._codebook_index_to_remove == remove_idx)
        mask_relabel = mask_at_i & (self._pool_index_to_add == new_candidate)
        self._pool_index_to_add[mask_relabel] = old_candidate
        self._S_indices[remove_idx] = new_candidate
        self._S_seq[remove_idx] = u_new

        # --- Step 5: Recompute deltas for entries at codeword = remove_idx ---
        if np.any(mask_at_i):
            cands_p = self._pool_index_to_add[mask_at_i]
            u_p = self._to_seq(cands_p)
            self.deltas[mask_at_i] = (
                self._pep_sum_pool[u_new]
                - self._pep_sum_pool[u_p]
                + x_row_new[u_p]
                - self._diag_const
            )

    def compute_objective(self, S: np.ndarray) -> float:
        """Compute union bound decode accuracy from running in-memory state.

        Uses the identity:
        sum_{i!=j in S} M(i,j) = (sum_{s in S} alpha_S(s) - sum_{s in S} X(s,s)) / 2
        """
        S_seq = self._to_seq(S)
        alpha_sum = float(np.sum(self._pep_sum_pool[S_seq]))
        diag_sum = float(np.sum(self._diag_X[S_seq]))
        total_pep = (alpha_sum - diag_sum) / 2.0
        return 1.0 - total_pep / len(S)


class ScoreSwapCache(ObjectiveSwapCache):
    """Swap cache for activity score objective.

    Computes marginal gains in mean score for all possible within-group swaps.
    """

    def __init__(
        self,
        scores: np.ndarray,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        name: str = "score",
    ):
        super().__init__(name, group_to_candidates)
        self.scores = scores
        self._codeword_to_group = codeword_to_group

    def build_cache(self, S: np.ndarray) -> None:
        """Build score delta cache."""
        self._codebook_index_to_remove, self._pool_index_to_add = (
            enumerate_within_group_swaps(
                S, self._codeword_to_group, self.group_to_candidates,
            )
        )

        if len(self._codebook_index_to_remove) == 0:
            self.deltas = np.array([])
            self._is_valid = True
            return

        # Score delta = score(new) - score(old)
        old_candidates = S[self._codebook_index_to_remove]
        self.deltas = self.scores[self._pool_index_to_add] - self.scores[old_candidates]
        self._is_valid = True

    def update_after_swap(
        self, remove_idx: int, old_candidate: int, new_candidate: int
    ) -> None:
        """Update score deltas incrementally after a swap."""
        if not self._is_valid:
            raise ValueError("Cache is not valid. Cannot update invalid cache.")

        score_diff = self.scores[old_candidate] - self.scores[new_candidate]
        mask_at_i = (self._codebook_index_to_remove == remove_idx)

        # All swaps at codeword remove_idx: their "old" score changed.
        self.deltas[mask_at_i] += score_diff

        # The single (remove_idx, new_candidate) entry becomes (remove_idx, old_candidate);
        # picks up another += score_diff, then is relabeled.
        mask_relabel = mask_at_i & (self._pool_index_to_add == new_candidate)
        self.deltas[mask_relabel] += score_diff
        self._pool_index_to_add[mask_relabel] = old_candidate

    def compute_objective(self, S: np.ndarray) -> float:
        """Compute mean score of selected candidates."""
        return float(np.mean(self.scores[S]))


class CrowdingSwapCache(ObjectiveSwapCache):
    """Swap cache for the MERFISH optical-crowding objective.

    Implements the Candidate C score O_C(S) = 1 - C(S) / C(S_0), where
    C(S) = sum_r TEV[r]^2 and C(S_0) is the realised C of the initial
    selection -- frozen at build_cache time. The score is 0 at S = S_0
    by construction and increases as the per-round TEV distribution
    contracts (lower magnitude and/or more uniform shape).

    Cache state is O(N + R): length-R TEV vector, length-N beta vector,
    length-N per-candidate Hamming-weight cache, plus scalar M (total
    expression sum, retained for diagnostics) and C_anchor. The per-swap
    beta update is one BLAS matvec against the codeword matrix.
    """

    # Default per-chunk swap count for _compute_crowding_deltas. Benchmarked
    # on a Zhang 2023-scale problem (9 workers sharing one node): 1M is ~30% faster
    # than the monolithic variant (per-chunk working set ~56 MiB fits in
    # L3) and keeps per-call tracemalloc peak under ~1.8 GiB at that shape.
    # Override per-instance via the chunk_size kwarg if profiling shows a
    # different L3 size benefits from a different value.
    _DEFAULT_CHUNK_SIZE: int = 1_000_000

    def __init__(
        self,
        codewords: np.ndarray,
        expression: np.ndarray,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        name: str = "crowding",
        chunk_size: int | None = None,
    ):
        super().__init__(name, group_to_candidates)
        self._codeword_to_group = codeword_to_group
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {chunk_size}."
            )
        self._chunk_size = chunk_size if chunk_size is not None else self._DEFAULT_CHUNK_SIZE
        self._codewords = codewords.astype(np.float64)
        self._expression = expression.astype(np.float64)
        self._pool_size, self._n_rounds = codewords.shape

        # Per-candidate Hamming weight: replaces the diagonal of the previous
        # N×N IP matrix. For binary c_j, IP[j,j] = <c_j, c_j> = sum(c_j).
        self._HW_per_cand = self._codewords.sum(axis=1)

        # Runtime state (set in build_cache)
        self._TEV = None
        self._beta = None
        self._S_indices = None
        self._M = None
        self._C_anchor = None

    def build_cache(self, S: np.ndarray) -> None:
        """Build cache for all possible swaps from current state S.

        Freezes C_anchor = C(S_0) -- the realised C of the initial
        selection -- and asserts the math preconditions that keep the
        score well-defined.
        """
        self._validate_preconditions(S)

        self._S_indices = S.astype(np.int32, copy=True)

        self._codebook_index_to_remove, self._pool_index_to_add = (
            enumerate_within_group_swaps(
                S, self._codeword_to_group, self.group_to_candidates,
            )
        )

        # TEV[r] = sum_i expression[i] * codewords[S[i]][r]
        self._TEV = (self._expression[:, None] * self._codewords[S]).sum(axis=0)

        # M = sum_r TEV[r] = sum_i e_i * HW(c_{S[i]})  -- maintained incrementally
        self._M = float(np.sum(self._TEV))

        # beta[j] = codewords[j] · TEV for entire pool
        self._beta = self._codewords @ self._TEV

        # Anchor: C(S_0) = sum_r TEV(S_0)[r]^2, frozen here. C_anchor > 0
        # follows from the preconditions: at least one position has
        # e_i > 0 AND HW(c_{S[i]}) > 0, guaranteeing at least one TEV[r] > 0.
        self._C_anchor = float(np.sum(self._TEV ** 2))

        self.deltas = self._compute_crowding_deltas()
        self._is_valid = True

    def update_after_swap(
        self, remove_idx: int, old_candidate: int, new_candidate: int
    ) -> None:
        """Update cache incrementally after a swap."""
        if not self._is_valid:
            raise ValueError("Cache is not valid. Cannot update invalid cache.")

        e_i = self._expression[remove_idx]
        hw_new = self._HW_per_cand[new_candidate]
        hw_old = self._HW_per_cand[old_candidate]

        # Update TEV: O(n_rounds)
        self._TEV += e_i * (
            self._codewords[new_candidate] - self._codewords[old_candidate]
        )

        # Update M: O(1). Mirrors the in-place TEV change.
        self._M += e_i * (hw_new - hw_old)

        # Update beta via one BLAS matvec: O(pool_size * n_rounds).
        # beta[j] = <c_j, TEV> by definition; recompute from the just-updated
        # TEV. More numerically stable than accumulating per-swap increments,
        # and avoids needing a precomputed N×N codeword inner-product matrix.
        self._beta = self._codewords @ self._TEV

        # Update swap bookkeeping
        mask_at_i = (self._codebook_index_to_remove == remove_idx)
        mask_relabel = mask_at_i & (self._pool_index_to_add == new_candidate)
        self._pool_index_to_add[mask_relabel] = old_candidate
        self._S_indices[remove_idx] = new_candidate

        self.deltas = self._compute_crowding_deltas()

    def compute_objective(self, S: np.ndarray) -> float:
        """Compute O_C(S) = 1 - C(S) / C(S_0).

        Uses the frozen C_anchor set at build_cache time; raises if the
        cache has not been built.
        """
        if self._C_anchor is None:
            raise ValueError(
                "Cannot compute objective: C_anchor not set. Call "
                "build_cache(S_0) first to freeze the anchor."
            )
        TEV = (self._expression[:, None] * self._codewords[S]).sum(axis=0)
        C = float(np.sum(TEV ** 2))
        return 1.0 - C / self._C_anchor

    def _compute_crowding_deltas(self) -> np.ndarray:
        """Vectorised per-swap delta of O_C = 1 - C(S) / C(S_0).

        Returns sum-form deltas: |S| * (O_C(S_k) - O_C(S)) for each
        enumerated swap k (positive = improvement). The |S| factor matches
        the sum-form convention documented on ObjectiveSwapCache; the
        optimizer's ``running += delta / |S|`` update then recovers O_C.

        Closed form: -|S| * Delta_C / C_anchor, where C evolves under a
        single swap c_old -> c_new at position p as

            Delta_C = 2 * e_p * (beta_new - beta_old)
                    + e_p^2 * (HW_new + HW_old - 2 * <c_new, c_old>)

        Implementation notes
        --------------------
        - <c_new, c_old> for all swap pairs is read from a (|S|, N) inner-
          product slab built with one BLAS matmul (cheap on (|S|, R) @
          (R, N), fast on zhang_2023_scale). The slab lives only inside
          this function and is freed before the function returns.
        - The per-swap gather + arithmetic chain runs in chunks of
          ``self._chunk_size`` swaps. Without chunking, seven
          ``(num_swaps,) float64`` arrays (~845 MiB each at zhang) are
          locked into local names through the delta_C expression,
          producing a ~9 GiB transient peak. Chunked, the per-chunk
          working set (~7 × chunk_size × 8 B) fits in L3 cache and the
          arithmetic chain stays warm — chunked is both ~5× smaller in
          peak and ~30% faster wall on the benchmark.
        """
        positions = self._codebook_index_to_remove
        new_candidates = self._pool_index_to_add
        num_swaps = len(positions)

        if num_swaps == 0:
            return np.empty(0, dtype=np.float64)

        # One BLAS matmul; held only for the chunked indexing below, then freed.
        IP_S = self._codewords[self._S_indices] @ self._codewords.T

        deltas_out = np.empty(num_swaps, dtype=np.float64)
        for s in range(0, num_swaps, self._chunk_size):
            e = min(s + self._chunk_size, num_swaps)
            pos_c = positions[s:e]
            new_c = new_candidates[s:e]
            old_c = self._S_indices[pos_c]

            e_i = self._expression[pos_c]
            beta_old = self._beta[old_c]
            beta_new = self._beta[new_c]
            ip = IP_S[pos_c, new_c]
            hw_old = self._HW_per_cand[old_c]
            hw_new = self._HW_per_cand[new_c]

            deltas_out[s:e] = (
                2.0 * e_i * (beta_new - beta_old)
                + e_i ** 2 * (hw_new + hw_old - 2.0 * ip)
            )

        del IP_S
        return -deltas_out * len(self._S_indices) / self._C_anchor

    def _validate_preconditions(self, S: np.ndarray) -> None:
        """Assert math preconditions needed for the Candidate C score.

        C(S_0) = sum_r TEV[r]^2 is strictly positive iff at least one
        round has TEV[r] > 0, which holds iff at least one position i
        satisfies e_i > 0 AND HW(c_{S[i]}) > 0. Positions with e_i = 0
        are crowding-neutral: they contribute nothing to TEV and their
        per-swap delta is identically zero, so the optimizer drives them
        purely by the decoding objective. This is the correct semantics
        for non-expressed panel genes (e.g. multi-tissue panels paired
        with a single-tissue expression vector).

        Non-negativity (e_i >= 0) is still required: a negative weight
        would let TEV[r] cancel to zero across positions and leave the
        delta formula ill-conditioned.

        R >= 2 is retained as a sanity guard; R = 1 collapses the
        objective to a function of M alone.

        Unlike the chi^2-anchored predecessor there is no upper bound on
        HW (the all-ones codeword is legal) and no anchor-positivity
        post-condition (C_anchor > 0 is implied by the preconditions
        above).
        """
        if self._n_rounds < 2:
            raise ValueError(
                f"CrowdingSwapCache requires R >= 2 imaging rounds; got "
                f"R = {self._n_rounds}."
            )
        if len(self._expression) != len(S):
            raise ValueError(
                f"expression length ({len(self._expression)}) must match "
                f"selection size ({len(S)})."
            )
        if np.any(self._expression < 0.0):
            n_bad = int(np.sum(self._expression < 0.0))
            raise ValueError(
                f"CrowdingSwapCache requires e_i >= 0 for every position; "
                f"got {n_bad}/{len(self._expression)} negative entries."
            )
        hw_S = self._HW_per_cand[S]
        if not np.all(hw_S > 0):
            n_bad = int(np.sum(hw_S <= 0))
            raise ValueError(
                f"CrowdingSwapCache requires HW(c_i) > 0 for every "
                f"selected codeword; got {n_bad}/{len(S)} codewords with "
                f"HW = 0 (all-zero codewords)."
            )
        # C_anchor > 0 requires at least one position with both
        # e_i > 0 AND HW(c_{S[i]}) > 0.
        if not np.any((self._expression > 0.0) & (hw_S > 0)):
            raise ValueError(
                "CrowdingSwapCache requires at least one position with "
                "e_i > 0 AND HW(c_{S[i]}) > 0 (otherwise C(S_0) = 0 and "
                "the normalized score 1 - C(S)/C(S_0) is undefined)."
            )


# =============================================================================
# Optimizer hierarchy
# =============================================================================


class ParetoOptimizer(ABC):
    """Base class for Pareto optimization of codebook selections.

    Provides structure for optimization methods that balance multiple objectives.
    """

    def __init__(
        self,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        max_iter: int = 100_000,
        max_patience: int = 2000,
        track_history: bool = False,
        verbose: bool = False,
    ):
        self.group_to_candidates = group_to_candidates
        self._codeword_to_group = codeword_to_group

        self.history = defaultdict(list)
        self.max_iter = max_iter
        self.max_patience = max_patience
        self.track_history = track_history
        self.verbose = verbose

    @abstractmethod
    def initialize(self, S: np.ndarray):
        """Initialize the optimizer with the given selection."""
        pass

    @abstractmethod
    def step(self, S: np.ndarray) -> np.ndarray:
        """Perform a single optimization step."""
        pass

    @abstractmethod
    def evaluate(self, S: np.ndarray) -> dict[str, float]:
        """Evaluate the performance of the current selection."""
        pass

    def _update_history(self, k: int, objectives: dict[str, float]):
        """Record N named objectives."""
        self.history['iteration'].append(k)
        for name, value in objectives.items():
            self.history[name].append(value)

    def _reset_history(self):
        """Clear optimization history."""
        self.history = defaultdict(list)

    def validate_solution(self, solution: np.ndarray, init: np.ndarray) -> bool:
        """Return True if solution[i] is in the group of codeword position i."""
        sol = np.asarray(solution).ravel()
        if len(sol) != len(self._codeword_to_group):
            return False
        try:
            for i, c in enumerate(sol):
                group = self._codeword_to_group[i]
                if int(c) not in set(self.group_to_candidates[group]):
                    return False
            return True
        except (KeyError, IndexError):
            return False

    @abstractmethod
    def plot_progress(self):
        """Plot optimization progress."""
        pass


class ScalarizedParetoOptimizer(ParetoOptimizer, ABC):
    """N-objective scalarization via weights dict.

    Combines multiple ObjectiveSwapCache instances using weighted sum
    of marginal gains for swap selection.
    """

    def __init__(
        self,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        weights: dict[str, float],
        temperature: float = 1.0,
        max_iter: int = 100_000,
        max_patience: int = 2000,
        track_history: bool = False,
        verbose: bool = False,
        store_all_solutions: bool = False,
    ):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")

        super().__init__(
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
            max_iter=max_iter,
            max_patience=max_patience,
            track_history=track_history,
            verbose=verbose,
        )

        self.weights = {k: v / total for k, v in weights.items()}
        self.temperature = temperature
        self._caches: list[ObjectiveSwapCache] = []
        self._running_objectives: dict[str, float] = {}

        self.store_all_solutions = store_all_solutions
        if self.track_history and self.store_all_solutions:
            self.all_solutions = []

        self.best_S = None
        self.best_objectives: dict[str, float] | None = None
        self.best_scalarized_objective = -np.inf

    def _compute_scalarized(self, objectives: dict[str, float]) -> float:
        """Compute weighted sum of objectives."""
        return sum(self.weights[name] * objectives[name] for name in objectives)

    def _compute_combined_deltas(self) -> np.ndarray:
        """Compute weighted combination of all cache deltas."""
        combined = np.zeros_like(self._caches[0].deltas)
        for cache in self._caches:
            combined += self.weights[cache.name] * cache.get_deltas()
        return combined

    def _initialize_state(self, S: np.ndarray):
        """Initialize optimizer state with the given selection."""
        initial_objectives = self.evaluate(S)
        initial_scalarized = self._compute_scalarized(initial_objectives)

        self.best_S = S.copy()
        self.best_objectives = dict(initial_objectives)
        self.best_scalarized_objective = initial_scalarized

        if self.track_history:
            self._reset_history()
            self._update_history(0, initial_objectives)
            if self.store_all_solutions:
                self.all_solutions.append(S.copy())

    def optimize(self, init: list, seed: int = None):
        """Run optimization starting from the initial selection."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        S = np.array(init)
        k = 0
        patience = 0
        self.initialize(S)

        while k < self.max_iter and patience < self.max_patience:
            S = self.step(S)
            objectives = self.evaluate(S)

            scalarized_obj = self._compute_scalarized(objectives)
            has_improved = scalarized_obj > self.best_scalarized_objective

            if has_improved:
                if self.verbose and k > 0:
                    print(f'Improvement at iteration {k}', flush=True)
                self.best_scalarized_objective = scalarized_obj
                self.best_S = S.copy()
                self.best_objectives = dict(objectives)

            if self.track_history:
                self._update_history(k + 1, objectives)
                if self.store_all_solutions:
                    self.all_solutions.append(S.copy())

            k += 1
            patience = 0 if has_improved else patience + 1

            if self.verbose and k % max(self.max_iter // 20, 1) == 0:
                print(f'Finished iteration {k} ({k * 20 // self.max_iter} / 20)', flush=True)

    def _update_history(self, k: int, objectives: dict[str, float]):
        """Update history with per-objective and scalarized values."""
        self.history['iteration'].append(k)
        for name, value in objectives.items():
            self.history[name].append(value)
        scalarized = self._compute_scalarized(objectives)
        self.history['scalarized_objective'].append(scalarized)

    def plot_progress(self):
        """Plot optimization progress showing individual and scalarized objectives."""
        if not self.track_history:
            raise ValueError("History is not being tracked. Set track_history=True.")

        # Plotting libraries load only here, so importing the optimizer stays light.
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        objective_names = [c.name for c in self._caches]
        n_objectives = len(objective_names)

        fig, axes = plt.subplots(n_objectives + 1, 1, figsize=(10, 4 * (n_objectives + 1)))
        if n_objectives + 1 == 1:
            axes = [axes]

        df = pd.DataFrame(self.history)

        for ax, name in zip(axes[:n_objectives], objective_names):
            sns.lineplot(x='iteration', y=name, data=df, ax=ax)
            ax.set_ylabel(name)
            ax.set_title(f'{name} (weight={self.weights.get(name, 0):.3f})')

        ax_scalar = axes[-1]
        sns.lineplot(x='iteration', y='scalarized_objective', data=df, ax=ax_scalar, color='green')
        ax_scalar.set_ylabel('Scalarized Objective')
        ax_scalar.set_title('Scalarized Objective')

        plt.tight_layout()
        plt.show()


class DUET(ScalarizedParetoOptimizer):
    """Concrete scalarized Pareto optimizer with annealing and softmax selection.

    Accepts a list of ObjectiveSwapCache instances and a weights dict.
    All decode accuracy computations use the union bound approximation
    derived from the pairwise error probability (PEP) matrix.
    """

    # `evaluate()` returns the incrementally-maintained _running_objectives, so
    # the scalarized objective accumulates float64 ULP drift across iterations.
    # The patience improvement check must require improvement above that drift
    # band; otherwise A↔B cycles at a local optimum reset patience indefinitely.
    # Objectives are O(1)-bounded, so 1e-12 sits well above ULP-drift (~1e-13)
    # and well below any meaningful objective change (>> 1e-6).
    _PATIENCE_ATOL: float = 1e-12

    def __init__(
        self,
        group_to_candidates: dict[str, list[int]],
        codeword_to_group: list[str],
        objective_caches: list[ObjectiveSwapCache],
        weights: dict[str, float],
        temperature: float = 1.0,
        max_iter: int = 100_000,
        max_patience: int = 2000,
        track_history: bool = False,
        verbose: bool = False,
        store_all_solutions: bool = False,
        # Annealing parameters
        annealing: bool = False,
        initial_temperature: float | None = None,
        final_temperature: float = 1e-6,
        cooling_rate: float = 0.9995,
        reheat_patience: int | None = None,
        reheat_factor: float = 5.0,
    ):
        # Validate caches match weights
        cache_names = {c.name for c in objective_caches}
        weight_names = set(weights.keys())
        if cache_names != weight_names:
            raise ValueError(
                f"Cache names {cache_names} don't match weight names {weight_names}"
            )

        super().__init__(
            group_to_candidates=group_to_candidates,
            codeword_to_group=codeword_to_group,
            weights=weights,
            temperature=temperature,
            max_iter=max_iter,
            max_patience=max_patience,
            track_history=track_history,
            verbose=verbose,
            store_all_solutions=store_all_solutions,
        )

        self._caches = list(objective_caches)

        # Annealing configuration
        self.annealing = annealing
        self.initial_temperature = float(initial_temperature) if initial_temperature is not None else None
        self.final_temperature = float(final_temperature)
        self.cooling_rate = float(cooling_rate)
        self.reheat_patience = reheat_patience
        self.reheat_factor = float(reheat_factor)

        # Annealing runtime state
        self._current_temperature: float | None = None
        self._initial_temperature: float | None = None
        self._reheat_patience_counter: int = 0

        # Structural within-group duplicate guard (consumed in step()).
        self._setup_within_group_guard()

    def _setup_within_group_guard(self) -> None:
        """Prepare the guard that forbids selecting the same candidate twice
        within one group (see step()).

        When groups are disjoint — every candidate belongs to exactly one group,
        as in the OPS per-row pool — "already in S" is equivalent to "already
        selected in its own group", so a single reusable O(pool_size) boolean
        membership array is sufficient and exact.

        When groups overlap — a candidate is a member of several groups, as in
        the MERFISH per-codeword pool — that equivalence breaks (a candidate can
        legitimately sit in a sibling slot of a *different* group). Those pools
        use quota 1 per group, so a within-group duplicate is structurally
        impossible and the guard is disabled (a no-op). The quota-1 precondition
        is asserted so a future overlapping-group pool with quota > 1 fails
        loudly rather than silently permitting duplicates.
        """
        members = [c for cands in self.group_to_candidates.values() for c in cands]
        disjoint = len(members) == len(set(members))
        if disjoint and members:
            # Reused across steps; cleared by toggling only the S entries back
            # off (O(|S|)), never a full O(pool_size) memset.
            self._dup_guard_in_S = np.zeros(max(members) + 1, dtype=bool)
        else:
            self._dup_guard_in_S = None
            if not disjoint:
                max_quota = max(Counter(self._codeword_to_group).values())
                if max_quota > 1:
                    raise NotImplementedError(
                        "within-group duplicate guard does not support "
                        "overlapping groups with quota > 1; a group-scoped "
                        "membership test would be required"
                    )

    def initialize(self, S: np.ndarray):
        """Initialize optimizer: build all caches, compute initial objectives."""
        # Build all caches
        for cache in self._caches:
            cache.build_cache(S)

        # Assert consistent swap enumeration across all caches
        if len(self._caches) > 1:
            ref_remove = self._caches[0]._codebook_index_to_remove
            ref_add = self._caches[0]._pool_index_to_add
            for cache in self._caches[1:]:
                assert np.array_equal(ref_remove, cache._codebook_index_to_remove), \
                    f"Swap enumeration mismatch: {self._caches[0].name} vs {cache.name}"
                assert np.array_equal(ref_add, cache._pool_index_to_add), \
                    f"Swap enumeration mismatch: {self._caches[0].name} vs {cache.name}"

        # Compute initial running objectives
        self._running_objectives = {
            cache.name: cache.compute_objective(S) for cache in self._caches
        }

        # Calibrate annealing temperature
        if self.annealing:
            if self.initial_temperature is not None:
                self._initial_temperature = self.initial_temperature
            else:
                # Auto-calibrate from initial combined deltas
                combined = self._compute_combined_deltas()
                if len(combined) > 1:
                    gap = np.max(combined) - np.median(combined)
                    self._initial_temperature = gap / np.log(3.0) if gap > 0 else 1.0
                else:
                    self._initial_temperature = 1.0

            self._current_temperature = self._initial_temperature
            if self.verbose:
                print(f'Annealing enabled: initial temperature = {self._initial_temperature:.6f}', flush=True)

        # Initialize best tracking
        self._initialize_state(S)

        # Record initial temperature in history
        if self.annealing and self.track_history:
            self.history['temperature'].append(self._get_current_temperature())

    def _get_current_temperature(self) -> float:
        """Return the effective temperature for swap selection."""
        if self.annealing and self._current_temperature is not None:
            return self._current_temperature
        return self.temperature

    def step(self, S: np.ndarray) -> np.ndarray:
        """Perform one optimization step.

        Computes combined deltas, selects swap via softmax, reads per-objective
        deltas BEFORE update, performs swap, updates all caches.
        """
        # Compute combined deltas
        combined_deltas = self._compute_combined_deltas()

        # Handle edge case: no swaps available
        if len(combined_deltas) == 0:
            return S

        # Get swap arrays from first cache (all caches have same enumeration)
        remove_arr = self._caches[0]._codebook_index_to_remove
        add_arr = self._caches[0]._pool_index_to_add

        # Structural within-group duplicate guard: forbid any swap that would
        # place a candidate already selected in its group into a second slot of
        # that group. Re-derived from the live S every step, so it holds under
        # any objective weighting (notably decode weight == 0 at lambda == 0.0,
        # where the diagonal duplicate penalty vanishes) and is immune to
        # staleness in the incrementally-maintained swap menu. Disabled (None)
        # for overlapping-group pools where it is unnecessary. Cost is
        # O(|S| + |swaps|), the same order as the combined-delta computation.
        in_S = self._dup_guard_in_S
        if in_S is not None:
            in_S[S] = True
            forbidden = in_S[add_arr]
            in_S[S] = False
            n_forbidden = int(forbidden.sum())
            if n_forbidden:
                if n_forbidden == forbidden.size:
                    # Every enumerated swap would duplicate within its group
                    # (all groups saturated) — make no move this step.
                    return S
                combined_deltas[forbidden] = -np.inf

        # Select swap via softmax
        temp = self._get_current_temperature()
        if temp == 0:
            max_score = np.max(combined_deltas)
            max_indices = np.where(combined_deltas == max_score)[0]
            idx = np.random.choice(max_indices)
        else:
            z = combined_deltas / max(temp, 1e-12)
            z -= np.max(z)
            exp_scores = np.exp(z)
            probs = exp_scores / np.sum(exp_scores)
            idx = np.random.choice(len(probs), p=probs)

        remove_idx = int(remove_arr[idx])
        add_candidate = int(add_arr[idx])

        # Read per-objective deltas BEFORE update
        per_objective_deltas = {
            cache.name: float(cache.deltas[idx]) for cache in self._caches
        }

        # Perform swap
        old_candidate = S[remove_idx]
        S[remove_idx] = add_candidate

        # Update all caches
        for cache in self._caches:
            cache.update_after_swap(remove_idx, old_candidate, add_candidate)

        # Update running objectives
        S_size = len(S)
        for name, delta in per_objective_deltas.items():
            self._running_objectives[name] += delta / S_size

        return S

    def evaluate(self, S: np.ndarray) -> dict[str, float]:
        """Return current running objectives (no recomputation)."""
        return dict(self._running_objectives)

    def compute_approximate_objectives(self, S: np.ndarray) -> dict[str, float]:
        """Recompute objectives from scratch (useful for drift validation)."""
        return {cache.name: cache.compute_objective(S) for cache in self._caches}

    def optimize(self, init: list, seed: int = None):
        """Run optimization with optional temperature annealing.

        Args:
            init: Initial codebook selection (list of candidate indices).
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        S = np.array(init)
        k = 0
        patience = 0
        self._reheat_patience_counter = 0
        self.initialize(S)

        while k < self.max_iter and patience < self.max_patience:
            S = self.step(S)
            objectives = self.evaluate(S)

            scalarized_obj = self._compute_scalarized(objectives)
            has_improved = (
                scalarized_obj > self.best_scalarized_objective + self._PATIENCE_ATOL
            )

            if has_improved:
                if self.verbose and k > 0:
                    print(f'Improvement at iteration {k}', flush=True)
                self.best_scalarized_objective = scalarized_obj
                self.best_S = S.copy()
                self.best_objectives = dict(objectives)
                self._reheat_patience_counter = 0
            else:
                self._reheat_patience_counter += 1

            if self.track_history:
                self._update_history(k + 1, objectives)
                if self.annealing:
                    self.history['temperature'].append(self._get_current_temperature())
                if self.store_all_solutions:
                    self.all_solutions.append(S.copy())

            # === Annealing logic ===
            if self.annealing and self._current_temperature is not None:
                # Geometric cooling
                if self._current_temperature > self.final_temperature:
                    self._current_temperature *= self.cooling_rate

                # Adaptive reheating: reset to initial_temperature
                if (self.reheat_patience is not None and
                        self._reheat_patience_counter >= self.reheat_patience):
                    self._current_temperature = self._initial_temperature
                    self._reheat_patience_counter = 0
                    if self.verbose:
                        print(f'Reheating at iteration {k} to T={self._current_temperature:.4f}', flush=True)

            k += 1
            patience = 0 if has_improved else patience + 1

            if self.verbose and k % max(self.max_iter // 20, 1) == 0:
                print(f'Finished iteration {k} ({k * 20 // self.max_iter} / 20)', flush=True)
