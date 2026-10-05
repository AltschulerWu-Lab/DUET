"""
MERFISHFactory: Factory for generating binary codeword candidates for MERFISH.

This module adapts the DUET framework for MERFISH codebook optimization by
generating all binary sequences of a specified length with configurable
Hamming weight constraints.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np

from duet.candidate_pool import CandidatePool
from duet.candidate_pool_factory import CandidatePoolFactory

logger = logging.getLogger(__name__)


class MERFISHFactory(CandidatePoolFactory):
    """Factory for MERFISH binary codebook candidates.

    Generates all binary sequences of a given length with specified Hamming
    weight constraints. All candidates are treated as members of a single
    pseudo-group named "MERFISH" to fit the DUET optimization framework.

    For MERFISH, we optimize purely for decoding accuracy (lambda=1), so all
    scores are set to 1.0.

    Example:
        >>> factory = MERFISHFactory(
        ...     seq_rounds=16,
        ...     codebook_size=140,
        ...     hamming_weights=[4, 5],
        ... )
        >>> candidates = factory.create()
        >>> print(f"Generated {candidates.pool_size} candidate codewords")
    """

    LARGE_CANDIDATE_WARNING_THRESHOLD = 50000

    def __init__(
        self,
        seq_rounds: int,
        codebook_size: int,
        hamming_weights: List[int] | None = None,
        hamming_weight_limits: Dict[int, int] | None = None,
        subsample_seed: int | None = None,
        required_codewords: list[Iterable[str]] | None = None,
        group_name: str = "MERFISH",
        per_codeword_sample_size: int | None = None,
        sample_seed: int | None = None,
    ):
        """Initialize MERFISH factory.

        Args:
            seq_rounds: Length of binary codewords (number of bits/rounds).
            codebook_size: Number of codewords to select for final codebook.
            hamming_weights: List of allowed Hamming weights. If None, all
                2^seq_rounds binary sequences are included (all weights 0 to seq_rounds).
            hamming_weight_limits: Optional per-Hamming-weight subsample cap.
                Keys must be in hamming_weights. Values must be positive ints.
            subsample_seed: Seed for the subsample RNG. Used only when at least
                one cap in hamming_weight_limits is binding.
            required_codewords: Optional positional list (or tuple) of
                length codebook_size; each element is an Iterable[str] of
                required codeword sequences. required_codewords[i] is the
                set of codewords that must appear in pos_i's sample (when
                per_codeword_sample_size is set) or in the pool (when it's
                not). Empty iterable at position i = no requirement at that
                position. None = no requirements anywhere. Must support
                len(); generators are not accepted.
            group_name: Pseudo-group name for the codebook (default: "MERFISH").
            per_codeword_sample_size: If set, build one group per codeword
                position (``pos_0`` .. ``pos_{codebook_size-1}``) whose
                membership is a random sample of this size drawn from the
                enumerated codeword pool, and assign each group quota=1. If
                None (default), use a single pseudo-group ``group_name`` with
                quota=codebook_size (original behavior).
            sample_seed: Seed for the per-codeword sampling RNG. Distinct from
                ``subsample_seed`` (which controls Hamming-weight-cap
                subsampling).

        Raises:
            ValueError: For invalid hamming_weights, hamming_weight_limits,
                required_codewords, per_codeword_sample_size, or if the
                required-codeword count for any weight exceeds that weight's
                cap.
        """
        # Validate seq_rounds
        if not isinstance(seq_rounds, int) or seq_rounds <= 0:
            raise ValueError(f"seq_rounds must be a positive integer, got {seq_rounds}")

        # Validate codebook_size
        if not isinstance(codebook_size, int) or codebook_size <= 0:
            raise ValueError(f"codebook_size must be a positive integer, got {codebook_size}")

        # Validate hamming_weights if provided
        if hamming_weights is not None:
            if not isinstance(hamming_weights, (list, tuple)):
                raise ValueError(f"hamming_weights must be a list, got {type(hamming_weights)}")
            for w in hamming_weights:
                if not isinstance(w, int):
                    raise ValueError(f"All Hamming weights must be integers, got {type(w)}")
                if w < 0 or w > seq_rounds:
                    raise ValueError(
                        f"Hamming weight {w} is invalid. Must be in range [0, {seq_rounds}]"
                    )
            if len(hamming_weights) != len(set(hamming_weights)):
                raise ValueError("hamming_weights must not contain duplicates")
            hamming_weights = sorted(hamming_weights)

        allowed_weights = (
            set(hamming_weights) if hamming_weights is not None
            else set(range(seq_rounds + 1))
        )

        # Validate hamming_weight_limits
        if hamming_weight_limits is not None:
            bad = set(hamming_weight_limits) - allowed_weights
            if bad:
                raise ValueError(
                    f"hamming_weight_limits keys {sorted(bad)} not in hamming_weights "
                    f"{sorted(allowed_weights)}"
                )
            for w, cap in hamming_weight_limits.items():
                if not isinstance(cap, int) or cap <= 0:
                    raise ValueError(
                        f"hamming_weight_limits[{w}] must be a positive int, got {cap}"
                    )

        # Validate required_codewords as a positional list (one Iterable[str]
        # per position). Build two derived stores:
        #   - self._required_by_position: list[set[str]] length codebook_size,
        #     drives per-position anchoring in the per-codeword sampling
        #     branch of create().
        #   - self._required_flat: set[str], union of all positional sets,
        #     drives HW-cap preservation in _enumerate_codewords (via
        #     _required_by_weight) and the post-sampling structural assert.
        self._required_by_position: List[set] = [set() for _ in range(codebook_size)]
        self._required_flat: set = set()
        if required_codewords is not None:
            if len(required_codewords) != codebook_size:
                raise ValueError(
                    f"required_codewords length {len(required_codewords)} != "
                    f"codebook_size {codebook_size}"
                )
            for i, seqs_at_i in enumerate(required_codewords):
                for seq in seqs_at_i:
                    if len(seq) != seq_rounds or any(c not in "01" for c in seq):
                        raise ValueError(
                            f"required codeword {seq!r} is not a length-{seq_rounds} binary string"
                        )
                    w = seq.count("1")
                    if w not in allowed_weights:
                        raise ValueError(
                            f"required codeword {seq!r} has HW={w}, not in hamming_weights "
                            f"{sorted(allowed_weights)}"
                        )
                    self._required_by_position[i].add(seq)
                    self._required_flat.add(seq)

        # _required_by_weight: union of required sequences by HW. Used by
        # _enumerate_codewords for HW-cap preservation. Built from
        # _required_flat so the contract for that path is unchanged.
        self._required_by_weight: Dict[int, set] = {}
        for seq in self._required_flat:
            self._required_by_weight.setdefault(seq.count("1"), set()).add(seq)

        # Cap-vs-required fail-fast check: with both inputs known at
        # construction, we can detect misconfiguration before any
        # enumeration work in .create().
        if hamming_weight_limits is not None:
            for w, cap in hamming_weight_limits.items():
                n_required = len(self._required_by_weight.get(w, ()))
                if n_required > cap:
                    raise ValueError(
                        f"HW={w}: {n_required} required codewords exceed cap {cap}"
                    )

        # Validate per_codeword_sample_size
        if per_codeword_sample_size is not None:
            if not isinstance(per_codeword_sample_size, int) or per_codeword_sample_size <= 0:
                raise ValueError(
                    f"per_codeword_sample_size must be a positive int or None, "
                    f"got {per_codeword_sample_size!r}"
                )
            # Per-position anchor count must fit in K.
            for i, anchors in enumerate(self._required_by_position):
                if len(anchors) > per_codeword_sample_size:
                    raise ValueError(
                        f"pos_{i}: anchor count {len(anchors)} exceeds "
                        f"per_codeword_sample_size {per_codeword_sample_size}"
                    )

        self.seq_rounds = seq_rounds
        self.codebook_size = codebook_size
        self.hamming_weights = hamming_weights
        self.hamming_weight_limits = dict(hamming_weight_limits) if hamming_weight_limits else {}
        self.subsample_seed = subsample_seed
        self.group_name = group_name
        self.per_codeword_sample_size = per_codeword_sample_size
        self.sample_seed = sample_seed

    def create(
        self,
        num_groups: int | None = None,  # Ignored for MERFISH
        seed: int | None = None,        # Ignored — subsampling is seeded via __init__'s subsample_seed
    ) -> CandidatePool:
        """Create the MERFISH CandidatePool.

        Args:
            num_groups: Ignored (MERFISH uses a single pseudo-group).
            seed: Ignored. Subsampling determinism is controlled by the
                ``subsample_seed`` argument to ``__init__``, not by this
                parameter; ``create()`` is kept seedless to match the
                ``CandidatePoolFactory`` interface.

        Returns:
            CandidatePool with the enumerated codewords. When no
            ``hamming_weight_limits`` entry binds, the pool contains every
            binary codeword whose Hamming weight is in ``hamming_weights``.
            When a cap binds for a weight, that weight contributes a
            uniform random subsample (size ``cap``) drawn from the
            non-required codewords plus all required codewords for that
            weight; combinatorial order within each weight is preserved.

        Raises:
            ValueError: If codebook_size exceeds number of candidates.
        """
        # Estimate candidate count and warn if large
        estimated_count = self._estimate_candidate_count()
        if estimated_count > self.LARGE_CANDIDATE_WARNING_THRESHOLD:
            logger.warning(
                f"Generating {estimated_count} candidate codewords. "
                f"This may require significant memory."
            )

        # Enumerate codewords
        codewords = self._enumerate_codewords()

        # Validate codebook_size
        if self.codebook_size > len(codewords):
            raise ValueError(
                f"codebook_size ({self.codebook_size}) exceeds number of "
                f"available candidates ({len(codewords)})"
            )

        # Log generation info
        weights_str = (
            str(self.hamming_weights) if self.hamming_weights is not None
            else f"all (0 to {self.seq_rounds})"
        )
        logger.info(
            f"Generated {len(codewords)} candidate codewords "
            f"(seq_rounds={self.seq_rounds}, hamming_weights={weights_str})"
        )

        # Build and return CandidatePool
        pool_size = len(codewords)
        if self.per_codeword_sample_size is None:
            # Original path: single pseudo-group with quota=codebook_size
            group_to_candidates = {
                self.group_name: list(range(pool_size)),
            }
            quotas = {self.group_name: self.codebook_size}
            pool_sequences = codewords
            pool_size_final = pool_size
        else:
            K = self.per_codeword_sample_size
            pool_size_input = pool_size  # len(codewords) computed above
            sample_size = min(K, pool_size_input)
            rng = np.random.default_rng(self.sample_seed)

            # Build a sequence->pool-index lookup once. The anchor sequences
            # come from _required_by_position, which validation already
            # ensured live in the allowed-weight set, so they must be in
            # `codewords`. A KeyError here is a structural drift bug.
            seq_to_pool_idx = {seq: i for i, seq in enumerate(codewords)}

            raw_samples: Dict[str, list] = {}
            for i in range(self.codebook_size):
                group_name = f"pos_{i}"
                anchor_indices = [
                    seq_to_pool_idx[seq] for seq in self._required_by_position[i]
                ]
                anchor_set = set(anchor_indices)
                # Draw (sample_size - len(anchors)) more from the non-anchor
                # pool, then prepend the anchors so the renumbering loop's
                # "ordered by first occurrence" semantics places anchors at
                # the front of the pruned pool (stable, seed-derived order
                # for anchors via Python's set->list ordering would not be
                # reproducible, so we sort anchor_indices for determinism).
                anchor_indices_sorted = sorted(anchor_indices)
                n_remaining = sample_size - len(anchor_indices_sorted)
                non_anchor_pool = np.array(
                    [j for j in range(pool_size_input) if j not in anchor_set],
                    dtype=np.int64,
                )
                if n_remaining > 0:
                    drawn = rng.choice(non_anchor_pool, size=n_remaining, replace=False)
                    sample = anchor_indices_sorted + drawn.tolist()
                else:
                    sample = anchor_indices_sorted
                raw_samples[group_name] = sample

            # Build the pruned sequence list: union of sampled indices, ordered
            # by first occurrence across groups (pos_0, pos_1, ...).
            old_to_new: Dict[int, int] = {}
            pool_sequences = []
            for group_name in raw_samples:
                for old_idx in raw_samples[group_name]:
                    if old_idx not in old_to_new:
                        old_to_new[old_idx] = len(pool_sequences)
                        pool_sequences.append(codewords[old_idx])

            # Renumber group memberships into the pruned sequence list.
            group_to_candidates = {
                group_name: [old_to_new[old_idx] for old_idx in old_indices]
                for group_name, old_indices in raw_samples.items()
            }
            quotas = {group_name: 1 for group_name in raw_samples}
            pool_size_final = len(pool_sequences)

            # Structural assert: every flat-required sequence must be in the
            # pruned pool (guaranteed by per-position anchoring; cheap defense
            # against future drift).
            pool_seq_set = set(pool_sequences)
            missing = self._required_flat - pool_seq_set
            assert not missing, (
                f"required_codewords not covered by per-position samples: {sorted(missing)}"
            )

        return CandidatePool(
            sequences=pool_sequences,
            group_to_candidates=group_to_candidates,
            scores=np.ones(pool_size_final, dtype=np.float64),
            quotas=quotas,
            metadata={
                "source": "MERFISH",
                "seq_rounds": self.seq_rounds,
                "codebook_size": self.codebook_size,
                "hamming_weights": self.hamming_weights,
                "num_candidates": pool_size_final,
                "per_codeword_sample_size": self.per_codeword_sample_size,
                "sample_seed": self.sample_seed,
            },
        )

    def _enumerate_codewords(self) -> List[str]:
        """Generate binary strings with allowed Hamming weights, applying
        per-weight subsample caps and preserving required codewords.

        The cap-vs-required cross-check has already happened in __init__,
        so this method only encounters consistent configurations.

        Returns:
            List of binary strings in combinatorial enumeration order.
        """
        codewords = []
        weights = (
            self.hamming_weights
            if self.hamming_weights is not None
            else range(self.seq_rounds + 1)
        )
        rng = np.random.default_rng(self.subsample_seed)

        for weight in weights:
            # Enumerate all C(seq_rounds, weight) codewords for this weight,
            # preserving combinatorial order.
            all_for_weight: List[str] = []
            for positions in combinations(range(self.seq_rounds), weight):
                bits = ['0'] * self.seq_rounds
                for pos in positions:
                    bits[pos] = '1'
                all_for_weight.append(''.join(bits))

            required = self._required_by_weight.get(weight, set())
            cap = self.hamming_weight_limits.get(weight)
            available = len(all_for_weight)

            if cap is None or available <= cap:
                if cap is not None:
                    logger.info(
                        f"HW={weight}: cap {cap} >= available {available}; using all"
                    )
                codewords.extend(all_for_weight)
                continue

            # Cap is binding. Partition into required and non-required by
            # index into the enumeration so the final output preserves
            # combinatorial order.
            required_idx = [i for i, s in enumerate(all_for_weight) if s in required]
            non_required_idx = [
                i for i, s in enumerate(all_for_weight) if s not in required
            ]
            n_to_draw = cap - len(required_idx)
            drawn = rng.choice(len(non_required_idx), size=n_to_draw, replace=False)
            chosen = sorted(
                set(required_idx).union(non_required_idx[i] for i in drawn)
            )
            per_weight = [all_for_weight[i] for i in chosen]
            logger.info(
                f"HW={weight}: subsampled {cap}/{available} "
                f"(C({self.seq_rounds},{weight})); preserved "
                f"{len(required_idx)} required"
            )
            codewords.extend(per_weight)
        return codewords

    def _estimate_candidate_count(self) -> int:
        """Estimate number of candidates without full enumeration.

        Uses binomial coefficients: sum of C(n, k) for each allowed weight k.

        Returns:
            Estimated number of candidate codewords.
        """
        weights = (
            self.hamming_weights
            if self.hamming_weights is not None
            else range(self.seq_rounds + 1)
        )

        total = sum(
            min(math.comb(self.seq_rounds, k), self.hamming_weight_limits.get(k, math.inf))
            for k in weights
        )
        return int(total)
