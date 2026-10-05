"""Greedy optimizers for codebook selection."""
from typing import Dict, Iterable, List

import numpy as np

from duet.codebook_evaluator import DecodingMetric, HammingDistance


class SivanandanOptimizer:
    """Greedy sgRNA selection following Sivanandan et al. 2023.

    Implements the deterministic greedy algorithm described in:
    Sivanandan et al. (2023) "Pooled optical screens in human cells"
    https://www.biorxiv.org/content/10.1101/2023.08.13.553051v3.full.pdf

    Algorithm:
        1. Partition groups into targeting genes and control groups.
        2. Sort targeting genes ascending by number of available candidates
           (most constrained first). This heuristic gives genes with fewer
           candidates priority access to the Hamming space.
        3. Append control groups after all targeting genes.
        4. For each group, sort candidates descending by score, then greedily
           select candidates that satisfy the minimum Hamming distance
           constraint against all previously selected candidates.

    The algorithm is fully deterministic. Ties in gene availability and
    within-gene scores are broken by input order (Python's stable sort).

    Note:
        ``gene_to_N`` and ``gene_to_indices`` must share the same keys.
    """

    def optimize(
        self,
        input_hamming_matrix,
        gene_to_indices,
        gene_to_N,
        target_min_hamming,
        scores,
        control_groups=None,
    ):
        """Select candidates greedily under a minimum Hamming distance constraint.

        Args:
            input_hamming_matrix: HammingMatrix instance for pairwise distances.
            gene_to_indices: Mapping from group name to list of candidate indices.
            gene_to_N: Mapping from group name to number of candidates to select.
            target_min_hamming: Minimum Hamming distance required between any
                two selected candidates.
            scores: Array of candidate scores indexed by candidate index.
                Candidates within each group are tried in descending score order.
            control_groups: Set of group names that are controls. Controls are
                processed after all targeting genes. If None or empty, all
                groups are treated as targeting genes.

        Returns:
            Set of selected candidate indices.
        """
        if control_groups is None:
            control_groups = set()

        # Partition into targeting and control groups
        targeting = []
        controls = []
        for gene in gene_to_N:
            if gene in control_groups:
                controls.append(gene)
            else:
                targeting.append(gene)

        # Sort targeting genes ascending by number of available candidates
        targeting.sort(key=lambda g: len(gene_to_indices[g]))

        # Controls go last
        gene_list = targeting + controls

        # Greedy selection
        S = set()
        for gene in gene_list:
            N = gene_to_N[gene]
            # Sort candidates descending by score
            indices = sorted(
                gene_to_indices[gene],
                key=lambda idx: scores[idx],
                reverse=True,
            )
            selected = 0
            for index in indices:
                min_hamming = input_hamming_matrix.min(S | {index})
                if min_hamming >= target_min_hamming:
                    S.add(index)
                    selected += 1
                    if selected == N:
                        break
        return S


class GreedyDistanceOptimizer:
    """Greedy codebook construction by maximizing minimum pairwise distance.

    Iteratively builds a codebook by processing groups in random order.
    For each group, selects the candidate that maximizes the minimum
    symmetrized distance to all previously selected candidates.

    Distances are symmetrized: d_sym(a, b) = min(d(a, b), d(b, a)). For
    symmetric metrics like HammingDistance this equals d(a, b) and does not
    change the selection. For asymmetric metrics like AsymmetricNLL or
    PositionVaryingAsymmetricNLL, taking the minimum captures the "most confusable"
    direction — two codewords are only well-separated if they are separated
    in both directions. This avoids the problem where inf costs from
    zero-probability channel transitions mask finite confusability in the
    opposite direction.

    The decoding metric parameter controls the distance metric:
    - HammingDistance (default): noise-channel-agnostic (symmetric)
    - Any NLL variant: noise-channel-aware (asymmetric variants are symmetrized)
    """

    def __init__(self, decoding_metric: DecodingMetric | None = None):
        self.decoding_metric = decoding_metric or HammingDistance()

    def optimize(
        self,
        sequences: np.ndarray,
        group_to_candidates: Dict[str, List[int]],
        quotas: Dict[str, int],
        seed: int,
    ) -> np.ndarray:
        """Select candidates greedily to maximize minimum pairwise distance.

        Args:
            sequences: Integer-encoded sequences, shape (pool_size, seq_length).
            group_to_candidates: Mapping from group name to candidate indices.
            quotas: Number of candidates to select per group.
            seed: Random seed for group and candidate shuffling.

        Returns:
            Array of selected candidate indices.
        """
        rng = np.random.default_rng(seed)

        # Shuffle group order
        groups = list(group_to_candidates.keys())
        rng.shuffle(groups)

        selected = []

        for group in groups:
            candidates = list(group_to_candidates[group])
            rng.shuffle(candidates)
            quota = quotas[group]

            for _ in range(quota):
                if not selected:
                    # First candidate overall: take first from shuffled order
                    selected.append(candidates[0])
                    candidates.remove(selected[-1])
                    continue

                # Compute symmetrized distance from each candidate to all selected.
                # d_sym(a, b) = min(d(a, b), d(b, a)) takes the most confusable
                # direction. For symmetric metrics (Hamming), this is just d
                # and doesn't affect rankings. For asymmetric NLL metrics,
                # this avoids inf values masking finite confusability.
                selected_seqs = sequences[selected]  # (num_selected, L)
                candidate_seqs = sequences[candidates]  # (num_candidates, L)

                dist_forward = self.decoding_metric.compute(
                    candidate_seqs, selected_seqs
                )  # (num_candidates, num_selected)
                dist_reverse = self.decoding_metric.compute(
                    selected_seqs, candidate_seqs
                ).T  # (num_selected, num_candidates).T = (num_candidates, num_selected)
                dist_matrix = np.minimum(dist_forward, dist_reverse)

                # For each candidate, get minimum distance to any selected
                min_dists = dist_matrix.min(axis=1)  # (num_candidates,)

                # Select candidate with maximum minimum distance
                best_idx = int(np.argmax(min_dists))
                selected.append(candidates[best_idx])
                candidates.pop(best_idx)

        return np.array(selected, dtype=int)


class GreedyHammingMOOptimizer:
    """Lambda-sweep scalarized greedy-Hamming MO baseline.

    Same algorithmic frame as DUET (greedy + lambda sweep), different signal
    (raw pairwise Hamming distance to the already-selected set, never PEP).
    Per-step normalization makes lambda scale-clean: λ=1 reduces to max-min
    Hamming greedy, λ=0 to max-score-only greedy (λ weights the distance
    term, matching DUET's convention that λ weights decoding accuracy).

    This class never instantiates `CodebookEvaluator` or computes a PEP
    matrix; that property is the experimental ablation point and is
    enforced by tests.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def optimize(
        self,
        sequences: np.ndarray,
        group_to_candidates: Dict[str, List[int]],
        quotas: Dict[str, int],
        scores: np.ndarray,
        lambdas: Iterable[float],
    ) -> Dict[float, np.ndarray]:
        """Run the lambda sweep, returning one selection per lambda.

        Note: GreedyDistanceOptimizer omits the `scores` argument because
        its objective is purely Hamming-based. GreedyHammingMOOptimizer
        requires `scores` because the lambda blend reads them directly.
        """
        rng = np.random.default_rng(self.seed)

        # Single shuffle per `optimize` call; shared across lambdas to keep
        # the lambda-sweep trace smooth (per-lambda reshuffles would inject
        # extra randomness that obscures the trade-off curve).
        groups = list(group_to_candidates.keys())
        rng.shuffle(groups)
        candidates_by_group = {
            g: list(group_to_candidates[g]) for g in groups
        }
        for g in groups:
            rng.shuffle(candidates_by_group[g])

        # Hamming distance precomputation (full pairwise); cheap for
        # synthetic-benchmark pool sizes and kept once across lambdas.
        sequences = np.asarray(sequences)
        n = sequences.shape[0]
        dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            dist[i] = (sequences != sequences[i]).sum(axis=1)

        results: Dict[float, np.ndarray] = {}
        for lambda_ in lambdas:
            results[float(lambda_)] = self._run_one_lambda(
                lambda_=float(lambda_),
                groups=groups,
                candidates_by_group=candidates_by_group,
                quotas=quotas,
                scores=scores,
                dist=dist,
            )
        return results

    @staticmethod
    def _run_one_lambda(
        lambda_: float,
        groups: List[str],
        candidates_by_group: Dict[str, List[int]],
        quotas: Dict[str, int],
        scores: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        selected: List[int] = []
        # Note: candidates_by_group is shared across lambdas, so we work on
        # a local copy of each group's list to track which have been chosen.
        per_group_remaining = {g: list(c) for g, c in candidates_by_group.items()}

        for g in groups:
            quota = quotas[g]
            for _ in range(quota):
                cands = per_group_remaining[g]
                if not cands:
                    break

                # Distance gain: min Hamming to any already-selected codeword.
                # First-pick convention: gain_d = 0 for all (per-step
                # normalization yields gain_d_norm = 0 → blend at λ > 0
                # collapses to argmax score).
                if not selected:
                    gain_d = np.zeros(len(cands))
                else:
                    sub = dist[np.ix_(cands, selected)]
                    gain_d = sub.min(axis=1).astype(float)

                gain_s = scores[cands].astype(float)

                # Symmetric per-step min-max normalization on BOTH axes.
                # If max == min on an axis, set normalized gain to 0.
                d_min, d_max = float(gain_d.min()), float(gain_d.max())
                gain_d_norm = (
                    np.zeros_like(gain_d)
                    if d_max == d_min
                    else (gain_d - d_min) / (d_max - d_min)
                )
                s_min, s_max = float(gain_s.min()), float(gain_s.max())
                gain_s_norm = (
                    np.zeros_like(gain_s)
                    if s_max == s_min
                    else (gain_s - s_min) / (s_max - s_min)
                )

                blend = lambda_ * gain_d_norm + (1.0 - lambda_) * gain_s_norm
                pick_local = int(np.argmax(blend))
                pick = cands[pick_local]
                selected.append(pick)
                per_group_remaining[g] = [c for c in cands if c != pick]

        return np.array(selected, dtype=int)


class GreedyNLLMOOptimizer:
    """Lambda-sweep scalarized greedy-NLL MO baseline.

    Same algorithmic frame as `GreedyHammingMOOptimizer` (greedy + lambda
    sweep, per-step min-max normalization on both axes, shuffled tie-break)
    but the pairwise decoding metric is supplied externally via a
    noise-channel-aware NLL function rather than fixed to Hamming distance.

    Like `GreedyHammingMOOptimizer`, this class never instantiates
    `CodebookEvaluator` or computes a PEP matrix. That property is the
    experimental ablation point and is enforced by tests
    (`test_no_pep_machinery_used`).

    Asymmetric NLL channels are symmetrized via min: d_sym(a, b) =
    min(d(a, b), d(b, a)) — matching `GreedyDistanceOptimizer`. This
    captures the "most confusable direction" and prevents inf costs
    from zero-probability channel transitions in one direction from
    masking finite confusability in the opposite direction.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def optimize(
        self,
        sequences: np.ndarray,
        group_to_candidates: Dict[str, List[int]],
        quotas: Dict[str, int],
        scores: np.ndarray,
        lambdas: Iterable[float],
        decoding_metric: DecodingMetric,
    ) -> Dict[float, np.ndarray]:
        """Run the lambda sweep, returning one selection per lambda.

        Args:
            sequences: Integer-encoded sequences, shape (pool_size, seq_length).
            group_to_candidates: Mapping from group name to candidate indices.
            quotas: Number of candidates to select per group.
            scores: Per-candidate scalar scores, used in the lambda blend.
            lambdas: Iterable of lambda values to sweep.
            decoding_metric: Noise-channel-aware NLL function (SymmetricNLL,
                PositionVaryingNLL, AsymmetricNLL, or PositionVaryingAsymmetricNLL).
                The runner should pass the same decoding metric it builds
                for DUET so all three methods see the same noise channel.

        Returns:
            Dict mapping lambda → array of selected candidate indices.
        """
        rng = np.random.default_rng(self.seed)

        # Single shuffle per `optimize` call; shared across lambdas to keep
        # the lambda-sweep trace smooth (matching GreedyHammingMOOptimizer).
        groups = list(group_to_candidates.keys())
        rng.shuffle(groups)
        candidates_by_group = {
            g: list(group_to_candidates[g]) for g in groups
        }
        for g in groups:
            rng.shuffle(candidates_by_group[g])

        sequences = np.asarray(sequences)
        # Full pairwise NLL matrix [n, n], symmetrized via min over both
        # directions. Divergence from GreedyDistanceOptimizer's on-the-fly
        # pattern is intentional: this optimizer is called once per cell
        # but its `dist` is read once per lambda × per pick, so amortizing
        # the cost upfront beats recomputing inside the lambda sweep. For
        # n up to a few thousand the n×n matrix is trivially small.
        d_forward = decoding_metric.compute(sequences, sequences)
        dist = np.minimum(d_forward, d_forward.T)

        results: Dict[float, np.ndarray] = {}
        for lambda_ in lambdas:
            results[float(lambda_)] = self._run_one_lambda(
                lambda_=float(lambda_),
                groups=groups,
                candidates_by_group=candidates_by_group,
                quotas=quotas,
                scores=scores,
                dist=dist,
            )
        return results

    @staticmethod
    def _run_one_lambda(
        lambda_: float,
        groups: List[str],
        candidates_by_group: Dict[str, List[int]],
        quotas: Dict[str, int],
        scores: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        # Body matches GreedyHammingMOOptimizer._run_one_lambda exactly —
        # only the `dist` matrix's semantics differ (NLL vs Hamming).
        selected: List[int] = []
        per_group_remaining = {g: list(c) for g, c in candidates_by_group.items()}

        for g in groups:
            quota = quotas[g]
            for _ in range(quota):
                cands = per_group_remaining[g]
                if not cands:
                    break

                if not selected:
                    gain_d = np.zeros(len(cands))
                else:
                    sub = dist[np.ix_(cands, selected)]
                    gain_d = sub.min(axis=1).astype(float)

                gain_s = scores[cands].astype(float)

                d_min, d_max = float(gain_d.min()), float(gain_d.max())
                gain_d_norm = (
                    np.zeros_like(gain_d)
                    if d_max == d_min
                    else (gain_d - d_min) / (d_max - d_min)
                )
                s_min, s_max = float(gain_s.min()), float(gain_s.max())
                gain_s_norm = (
                    np.zeros_like(gain_s)
                    if s_max == s_min
                    else (gain_s - s_min) / (s_max - s_min)
                )

                blend = lambda_ * gain_d_norm + (1.0 - lambda_) * gain_s_norm
                pick_local = int(np.argmax(blend))
                pick = cands[pick_local]
                selected.append(pick)
                per_group_remaining[g] = [c for c in cands if c != pick]

        return np.array(selected, dtype=int)
