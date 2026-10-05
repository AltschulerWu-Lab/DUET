"""
Evaluation metrics for benchmark experiments.

This module provides:
- Functions to create evaluators and evaluate guide selections
- Hypervolume indicator for comparing Pareto fronts
- Utilities for non-dominated sorting
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
# pymoo (the [benchmark] extra) is imported inside the hypervolume and
# non-dominated-sorting helpers below, so the evaluation functions
# (evaluate_codebooks_by_sequence, compute_metrics) work without it.
from duet.candidate_pool import CandidatePool
from duet.codebook_evaluator import CodebookEvaluator, ErrorCorrectionMetrics
from duet.evaluator_config import EvaluatorConfig, create_dna_evaluator, create_evaluator
from duet.utils import Stopwatch, log_memory


# =============================================================================
# Shared Union Evaluation Core
# =============================================================================


@dataclass(frozen=True)
class CodebookEvalResult:
    """Ground-truth eval output for one codebook: per-codeword accuracy +
    error-correction metrics, in codebook order."""

    codeword_accuracy: np.ndarray
    error_metrics: ErrorCorrectionMetrics

    @property
    def mean_accuracy(self) -> float:
        return float(self.codeword_accuracy.mean())


def _evaluate_codebooks_over_union(
    union_sequences: list[str],
    codebooks_in_union_space: list[np.ndarray],
    *,
    eval_config: EvaluatorConfig,
    alphabet_size: int,
    n_jobs: int | None = None,
    device: str = "cpu",
    mem_budget_gb: float = 20.0,
) -> list[CodebookEvalResult]:
    """Build ONE evaluator over ``union_sequences``, initialize the cache once,
    and batch-evaluate every codebook (given as arrays of union-row indices) in a
    single matmul. Returns one CodebookEvalResult per input codebook, in input
    order. Empty ``codebooks_in_union_space`` -> ``[]`` (no evaluator built)."""
    if not codebooks_in_union_space:
        return []
    evaluator = create_evaluator(union_sequences, eval_config, alphabet_size)
    log_memory("before evaluation cache init")
    with Stopwatch("Evaluation cache initialization"):
        evaluator.initialize_cache(n_jobs=n_jobs, device=device)
    log_memory("after evaluation cache init")
    with Stopwatch("Per-solution evaluation (matmul)"):
        cw_accuracy, metrics_list = evaluator.batch_get_accuracy_with_error_metrics(
            codebooks_in_union_space, mem_budget_gb=mem_budget_gb,
        )
    return [
        CodebookEvalResult(codeword_accuracy=acc, error_metrics=metrics)
        for acc, metrics in zip(cw_accuracy, metrics_list)
    ]


def _build_multiset_union(
    codebooks: Sequence[Sequence[str]],
) -> tuple[list[str], list[np.ndarray]]:
    """Max-multiplicity multiset union over sequence strings (MERFISH).

    Union multiplicity of a sequence = the MAX number of times it occurs in any
    single codebook. Identical codewords are shared across codebooks (one union
    row), while duplicate codewords WITHIN a codebook stay distinct competitor
    columns. Returns (union_sequences, codebooks_in_union_space) in input order.
    """
    max_mult: dict[str, int] = {}
    for seqs in codebooks:
        for seq, c in Counter(seqs).items():
            max_mult[seq] = max(max_mult.get(seq, 0), c)

    union_sequences: list[str] = []
    base: dict[str, int] = {}
    for seq, m in max_mult.items():          # first-encounter order => deterministic
        base[seq] = len(union_sequences)
        union_sequences.extend([seq] * m)

    codebooks_in_union_space: list[np.ndarray] = []
    for seqs in codebooks:
        occ: Counter = Counter()
        rows = []
        for seq in seqs:
            k = occ[seq]
            occ[seq] += 1
            rows.append(base[seq] + k)       # k-th occurrence -> k-th union copy
        codebooks_in_union_space.append(np.asarray(rows, dtype=int))
    return union_sequences, codebooks_in_union_space


def evaluate_codebooks_by_sequence(
    codebooks: Sequence[Sequence[str]],
    *,
    eval_config: EvaluatorConfig,
    alphabet_size: int,
    n_jobs: int | None = None,
    device: str = "cpu",
    mem_budget_gb: float = 20.0,
) -> list[CodebookEvalResult]:
    """MERFISH union eval. Positional: one CodebookEvalResult per input codebook,
    in input order. No scores. The union is a max-multiplicity multiset over
    sequence strings, so duplicate codewords within a codebook decode at ~100%
    error (they stay distinct competitor columns)."""
    if not codebooks:
        return []
    union_sequences, codebooks_in_union_space = _build_multiset_union(codebooks)
    return _evaluate_codebooks_over_union(
        union_sequences, codebooks_in_union_space,
        eval_config=eval_config, alphabet_size=alphabet_size,
        n_jobs=n_jobs, device=device, mem_budget_gb=mem_budget_gb,
    )


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_indices(
    indices: Iterable[int],
    scores: np.ndarray,
    evaluator: CodebookEvaluator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate decode accuracy and scores for a set of candidate indices.

    Args:
        indices: Iterable of candidate indices to evaluate.
        scores: Array of scores for all candidates.
        evaluator: Initialized CodebookEvaluator instance.

    Returns:
        Tuple of (decode_accuracies, scores) arrays.
            - decode_accuracies: Per-codeword decoding accuracy
            - scores: Score for each selected candidate

    Raises:
        RuntimeError: If evaluator cache is not initialized.
    """
    idx = np.asarray(list(indices), dtype=int)
    decode_acc = evaluator.get_codeword_accuracy(indices=idx)
    score_values = scores[idx]
    return decode_acc, score_values


def evaluate_codebook(
    indices: Iterable[int],
    candidates: CandidatePool,
    eval_config: EvaluatorConfig,
    alphabet_size: int,
    n_jobs: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, ErrorCorrectionMetrics]:
    """Evaluate a codebook by creating a per-codebook evaluator.

    Instead of subsetting a giant evaluator built over all candidates,
    this creates a small CodebookEvaluator containing only the selected
    codewords, initializes it, and computes accuracy and error metrics.
    Memory usage scales with codebook size, not pool size.

    Args:
        indices: Candidate indices defining the codebook to evaluate.
        candidates: CandidatePool instance (provides sequences and scores).
        eval_config: EvaluatorConfig for the ground truth evaluator.
        alphabet_size: Alphabet size (4 for DNA, 16 for hex, etc.).
        n_jobs: Number of parallel workers for evaluator initialization.

    Returns:
        Tuple of (decode_accuracies, scores, error_metrics):
            - decode_accuracies: Per-codeword decoding accuracy array
            - scores: Per-codeword activity scores array
            - error_metrics: ErrorCorrectionMetrics instance
    """
    idx = np.asarray(list(indices), dtype=int)
    codebook_sequences = [candidates.sequences[i] for i in idx]
    evaluator = create_evaluator(codebook_sequences, eval_config, alphabet_size)
    evaluator.initialize_cache(n_jobs=n_jobs)
    decode_acc = evaluator.get_codeword_accuracy()
    error_metrics = evaluator.get_error_correction_metrics()
    scores = candidates.scores[idx]
    return decode_acc, scores, error_metrics


def evaluate_solutions(
    solutions: dict[str, np.ndarray],
    candidates: CandidatePool,
    eval_config: EvaluatorConfig,
    alphabet_size: int,
    n_jobs: int | None = None,
    device: str = "cpu",
    mem_budget_gb: float = 20.0,
) -> dict[str, tuple[np.ndarray, np.ndarray, ErrorCorrectionMetrics]]:
    """Evaluate multiple codebooks using a single shared evaluator.

    Builds one evaluator over the union of all codewords, initializes the
    cache once, and evaluates all solutions simultaneously via
    CodebookEvaluator.batch_get_accuracy_with_error_metrics().

    Args:
        solutions: Mapping from solution name to array of candidate pool
            indices defining that solution's codebook.
        candidates: CandidatePool instance (provides sequences and scores).
        eval_config: EvaluatorConfig for the ground truth evaluator.
        alphabet_size: Alphabet size (4 for DNA, 16 for hex, etc.).
        n_jobs: Number of parallel workers for evaluator initialization.
        device: Computation device ("cpu" or "gpu").
        mem_budget_gb: Memory budget in GB for batch size estimation.

    Returns:
        Dict mapping solution name to (decode_accuracies, scores,
        error_metrics) tuple.
    """
    # 1. Union of all solution indices (SET — distinct pool indices; preserves OPS
    #    decode semantics for codebooks where two distinct candidates share a sequence).
    all_indices = set()
    for idx in solutions.values():
        all_indices.update(int(i) for i in idx)
    union_pool_indices = sorted(all_indices)
    pool_to_union = {p: u for u, p in enumerate(union_pool_indices)}
    union_sequences = [candidates.sequences[i] for i in union_pool_indices]

    # 2. Translate each solution to union-index space (input/insertion order preserved).
    solution_names = list(solutions.keys())
    codebooks_in_union_space = [
        np.array([pool_to_union[int(i)] for i in solutions[name]])
        for name in solution_names
    ]

    # 3. Shared core: one cache build + one batch matmul (Stopwatch/log_memory live there).
    core_results = _evaluate_codebooks_over_union(
        union_sequences, codebooks_in_union_space,
        eval_config=eval_config, alphabet_size=alphabet_size,
        n_jobs=n_jobs, device=device, mem_budget_gb=mem_budget_gb,
    )

    # 4. Gather scores post-core (OPS-only) and package in the legacy 3-tuple shape.
    results = {}
    for name, res in zip(solution_names, core_results):
        scores = candidates.scores[np.asarray(solutions[name], dtype=int)]
        results[name] = (res.codeword_accuracy, scores, res.error_metrics)
    return results


def evaluate_mean_objectives(
    indices: Iterable[int],
    scores: np.ndarray,
    evaluator: CodebookEvaluator,
) -> Tuple[float, float]:
    """Evaluate mean decode accuracy and mean score.

    Args:
        indices: Iterable of candidate indices to evaluate.
        scores: Array of scores for all candidates.
        evaluator: Initialized CodebookEvaluator instance.

    Returns:
        Tuple of (mean_decode_accuracy, mean_score).
    """
    decode_acc, score_values = evaluate_indices(indices, scores, evaluator)
    return float(np.mean(decode_acc)), float(np.mean(score_values))


# =============================================================================
# Hypervolume Indicator for Pareto Front Comparison
# =============================================================================


@dataclass
class FrontResult:
    """Results for a single Pareto front.

    Attributes:
        hypervolume: Raw hypervolume value.
        normalized_hv: Hypervolume normalized by ideal hypervolume.
        n_solutions: Number of solutions in the Pareto front.
        pareto_front: Array of Pareto-optimal points.
    """

    hypervolume: float
    normalized_hv: float
    n_solutions: int
    pareto_front: np.ndarray


@dataclass
class ComparisonResults:
    """Results from comparing multiple Pareto fronts.

    Attributes:
        fronts: List of FrontResult objects, one per input front.
        reference_point: Reference point used for hypervolume calculation.
        ideal_hypervolume: Theoretical maximum hypervolume.
    """

    fronts: List[FrontResult]
    reference_point: np.ndarray
    ideal_hypervolume: float

    def get_best_front(self, metric: str = "hypervolume") -> FrontResult:
        """Get the best performing front based on specified metric.

        Args:
            metric: One of "hypervolume", "normalized_hv", or "n_solutions".

        Returns:
            FrontResult with highest value for the specified metric.

        Raises:
            ValueError: If metric is not recognized.
        """
        if metric == "hypervolume":
            return max(self.fronts, key=lambda f: f.hypervolume)
        elif metric == "normalized_hv":
            return max(self.fronts, key=lambda f: f.normalized_hv)
        elif metric == "n_solutions":
            return max(self.fronts, key=lambda f: f.n_solutions)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def to_dict(self) -> dict:
        """Convert results to dictionary format for backward compatibility."""
        result_dict = {
            "reference_point": self.reference_point,
            "ideal_hypervolume": self.ideal_hypervolume,
        }

        for i, front_result in enumerate(self.fronts):
            result_dict[f"front_{i}"] = {
                "hypervolume": front_result.hypervolume,
                "normalized_hv": front_result.normalized_hv,
                "n_solutions": front_result.n_solutions,
                "pareto_front": front_result.pareto_front,
            }

        return result_dict


def calculate_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Calculate hypervolume for a set of 2D points.

    Args:
        points: Array of shape (n_points, 2).
        ref_point: Reference point [x_ref, y_ref].

    Returns:
        Hypervolume value.
    """
    from pymoo.indicators.hv import HV

    ind = HV(ref_point=ref_point)
    return ind(points)


def filter_dominated_solutions(points: np.ndarray) -> np.ndarray:
    """Filter out dominated solutions, keeping only Pareto-optimal ones.

    Args:
        points: Array of shape (n_points, 2).

    Returns:
        Non-dominated points array.
    """
    from pymoo.util.nds.non_dominated_sorting import find_non_dominated

    # pymoo assumes minimization, so negate for maximization
    non_dom_indices = find_non_dominated(points)
    return points[non_dom_indices]


def compute_igd(
    approx_front: np.ndarray,
    true_front: np.ndarray,
    maximize: bool = False,
) -> float:
    """Inverted Generational Distance from `approx_front` to `true_front`.

    IGD = mean over true_front points of the minimum Euclidean distance to
    any point in approx_front. Lower is better; 0 iff approx_front contains
    every point of true_front.

    Args:
        approx_front: Recovered front, shape (n_approx, 2).
        true_front: Reference (true) front, shape (n_true, 2).
        maximize: If True, both fronts are negated before passing to pymoo
            (which assumes minimization), matching the convention used by
            ``compare_pareto_fronts``.
    """
    from pymoo.indicators.igd import IGD
    if maximize:
        approx_front = -np.asarray(approx_front)
        true_front = -np.asarray(true_front)
    return float(IGD(true_front)(approx_front))


def compute_hvr(
    approx_front: np.ndarray,
    true_front: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    maximize: bool = False,
) -> float:
    """Hypervolume Ratio: HV(approx_front) / HV(true_front).

    Both fronts are non-domination filtered, then evaluated against a
    shared reference point — either the supplied ``ref_point`` or one
    auto-computed from the union of both fronts (``worst + 0.1*delta``,
    matching ``compare_pareto_fronts``'s auto-ref convention).

    HVR = 1.0 when ``approx_front`` recovers exactly the same hypervolume
    as ``true_front``. Below 1.0 means the recovered front dominates a
    smaller region than the true PF.

    **Auto-ref caveat.** With ``ref_point=None``, the auto-ref is computed
    from the union of ``approx_front`` and ``true_front`` only. The 2-D
    synthetic benchmark's runner computes per-cell HVR by dividing HVs
    from ``compare_pareto_fronts``, whose auto-ref uses the union of
    *all* method fronts in the cell — so a standalone ``compute_hvr``
    call on the same approx/true pair can produce a different number.
    Pass an explicit ``ref_point`` for cross-call consistency.

    Args:
        approx_front: Recovered front, shape (n_approx, d).
        true_front: Reference (true) front, shape (n_true, d).
        ref_point: Shared reference point for both HV computations. If
            None, auto-computed from the union of both fronts using the
            same ``worst + 0.1*delta`` convention as
            ``compare_pareto_fronts``.
        maximize: If True, both fronts and the reference point are
            negated before passing to pymoo (which assumes minimization).

    Returns:
        HV(approx) / HV(true). Returns ``float('nan')`` if HV(true) == 0.
    """
    from pymoo.util.nds.non_dominated_sorting import find_non_dominated

    approx = np.asarray(approx_front, dtype=np.float64)
    true = np.asarray(true_front, dtype=np.float64)

    if maximize:
        approx = -approx
        true = -true
        if ref_point is not None:
            ref_point = -np.asarray(ref_point, dtype=np.float64)

    # Non-domination filter (pymoo minimization convention).
    approx = approx[find_non_dominated(approx)]
    true = true[find_non_dominated(true)]

    if ref_point is None:
        union = np.vstack([approx, true])
        best_point = np.min(union, axis=0)
        worst_point = np.max(union, axis=0)
        delta = worst_point - best_point
        ref_point = worst_point + 0.1 * delta
    else:
        ref_point = np.asarray(ref_point, dtype=np.float64)

    hv_true = calculate_hypervolume(true, ref_point)
    if hv_true == 0:
        return float("nan")
    hv_approx = calculate_hypervolume(approx, ref_point)
    return float(hv_approx / hv_true)


def compare_pareto_fronts(
    fronts: List[np.ndarray],
    ref_point: Optional[np.ndarray] = None,
    maximize: bool = False,
) -> ComparisonResults:
    """Compare multiple Pareto fronts using hypervolume indicator.

    Args:
        fronts: List of arrays, each of shape (n_points, 2).
        ref_point: Reference point for HV calculation. If None, auto-computed.
        maximize: If True, assumes maximization problem (will negate values).

    Returns:
        ComparisonResults object containing metrics for each front.

    Raises:
        ValueError: If any front is empty.
    """
    if any(len(front) == 0 for front in fronts):
        raise ValueError("All fronts must contain at least one point.")

    # Handle maximization problems by negating objectives
    if maximize:
        fronts = [-arr for arr in fronts]
        if ref_point is not None:
            ref_point = -np.array(ref_point)

    # Filter to keep only non-dominated solutions
    pareto_fronts = []
    for front in fronts:
        pareto = filter_dominated_solutions(front)
        pareto_fronts.append(pareto)

    # Collect all Pareto-optimal points for reference point calculation
    all_pareto_points = np.vstack([p for p in pareto_fronts if len(p) > 0])

    # Auto-compute reference point if not provided
    if ref_point is None:
        best_point = np.min(all_pareto_points, axis=0)
        worst_point = np.max(all_pareto_points, axis=0)
        delta = worst_point - best_point
        ref_point = worst_point + 0.1 * delta

    # Calculate best point for ideal hypervolume
    best_point = np.min(all_pareto_points, axis=0)
    ideal_hv = np.prod(ref_point - best_point)

    # Calculate hypervolumes and create FrontResult objects
    front_results = []
    for pareto in pareto_fronts:
        hv = calculate_hypervolume(pareto, ref_point)
        normalized_hv = hv / ideal_hv if ideal_hv > 0 else 0

        # Convert back to original space if maximization
        original_pareto = -pareto if maximize else pareto

        front_result = FrontResult(
            hypervolume=hv,
            normalized_hv=normalized_hv,
            n_solutions=len(pareto),
            pareto_front=original_pareto,
        )
        front_results.append(front_result)

    # Convert reference point back to original space if maximization
    original_ref_point = -ref_point if maximize else ref_point

    return ComparisonResults(
        fronts=front_results,
        reference_point=original_ref_point,
        ideal_hypervolume=ideal_hv,
    )


def compute_metrics(accuracies: np.ndarray) -> dict:
    """Compute decode accuracy summary statistics.

    Domain-agnostic: operates on a 1D numpy array of per-codeword accuracies.
    Used by MERFISH (ported from run_merfish.py) and available for OPS.
    """
    p5 = np.percentile(accuracies, 5)
    p10 = np.percentile(accuracies, 10)
    p95 = np.percentile(accuracies, 95)

    return {
        "Mean decode accuracy": float(accuracies.mean()),
        "Std decode accuracy": float(accuracies.std()),
        "5th percentile decode accuracy": float(p5),
        "10th percentile decode accuracy": float(p10),
        "95th percentile decode accuracy": float(p95),
        "Mean decode accuracy (<= 5th pct)": float(accuracies[accuracies <= p5].mean()) if (accuracies <= p5).any() else float(p5),
        "Mean decode accuracy (<= 10th pct)": float(accuracies[accuracies <= p10].mean()) if (accuracies <= p10).any() else float(p10),
        "95th/5th percentile ratio": float(p95 / p5) if p5 > 0 else float("inf"),
    }


def compute_duet_objective(
    pep_matrix: np.ndarray, codebook: np.ndarray
) -> float:
    """DUET's union-bound decode objective for one codebook.

    Equals ``DecodingSwapCache.compute_objective`` in
    ``src/duet/pareto_optimization.py``: the ordered off-diagonal sum (both
    ``M[i, j]`` and ``M[j, i]`` counted) of the codebook's PEP submatrix,
    divided by ``|S|``, subtracted from 1. The diagonal of ``pep_matrix``
    is irrelevant — only off-diagonal entries are summed.

    Args:
        pep_matrix: Raw asymmetric PEP matrix. ``M[i, j]`` is the pairwise
            error probability — the probability codeword ``j``'s decoding
            cost beats ``i``'s — *not* a decode-outcome distribution.
        codebook: Integer index array selecting the codebook members.

    Returns:
        Objective value (<= 1; can be negative). Rows of ``M`` are not
        stochastic — many ``j``'s can each beat a given ``i`` — so the
        off-diagonal sum can exceed ``|S|``, making the objective negative.
        This is the standard behavior of a union bound and matches what
        ``DecodingSwapCache.compute_objective`` returns.
    """
    sub = pep_matrix[np.ix_(codebook, codebook)]
    offdiag_sum = float(sub.sum() - np.trace(sub))
    return 1.0 - offdiag_sum / len(codebook)