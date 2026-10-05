"""Shared machinery for baseline-objectives experiments.

Currently hosts the noise-and-decoding-metric factory used by benchmarks
that sweep over the four canonical noise channels (`symmetric`,
`position_varying`, `asymmetric`, `position_varying_asymmetric`).

Also hosts the `PoolConfig`, `EvaluatorRunConfig`, and
`BaselineObjectivesConfig` dataclasses for configuring synthetic-pool
construction, evaluator setup, and enumeration-mode runs.

Also hosts the `OBJECTIVES` constant — the single source of truth
for the five baseline-objective columns, their display labels, and
their argmax/argmin direction — the shared `process_trial` inner loop
that both runners (enumeration and sampling) invoke, and the
`sample_codebooks` helper used by the sampling runner.

The `mean_pairwise_nll` / `min_pairwise_nll` columns reduce the centered,
direction-averaged matrix from `compute_pairwise_nll` — see that function for
the definition and for why it is not the raw negative log-likelihood.

History: the noise factory was previously defined inline in
`scripts/benchmark/synthetic/run_2d_synthetic_benchmark.py`. It was
moved here when the baseline-objectives `process_trial` inner loop
needed to live in `src/`; that move required moving the noise
factory too, since `process_trial` calls it directly.
"""
from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np

from duet.benchmark.metrics import compute_duet_objective
from duet.candidate_pool import CandidatePool
from duet.codebook_evaluator import (
    AsymmetricNLL,
    AsymmetricChannel,
    CodebookEvaluator,
    DecodingMetric,
    HammingDistance,
    NoiseChannel,
    PositionVaryingAsymmetricNLL,
    PositionVaryingAsymmetricChannel,
    PositionVaryingEpsilon,
    PositionVaryingNLL,
    UniqueMinimum,
    SymmetricEpsilon,
    SymmetricNLL,
)

logger = logging.getLogger(__name__)


_EPSILON_FLOOR = 1e-10


def _positional_ramp(lo: float, hi: float, length: int) -> List[float]:
    """Linear ramp from `lo` (at position 0) to `hi` (at position length-1).

    Average over positions equals (lo + hi) / 2. For length == 1 the
    index formula `i / (length - 1)` would divide by zero, so the ramp
    degenerates to that midpoint — preserving the average-equals-x
    property of the noise factory.
    """
    if length == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (length - 1) for i in range(length)]


def _channel_class_partition(q: int) -> Tuple[int, int, int]:
    """Return (n_clean, n_normal, n_noisy) for the channel partition.

    Even q: q/2 clean + q/2 noisy. Odd q: (q-1)/2 clean + 1 normal +
    (q-1)/2 noisy. The "normal" slot only exists for odd alphabets and
    uses eps == x exactly.
    """
    n_clean = q // 2
    n_normal = 1 if q % 2 == 1 else 0
    n_noisy = q - n_clean - n_normal
    return n_clean, n_normal, n_noisy


def _class_factor_per_symbol(q: int) -> np.ndarray:
    """Per-symbol class factor: clean=0.5, normal=1.0, noisy=1.5.

    Symbol indices are assigned by class in order. Sum across all q
    symbols equals q, which keeps the mean-per-bit-error invariant clean.
    """
    n_clean, n_normal, n_noisy = _channel_class_partition(q)
    factor = np.empty(q, dtype=np.float64)
    factor[:n_clean] = 0.5
    factor[n_clean : n_clean + n_normal] = 1.0
    factor[n_clean + n_normal :] = 1.5
    return factor


def _uniform_row_channel_matrix(eps_per_symbol: np.ndarray, q: int) -> np.ndarray:
    """Build a q×q channel matrix where row a is the uniform-eps row for
    that symbol: diagonal = 1 - eps(a), off-diagonals = eps(a) / (q - 1).
    """
    ch = np.empty((q, q), dtype=np.float64)
    for a in range(q):
        eps_a = float(eps_per_symbol[a])
        ch[a, :] = eps_a / (q - 1)
        ch[a, a] = 1.0 - eps_a
    return ch


def _create_noise_and_decoding_metric(
    noise_type: str,
    error_rate: float,
    seq_length: int,
    alphabet_size: int,
) -> Tuple[NoiseChannel, DecodingMetric]:
    """Construct (noise channel, NLL decoding metric) for the configured cell.

    Per the 2026-05-10 noise-channel redesign, all four channels satisfy two
    invariants:
    - Per-position, per-symbol eps stays ≤ 0.5 for any x in the sweep
      [0.05, 0.1, 0.2].
    - Mean per-bit error rate (uniform symbol distribution) = error_rate.
    """
    x = error_rate
    L = seq_length
    q = alphabet_size

    if noise_type == "symmetric":
        noise = SymmetricEpsilon(epsilon=x, alphabet_size=q)
        decoding_metric = SymmetricNLL(epsilon=x, alphabet_size=q)
    elif noise_type == "position_varying":
        # Symmetric ramp [0.5x, 1.5x] across positions: max = 1.5x ≤ 0.3 at x=0.2.
        eps = np.maximum(
            np.array(_positional_ramp(0.5 * x, 1.5 * x, L)),
            _EPSILON_FLOOR,
        )
        noise = PositionVaryingEpsilon(epsilons=eps, alphabet_size=q)
        decoding_metric = PositionVaryingNLL(epsilons=eps, alphabet_size=q)
    elif noise_type == "asymmetric":
        # Clean / normal / noisy partition; uniform-eps row per symbol.
        class_factor = _class_factor_per_symbol(q)
        eps_per_symbol = np.maximum(class_factor * x, _EPSILON_FLOOR)
        ch = _uniform_row_channel_matrix(eps_per_symbol, q)
        noise = AsymmetricChannel(channel_matrix=ch)
        decoding_metric = AsymmetricNLL(channel_matrix=ch)
    elif noise_type == "position_varying_asymmetric":
        # Multiplicative: eps[p, c] = class_factor[c] * position_factor[p] * x.
        # Max = 1.5 * 1.5 * x = 2.25x; at x=0.2 that's 0.45 < 0.5 ✓.
        class_factor = _class_factor_per_symbol(q)
        position_factor = np.array(_positional_ramp(0.5, 1.5, L))
        eps_pq = np.maximum(
            position_factor[:, None] * class_factor[None, :] * x,
            _EPSILON_FLOOR,
        )
        chs = np.empty((L, q, q), dtype=np.float64)
        for p in range(L):
            chs[p] = _uniform_row_channel_matrix(eps_pq[p], q)
        noise = PositionVaryingAsymmetricChannel(channel_matrices=chs)
        decoding_metric = PositionVaryingAsymmetricNLL(channel_matrices=chs)
    else:
        raise ValueError(f"Unknown noise channel type: {noise_type!r}")

    return noise, decoding_metric


@dataclass
class PoolConfig:
    num_groups: int
    candidates_per_group: int
    seq_length: int
    alphabet_size: int
    quota: int


@dataclass
class EvaluatorRunConfig:
    num_samples: int
    num_cpus: int = 1
    scratch_dir: Optional[str] = None


@dataclass
class BaselineObjectivesConfig:
    """Config for the enumeration-mode DUET-vs-baseline-objectives runner.

    Sweeps `noise_channels` × `trials` × all enumerated codebooks at a single
    `error_rate`. The DUET optimizer is never run — only the evaluator
    accuracy + PEP cache, plus the four pair-reduction baselines.
    """
    outdir: str
    seed: int
    trials: int
    pool: PoolConfig
    evaluator: EvaluatorRunConfig
    noise_channels: List[str]
    error_rate: float

    @classmethod
    def from_yaml(cls, path: str) -> "BaselineObjectivesConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            outdir=raw["outdir"],
            seed=raw["seed"],
            trials=raw["trials"],
            pool=PoolConfig(**raw["pool"]),
            evaluator=EvaluatorRunConfig(**raw["evaluator"]),
            noise_channels=list(raw["noise_channels"]),
            error_rate=float(raw["error_rate"]),
        )


Direction = Literal["max", "min"]


OBJECTIVES: List[Tuple[str, str, Direction]] = [
    ("duet_objective",        "DUET objective",                "max"),
    ("mean_pairwise_nll",     "Mean negative\nlog-likelihood", "max"),
    ("min_pairwise_nll",      "Minimum negative\nlog-likelihood", "max"),
    ("mean_pairwise_hamming", "Mean Hamming\ndistance",        "max"),
    ("min_pairwise_hamming",  "Minimum Hamming\ndistance",     "max"),
]
# Direction for `duet_objective` is "max": `compute_duet_objective` returns
# `1 - offdiag_pep_sum / |S|`, a survival-bound (larger = better), not an
# error-bound. All five objectives are max-direction, so no sign-flip is
# needed in the boxplot Spearman view.


# Objective values are rounded to this many decimals before any rank statistic
# (Spearman, argmax-tie detection). Two codebooks whose objective values are
# mathematically equal can differ in the last bits when the value is a sum of
# floats — `mean_pairwise_nll` sums `C(|S|, 2)` pair values, and the result
# depends on the addend order — which makes `scipy.stats.spearmanr` rank
# pseudo-ties that should have been rank-averaged, and makes an exact `==`
# argmax comparison miss genuinely tied codebooks. Nine decimals is far below
# the smallest real gap between distinct objective values (order 1e-2) and far
# above float summation error (order 1e-15).
RANK_ROUND_DECIMALS = 9


def round_for_ranking(values) -> np.ndarray:
    """Round objective values to `RANK_ROUND_DECIMALS` for rank statistics.

    Restores exact ties between values that are mathematically equal but
    differ in their last bits. Apply before `spearmanr` and before any
    exact-equality argmax comparison; never to values that are reported
    directly.

    Only the objective (y) axis is rounded — `decode_accuracy` (the x axis at
    every call site) is not. This was measured, not assumed: rounding
    `decode_accuracy` too changes rho in every cell, but by at most
    |delta rho| = 1.0e-5, and no median cell moves at four decimals, so
    leaving the accuracy axis unrounded is a deliberate choice, not an
    oversight worth re-opening.
    """
    return np.round(np.asarray(values, dtype=np.float64), RANK_ROUND_DECIMALS)


GetCodebooks = Callable[[CandidatePool], List[np.ndarray]]


def _pair_values(matrix: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Values of distinct unordered (i, j) pairs (i < j) from the codebook
    submatrix of `matrix`.

    Reused by all three pair reductions in process_trial: min/mean Hamming
    and mean NLL all reduce over `C(|S|, 2)` pairs of the codebook submatrix.
    """
    sub = matrix[np.ix_(codebook, codebook)]
    iu, ju = np.triu_indices(len(codebook), k=1)
    return sub[iu, ju]


def compute_pairwise_nll(decoding_metric, sequences: np.ndarray) -> np.ndarray:
    """Full pairwise NLL matrix on `sequences`, centered and direction-averaged.

    `decoding_metric.compute(x, y)` is a raw negative log-likelihood: for
    `AsymmetricNLL` and `PositionVaryingAsymmetricNLL` it sums over *all*
    positions, so `d(x, x) > 0` and matched positions contribute a
    composition-dependent offset. Subtracting the row diagonal turns each entry
    into the log-likelihood ratio `log[ P(x | x) / P(x | y) ]`, which vanishes
    where the two codewords agree — i.e. makes it a distance. Averaging the two
    directions makes it symmetric, which any pairwise criterion must be, since
    the designer does not know which of the two codewords will be transmitted.

    Row- versus column-centering is not a modelling choice: after the average
    both give `S[i, j] = 0.5 * (D[i,j] + D[j,i] - D[i,i] - D[j,j])`.

    Every metric here is additive over positions. Given that premise, on a
    binary alphabet the result is exactly a per-position weighted Hamming
    distance, `sum_p w_p * 1(x_p != y_p)` with

        w_p = 0.5 * [ log((1 - e1_p) / e0_p) + log((1 - e0_p) / e1_p) ]

    for `e0_p = P(1 | 0)` and `e1_p = P(0 | 1)` at position p. `w_p` is
    positive only while both `e0_p` and `e1_p` stay below 0.5; past that
    threshold the corresponding log term flips sign and the sum is no longer
    a distance — an unstated precondition that the channel factory
    (`_create_noise_and_decoding_metric`) satisfies across the configured
    sweep (max eps 0.45 at `error_rate = 0.2`) but that this function does not
    itself enforce. The weights are
    uniform whenever the channel does not vary by position (both the symmetric
    and the asymmetric channel), in which case this is Hamming distance
    rescaled and cannot change any ranking.

    History: this previously symmetrized with `np.minimum(D, D.T)`, inherited
    from `GreedyNLLMOOptimizer` which inherited it from
    `GreedyDistanceOptimizer`, where it was introduced narrowly to stop an
    `inf` from a zero-probability channel transition in one direction from
    masking finite confusability in the other. That rationale does not apply
    here — none of the four benchmark channels has a zero entry. Fixing the
    not-a-distance problem above (by centering) does not by itself justify
    `min`: taking `min` of the *centered* log-likelihood ratios `D'` would
    still be actively harmful, a separate problem from the first — `D'[i, j]`
    and `D'[j, i]` sum to a conserved multiple of the Hamming distance (a
    constant independent of which codeword is transmitted), so `min` reports
    how *unevenly* that fixed total splits between the two directions, a
    tiebreaker anti-correlated with true pairwise error rather than
    correlated with it. (This conserved-sum argument is about the centered
    ratios `D'`, not the raw `D` that the historical `min(D, D.T)` operated
    on: the raw directional sum `D[i,j] + D[j,i]` is `w * d_H + D[i,i] +
    D[j,j]`, not a clean conserved quantity, because of the same
    composition-dependent offset described above.) That is why the fix here
    averages the two centered directions instead of taking their `min`. The
    greedy optimizers still use `min` on the raw matrix; see
    `src/duet/greedy_optimizers.py`.
    """
    D = decoding_metric.compute(sequences, sequences)
    D = D - np.diag(D)[:, None]
    return 0.5 * (D + D.T)


def process_trial(
    config: BaselineObjectivesConfig,
    pool: CandidatePool,
    trial_seed: int,
    get_codebooks: GetCodebooks,
    evaluator_seed: int = None,
) -> List[dict]:
    """Evaluate codebooks for one trial across all noise channels.

    The codebook list comes from `get_codebooks(pool)`. The rest of the
    loop — Hamming matrix, per-noise evaluator, accuracy, DUET objective,
    NLL pair reductions — is identical to the prior in-script
    implementation.

    Args:
        trial_seed: Written to the parquet's `seed` column. For the
            enumeration path this is also the evaluator's MC seed
            (see `evaluator_seed`).
        evaluator_seed: Seed passed to `CodebookEvaluator`. When `None`
            (the enumeration path), defaults to `trial_seed` —
            preserving the prior behavior. The sampling path passes
            a `SeedSequence(trial_seed).spawn(2)[1]`-derived seed
            here so codebook-sampling RNG and evaluator MC RNG are
            independent.
    """
    if evaluator_seed is None:
        evaluator_seed = trial_seed
    codebooks = get_codebooks(pool)

    # Noise-agnostic — computed once per trial.
    hamming_full = HammingDistance().compute(pool.sequences, pool.sequences)
    hamming_pairs_per_cb = [_pair_values(hamming_full, cb) for cb in codebooks]
    min_hamming = np.array([p.min() for p in hamming_pairs_per_cb], dtype=np.int64)
    mean_hamming = np.array([p.mean() for p in hamming_pairs_per_cb], dtype=np.float64)

    rows: List[dict] = []
    for noise_type in config.noise_channels:
        logger.info("  Noise=%s", noise_type)
        noise_channel, decoding_metric = _create_noise_and_decoding_metric(
            noise_type,
            config.error_rate,
            config.pool.seq_length,
            config.pool.alphabet_size,
        )
        evaluator = CodebookEvaluator(
            codebook=pool.sequences,
            noise_channel=noise_channel,
            decoding_metric=decoding_metric,
            decoding_rule=UniqueMinimum(),
            n_samples=config.evaluator.num_samples,
            seed=evaluator_seed,
            scratch_dir=config.evaluator.scratch_dir,
        )
        evaluator.initialize_cache(n_jobs=config.evaluator.num_cpus)

        accuracy = evaluator.batch_get_accuracy(codebooks)
        pep_matrix = evaluator.get_pairwise_error_from_cache()
        duet_obj = np.array(
            [compute_duet_objective(pep_matrix, cb) for cb in codebooks],
            dtype=np.float64,
        )
        nll_sym = compute_pairwise_nll(decoding_metric, pool.sequences)
        mean_nll = np.array(
            [_pair_values(nll_sym, cb).mean() for cb in codebooks],
            dtype=np.float64,
        )
        min_nll = np.array(
            [_pair_values(nll_sym, cb).min() for cb in codebooks],
            dtype=np.float64,
        )
        evaluator.close()

        logger.info(
            "    accuracy: min=%.3f mean=%.3f max=%.3f",
            float(accuracy.min()), float(accuracy.mean()), float(accuracy.max()),
        )

        for idx, cb in enumerate(codebooks):
            rows.append({
                "trial": None,  # filled in by caller
                "seed": trial_seed,
                "noise_channel": noise_type,
                "error_rate": config.error_rate,
                "codebook_index": idx,
                "codebook_members": ",".join(str(int(c)) for c in cb),
                "decode_accuracy": float(accuracy[idx]),
                "duet_objective": float(duet_obj[idx]),
                "min_pairwise_hamming": int(min_hamming[idx]),
                "mean_pairwise_hamming": float(mean_hamming[idx]),
                "mean_pairwise_nll": float(mean_nll[idx]),
                "min_pairwise_nll": float(min_nll[idx]),
            })

    return rows


def sample_codebooks(
    group_to_candidates: dict,
    quotas: dict,
    num_codebooks: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Sample `num_codebooks` random codebooks from a quota-constrained
    candidate pool.

    Each codebook is formed by drawing `quota[g]` candidates uniformly
    without replacement from each group `g`, then concatenating across
    groups. Across the `num_codebooks` draws, sampling is *with*
    replacement at the codebook level — duplicates across draws are
    kept (silent deduplication would bias the sample distribution).

    Args:
        group_to_candidates: Mapping `group_id -> 1-D array of candidate
            indices in that group`. Must match the convention used by
            `enumerate_all_codebooks`.
        quotas: Mapping `group_id -> int` specifying how many candidates
            to sample from each group.
        num_codebooks: Number of codebook draws to produce.
        rng: A numpy `Generator`. The caller is responsible for seeding
            (typically via `SeedSequence(trial_seed).spawn(2)[0]`).

    Returns:
        List of `num_codebooks` arrays. Each array has length
        `sum(quotas.values())` and dtype `np.int64`.

    Warns:
        Logs a warning when the sampled set contains duplicate codebooks
        — this signals `num_codebooks` is too large for the candidate
        space and the sampling is behaving like near-enumeration.
    """
    sampled: List[np.ndarray] = []
    seen: set = set()
    n_duplicates = 0

    for _ in range(num_codebooks):
        parts: List[np.ndarray] = []
        for group_id, candidates in group_to_candidates.items():
            q = quotas[group_id]
            picks = rng.choice(candidates, size=q, replace=False)
            parts.append(picks)
        codebook = np.concatenate(parts).astype(np.int64)
        key = tuple(sorted(codebook.tolist()))
        if key in seen:
            n_duplicates += 1
        else:
            seen.add(key)
        sampled.append(codebook)

    if n_duplicates > 0:
        logger.warning(
            "sample_codebooks: %d / %d sampled codebooks were duplicates "
            "of an earlier draw. Consider reducing num_codebooks or "
            "expanding the candidate pool.",
            n_duplicates, num_codebooks,
        )

    return sampled


@dataclass
class SampledBaselineObjectivesConfig:
    """Config for the sampling-mode DUET-vs-baseline-objectives runner.

    Same fields as BaselineObjectivesConfig plus `num_codebooks`. No
    nested `sampling:` block — a one-field block is more ceremony than
    it's worth. Add a `sampling:` block here if a second sampling knob
    (stratification, replacement policy, etc.) ever lands.
    """
    outdir: str
    seed: int
    trials: int
    pool: PoolConfig
    evaluator: EvaluatorRunConfig
    noise_channels: List[str]
    error_rate: float
    num_codebooks: int

    @classmethod
    def from_yaml(cls, path: str) -> "SampledBaselineObjectivesConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            outdir=raw["outdir"],
            seed=raw["seed"],
            trials=raw["trials"],
            pool=PoolConfig(**raw["pool"]),
            evaluator=EvaluatorRunConfig(**raw["evaluator"]),
            noise_channels=list(raw["noise_channels"]),
            error_rate=float(raw["error_rate"]),
            num_codebooks=int(raw["num_codebooks"]),
        )
