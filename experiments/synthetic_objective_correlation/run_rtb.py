#!/usr/bin/env python3
"""DUET-vs-baseline-objectives runner, RANDOM-TIE-BREAK variant (enumeration mode).

This is a self-contained sibling of
``scripts/benchmark/synthetic/run_duet_vs_baseline_objectives.py``. It produces
the byte-compatible ``objective_correlation.parquet`` schema (same columns), so
the shared visualizers render it unchanged, but it decodes with RANDOM TIE-BREAK
instead of the reject (``UniqueMinimum``) rule.

Why a local runner rather than a flag on the core evaluator: the tie-break only
matters for this didactic synthetic experiment. The OPS and MERFISH benchmarks
deliberately keep the reject decoder (an ambiguous read is dropped, matching the
real assays), so the shared ``CodebookEvaluator`` / ``UniqueMinimum`` path is
left untouched. Everything else here is imported unchanged from ``duet``. At the
time this docstring was first written (07-24), that made the tie handling the
ONLY difference from the 06-11 reject run; as of this 08-05 copy there is a
second difference (see below), because the shared ``compute_pairwise_nll``
this runner imports was itself redefined after 06-11, not because this
runner diverged from the 06-11 design:

  * the pool, codebook enumeration, and the four noise channels come from the
    same ``duet`` helpers the reject runner uses (identical channels);
  * the Hamming and NLL objective columns are decoder-independent and are
    computed exactly as in the reject runner;
  * only the decode-accuracy and the DUET-objective PEP are recomputed under
    random tie-break, locally, in ``_rtb_accuracy_and_pep`` below.

Random tie-break semantics: for a read whose true codeword is among the
minimum-cost codewords of the selected codebook, credit ``1 / (#codewords tied
at the minimum)``; otherwise 0. This is the exact expected accuracy of a decoder
that breaks ties by a uniform random draw (no coin-flip Monte-Carlo variance).
The DUET PEP is made consistent by counting a tie as half: ``pep[i, j] =
P(cost_j < cost_i) + 0.5 * P(cost_j == cost_i)``.

The executable code in this 08-05 copy is identical to the 07-24 runner; only
this docstring differs. The pairwise NLL columns differ from 07-24's output
only because ``duet.benchmark.baseline_objectives.compute_pairwise_nll``, which
this runner imports unchanged, was redefined from ``min(D, D.T)`` on the raw
negative log-likelihood to ``0.5 * (D + D.T)`` on the diagonal-centered
matrix. Everything else (the pool, the enumeration, the four channels, the
Hamming columns, and the random-tie-break decode) is the same code producing
the same numbers.

Usage (run.sh runs it from this folder):
    python run_rtb.py --config config.yaml -v
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from duet.benchmark.baseline_objectives import (
    BaselineObjectivesConfig,
    _create_noise_and_decoding_metric,
    compute_pairwise_nll,
)
from duet.benchmark.exhaustive import enumerate_all_codebooks
from duet.benchmark.metrics import compute_duet_objective
from duet.benchmark.synthetic_pool import create_2d_synthetic_pool
from duet.candidate_pool import CandidatePool
from duet.codebook_evaluator import HammingDistance

logger = logging.getLogger(__name__)

# Absolute tolerance for treating two decoding costs as tied. Symmetric-channel
# costs are integer multiples of a positive weight, so ties are exact; the NLL
# channels have a smallest non-tie cost gap on the order of 1e-2, so 1e-9 cleanly
# separates genuine ties (mathematically equal, up to float roundoff) from
# distinct costs.
_TIE_TOL = 1e-9


def _pair_values(matrix: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Values of distinct unordered (i, j) pairs (i < j) of the codebook submatrix.

    Local copy of the identical helper in ``duet.benchmark.baseline_objectives``
    so this runner does not depend on that module's private API.
    """
    sub = matrix[np.ix_(codebook, codebook)]
    iu, ju = np.triu_indices(len(codebook), k=1)
    return sub[iu, ju]


def _rtb_accuracy_and_pep(
    pool_sequences: np.ndarray,
    noise_channel,
    decoding_metric,
    n_samples: int,
    seed: int,
    codebooks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Random-tie-break decode accuracy per codebook, plus the tie-halved PEP.

    Reuses the (unmodified) ``noise_channel`` and ``decoding_metric`` strategies
    for read generation and cost scoring; only the tie handling is local.

    Returns:
        (accuracy, pep) where ``accuracy`` has shape ``(len(codebooks),)`` and
        ``pep`` is the full ``(num_pool, num_pool)`` tie-halved PEP indexed by
        pool index (diagonal is ignored by ``compute_duet_objective``).
    """
    num_pool = len(pool_sequences)
    rng = np.random.default_rng(seed)

    # cost_cache[c]: (n_samples, num_pool) cost from codeword c's reads to every
    # pool codeword. Generation + scoring use the production strategy objects.
    cost_cache = np.empty((num_pool, n_samples, num_pool), dtype=np.float64)
    for c in range(num_pool):
        reads = noise_channel.generate(pool_sequences[c], n_samples, rng)
        cost_cache[c] = decoding_metric.compute(reads, pool_sequences)

    # Tie-halved PEP: pep[i, j] = P(cost_j < cost_i) + 0.5 * P(cost_j == cost_i),
    # averaged over transmitter i's reads. Consistent with random tie-break so
    # the DUET objective (a union bound over this PEP) matches the accuracy x-axis.
    pep = np.zeros((num_pool, num_pool), dtype=np.float64)
    for i in range(num_pool):
        di = cost_cache[i]
        ci = di[:, i][:, None]
        strict = di < ci - _TIE_TOL
        tie = np.abs(di - ci) <= _TIE_TOL
        pep[i] = strict.mean(axis=0) + 0.5 * tie.mean(axis=0)

    accuracy = np.empty(len(codebooks), dtype=np.float64)
    for k, cb in enumerate(codebooks):
        cb = np.asarray(cb)
        per_cw_sum = 0.0
        for pos, c in enumerate(cb):
            dsub = cost_cache[c][:, cb]                 # (n_samples, |cb|)
            true_cost = dsub[:, pos]
            min_cost = dsub.min(axis=1)
            n_at_min = (dsub <= min_cost[:, None] + _TIE_TOL).sum(axis=1)
            true_is_min = true_cost <= min_cost + _TIE_TOL
            per_cw_sum += np.mean(np.where(true_is_min, 1.0 / n_at_min, 0.0))
        accuracy[k] = per_cw_sum / len(cb)

    return accuracy, pep


GetCodebooks = Callable[[CandidatePool], List[np.ndarray]]


def process_trial_rtb(
    config: BaselineObjectivesConfig,
    pool: CandidatePool,
    trial_seed: int,
    get_codebooks: GetCodebooks,
) -> List[dict]:
    """Evaluate every codebook for one trial across all noise channels (rtb decode).

    Mirrors ``duet.benchmark.baseline_objectives.process_trial`` exactly, except
    ``decode_accuracy`` and ``duet_objective`` come from ``_rtb_accuracy_and_pep``
    (random tie-break) instead of the reject ``CodebookEvaluator`` path.
    """
    codebooks = get_codebooks(pool)

    # Noise-agnostic objectives, computed once per trial (identical to reject run).
    hamming_full = HammingDistance().compute(pool.sequences, pool.sequences)
    hamming_pairs = [_pair_values(hamming_full, cb) for cb in codebooks]
    min_hamming = np.array([p.min() for p in hamming_pairs], dtype=np.int64)
    mean_hamming = np.array([p.mean() for p in hamming_pairs], dtype=np.float64)

    rows: List[dict] = []
    for noise_type in config.noise_channels:
        logger.info("  Noise=%s", noise_type)
        noise_channel, decoding_metric = _create_noise_and_decoding_metric(
            noise_type,
            config.error_rate,
            config.pool.seq_length,
            config.pool.alphabet_size,
        )

        accuracy, pep_matrix = _rtb_accuracy_and_pep(
            pool.sequences,
            noise_channel,
            decoding_metric,
            config.evaluator.num_samples,
            trial_seed,
            codebooks,
        )
        duet_obj = np.array(
            [compute_duet_objective(pep_matrix, cb) for cb in codebooks],
            dtype=np.float64,
        )

        nll_sym = compute_pairwise_nll(decoding_metric, pool.sequences)
        mean_nll = np.array(
            [_pair_values(nll_sym, cb).mean() for cb in codebooks], dtype=np.float64
        )
        min_nll = np.array(
            [_pair_values(nll_sym, cb).min() for cb in codebooks], dtype=np.float64
        )

        logger.info(
            "    rtb accuracy: min=%.3f mean=%.3f max=%.3f",
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


def run_baseline_objectives_rtb(config: BaselineObjectivesConfig) -> pd.DataFrame:
    """Run all trials, enumerating every codebook, and write the parquet."""
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for trial in range(config.trials):
        trial_seed = config.seed + trial
        logger.info("Trial %d/%d (seed=%d)", trial + 1, config.trials, trial_seed)

        pool = create_2d_synthetic_pool(
            num_groups=config.pool.num_groups,
            candidates_per_group=config.pool.candidates_per_group,
            seq_length=config.pool.seq_length,
            alphabet_size=config.pool.alphabet_size,
            quota=config.pool.quota,
            seed=trial_seed,
        )

        def get_codebooks(p):
            return enumerate_all_codebooks(p.group_to_candidates, p.quotas)

        trial_rows = process_trial_rtb(config, pool, trial_seed, get_codebooks)
        for row in trial_rows:
            row["trial"] = trial + 1
        all_rows.extend(trial_rows)

    df = pd.DataFrame(all_rows)
    out_path = outdir / "objective_correlation.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(df), out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DUET-vs-baseline-objectives runner, random-tie-break variant.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = BaselineObjectivesConfig.from_yaml(args.config)
    logger.info("Config loaded: %s", args.config)

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(args.config).resolve()
    dst = (outdir / "config.yaml").resolve()
    if src != dst:
        shutil.copy2(src, dst)

    df = run_baseline_objectives_rtb(config)
    print(f"Done. {len(df)} rows -> {outdir / 'objective_correlation.parquet'}")


if __name__ == "__main__":
    main()
