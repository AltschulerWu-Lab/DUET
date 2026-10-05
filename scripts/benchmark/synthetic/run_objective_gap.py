#!/usr/bin/env python3
"""
DUET objective-gap experiment runner.

Enumerates every codebook in the small synthetic regime and computes, per
codebook, its decode accuracy (the evaluator's accuracy metric) and DUET's
union-bound decode surrogate. The DUET optimizer is never run — only the
evaluator (for accuracy and the PEP matrix) and codebook enumeration.

The surrogate is DUET's actual decode objective, `DecodingSwapCache.
compute_objective` in src/duet/pareto_optimization.py:

    surrogate(S) = 1 - (sum_{i != j in S} M[i, j]) / |S|

where M is the raw asymmetric PEP probability matrix.

Usage:
    python -m scripts.benchmark.synthetic.run_objective_gap \
        --config <path/to/config.yaml> -v
"""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

from duet.benchmark.exhaustive import enumerate_all_codebooks
from duet.benchmark.metrics import compute_duet_objective  # noqa: F401  re-exported
from duet.benchmark.synthetic_pool import create_2d_synthetic_pool
from duet.codebook_evaluator import CodebookEvaluator, UniqueMinimum
from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
    EvaluatorRunConfig,
    PoolConfig,
    _create_noise_and_decoding_metric,
)

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveGapConfig:
    """Config for the objective-gap experiment.

    Reuses PoolConfig / EvaluatorRunConfig from the 2-D synthetic runner.
    `noise_channel` and `error_rate` are scalars — this is a single-cell
    experiment, so the 2-D config's list shape is not mirrored. There is no
    `duet:` block: the DUET optimizer is never run.
    """
    outdir: str
    seed: int
    trials: int
    pool: PoolConfig
    evaluator: EvaluatorRunConfig
    noise_channel: str
    error_rate: float

    @classmethod
    def from_yaml(cls, path: str) -> "ObjectiveGapConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            outdir=raw["outdir"],
            seed=raw["seed"],
            trials=raw["trials"],
            pool=PoolConfig(**raw["pool"]),
            evaluator=EvaluatorRunConfig(**raw["evaluator"]),
            noise_channel=raw["noise_channel"],
            error_rate=raw["error_rate"],
        )


def process_trial(
    config: ObjectiveGapConfig, pool, trial_seed: int
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Evaluate every codebook for one trial.

    Constructs the evaluator identically to `process_cell` in
    run_2d_synthetic_benchmark.py — same noise factory, and
    `decoding_rule=UniqueMinimum()`. The decoding rule is not incidental:
    the surrogate-vs-accuracy gap is specific to UniqueMinimum, which is the
    regime DUET operates in.

    Accuracy and the PEP matrix are read from the SAME initialized cache, so
    the gap is a genuine objective-function property, not Monte-Carlo noise
    between two independent estimates.

    Returns:
        (codebooks, accuracy, surrogate) — accuracy and surrogate are 1D
        arrays aligned with the codebooks list.
    """
    noise_channel, decoding_metric = _create_noise_and_decoding_metric(
        config.noise_channel,
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
        seed=trial_seed,
    )
    evaluator.initialize_cache(n_jobs=config.evaluator.num_cpus)

    codebooks = enumerate_all_codebooks(pool.group_to_candidates, pool.quotas)
    accuracy = evaluator.batch_get_accuracy(codebooks)
    pep_matrix = evaluator.get_pairwise_error_from_cache()
    duet_obj = np.array(
        [compute_duet_objective(pep_matrix, cb) for cb in codebooks],
        dtype=np.float64,
    )
    evaluator.close()
    return codebooks, accuracy, duet_obj


def run_objective_gap(config: ObjectiveGapConfig) -> pd.DataFrame:
    """Run all trials and write objective_gap.parquet.

    Per trial t (0-based), the pool seed is `config.seed + t`, matching the
    2-D synthetic runner's per-trial seeding.
    """
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
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
        codebooks, accuracy, duet_obj = process_trial(config, pool, trial_seed)

        for idx, (cb, acc, obj) in enumerate(zip(codebooks, accuracy, duet_obj)):
            rows.append({
                "trial": trial + 1,
                "seed": trial_seed,
                "noise_channel": config.noise_channel,
                "error_rate": config.error_rate,
                "codebook_index": idx,
                "codebook_members": ",".join(str(int(c)) for c in cb),
                "decode_accuracy": float(acc),
                "duet_objective": float(obj),
            })

    df = pd.DataFrame(rows)
    df.to_parquet(outdir / "objective_gap.parquet", index=False)
    logger.info("Wrote %d rows -> %s", len(df), outdir / "objective_gap.parquet")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DUET objective-gap experiment runner.",
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

    config = ObjectiveGapConfig.from_yaml(args.config)
    logger.info("Config loaded: %s", args.config)

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(args.config).resolve()
    dst = (outdir / "config.yaml").resolve()
    if src != dst:
        shutil.copy2(src, dst)

    df = run_objective_gap(config)
    print(f"Done. {len(df)} rows -> {outdir / 'objective_gap.parquet'}")


if __name__ == "__main__":
    main()
