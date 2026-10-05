#!/usr/bin/env python3
"""
DUET-vs-baseline-objectives experiment runner (sampling mode).

Thin wrapper around `duet.benchmark.baseline_objectives.process_trial`
that injects `sample_codebooks` in place of `enumerate_all_codebooks`.
Used when the codebook space is too large to enumerate (e.g., |S| ≥ 1000).

Writes `objective_correlation_sampled.parquet`. Each completed trial
triggers an incremental atomic write of `*.partial.parquet` so a crash
mid-run leaves all previously-completed trials usable.

Usage:
    python -m scripts.benchmark.synthetic.run_duet_vs_baseline_objectives_sampled \
        --config <path/to/config.yaml> -v
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from duet.benchmark.baseline_objectives import (
    SampledBaselineObjectivesConfig,
    process_trial,
    sample_codebooks,
)
from duet.benchmark.synthetic_pool import create_2d_synthetic_pool

logger = logging.getLogger(__name__)


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Write `df` to `target` via a tmp + fsync + rename sequence so a
    SIGTERM/OOM mid-write cannot corrupt the existing target file."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp, target)


def run_sampled_baseline_objectives(
    config: SampledBaselineObjectivesConfig,
) -> pd.DataFrame:
    """Run all trials with sampled codebooks; write incrementally to
    `*.partial.parquet`, rename to final on completion."""
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    final_path = outdir / "objective_correlation_sampled.parquet"
    partial_path = outdir / "objective_correlation_sampled.partial.parquet"

    all_rows: List[dict] = []
    for trial in range(config.trials):
        trial_seed = config.seed + trial
        logger.info(
            "Trial %d/%d (seed=%d)", trial + 1, config.trials, trial_seed
        )

        # Split trial_seed into independent streams for codebook sampling
        # and evaluator MC sampling. See baseline_objectives.py:
        # "RNG isolation rationale" in the spec.
        sampling_seed, evaluator_seed = np.random.SeedSequence(trial_seed).spawn(2)
        rng = np.random.default_rng(sampling_seed)

        pool = create_2d_synthetic_pool(
            num_groups=config.pool.num_groups,
            candidates_per_group=config.pool.candidates_per_group,
            seq_length=config.pool.seq_length,
            alphabet_size=config.pool.alphabet_size,
            quota=config.pool.quota,
            seed=trial_seed,
        )

        def get_codebooks(p):
            return sample_codebooks(
                p.group_to_candidates, p.quotas, config.num_codebooks, rng,
            )

        # Pass the spawned evaluator_seed explicitly so the parquet's
        # `seed` column reflects the user's original trial_seed while
        # the MC RNG uses the independent spawned stream.
        evaluator_seed_int = int(
            evaluator_seed.generate_state(1, dtype=np.uint32)[0]
        )
        trial_rows = process_trial(
            config, pool, trial_seed, get_codebooks,
            evaluator_seed=evaluator_seed_int,
        )
        # Rename codebook_index → sample_index for downstream
        # disambiguation between enumeration and sampled parquets.
        for row in trial_rows:
            row["trial"] = trial + 1
            row["sample_index"] = row.pop("codebook_index")
        all_rows.extend(trial_rows)

        # Atomic incremental write after each completed trial.
        _atomic_write_parquet(pd.DataFrame(all_rows), partial_path)
        logger.info("Wrote partial parquet: %s (%d rows)", partial_path, len(all_rows))

    df = pd.DataFrame(all_rows)
    _atomic_write_parquet(df, final_path)
    if partial_path.exists():
        partial_path.unlink()
    logger.info("Wrote %d rows -> %s", len(df), final_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DUET-vs-baseline-objectives experiment runner (sampling mode).",
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

    config = SampledBaselineObjectivesConfig.from_yaml(args.config)
    logger.info("Config loaded: %s", args.config)

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(args.config).resolve()
    dst = (outdir / "config.yaml").resolve()
    if src != dst:
        shutil.copy2(src, dst)

    df = run_sampled_baseline_objectives(config)
    print(f"Done. {len(df)} rows -> {outdir / 'objective_correlation_sampled.parquet'}")


if __name__ == "__main__":
    main()
