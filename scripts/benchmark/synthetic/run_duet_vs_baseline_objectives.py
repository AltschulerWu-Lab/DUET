#!/usr/bin/env python3
"""
DUET-vs-baseline-objectives experiment runner (enumeration mode).

Thin wrapper around `duet.benchmark.baseline_objectives.process_trial`.
Wires the enumeration codebook source (`enumerate_all_codebooks`), loops
over trials, and writes `objective_correlation.parquet`.

Usage:
    python -m scripts.benchmark.synthetic.run_duet_vs_baseline_objectives \
        --config <path/to/config.yaml> -v
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from duet.benchmark.baseline_objectives import (
    BaselineObjectivesConfig,
    process_trial,
)
from duet.benchmark.exhaustive import enumerate_all_codebooks
from duet.benchmark.synthetic_pool import create_2d_synthetic_pool

logger = logging.getLogger(__name__)


def run_baseline_objectives(config: BaselineObjectivesConfig) -> pd.DataFrame:
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

        trial_rows = process_trial(config, pool, trial_seed, get_codebooks)
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
        description="DUET-vs-baseline-objectives experiment runner (enumeration mode).",
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

    df = run_baseline_objectives(config)
    print(f"Done. {len(df)} rows -> {outdir / 'objective_correlation.parquet'}")


if __name__ == "__main__":
    main()
