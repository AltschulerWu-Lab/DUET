#!/usr/bin/env python3
"""Thin CLI wrapper around duet.ops_benchmark.run_ops_benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from duet.ops_benchmark import OpsBenchmarkConfig, run_ops_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DUET vs baselines with n trials.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress matplotlib's per-font DEBUG output under -vv (not our signal).
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    config_path = Path(args.config)
    config = OpsBenchmarkConfig.from_yaml(config_path)
    run_ops_benchmark(config, config_path=config_path)


if __name__ == "__main__":
    main()
