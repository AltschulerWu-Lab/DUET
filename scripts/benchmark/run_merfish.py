#!/usr/bin/env python3
"""Thin CLI wrapper around duet.merfish_benchmark.run_merfish_benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from duet.merfish_benchmark import MerfishBenchmarkConfig, run_merfish_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MERFISH codebook optimization via DUET.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v INFO; -vv DEBUG")
    parser.add_argument("--debug-pep", action="store_true", default=False,
                        help="Generate PEP matrix debug visualizations")
    parser.add_argument("--init-source", type=str,
                        choices=["random", "warm_start"],
                        default=None, help="Override initialization type")
    parser.add_argument("--init-path", type=str, default=None,
                        help="Override initialization codebook path (required with --init-source warm_start)")
    parser.add_argument("--noise-percent", type=float, default=None,
                        help="Override noise percentage (0-100)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--outdir", type=str, default=None, help="Override output directory")
    return parser.parse_args()


def _apply_cli_overrides(d: dict, args: argparse.Namespace) -> dict:
    """Layer CLI overrides onto the YAML dict BEFORE from_dict runs."""
    if args.init_path is not None and args.init_source != "warm_start":
        raise ValueError("--init-path requires --init-source warm_start")
    if args.init_source is not None:
        init_block = d.setdefault("initialization", {})
        init_block["type"] = args.init_source
        if args.init_source == "warm_start":
            if args.init_path is None and "path" not in init_block:
                raise ValueError("--init-source warm_start requires --init-path (or initialization.path in YAML)")
            if args.init_path is not None:
                init_block["path"] = args.init_path
    if args.noise_percent is not None:
        d.setdefault("initialization", {})["noise_percent"] = args.noise_percent
    if args.seed is not None:
        d["seed"] = args.seed
    if args.outdir is not None:
        d["outdir"] = args.outdir
    return d


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
    )
    # Suppress matplotlib's per-font DEBUG output under -vv (not our signal).
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    config_path = Path(args.config)
    with open(config_path) as f:
        d = yaml.safe_load(f)
    d = _apply_cli_overrides(d, args)

    config = MerfishBenchmarkConfig.from_dict(d, config_dir=config_path.parent)
    run_merfish_benchmark(config, args=args)


if __name__ == "__main__":
    main()
