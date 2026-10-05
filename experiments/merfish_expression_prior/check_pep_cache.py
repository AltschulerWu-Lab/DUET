#!/usr/bin/env python
"""Pre-flight: every panel's candidate pool must hit the existing PEP cache.

A miss would make run_merfish.py build a new 100,000-codeword PEP (~4 h on 4 GPUs
and ~38 GB, per panel). This builds each panel's candidate pool with the runner's
own build_candidates and asks PEPMatrixProvider for the cache status of its
duet.pep config, without computing anything.

Run from this folder: the channel_matrix_path strings in the fingerprint are
relative to the working directory.

    python check_pep_cache.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from duet.merfish_benchmark.config import MerfishBenchmarkConfig
from duet.merfish_benchmark.runner import build_candidates
from duet.providers import PEPMatrixProvider

ALPHABET_SIZE = 2  # binary MERFISH codewords, as in the runner


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    base = args.config.resolve().parent
    cfg = yaml.safe_load(args.config.read_text())
    out = (base / cfg["outdir"]).resolve()

    misses = []
    for panel in cfg["panels"]:
        config = MerfishBenchmarkConfig.from_yaml(out / panel["id"] / "run_config.yaml")
        # The runner builds the PEP over unique_sequences (runner/core.py), so hash those.
        pool = list(build_candidates(config, config.seed).candidates.unique_sequences)
        status = PEPMatrixProvider(config.duet_cache_dir or config.cache_dir).cache_status(
            pool, config.duet_pep, ALPHABET_SIZE)
        hit = status["sym_hit"]
        print(f"[{'ok' if hit else 'MISS'}] {panel['id']}: PEP fingerprint "
              f"{status['fingerprint']} (symmetrized cache {'found' if hit else 'absent'})")
        misses += [] if hit else [panel["id"]]
    if misses:
        sys.exit(f"PEP cache miss for {misses}: these panels would rebuild the PEP")


if __name__ == "__main__":
    main()
