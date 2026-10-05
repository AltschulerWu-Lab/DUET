# src/duet/merfish_benchmark/init.py
"""MERFISH-specific initialization parser (decoupled from baselines)."""

from __future__ import annotations

from duet.initialization import CodebookWarmStart, InitializationStrategy, RandomInit
from duet.merfish_benchmark.config import InitializationConfig


def parse_init(cfg: InitializationConfig) -> InitializationStrategy:
    if cfg.type == "random":
        return RandomInit()
    if cfg.type == "warm_start":
        if cfg.path is None:
            raise ValueError("initialization.type=warm_start requires `path`")
        return CodebookWarmStart(
            codebook_path=cfg.path,
            noise_percent=cfg.noise_percent,
            description_tag="warm_start",
            sequence_col=cfg.sequence_col,
        )
    raise ValueError(f"Unknown initialization type: {cfg.type!r}")
