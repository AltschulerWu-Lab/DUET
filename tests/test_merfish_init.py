"""Tests for the refactored MERFISH parse_init."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import pandas as pd
import pytest

from duet.merfish_benchmark.config import InitializationConfig
from duet.merfish_benchmark.init import parse_init
from duet.initialization import CodebookWarmStart, RandomInit


def test_parse_init_random():
    cfg = InitializationConfig(type="random")
    strategy = parse_init(cfg)
    assert isinstance(strategy, RandomInit)


def test_parse_init_warm_start(tmp_path):
    cb_csv = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["A"], "Sequence": ["0000"]}).to_csv(cb_csv, index=False)
    cfg = InitializationConfig(type="warm_start", path=cb_csv, sequence_col="Sequence", noise_percent=2.5)
    strategy = parse_init(cfg)
    assert isinstance(strategy, CodebookWarmStart)
    assert strategy.codebook_path == cb_csv
    assert strategy.sequence_col == "Sequence"
    assert strategy.noise_percent == 2.5


def test_parse_init_warm_start_missing_path():
    # InitializationConfig.__post_init__ already validates warm_start requires path,
    # so the ValueError fires at construction time — we catch it here.
    with pytest.raises(ValueError, match="path"):
        cfg = InitializationConfig(type="warm_start")
        parse_init(cfg)
