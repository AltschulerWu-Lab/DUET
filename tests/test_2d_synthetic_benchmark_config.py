"""Tests for TwoDSyntheticBenchmarkConfig.from_yaml — optional greedy baselines.

Covers the post-refactor validation contract:
- Missing greedy block → None in the parsed config.
- Block present with valid lambda → parses to GreedyMOConfig.
- Block present but null / lambda-missing / lambda-empty → ValueError.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import pytest
import yaml

from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
    TwoDSyntheticBenchmarkConfig,
)


def _minimal_config_dict() -> dict:
    """Return a minimal top-level config dict with no greedy blocks.

    Tests add `greedy_hamming_mo` / `greedy_nll_mo` keys as needed. All
    other required keys are populated with cheap values — these tests
    only exercise from_yaml's parsing, not the runner itself.
    """
    return {
        "outdir": "test_out",
        "seed": 42,
        "trials": 1,
        "pool": {
            "num_groups": 2,
            "candidates_per_group": 2,
            "seq_length": 4,
            "alphabet_size": 2,
            "quota": 1,
        },
        "evaluator": {"num_samples": 100, "num_cpus": 1},
        "noise_channels": ["symmetric"],
        "error_rates": [0.05],
        "duet": {
            "optimizer": {
                "lambda": [0.5],
                "temperature": 0.0,
                "max_iter": 100,
                "max_patience": 30,
                "num_cpus": 1,
            }
        },
    }


def _write_config(tmp_path, cfg_dict: dict):
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg_dict, f)
    return path


def test_both_greedy_blocks_absent(tmp_path):
    cfg = _minimal_config_dict()
    path = _write_config(tmp_path, cfg)
    config = TwoDSyntheticBenchmarkConfig.from_yaml(str(path))
    assert config.greedy_hamming_mo is None
    assert config.greedy_nll_mo is None


def test_both_greedy_blocks_present_with_valid_lambda(tmp_path):
    cfg = _minimal_config_dict()
    cfg["greedy_hamming_mo"] = {"lambda": [0.0, 0.5, 1.0]}
    cfg["greedy_nll_mo"] = {"lambda": [0.0, 0.5, 1.0]}
    path = _write_config(tmp_path, cfg)
    config = TwoDSyntheticBenchmarkConfig.from_yaml(str(path))
    assert config.greedy_hamming_mo is not None
    assert config.greedy_hamming_mo.lambda_ == [0.0, 0.5, 1.0]
    assert config.greedy_nll_mo is not None
    assert config.greedy_nll_mo.lambda_ == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("key", ["greedy_hamming_mo", "greedy_nll_mo"])
def test_greedy_block_present_but_null_raises(tmp_path, key):
    cfg = _minimal_config_dict()
    cfg[key] = None
    path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match=rf"{key!r}.*null"):
        TwoDSyntheticBenchmarkConfig.from_yaml(str(path))


@pytest.mark.parametrize("key", ["greedy_hamming_mo", "greedy_nll_mo"])
def test_greedy_block_lambda_missing_raises(tmp_path, key):
    cfg = _minimal_config_dict()
    cfg[key] = {"some_other_field": 1}  # block present, no `lambda` sub-key
    path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match=rf"{key!r}.*lambda.*missing or empty"):
        TwoDSyntheticBenchmarkConfig.from_yaml(str(path))


@pytest.mark.parametrize("key", ["greedy_hamming_mo", "greedy_nll_mo"])
def test_greedy_block_empty_lambda_raises(tmp_path, key):
    cfg = _minimal_config_dict()
    cfg[key] = {"lambda": []}
    path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match=rf"{key!r}.*lambda.*missing or empty"):
        TwoDSyntheticBenchmarkConfig.from_yaml(str(path))
