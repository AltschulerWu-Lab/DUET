"""Tests for src/duet/merfish_benchmark/config.py — config-relative path resolution."""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import pytest
import yaml

from duet.merfish_benchmark.config import (
    MerfishBenchmarkConfig,
    resolve_config_path,
)


def test_resolve_config_path_returns_absolute_paths_unchanged(tmp_path):
    abs_path = "/some/absolute/path/file.csv"
    result = resolve_config_path(abs_path, tmp_path)
    assert str(result) == abs_path


def test_resolve_config_path_resolves_relative_against_config_dir(tmp_path):
    # tmp_path / "configs" / "a.yaml" relative to "../data/foo.csv"
    # should resolve to tmp_path / "data" / "foo.csv"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    result = resolve_config_path("../data/foo.csv", config_dir)
    assert result == (tmp_path / "data" / "foo.csv").resolve()


def test_resolve_config_path_passthrough_when_config_dir_none():
    """Backward-compat: no config_dir means no resolution."""
    result = resolve_config_path("relative/path.csv", None)
    assert result == Path("relative/path.csv")


def test_resolve_config_path_none_input_returns_none(tmp_path):
    assert resolve_config_path(None, tmp_path) is None


def test_from_yaml_resolves_outdir_relative_to_config_dir(tmp_path):
    """End-to-end: yaml with relative outdir resolves against its parent dir."""
    config_dir = tmp_path / "archive" / "demo"
    config_dir.mkdir(parents=True)
    yaml_data = {
        "outdir": "../../results/demo",
        "candidates": {"seq_rounds": 4, "codebook_size": 8, "hamming_weights": [2]},
        "evaluator": {
            "noise_channel": {"type": "symmetric", "epsilon": 0.05},
            "decoding_metric": {"type": "hamming"},
            "decoding_rule": {"type": "unique_minimum"},
            "num_samples": 100,
            "num_cpus": 1,
            "seed": 0,
        },
        "duet": {
            "use_mmap": False,
            "device": "cpu",
            "force_rebuild": False,
            "lambda": [1.0], "temperature": 0.0, "max_iter": 10,
            "max_patience": 5, "avoid_duplicates": False, "num_cpus": 1,
            "pep": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.05},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 100,
                "num_cpus": 1,
                "seed": 0,
            },
        },
        "seed": 0,
    }
    config_path = config_dir / "demo.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(yaml_data, f)
    config = MerfishBenchmarkConfig.from_yaml(config_path)
    # outdir = config_dir / "../../results/demo" = tmp_path / "results/demo"
    assert config.outdir == (tmp_path / "results" / "demo").resolve()


def test_from_yaml_leaves_absolute_outdir_unchanged(tmp_path):
    """Absolute outdir is not touched by the resolver."""
    config_dir = tmp_path / "archive" / "demo"
    config_dir.mkdir(parents=True)
    abs_outdir = "/var/run/outputs"
    yaml_data = {
        "outdir": abs_outdir,
        "candidates": {"seq_rounds": 4, "codebook_size": 8, "hamming_weights": [2]},
        "evaluator": {
            "noise_channel": {"type": "symmetric", "epsilon": 0.05},
            "decoding_metric": {"type": "hamming"},
            "decoding_rule": {"type": "unique_minimum"},
            "num_samples": 100,
            "num_cpus": 1,
            "seed": 0,
        },
        "duet": {
            "use_mmap": False,
            "device": "cpu",
            "force_rebuild": False,
            "lambda": [1.0], "temperature": 0.0, "max_iter": 10,
            "max_patience": 5, "avoid_duplicates": False, "num_cpus": 1,
            "pep": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.05},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 100,
                "num_cpus": 1,
                "seed": 0,
            },
        },
        "seed": 0,
    }
    config_path = config_dir / "demo.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(yaml_data, f)
    config = MerfishBenchmarkConfig.from_yaml(config_path)
    assert str(config.outdir) == abs_outdir
