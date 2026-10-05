# tests/test_ops_benchmark_config.py
"""Tests for OpsBenchmarkConfig parsing (replaces test_benchmark_duet_config)."""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import pytest
import yaml

from duet.ops_benchmark import OpsBenchmarkConfig


@pytest.fixture
def minimal_yaml(tmp_path):
    d = {
        "outdir": str(tmp_path / "out"),
        "trials": 1,
        "candidate_pool": {
            "source": "synthetic",
            "seq_rounds": 4,
            "quota": 1,
            "num_controls": 0,
            "num_groups": 4,
            "candidates_per_group": 3,
            "alphabet_size": 4,
        },
        "evaluator": {
            "noise_channel": {"type": "symmetric", "epsilon": 0.1},
            "decoding_metric": {"type": "hamming"},
            "decoding_rule": {"type": "unique_minimum"},
            "num_samples": 200,
            "num_cpus": 1,
        },
        "duet": {
            "optimizer": {"lambda": [0.9]},
            "pep": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.1},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 200, "num_cpus": 1,
            },
        },
    }
    path = tmp_path / "ops.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


class TestEvalMemBudget:
    def test_default(self, minimal_yaml):
        cfg = OpsBenchmarkConfig.from_yaml(minimal_yaml)
        assert cfg.eval_mem_budget_gb == 20.0

    def test_explicit_on_evaluator(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        d["evaluator"]["mem_budget_gb"] = 64.0
        p = tmp_path / "ops2.yaml"
        p.write_text(yaml.safe_dump(d))
        cfg = OpsBenchmarkConfig.from_yaml(p)
        assert cfg.eval_mem_budget_gb == 64.0

    def test_rejects_nested_duet_eval_mem(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        d["duet"]["eval_mem_budget_gb"] = 64.0
        p = tmp_path / "ops3.yaml"
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="evaluator.mem_budget_gb"):
            OpsBenchmarkConfig.from_yaml(p)


class TestLegacyKeysRejected:
    @pytest.mark.parametrize("legacy_key,legacy_value", [
        ("guides_per_gene", 1),
        ("num_ntc", 0),
        ("num_genes", 4),
        ("min_guide_rank", 10),
        ("max_ntc_pairs", 1000),
    ])
    def test_legacy_candidate_pool_key_rejected(self, tmp_path, legacy_key, legacy_value):
        d = {
            "outdir": str(tmp_path / "out"),
            "trials": 1,
            "candidate_pool": {
                "source": "synthetic",
                "seq_rounds": 4,
                "quota": 1,
                "num_controls": 0,
                "num_groups": 4,
                "candidates_per_group": 3,
                "alphabet_size": 4,
                legacy_key: legacy_value,  # injected legacy alias
            },
            "evaluator": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.1},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 200,
                "num_cpus": 1,
            },
        }
        p = tmp_path / "legacy.yaml"
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match=legacy_key):
            OpsBenchmarkConfig.from_yaml(p)

    def test_legacy_section_name_rejected(self, tmp_path):
        """Top-level section rename: guide_candidates -> candidate_pool."""
        d = {
            "outdir": str(tmp_path / "out"),
            "trials": 1,
            "guide_candidates": {
                "source": "synthetic",
                "seq_rounds": 4,
                "quota": 1,
                "num_controls": 0,
                "num_groups": 4,
                "candidates_per_group": 3,
                "alphabet_size": 4,
            },
            "evaluator": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.1},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 200,
                "num_cpus": 1,
            },
        }
        p = tmp_path / "legacy.yaml"
        p.write_text(yaml.safe_dump(d))
        with pytest.raises((KeyError, ValueError)):
            OpsBenchmarkConfig.from_yaml(p)


class TestDeprecatedOptimizerKeys:
    def test_crowding_weight_rejected(self, tmp_path):
        d = {
            "outdir": str(tmp_path / "out"),
            "trials": 1,
            "candidate_pool": {
                "source": "synthetic", "seq_rounds": 4, "quota": 1,
                "num_controls": 0, "num_groups": 4, "candidates_per_group": 3,
                "alphabet_size": 4,
            },
            "evaluator": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.1},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 200, "num_cpus": 1,
            },
            "duet": {
                "optimizer": {"lambda": [0.9], "crowding_weight": 0.5},
                "pep": {
                    "noise_channel": {"type": "symmetric", "epsilon": 0.1},
                    "decoding_metric": {"type": "hamming"},
                    "decoding_rule": {"type": "unique_minimum"},
                    "num_samples": 200, "num_cpus": 1,
                },
            },
        }
        p = tmp_path / "deprecated.yaml"
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="crowding_weight"):
            OpsBenchmarkConfig.from_yaml(p)


class TestDeviceResolution:
    def test_top_level_inherited_by_both_steps(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        d["device"] = "gpu:all"
        p = tmp_path / "g.yaml"; p.write_text(yaml.safe_dump(d))
        cfg = OpsBenchmarkConfig.from_yaml(p)
        assert cfg.duet_device == "gpu:all"
        assert cfg.eval_device == "gpu:all"

    def test_step_overrides(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        d["device"] = "gpu:all"
        d["duet"]["device"] = "cpu"
        p = tmp_path / "g2.yaml"; p.write_text(yaml.safe_dump(d))
        cfg = OpsBenchmarkConfig.from_yaml(p)
        assert cfg.duet_device == "cpu"
        assert cfg.eval_device == "gpu:all"

    def test_null_device_normalizes_to_cpu(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        d["device"] = None
        p = tmp_path / "g3.yaml"; p.write_text(yaml.safe_dump(d))
        cfg = OpsBenchmarkConfig.from_yaml(p)
        assert cfg.eval_device == "cpu"
        assert cfg.duet_device == "cpu"

    def test_duet_absent_still_resolves_eval_device(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        del d["duet"]
        d["device"] = "gpu:all"
        p = tmp_path / "g4.yaml"; p.write_text(yaml.safe_dump(d))
        cfg = OpsBenchmarkConfig.from_yaml(p)
        assert cfg.run_duet is False
        assert cfg.eval_device == "gpu:all"   # NOT silently cpu

    def test_requires_duet_pep(self, minimal_yaml, tmp_path):
        d = yaml.safe_load(minimal_yaml.read_text())
        del d["duet"]["pep"]
        p = tmp_path / "g5.yaml"; p.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="duet.pep` is required"):
            OpsBenchmarkConfig.from_yaml(p)
