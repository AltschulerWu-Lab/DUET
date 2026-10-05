# tests/test_merfish_benchmark_config.py
"""Tests for MERFISH benchmark config dataclasses."""

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import pandas as pd
import pytest

from duet.merfish_benchmark.config import (
    MerfishCandidatesConfig,
    ExpressionConfig,
    RandomGenesConfig,
    FromCsvGenesConfig,
    parse_genes_config,
    CrowdingConfig,
    InitializationConfig,
    BaselineConfig,
    MerfishBenchmarkConfig,
)


def test_defaults_when_new_fields_absent():
    cfg = MerfishCandidatesConfig.from_dict({
        "seq_rounds": 32,
        "codebook_size": 1147,
        "hamming_weights": [3, 4, 5],
    })
    assert cfg.hamming_weight_limits is None
    assert cfg.subsample_seed is None


def test_parses_hamming_weight_limits():
    cfg = MerfishCandidatesConfig.from_dict({
        "seq_rounds": 32,
        "codebook_size": 1147,
        "hamming_weights": [3, 4, 5],
        "hamming_weight_limits": {5: 59080},
        "subsample_seed": 7,
    })
    assert cfg.hamming_weight_limits == {5: 59080}
    assert cfg.subsample_seed == 7


def test_subsample_seed_zero_is_preserved():
    """A YAML-supplied `subsample_seed: 0` must round-trip as 0, not None.

    Guards against `or`-based fallback logic that treats 0 as "omitted"
    (see runner-side handling in Task 5).
    """
    cfg = MerfishCandidatesConfig.from_dict({
        "seq_rounds": 32,
        "codebook_size": 1147,
        "hamming_weights": [3, 4, 5],
        "subsample_seed": 0,
    })
    assert cfg.subsample_seed == 0
    assert cfg.subsample_seed is not None


# ---------------------------------------------------------------------------
# New schema tests (Task 2)
# ---------------------------------------------------------------------------


def test_expression_config_defaults():
    cfg = ExpressionConfig.from_dict({"path": "/tmp/expr.csv"})
    assert cfg.path == Path("/tmp/expr.csv")
    assert cfg.gene_col == "gene_name"
    assert cfg.expression_col == "mean_raw_counts"


def test_expression_config_overrides():
    cfg = ExpressionConfig.from_dict({
        "path": "/tmp/expr.csv", "gene_col": "symbol", "expression_col": "counts",
    })
    assert cfg.gene_col == "symbol"
    assert cfg.expression_col == "counts"


def test_random_genes_config_defaults():
    cfg = RandomGenesConfig.from_dict({"pool_filter": "top_percentile", "pool_filter_value": 1})
    assert cfg.pool_filter == "top_percentile"
    assert cfg.pool_filter_value == 1
    assert cfg.exclude_controls is True


def test_random_genes_config_requires_filter():
    with pytest.raises(KeyError):
        RandomGenesConfig.from_dict({})


def test_from_csv_genes_config_defaults():
    cfg = FromCsvGenesConfig.from_dict({"path": "/tmp/cb.csv"})
    assert cfg.path == Path("/tmp/cb.csv")
    assert cfg.gene_col == "Gene"


def test_from_csv_genes_config_override_gene_col():
    cfg = FromCsvGenesConfig.from_dict({"path": "/tmp/cb.csv", "gene_col": "name"})
    assert cfg.gene_col == "name"


def test_parse_genes_config_random():
    cfg = parse_genes_config({"type": "random", "pool_filter": "top_k", "pool_filter_value": 1000})
    assert isinstance(cfg, RandomGenesConfig)


def test_parse_genes_config_from_csv():
    cfg = parse_genes_config({"type": "from_csv", "path": "/tmp/cb.csv"})
    assert isinstance(cfg, FromCsvGenesConfig)


def test_parse_genes_config_unknown_type():
    with pytest.raises(ValueError, match="genes.type"):
        parse_genes_config({"type": "magic"})


def test_crowding_config_defaults():
    cfg = CrowdingConfig.from_dict({})
    assert cfg.n_trials == 5
    assert cfg.total_reads == 100_000
    assert cfg.cell_size_um == 100.0
    assert cfg.wavelength_nm == 500.0
    assert cfg.numerical_aperture == 1.4
    assert cfg.expansion_factor == 1.0


def test_crowding_config_defaults_to_abbe_box():
    from duet.merfish_benchmark.config import CrowdingConfig
    c = CrowdingConfig.from_dict({})
    assert c.diffraction_model == "abbe"
    assert c.neighborhood == "box"
    c2 = CrowdingConfig.from_dict({"diffraction_model": "rayleigh", "neighborhood": "disk"})
    assert (c2.diffraction_model, c2.neighborhood) == ("rayleigh", "disk")


def test_initialization_config_random():
    cfg = InitializationConfig.from_dict({"type": "random"})
    assert cfg.type == "random"
    assert cfg.path is None


def test_initialization_config_warm_start_requires_path():
    with pytest.raises(KeyError):
        InitializationConfig.from_dict({"type": "warm_start"})


def test_initialization_config_warm_start_defaults():
    cfg = InitializationConfig.from_dict({"type": "warm_start", "path": "/tmp/cb.csv"})
    assert cfg.type == "warm_start"
    assert cfg.path == Path("/tmp/cb.csv")
    assert cfg.sequence_col == "Sequence"
    assert cfg.noise_percent == 0.0


def test_initialization_config_unknown_type():
    with pytest.raises(ValueError, match="initialization.type"):
        InitializationConfig.from_dict({"type": "magic"})


def test_initialization_config_warm_start_direct_construction_requires_path():
    with pytest.raises(ValueError, match="path is required"):
        InitializationConfig(type="warm_start", path=None)


def test_baseline_config_defaults():
    cfg = BaselineConfig.from_dict({"name": "chen", "path": "/tmp/cb.csv"})
    assert cfg.name == "chen"
    assert cfg.gene_col == "Gene"
    assert cfg.sequence_col == "Sequence"


def test_baseline_config_overrides():
    cfg = BaselineConfig.from_dict({
        "name": "zhang_v2", "path": "/tmp/cb.csv", "gene_col": "name", "sequence_col": "barcode",
    })
    assert cfg.gene_col == "name"
    assert cfg.sequence_col == "barcode"


def test_baseline_parse_list_empty():
    assert BaselineConfig.parse_list(None) == []
    assert BaselineConfig.parse_list([]) == []


def test_baseline_parse_list_two_entries():
    entries = BaselineConfig.parse_list([
        {"name": "a", "path": "/tmp/a.csv"},
        {"name": "b", "path": "/tmp/b.csv", "gene_col": "name"},
    ])
    assert len(entries) == 2
    assert entries[1].gene_col == "name"


def test_baseline_parse_list_duplicate_names_error():
    with pytest.raises(ValueError, match="duplicate"):
        BaselineConfig.parse_list([
            {"name": "a", "path": "/tmp/a.csv"},
            {"name": "a", "path": "/tmp/other.csv"},
        ])


def test_baseline_parse_list_rejects_old_struct():
    with pytest.raises(ValueError, match="list of dicts"):
        BaselineConfig.parse_list({"codebook_1": "/tmp/a.csv"})


def _minimal_d():
    return {
        "outdir": "/tmp/out",
        "candidates": {"seq_rounds": 32, "codebook_size": 4, "hamming_weights": [2]},
        "evaluator": {
            "noise_channel": {"type": "symmetric", "epsilon": 0.05},
            "decoding_metric": {"type": "hamming"},
            "decoding_rule": {"type": "unique_minimum"},
            "num_samples": 10, "num_cpus": 1,
        },
        "duet": {
            "lambda": [1.0], "max_iter": 1, "max_patience": 1,
            "pep": {
                "noise_channel": {"type": "symmetric", "epsilon": 0.05},
                "decoding_metric": {"type": "hamming"},
                "decoding_rule": {"type": "unique_minimum"},
                "num_samples": 10, "num_cpus": 1,
            },
        },
    }


def test_merfish_config_minimal_round_trips():
    cfg = MerfishBenchmarkConfig.from_dict(_minimal_d())
    assert cfg.candidates.codebook_size == 4
    assert cfg.crowding is None
    assert cfg.expression is None
    assert cfg.genes is None
    assert cfg.initialization.type == "random"
    assert cfg.initialization.path is None
    assert cfg.baselines == []
    assert cfg.duet_optimizer.lambda_ == [1.0]


def test_merfish_config_rejects_old_optical_crowding():
    d = _minimal_d()
    d["optical_crowding"] = {"expression_data": {"path": "/tmp/e.csv"}}
    with pytest.raises(ValueError, match="optical_crowding"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_rejects_old_duet_initialization():
    d = _minimal_d()
    d["duet"]["initialization"] = {"type": "random"}
    with pytest.raises(ValueError, match="duet.initialization"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_rejects_old_duet_optimizer():
    d = _minimal_d()
    d["duet"] = {"optimizer": {"alpha": [0.0]}}
    with pytest.raises(ValueError, match="duet.optimizer"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_crowding_requires_expression(tmp_path):
    cb = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["G"] * 4, "Sequence": ["00"] * 4}).to_csv(cb, index=False)
    d = _minimal_d()
    d["crowding"] = {"n_trials": 1}
    d["genes"] = {"type": "from_csv", "path": str(cb)}
    with pytest.raises(ValueError, match="expression"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_crowding_requires_genes(tmp_path):
    d = _minimal_d()
    d["crowding"] = {"n_trials": 1}
    d["expression"] = {"path": "/tmp/expr.csv"}
    with pytest.raises(ValueError, match="genes"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_from_csv_genes_missing_file(tmp_path):
    d = _minimal_d()
    d["expression"] = {"path": "/tmp/expr.csv"}
    d["genes"] = {"type": "from_csv", "path": str(tmp_path / "does_not_exist.csv")}
    with pytest.raises(ValueError, match="genes.path"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_from_csv_genes_missing_column(tmp_path):
    cb = tmp_path / "cb.csv"
    pd.DataFrame({"WRONG": ["G"] * 4, "Sequence": ["00"] * 4}).to_csv(cb, index=False)
    d = _minimal_d()
    d["expression"] = {"path": "/tmp/expr.csv"}
    d["genes"] = {"type": "from_csv", "path": str(cb), "gene_col": "Gene"}
    with pytest.raises(ValueError, match="gene_col"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_warm_start_missing_file(tmp_path):
    d = _minimal_d()
    d["initialization"] = {"type": "warm_start", "path": str(tmp_path / "missing.csv")}
    with pytest.raises(ValueError, match="initialization.path"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_config_warm_start_missing_column(tmp_path):
    cb = tmp_path / "cb.csv"
    pd.DataFrame({"Gene": ["G"] * 4, "BAD": ["00"] * 4}).to_csv(cb, index=False)
    d = _minimal_d()
    d["initialization"] = {"type": "warm_start", "path": str(cb), "sequence_col": "Sequence"}
    with pytest.raises(ValueError, match="sequence_col"):
        MerfishBenchmarkConfig.from_dict(d)


def test_eval_knobs_default_and_parse():
    import yaml
    from pathlib import Path
    from duet.merfish_benchmark.config import MerfishBenchmarkConfig

    cfg_path = Path(__file__).parent / "regression" / "data" / "merfish_config_post.yaml"
    cfg = MerfishBenchmarkConfig.from_yaml(cfg_path)
    assert cfg.eval_device == "cpu"            # no top-level/evaluator device -> cpu
    assert cfg.eval_mem_budget_gb == 20.0      # default preserved

    d = yaml.safe_load(cfg_path.read_text())
    d["evaluator"]["device"] = "gpu:all"
    d["evaluator"]["mem_budget_gb"] = 48.0
    cfg2 = MerfishBenchmarkConfig.from_dict(d, config_dir=cfg_path.parent)
    assert cfg2.eval_device == "gpu:all"
    assert cfg2.eval_mem_budget_gb == 48.0


def test_merfish_requires_duet_pep():
    d = _minimal_d()
    del d["duet"]["pep"]
    with pytest.raises(ValueError, match="duet.pep` is required"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_rejects_pep_runtime_keys():
    d = _minimal_d()
    d["duet"]["pep"]["device"] = "gpu:all"
    with pytest.raises(ValueError, match="duet.device"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_rejects_moved_top_level_eval_keys():
    d = _minimal_d()
    d["eval_device"] = "gpu:all"
    with pytest.raises(ValueError, match="evaluator.device"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_rejects_nested_duet_eval_mem():
    d = _minimal_d()
    d["duet"]["eval_mem_budget_gb"] = 64.0
    with pytest.raises(ValueError, match="evaluator.mem_budget_gb"):
        MerfishBenchmarkConfig.from_dict(d)


def test_merfish_top_level_device_inherited_by_both_steps():
    d = _minimal_d()
    d["device"] = "gpu:all"
    cfg = MerfishBenchmarkConfig.from_dict(d)
    assert cfg.duet_device == "gpu:all"
    assert cfg.eval_device == "gpu:all"


def test_merfish_step_overrides_beat_top_level():
    d = _minimal_d()
    d["device"] = "gpu:all"
    d["duet"]["device"] = "cpu"
    d["evaluator"]["device"] = "gpu:all"
    cfg = MerfishBenchmarkConfig.from_dict(d)
    assert cfg.duet_device == "cpu"
    assert cfg.eval_device == "gpu:all"
