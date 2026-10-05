# tests/test_evaluator_config_runtime_fields.py
"""device/mem_budget_gb are parsed but must NOT enter the PEP fingerprint."""
from duet.evaluator_config import EvaluatorConfig

_BASE = {
    "noise_channel": {"type": "symmetric", "epsilon": 0.1},
    "decoding_metric": {"type": "hamming"},
    "decoding_rule": {"type": "unique_minimum"},
    "num_samples": 100,
    "num_cpus": 1,
}


def test_device_and_mem_parse_and_default_none():
    cfg = EvaluatorConfig.from_dict(_BASE)
    assert cfg.device is None
    assert cfg.mem_budget_gb is None
    cfg2 = EvaluatorConfig.from_dict({**_BASE, "device": "gpu:all", "mem_budget_gb": 20.0})
    assert cfg2.device == "gpu:all"
    assert cfg2.mem_budget_gb == 20.0


def test_device_mem_excluded_from_to_dict_and_fingerprint():
    cpu = EvaluatorConfig.from_dict({**_BASE, "device": "cpu", "mem_budget_gb": 8.0})
    gpu = EvaluatorConfig.from_dict({**_BASE, "device": "gpu:all", "mem_budget_gb": 64.0})
    assert cpu.to_dict() == gpu.to_dict()  # device/mem do not affect the cache key
    assert "device" not in cpu.to_dict()
    assert "mem_budget_gb" not in cpu.to_dict()
