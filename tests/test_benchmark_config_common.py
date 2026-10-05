# tests/test_benchmark_config_common.py
"""Unit tests for the shared benchmark config resolution helper."""
import logging

import pytest

from duet.benchmark import config_common as cc


def test_normalized_top_device():
    assert cc.normalized_top_device({}) == "cpu"
    assert cc.normalized_top_device({"device": None}) == "cpu"
    assert cc.normalized_top_device({"device": ""}) == "cpu"
    assert cc.normalized_top_device({"device": "gpu:all"}) == "gpu:all"


def test_resolve_step_device_block_wins():
    assert cc.resolve_step_device("cpu", "gpu:all", step_name="eval") == "gpu:all"


def test_resolve_step_device_inherits_when_unset():
    assert cc.resolve_step_device("gpu:all", None, step_name="eval") == "gpu:all"
    assert cc.resolve_step_device("gpu:all", "", step_name="eval") == "gpu:all"


def test_resolve_step_device_logs_only_on_nondefault_inherit(caplog):
    with caplog.at_level(logging.INFO, logger="duet.benchmark.config_common"):
        cc.resolve_step_device("gpu:all", None, step_name="DUET")
    assert any("inheriting device" in r.message for r in caplog.records)
    caplog.clear()
    # Silent when the step explicitly repeats the top value:
    with caplog.at_level(logging.INFO, logger="duet.benchmark.config_common"):
        cc.resolve_step_device("gpu:all", "gpu:all", step_name="DUET")
    assert not caplog.records
    caplog.clear()
    # Silent when inheriting the plain cpu default:
    with caplog.at_level(logging.INFO, logger="duet.benchmark.config_common"):
        cc.resolve_step_device("cpu", None, step_name="DUET")
    assert not caplog.records


def test_resolve_step_mem():
    assert cc.resolve_step_mem(None, 64.0, 32.0) == 64.0   # block wins
    assert cc.resolve_step_mem(48.0, None, 32.0) == 48.0   # inherit top
    assert cc.resolve_step_mem(None, None, 32.0) == 32.0   # default


def test_resolve_scratch():
    assert cc.resolve_scratch("/top", "/block") == "/block"
    assert cc.resolve_scratch("/top", None) == "/top"
    assert cc.resolve_scratch(None, None) is None


def test_require_duet_pep():
    cc.require_duet_pep({"pep": {"noise_channel": {}}})  # no raise
    with pytest.raises(ValueError, match="duet.pep` is required"):
        cc.require_duet_pep({"optimizer": {}})


def test_reject_pep_runtime_keys():
    cc.reject_pep_runtime_keys({"pep": {"num_samples": 5}})  # no raise
    with pytest.raises(ValueError, match="duet.device"):
        cc.reject_pep_runtime_keys({"pep": {"device": "gpu:all"}})
    with pytest.raises(ValueError, match="duet.device"):
        cc.reject_pep_runtime_keys({"pep": {"mem_budget_gb": 8.0}})


def test_reject_nested_eval_mem():
    cc.reject_nested_eval_mem({"sym_mem_budget_gb": 32.0})  # no raise
    with pytest.raises(ValueError, match="evaluator.mem_budget_gb"):
        cc.reject_nested_eval_mem({"eval_mem_budget_gb": 64.0})
