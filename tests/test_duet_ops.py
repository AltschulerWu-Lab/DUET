# tests/test_duet_ops.py
"""Smoke test: run_duet_ops on a tiny pool completes and returns expected shape."""
import numpy as np

from duet.candidate_pool_factory import create_pool_from_source
from duet.runner import DuetOptimizerConfig, run_duet_ops
from duet.evaluator_config import EvaluatorConfig


def test_run_duet_ops_tiny_pool():
    candidates = create_pool_from_source(
        source="synthetic",
        seq_rounds=4,
        quota=1,
        num_controls=0,
        num_groups=4,
        candidates_per_group=3,
        alphabet_size=4,
        seed=0,
    )
    pep_cfg = EvaluatorConfig.from_dict({
        "noise_channel": {"type": "symmetric", "epsilon": 0.1},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 200,
        "num_cpus": 1,
    })
    opt_cfg = DuetOptimizerConfig(lambda_=[0.0, 1.0], max_iter=100, max_patience=20, num_cpus=1)
    init = candidates.sample_initial_selection(strategy="random", seed=0)

    result = run_duet_ops(
        candidates=candidates,
        pep_config=pep_cfg,
        optimizer_config=opt_cfg,
        init=init,
        alphabet_size=4,
        seed=0,
    )
    assert set(result.keys()) >= {"lambdas", "best_indices", "histories", "pep_count_matrix", "n_samples"}
    assert result["lambdas"] == [0.0, 1.0]
    assert len(result["best_indices"]) == 2
    assert len(result["best_indices"][0]) == candidates.total_selections
