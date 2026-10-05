# tests/test_duet_merfish.py
"""Smoke test: run_duet_merfish on a tiny codebook completes."""
import numpy as np

from duet.runner import DuetOptimizerConfig, run_duet_merfish
from duet.evaluator_config import EvaluatorConfig
from duet.merfish_factory import MERFISHFactory


def test_run_duet_merfish_tiny():
    candidates = MERFISHFactory(
        seq_rounds=10, codebook_size=6, hamming_weights=[4],
    ).create()
    codewords = candidates.get_sequences_as_array(alphabet_size=2).astype(np.float64)
    expression = np.ones(6, dtype=np.float64)
    pep_cfg = EvaluatorConfig.from_dict({
        "noise_channel": {"type": "symmetric", "epsilon": 0.1},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 200,
        "num_cpus": 1,
    })
    opt_cfg = DuetOptimizerConfig(lambda_=[0.0, 0.5], max_iter=100, max_patience=20, num_cpus=1)
    init = candidates.sample_initial_selection(strategy="random", seed=0)

    result = run_duet_merfish(
        candidates=candidates,
        pep_config=pep_cfg,
        optimizer_config=opt_cfg,
        init=init,
        codewords=codewords,
        expression=expression,
        alphabet_size=2,
        seed=0,
    )
    assert result["lambdas"] == [0.0, 0.5]
    assert len(result["best_indices"][0]) == candidates.total_selections
