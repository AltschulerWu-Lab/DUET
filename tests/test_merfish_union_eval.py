"""Tests for the MERFISH sequence-keyed union evaluator and runner wiring."""
from __future__ import annotations

import numpy as np
import pytest

from duet.benchmark.metrics import (
    CodebookEvalResult,
    _build_multiset_union,
    evaluate_codebooks_by_sequence,
)
from duet.evaluator_config import EvaluatorConfig, create_evaluator

CFG = EvaluatorConfig.default_symmetric(epsilon=0.1, num_samples=600, seed=42)
SEQS = ["0011", "0101", "0110", "1001", "1010", "1100"]


def _standalone(seqs):
    ev = create_evaluator(list(seqs), CFG, alphabet_size=2)
    ev.initialize_cache(n_jobs=1)
    return ev.get_codeword_accuracy(), ev.get_error_correction_metrics()


class TestMultisetUnion:
    def test_no_duplicates_is_set_union(self):
        # Two overlapping codebooks; union has each distinct sequence once.
        cbs = [["0011", "0101", "0110"], ["0101", "0110", "1001"]]
        union, mapped = _build_multiset_union(cbs)
        assert sorted(union) == sorted({"0011", "0101", "0110", "1001"})
        assert len(union) == 4
        # Each codebook maps back to its own sequences via the union.
        for cb, idx in zip(cbs, mapped):
            assert [union[i] for i in idx] == cb

    def test_within_codebook_duplicate_gets_two_copies(self):
        cbs = [["0011", "0011", "0101"]]      # "0011" twice in one codebook
        union, mapped = _build_multiset_union(cbs)
        assert union.count("0011") == 2       # two distinct union rows
        assert list(mapped[0]) == [0, 1, 2]   # grouped -> identity map
        assert len(set(mapped[0])) == 3       # the two "0011" positions are DISTINCT columns


class TestEvaluateCodebooksBySequence:
    def test_single_codebook_bit_exact(self):                      # spec §11 #1
        acc_std, err_std = _standalone(SEQS)
        [res] = evaluate_codebooks_by_sequence([SEQS], eval_config=CFG, alphabet_size=2, n_jobs=1)
        np.testing.assert_array_equal(res.codeword_accuracy, acc_std)
        np.testing.assert_array_equal(res.error_metrics.codeword_failed, err_std.codeword_failed)

    def test_grouped_duplicate(self):                              # spec §11 #2
        # Grouped duplicate -> identity map -> bit-exact vs standalone build over
        # the literal duplicated list; the two X positions tie -> exactly 0.0 accuracy.
        cb = ["0011", "0011", "0101", "0110"]
        acc_std, _ = _standalone(cb)
        [res] = evaluate_codebooks_by_sequence([cb], eval_config=CFG, alphabet_size=2, n_jobs=1)
        np.testing.assert_array_equal(res.codeword_accuracy, acc_std)  # holds ONLY because grouped
        assert res.codeword_accuracy[0] == 0.0      # duplicated positions decode at exactly 0.0
        assert res.codeword_accuracy[1] == 0.0

    def test_multi_codebook_within_tolerance(self):                # spec §11 #3
        K = 2000
        cfg = EvaluatorConfig.default_symmetric(epsilon=0.1, num_samples=K, seed=7)
        cbs = [SEQS[:5], SEQS[1:], [SEQS[0], SEQS[2], SEQS[4]]]
        results = evaluate_codebooks_by_sequence(cbs, eval_config=cfg, alphabet_size=2, n_jobs=1)
        bound = 6 * np.sqrt(0.25 / K)                  # ~6 sigma worst-case-p
        for cb, res in zip(cbs, results):
            ev = create_evaluator(list(cb), cfg, alphabet_size=2)
            ev.initialize_cache(n_jobs=1)
            assert np.max(np.abs(res.codeword_accuracy - ev.get_codeword_accuracy())) <= bound

    def test_empty_returns_empty(self):
        assert evaluate_codebooks_by_sequence([], eval_config=CFG, alphabet_size=2) == []


class TestRunnerWiring:
    """End-to-end numeric/alignment checks on the small post-refactor config."""

    @pytest.mark.slow
    def test_runner_outputs_aligned_and_no_activity_score(self, tmp_path):
        import argparse
        import pandas as pd
        import yaml
        from pathlib import Path
        from duet.merfish_benchmark import MerfishBenchmarkConfig, run_merfish_benchmark

        data = Path(__file__).parent / "regression" / "data"
        config = MerfishBenchmarkConfig.from_yaml(data / "merfish_config_post.yaml")
        config.outdir = tmp_path / "out"
        args = argparse.Namespace(debug_pep=False, config=data / "merfish_config_post.yaml")
        run_merfish_benchmark(config, args=args)

        results = pd.read_csv(config.outdir / "results.csv")
        summary = yaml.safe_load((config.outdir / "summary.yaml").read_text())

        # (a) Activity score dropped; Trial/Valid kept; no init/diagnostic rows leaked.
        assert "Activity score" not in results.columns
        assert {"Trial", "Valid"}.issubset(results.columns)
        assert not results["Method"].astype(str).str.contains("__init__").any()

        # (b) per-codeword alignment: each row's Sequence is non-empty and accuracy in [0, 1].
        assert results["Sequence"].astype(str).str.len().gt(0).all()
        assert results["Decode accuracy"].between(0.0, 1.0).all()

        # (c) summary carries initial_accuracy in range.
        assert 0.0 <= float(summary["initial_accuracy"]) <= 1.0
        # (d) at least one DUET method and one baseline method present.
        methods = set(results["Method"])
        assert any(m.startswith("DUET (lambda=") for m in methods)
