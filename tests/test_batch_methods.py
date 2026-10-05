"""Tests for CodebookEvaluator batch evaluation methods."""

import numpy as np
import pytest

from duet.codebook_evaluator import (
    CodebookEvaluator,
    ErrorCorrectionMetrics,
    SymmetricEpsilon,
    SymmetricNLL,
    UniqueMinimum,
)


def _make_evaluator():
    """Create a small evaluator with initialized cache for testing.

    Returns (evaluator, codebooks) where:
    - 3 groups × 4 candidates = 12 codewords, seq_length=6, alphabet=4
    - K=200 samples, symmetric noise at epsilon=0.1
    - codebooks: all 4^3 = 64 possible codebooks (quota=1 per group)
    """
    rng = np.random.default_rng(42)
    num_groups, candidates_per_group = 3, 4
    pool_size = num_groups * candidates_per_group
    seq_length = 6
    alphabet_size = 4

    sequences = rng.integers(0, alphabet_size, size=(pool_size, seq_length), dtype=np.int8)

    evaluator = CodebookEvaluator(
        codebook=sequences,
        noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=alphabet_size),
        decoding_metric=SymmetricNLL(epsilon=0.1, alphabet_size=alphabet_size),
        decoding_rule=UniqueMinimum(),
        n_samples=200,
        seed=42,
    )
    evaluator.initialize_cache(n_jobs=1)

    # Build all 4^3 = 64 codebooks (quota=1 from each of 3 groups)
    from itertools import product
    group_candidates = [
        list(range(g * candidates_per_group, (g + 1) * candidates_per_group))
        for g in range(num_groups)
    ]
    codebooks = [
        np.array(combo) for combo in product(*group_candidates)
    ]

    return evaluator, codebooks


class TestBatchGetAccuracy:
    """batch_get_accuracy must match sequential get_accuracy calls."""

    def test_matches_sequential(self):
        evaluator, codebooks = _make_evaluator()

        # Sequential: call get_accuracy for each codebook
        sequential = np.array([
            evaluator.get_accuracy(cb) for cb in codebooks
        ])

        # Batch
        batch = evaluator.batch_get_accuracy(codebooks)

        assert batch.shape == (len(codebooks),)
        np.testing.assert_allclose(batch, sequential, atol=1e-15)

    def test_single_codebook(self):
        evaluator, codebooks = _make_evaluator()
        single = [codebooks[0]]

        result = evaluator.batch_get_accuracy(single)

        assert result.shape == (1,)
        expected = evaluator.get_accuracy(codebooks[0])
        np.testing.assert_allclose(result[0], expected, atol=1e-15)

    def test_small_memory_budget_still_correct(self):
        evaluator, codebooks = _make_evaluator()

        # Tiny budget forces multiple batches
        result_small = evaluator.batch_get_accuracy(codebooks, mem_budget_gb=0.0001)
        result_large = evaluator.batch_get_accuracy(codebooks, mem_budget_gb=20.0)

        np.testing.assert_allclose(result_small, result_large, atol=1e-15)

    def test_requires_initialized_cache(self):
        rng = np.random.default_rng(1)
        sequences = rng.integers(0, 4, size=(6, 4), dtype=np.int8)
        evaluator = CodebookEvaluator(
            codebook=sequences,
            noise_channel=SymmetricEpsilon(epsilon=0.1, alphabet_size=4),
            decoding_metric=SymmetricNLL(epsilon=0.1, alphabet_size=4),
            decoding_rule=UniqueMinimum(),
            n_samples=50,
            seed=1,
        )
        with pytest.raises(RuntimeError, match="Cache must be initialized"):
            evaluator.batch_get_accuracy([np.array([0, 1, 2])])


class TestBatchGetCodewordAccuracy:
    """batch_get_codeword_accuracy must match sequential get_codeword_accuracy."""

    def test_matches_sequential(self):
        evaluator, codebooks = _make_evaluator()

        sequential = [
            evaluator.get_codeword_accuracy(cb) for cb in codebooks
        ]

        batch = evaluator.batch_get_codeword_accuracy(codebooks)

        assert len(batch) == len(codebooks)
        for i, cb in enumerate(codebooks):
            assert batch[i].shape == (len(cb),)
            np.testing.assert_allclose(batch[i], sequential[i], atol=1e-15)

    def test_mean_matches_batch_get_accuracy(self):
        evaluator, codebooks = _make_evaluator()

        cw_acc = evaluator.batch_get_codeword_accuracy(codebooks)
        mean_acc = evaluator.batch_get_accuracy(codebooks)

        per_codebook_mean = np.array([a.mean() for a in cw_acc])
        np.testing.assert_allclose(per_codebook_mean, mean_acc, atol=1e-15)


class TestBatchGetErrorCorrectionMetrics:
    """batch_get_error_correction_metrics must match sequential."""

    def test_matches_sequential(self):
        evaluator, codebooks = _make_evaluator()

        # Sequential
        sequential = [
            evaluator.get_error_correction_metrics(cb) for cb in codebooks
        ]

        # Batch
        batch = evaluator.batch_get_error_correction_metrics(codebooks)

        assert len(batch) == len(codebooks)
        for i in range(len(codebooks)):
            np.testing.assert_allclose(
                batch[i].codeword_no_error, sequential[i].codeword_no_error,
                atol=1e-15, err_msg=f"codebook {i}: codeword_no_error",
            )
            np.testing.assert_allclose(
                batch[i].codeword_corrected, sequential[i].codeword_corrected,
                atol=1e-15, err_msg=f"codebook {i}: codeword_corrected",
            )
            np.testing.assert_allclose(
                batch[i].codeword_failed, sequential[i].codeword_failed,
                atol=1e-15, err_msg=f"codebook {i}: codeword_failed",
            )
            assert batch[i].no_error_rate == pytest.approx(sequential[i].no_error_rate, abs=1e-15)
            assert batch[i].corrected_rate == pytest.approx(sequential[i].corrected_rate, abs=1e-15)
            assert batch[i].failed_rate == pytest.approx(sequential[i].failed_rate, abs=1e-15)

    def test_rates_sum_to_one(self):
        evaluator, codebooks = _make_evaluator()

        batch = evaluator.batch_get_error_correction_metrics(codebooks)

        for i, ecm in enumerate(batch):
            total = ecm.no_error_rate + ecm.corrected_rate + ecm.failed_rate
            assert total == pytest.approx(1.0, abs=1e-10), \
                f"codebook {i}: rates sum to {total}"


class TestBatchGetAccuracyWithErrorMetrics:
    """Combined method must equal individual batch methods."""

    def test_accuracy_matches_batch_get_codeword_accuracy(self):
        evaluator, codebooks = _make_evaluator()

        cw_acc_standalone = evaluator.batch_get_codeword_accuracy(codebooks)
        cw_acc_combined, _ = evaluator.batch_get_accuracy_with_error_metrics(codebooks)

        assert len(cw_acc_combined) == len(cw_acc_standalone)
        for i in range(len(codebooks)):
            np.testing.assert_allclose(
                cw_acc_combined[i], cw_acc_standalone[i], atol=1e-15
            )

    def test_metrics_match_batch_get_error_correction_metrics(self):
        evaluator, codebooks = _make_evaluator()

        metrics_standalone = evaluator.batch_get_error_correction_metrics(codebooks)
        _, metrics_combined = evaluator.batch_get_accuracy_with_error_metrics(codebooks)

        for i in range(len(codebooks)):
            np.testing.assert_allclose(
                metrics_combined[i].codeword_corrected,
                metrics_standalone[i].codeword_corrected,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                metrics_combined[i].codeword_failed,
                metrics_standalone[i].codeword_failed,
                atol=1e-15,
            )
            assert metrics_combined[i].no_error_rate == pytest.approx(
                metrics_standalone[i].no_error_rate, abs=1e-15
            )


class TestRaggedCodebookSupport:
    """Batch methods must handle codebooks of different sizes and reject empty codebooks."""

    def _make_ragged_codebooks(self):
        # Pool has 12 candidates across 3 groups. Pick indices freely — shape
        # matters more than group structure for these tests.
        return [
            np.array([0, 4, 8]),        # size 3
            np.array([0, 4]),           # size 2
            np.array([0, 4, 8, 1]),     # size 4
        ]

    def test_batch_get_accuracy_matches_sequential(self):
        evaluator, _ = _make_evaluator()
        codebooks = self._make_ragged_codebooks()

        sequential = np.array([
            evaluator.get_accuracy(cb) for cb in codebooks
        ])
        batch = evaluator.batch_get_accuracy(codebooks)

        assert batch.shape == (len(codebooks),)
        np.testing.assert_allclose(batch, sequential, atol=1e-15)

    def test_batch_get_codeword_accuracy_matches_sequential(self):
        evaluator, _ = _make_evaluator()
        codebooks = self._make_ragged_codebooks()

        sequential = [
            evaluator.get_codeword_accuracy(cb) for cb in codebooks
        ]
        batch = evaluator.batch_get_codeword_accuracy(codebooks)

        assert len(batch) == len(codebooks)
        for i, cb in enumerate(codebooks):
            assert batch[i].shape == (len(cb),), \
                f"codebook {i}: expected shape {(len(cb),)}, got {batch[i].shape}"
            np.testing.assert_allclose(batch[i], sequential[i], atol=1e-15)

    def test_batch_get_error_correction_metrics_matches_sequential(self):
        evaluator, _ = _make_evaluator()
        codebooks = self._make_ragged_codebooks()

        sequential = [
            evaluator.get_error_correction_metrics(cb) for cb in codebooks
        ]
        batch = evaluator.batch_get_error_correction_metrics(codebooks)

        assert len(batch) == len(codebooks)
        for i, cb in enumerate(codebooks):
            assert batch[i].codeword_no_error.shape == (len(cb),)
            np.testing.assert_allclose(
                batch[i].codeword_no_error, sequential[i].codeword_no_error,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                batch[i].codeword_corrected, sequential[i].codeword_corrected,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                batch[i].codeword_failed, sequential[i].codeword_failed,
                atol=1e-15,
            )

    def test_batch_get_accuracy_with_error_metrics_matches_standalone(self):
        evaluator, _ = _make_evaluator()
        codebooks = self._make_ragged_codebooks()

        cw_acc_standalone = evaluator.batch_get_codeword_accuracy(codebooks)
        cw_acc_combined, metrics_combined = (
            evaluator.batch_get_accuracy_with_error_metrics(codebooks)
        )
        metrics_standalone = evaluator.batch_get_error_correction_metrics(codebooks)

        assert len(cw_acc_combined) == len(codebooks)
        for i in range(len(codebooks)):
            np.testing.assert_allclose(
                cw_acc_combined[i], cw_acc_standalone[i], atol=1e-15
            )
            np.testing.assert_allclose(
                metrics_combined[i].codeword_no_error,
                metrics_standalone[i].codeword_no_error,
                atol=1e-15,
            )

    def test_empty_codebook_rejected(self):
        evaluator, _ = _make_evaluator()
        with pytest.raises(ValueError, match="non-empty"):
            evaluator.batch_get_accuracy([np.array([], dtype=int)])

    def test_empty_codebook_in_mixed_list_rejected(self):
        evaluator, _ = _make_evaluator()
        mixed = [np.array([0, 4, 8]), np.array([], dtype=int)]
        with pytest.raises(ValueError, match="non-empty"):
            evaluator.batch_get_accuracy(mixed)
