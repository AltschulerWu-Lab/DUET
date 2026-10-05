"""Tests for evaluate_solutions (formerly evaluate_codebook_batch)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from duet.codebook_evaluator import (
    CodebookEvaluator,
    ErrorCorrectionMetrics,
    SymmetricEpsilon,
    HammingDistance,
    UniqueMinimum,
)
from duet.benchmark.metrics import evaluate_solutions
from duet.evaluator_config import EvaluatorConfig, ComponentConfig
from duet.candidate_pool import CandidatePool


# =============================================================================
# CSR Block Extraction Tests (unchanged — tests a general technique)
# =============================================================================


class TestCSRBlockExtraction:
    """Test that manual indptr-offset CSR slicing matches scipy slicing."""

    def test_middle_block_matches_scipy(self):
        rng = np.random.default_rng(42)
        N_rows, N_cols = 500, 30
        density = 0.15
        mask = rng.random((N_rows, N_cols)) < density
        M = csr_matrix(mask.astype(bool))

        row_start, row_end = 100, 300
        ip_start = int(M.indptr[row_start])
        ip_end = int(M.indptr[row_end])

        sub_indptr = M.indptr[row_start: row_end + 1] - ip_start
        sub_indices = M.indices[ip_start:ip_end]
        sub_data = M.data[ip_start:ip_end]

        block_manual = csr_matrix(
            (sub_data, sub_indices, sub_indptr),
            shape=(row_end - row_start, N_cols),
        )

        block_scipy = M[row_start:row_end]

        np.testing.assert_array_equal(
            block_manual.toarray(), block_scipy.toarray()
        )

    def test_first_block(self):
        rng = np.random.default_rng(123)
        M = csr_matrix(rng.random((200, 20)) < 0.2)

        row_start, row_end = 0, 50
        ip_start = int(M.indptr[row_start])
        ip_end = int(M.indptr[row_end])
        sub_indptr = M.indptr[row_start: row_end + 1] - ip_start
        sub_indices = M.indices[ip_start:ip_end]
        sub_data = M.data[ip_start:ip_end]

        block_manual = csr_matrix(
            (sub_data, sub_indices, sub_indptr),
            shape=(row_end - row_start, 20),
        )
        np.testing.assert_array_equal(
            block_manual.toarray(), M[:50].toarray()
        )

    def test_last_block(self):
        rng = np.random.default_rng(456)
        M = csr_matrix(rng.random((200, 20)) < 0.2)

        row_start, row_end = 150, 200
        ip_start = int(M.indptr[row_start])
        ip_end = int(M.indptr[row_end])
        sub_indptr = M.indptr[row_start: row_end + 1] - ip_start
        sub_indices = M.indices[ip_start:ip_end]
        sub_data = M.data[ip_start:ip_end]

        block_manual = csr_matrix(
            (sub_data, sub_indices, sub_indptr),
            shape=(row_end - row_start, 20),
        )
        np.testing.assert_array_equal(
            block_manual.toarray(), M[150:200].toarray()
        )


# =============================================================================
# evaluate_solutions Tests
# =============================================================================


DNA_BASES = "ATCG"


def _make_synthetic_pool_and_solutions():
    """Build a small candidate pool and solutions with known overlap structure."""
    rng = np.random.default_rng(42)
    N, L = 20, 6
    alphabet_size = 4

    sequences = [
        "".join(DNA_BASES[b] for b in rng.integers(0, alphabet_size, size=L))
        for _ in range(N)
    ]
    scores = rng.random(N)
    groups = [f"gene_{i // 5}" for i in range(N)]
    group_to_candidates: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        group_to_candidates.setdefault(g, []).append(idx)

    candidates = CandidatePool(
        sequences=sequences,
        scores=scores,
        group_to_candidates=group_to_candidates,
        quotas={g: 1 for g in sorted(set(groups))},
    )

    solutions = {
        "sol_A": np.array([0, 5, 10, 15]),
        "sol_B": np.array([0, 5, 10, 16]),
        "sol_D": np.array([1, 6, 11, 17]),
        "sol_E": np.array([0, 5, 10, 15]),  # identical to A
    }

    eval_config = EvaluatorConfig.default_symmetric(
        epsilon=0.1,
        num_samples=100,
        num_cpus=1,
        seed=42,
    )

    return candidates, solutions, eval_config, alphabet_size


class TestEvaluateSolutions:
    """Tests for the renamed evaluate_solutions function."""

    def test_identical_solutions_produce_identical_results(self):
        candidates, solutions, eval_config, alphabet_size = (
            _make_synthetic_pool_and_solutions()
        )
        results = evaluate_solutions(
            solutions, candidates, eval_config, alphabet_size,
        )

        acc_a, scores_a, ecm_a = results["sol_A"]
        acc_e, scores_e, ecm_e = results["sol_E"]

        np.testing.assert_array_equal(acc_a, acc_e)
        np.testing.assert_array_equal(scores_a, scores_e)
        np.testing.assert_array_equal(ecm_a.codeword_corrected, ecm_e.codeword_corrected)
        np.testing.assert_array_equal(ecm_a.codeword_failed, ecm_e.codeword_failed)

    def test_single_solution(self):
        candidates, solutions, eval_config, alphabet_size = (
            _make_synthetic_pool_and_solutions()
        )
        single = {"only": solutions["sol_A"]}

        results = evaluate_solutions(
            single, candidates, eval_config, alphabet_size,
        )

        assert "only" in results
        acc, scores, ecm = results["only"]
        assert len(acc) == 4
        assert len(scores) == 4

    def test_error_metrics_rates_sum_to_one(self):
        candidates, solutions, eval_config, alphabet_size = (
            _make_synthetic_pool_and_solutions()
        )
        results = evaluate_solutions(
            solutions, candidates, eval_config, alphabet_size,
        )
        for name, (acc, scores, ecm) in results.items():
            total = ecm.no_error_rate + ecm.corrected_rate + ecm.failed_rate
            assert total == pytest.approx(1.0, abs=1e-10), \
                f"{name}: rates sum to {total}"

    def test_variable_size_codebooks(self):
        """evaluate_solutions tolerates solutions dicts with mixed-size codebooks.

        Regression for: Feldman ED=2 / Sivanandan ED=3 may return incomplete
        codebooks. Before the list-of-arrays fix, _batch_matmul_core raised
        ValueError because of a uniform-size assumption.

        Cross-check against evaluate_codebook is intentionally avoided:
        evaluate_solutions seeds the internal noise-sampling RNG via
        SeedSequence.spawn(n) where n = |union of all solution indices|, while
        evaluate_codebook spawns over a single codebook. Different n produces
        different spawned child seeds, so the two paths will not match bit-
        for-bit. Numerical batch-vs-sequential correctness is covered by
        TestRaggedCodebookSupport in tests/test_batch_methods.py, which
        compares evaluator.batch_get_* against evaluator.get_* on the *same*
        evaluator.
        """
        candidates, _, eval_config, alphabet_size = (
            _make_synthetic_pool_and_solutions()
        )
        solutions = {
            "full_A": np.array([0, 5, 10, 15]),       # size 4 (one per group)
            "full_B": np.array([1, 6, 11, 16]),       # size 4
            "partial": np.array([2, 7, 12]),           # size 3 (group 3 missing)
            "full_A_dup": np.array([0, 5, 10, 15]),    # identical to full_A
        }

        batch = evaluate_solutions(
            solutions, candidates, eval_config, alphabet_size,
        )

        assert set(batch.keys()) == set(solutions.keys())
        for name, indices in solutions.items():
            acc, scores, ecm = batch[name]
            assert acc.shape == (len(indices),), \
                f"{name}: expected acc shape {(len(indices),)}, got {acc.shape}"
            assert scores.shape == (len(indices),)
            assert ecm.codeword_no_error.shape == (len(indices),)
            assert ecm.codeword_corrected.shape == (len(indices),)
            assert ecm.codeword_failed.shape == (len(indices),)

            # Invariant: per-codeword rates sum to 1.
            total = ecm.no_error_rate + ecm.corrected_rate + ecm.failed_rate
            assert total == pytest.approx(1.0, abs=1e-10), \
                f"{name}: rates sum to {total}"

        # Internal consistency: identical solutions within the same batch
        # call must produce identical per-solution metrics (same union, same
        # seeds, same evaluator).
        np.testing.assert_array_equal(batch["full_A"][0], batch["full_A_dup"][0])
        np.testing.assert_array_equal(
            batch["full_A"][2].codeword_corrected,
            batch["full_A_dup"][2].codeword_corrected,
        )
        np.testing.assert_array_equal(
            batch["full_A"][2].codeword_failed,
            batch["full_A_dup"][2].codeword_failed,
        )


# =============================================================================
# Intermediate Row Sums Test (unchanged)
# =============================================================================


class TestIntermediateRowSums:
    """Verify matmul competitor counts match scipy fancy indexing."""

    def test_intermediate_row_sums(self):
        rng = np.random.default_rng(99)
        N, K = 10, 50
        N_cols = N

        total_rows = N * K
        mask = rng.random((total_rows, N_cols)) < 0.3
        M = csr_matrix(mask.astype(bool))

        sol_a_indices = np.array([0, 2, 4, 6, 8])
        sol_b_indices = np.array([1, 3, 5, 7, 9])
        masks_matrix = np.zeros((N_cols, 2), dtype=np.int16)
        masks_matrix[sol_a_indices, 0] = 1
        masks_matrix[sol_b_indices, 1] = 1

        row_sums_matmul = (M @ masks_matrix)

        cw = 2
        rows = np.arange(cw * K, (cw + 1) * K)
        submat_a = M[rows, :][:, sol_a_indices]
        expected_sums_a = np.asarray(submat_a.sum(axis=1)).flatten()

        actual_sums_a = np.asarray(row_sums_matmul[rows, 0]).flatten()
        np.testing.assert_array_equal(actual_sums_a, expected_sums_a)

        submat_b = M[rows, :][:, sol_b_indices]
        expected_sums_b = np.asarray(submat_b.sum(axis=1)).flatten()
        actual_sums_b = np.asarray(row_sums_matmul[rows, 1]).flatten()
        np.testing.assert_array_equal(actual_sums_b, expected_sums_b)


# =============================================================================
# Shared Union Eval Core Tests
# =============================================================================


class TestSharedCore:
    """The shared union core matches a standalone per-codebook build bit-for-bit
    when the union is a single codebook in codebook order (identity map)."""

    def test_core_single_codebook_bit_exact(self):
        from duet.benchmark.metrics import (
            CodebookEvalResult,
            _evaluate_codebooks_over_union,
        )
        from duet.evaluator_config import EvaluatorConfig, create_evaluator

        seqs = ["0011", "0101", "0110", "1001", "1010", "1100"]
        cfg = EvaluatorConfig.default_symmetric(epsilon=0.1, num_samples=500, seed=42)

        ev = create_evaluator(seqs, cfg, alphabet_size=2)
        ev.initialize_cache(n_jobs=1)
        acc_std = ev.get_codeword_accuracy()
        err_std = ev.get_error_correction_metrics()

        results = _evaluate_codebooks_over_union(
            seqs, [np.arange(len(seqs))],
            eval_config=cfg, alphabet_size=2, n_jobs=1,
        )
        assert len(results) == 1
        res = results[0]
        assert isinstance(res, CodebookEvalResult)
        np.testing.assert_array_equal(res.codeword_accuracy, acc_std)
        np.testing.assert_array_equal(res.error_metrics.codeword_no_error, err_std.codeword_no_error)
        np.testing.assert_array_equal(res.error_metrics.codeword_corrected, err_std.codeword_corrected)
        np.testing.assert_array_equal(res.error_metrics.codeword_failed, err_std.codeword_failed)
        assert res.mean_accuracy == pytest.approx(float(acc_std.mean()))

    def test_core_empty_returns_empty(self):
        from duet.benchmark.metrics import _evaluate_codebooks_over_union
        from duet.evaluator_config import EvaluatorConfig
        cfg = EvaluatorConfig.default_symmetric(epsilon=0.1, num_samples=10, seed=42)
        assert _evaluate_codebooks_over_union([], [], eval_config=cfg, alphabet_size=2) == []


class TestEvaluateSolutionsDelegation:
    def test_return_shape_and_scores_unchanged(self):
        from duet.benchmark.metrics import evaluate_solutions

        candidates, solutions, eval_config, alphabet_size = _make_synthetic_pool_and_solutions()
        out = evaluate_solutions(solutions, candidates, eval_config, alphabet_size, n_jobs=1)

        assert set(out.keys()) == set(solutions.keys())
        acc, sc, err = out["sol_A"]                  # exactly a 3-tuple, in this order
        assert acc.shape == (len(solutions["sol_A"]),)
        np.testing.assert_array_equal(sc, candidates.scores[solutions["sol_A"]])  # scores post-core
        assert hasattr(err, "codeword_failed")
