"""Unit tests for src/duet/benchmark/baseline_objectives.py."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import numpy as np
import pandas as pd
import pytest


def test_objectives_constant_has_five_entries():
    """OBJECTIVES is the single source of truth for the five baseline-objective
    columns. Order and shape must match the rows of the manuscript figures."""
    from duet.benchmark.baseline_objectives import OBJECTIVES

    assert len(OBJECTIVES) == 5
    column_names = [col for col, _, _ in OBJECTIVES]
    assert column_names == [
        "duet_objective",
        "mean_pairwise_nll",
        "min_pairwise_nll",
        "mean_pairwise_hamming",
        "min_pairwise_hamming",
    ]


def test_objectives_direction_field():
    """All five objectives are max-direction: `compute_duet_objective`
    returns `1 - error_bound` (a survival bound, larger = better), and
    the four baselines are larger-is-better confusability scores. The
    boxplot regret extractor depends on this map being correct."""
    from duet.benchmark.baseline_objectives import OBJECTIVES

    expected_direction = {
        "duet_objective": "max",
        "mean_pairwise_nll": "max",
        "min_pairwise_nll": "max",
        "mean_pairwise_hamming": "max",
        "min_pairwise_hamming": "max",
    }
    actual_direction = {col: direction for col, _, direction in OBJECTIVES}
    assert actual_direction == expected_direction


def test_sample_codebooks_shape():
    """Output is a list of `num_codebooks` arrays, each of length
    sum(quotas)."""
    from duet.benchmark.baseline_objectives import sample_codebooks

    rng = np.random.default_rng(0)
    group_to_candidates = {0: np.array([0, 1, 2, 3, 4, 5, 6, 7])}
    quotas = {0: 3}

    sampled = sample_codebooks(group_to_candidates, quotas, num_codebooks=5, rng=rng)

    assert len(sampled) == 5
    for cb in sampled:
        assert cb.shape == (3,)
        assert cb.dtype == np.int64 or cb.dtype == np.int32
        assert set(cb.tolist()).issubset(set(range(8)))
        assert len(set(cb.tolist())) == 3  # no duplicates within a codebook


def test_sample_codebooks_reproducibility():
    """Same rng seed → same sampled codebooks."""
    from duet.benchmark.baseline_objectives import sample_codebooks

    group_to_candidates = {0: np.arange(8)}
    quotas = {0: 3}

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    sampled_a = sample_codebooks(group_to_candidates, quotas, num_codebooks=10, rng=rng_a)
    sampled_b = sample_codebooks(group_to_candidates, quotas, num_codebooks=10, rng=rng_b)

    assert len(sampled_a) == len(sampled_b)
    for cb_a, cb_b in zip(sampled_a, sampled_b):
        np.testing.assert_array_equal(cb_a, cb_b)


def test_sample_codebooks_coverage():
    """Every candidate appears in at least one sampled codebook over a
    moderate draw count. Catches off-by-one errors that skip candidate 0
    or the last candidate."""
    from duet.benchmark.baseline_objectives import sample_codebooks

    rng = np.random.default_rng(7)
    group_to_candidates = {0: np.arange(8)}
    quotas = {0: 3}

    sampled = sample_codebooks(group_to_candidates, quotas, num_codebooks=200, rng=rng)
    appeared = set()
    for cb in sampled:
        appeared.update(cb.tolist())
    assert appeared == set(range(8))


def test_sample_codebooks_uniformity():
    """Each candidate appears with frequency within 10% of expected.
    Catches off-by-one errors that double-count one candidate."""
    from duet.benchmark.baseline_objectives import sample_codebooks

    rng = np.random.default_rng(11)
    n_candidates = 8
    quota = 3
    n_draws = 10000
    group_to_candidates = {0: np.arange(n_candidates)}
    quotas = {0: quota}

    sampled = sample_codebooks(
        group_to_candidates, quotas, num_codebooks=n_draws, rng=rng
    )
    counts = np.zeros(n_candidates, dtype=np.int64)
    for cb in sampled:
        for c in cb:
            counts[int(c)] += 1

    # Each candidate is drawn (quota / n_candidates) of the time per draw,
    # so expected count = n_draws * quota / n_candidates.
    expected = n_draws * quota / n_candidates
    for c, observed in enumerate(counts):
        ratio = observed / expected
        assert abs(ratio - 1.0) < 0.10, (
            f"Candidate {c}: observed {observed}, expected {expected:.0f}, "
            f"ratio {ratio:.3f} (must be within 0.9–1.1)"
        )


def test_sample_codebooks_warns_on_duplicates(caplog):
    """When duplicate codebooks appear in the sampled set, the runner emits
    a warning: this is a useful diagnostic that num_codebooks is too large
    for the candidate-pool size (the sampling becomes near-enumeration)."""
    import logging
    from duet.benchmark.baseline_objectives import sample_codebooks

    rng = np.random.default_rng(0)
    # 1 group of 3 candidates, quota 2 → only C(3, 2) = 3 distinct codebooks.
    # Sampling 50 draws guarantees duplicates.
    group_to_candidates = {0: np.arange(3)}
    quotas = {0: 2}

    with caplog.at_level(logging.WARNING, logger="duet.benchmark.baseline_objectives"):
        sample_codebooks(group_to_candidates, quotas, num_codebooks=50, rng=rng)

    duplicate_warnings = [
        record for record in caplog.records
        if "duplicate" in record.message.lower()
    ]
    assert len(duplicate_warnings) >= 1


def test_seed_sequence_spawn_produces_independent_streams():
    """Sanity check: SeedSequence(seed).spawn(2) yields two seeds that
    produce different streams from `np.random.default_rng`. This is the
    property the sampled runner relies on to keep codebook sampling
    independent of evaluator MC sampling."""
    seed = 42
    s0, s1 = np.random.SeedSequence(seed).spawn(2)

    draws_a = np.random.default_rng(s0).standard_normal(8)
    draws_b = np.random.default_rng(s1).standard_normal(8)

    assert not np.allclose(draws_a, draws_b), (
        "spawn(2)[0] and spawn(2)[1] produced identical streams — "
        "isolation is broken."
    )

    # Reproducibility: the spawn outputs must be deterministic in seed.
    s0_again, s1_again = np.random.SeedSequence(seed).spawn(2)
    np.testing.assert_array_equal(
        np.random.default_rng(s0).standard_normal(8),
        np.random.default_rng(s0_again).standard_normal(8),
    )
    np.testing.assert_array_equal(
        np.random.default_rng(s1).standard_normal(8),
        np.random.default_rng(s1_again).standard_normal(8),
    )


def test_sampled_runner_smoke(tmp_path):
    """End-to-end smoke: tiny sampled run completes, writes the expected
    parquet, and the parquet has `sample_index` (not `codebook_index`)."""
    from duet.benchmark.baseline_objectives import (
        EvaluatorRunConfig,
        PoolConfig,
        SampledBaselineObjectivesConfig,
    )
    from scripts.benchmark.synthetic.run_duet_vs_baseline_objectives_sampled import (
        run_sampled_baseline_objectives,
    )

    config = SampledBaselineObjectivesConfig(
        outdir=str(tmp_path / "out"),
        seed=42,
        trials=2,
        pool=PoolConfig(
            num_groups=1, candidates_per_group=4,
            seq_length=4, alphabet_size=2, quota=2,
        ),
        evaluator=EvaluatorRunConfig(num_samples=50, num_cpus=1),
        noise_channels=["symmetric"],
        error_rate=0.2,
        num_codebooks=5,
    )

    df = run_sampled_baseline_objectives(config)
    assert "sample_index" in df.columns
    assert "codebook_index" not in df.columns
    # 2 trials × 1 noise × 5 sampled = 10 rows.
    assert len(df) == 10

    # The partial file should have been cleaned up after the final write.
    assert not (tmp_path / "out" / "objective_correlation_sampled.partial.parquet").exists()
    assert (tmp_path / "out" / "objective_correlation_sampled.parquet").exists()


def test_sampled_runner_reproducibility(tmp_path):
    """Same trial seed → same codebooks and same accuracies across two
    independent runs."""
    from duet.benchmark.baseline_objectives import (
        EvaluatorRunConfig,
        PoolConfig,
        SampledBaselineObjectivesConfig,
    )
    from scripts.benchmark.synthetic.run_duet_vs_baseline_objectives_sampled import (
        run_sampled_baseline_objectives,
    )

    base = SampledBaselineObjectivesConfig(
        outdir="",
        seed=42,
        trials=1,
        pool=PoolConfig(
            num_groups=1, candidates_per_group=4,
            seq_length=4, alphabet_size=2, quota=2,
        ),
        evaluator=EvaluatorRunConfig(num_samples=50, num_cpus=1),
        noise_channels=["symmetric"],
        error_rate=0.2,
        num_codebooks=5,
    )

    from dataclasses import replace
    cfg_a = replace(base, outdir=str(tmp_path / "a"))
    cfg_b = replace(base, outdir=str(tmp_path / "b"))

    df_a = run_sampled_baseline_objectives(cfg_a)
    df_b = run_sampled_baseline_objectives(cfg_b)

    # Sort to make order-independent comparison.
    sort_keys = ["sample_index"]
    df_a = df_a.sort_values(sort_keys).reset_index(drop=True)
    df_b = df_b.sort_values(sort_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_a, df_b)
