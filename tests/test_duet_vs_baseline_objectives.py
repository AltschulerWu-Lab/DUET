"""Tests for the DUET-vs-baseline-objectives runner and visualizer."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def test_pair_values_extracts_distinct_unordered_pairs():
    """`_pair_values` returns values of (i, j) pairs with i < j only — no
    diagonal, no double-counting. Verified against a hand-built symmetric
    matrix and a known 3-member codebook."""
    from duet.benchmark.baseline_objectives import (
        _pair_values,
    )

    # 4x4 symmetric matrix with distinct off-diagonal values.
    matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 4, 5],
        [2, 4, 0, 6],
        [3, 5, 6, 0],
    ], dtype=float)
    codebook = np.array([0, 1, 3])  # picks rows/cols 0, 1, 3

    # Distinct unordered pairs in the submatrix:
    #   (0,1) -> matrix[0,1] = 1
    #   (0,3) -> matrix[0,3] = 3
    #   (1,3) -> matrix[1,3] = 5
    result = _pair_values(matrix, codebook)
    assert sorted(result.tolist()) == [1.0, 3.0, 5.0]


def test_hamming_reductions_match_analytic():
    """Min and mean of `_pair_values` on a hand-built pairwise Hamming
    matrix match analytic values for a known 3-member codebook."""
    from duet.benchmark.baseline_objectives import (
        _pair_values,
    )
    from duet.codebook_evaluator import HammingDistance

    # 4 sequences of length 4, alphabet {0, 1}.
    sequences = np.array([
        [0, 0, 0, 0],  # s0
        [1, 1, 0, 0],  # s1 — differs from s0 at positions 0, 1
        [0, 1, 1, 0],  # s2 — differs from s0 at positions 1, 2
        [1, 1, 1, 1],  # s3 — differs from s0 at all 4 positions
    ])
    full = HammingDistance().compute(sequences, sequences)
    # Sanity: HammingDistance returns the full pairwise matrix.
    assert full.shape == (4, 4)
    assert full[0, 3] == 4

    codebook = np.array([0, 1, 3])
    pair_vals = _pair_values(full, codebook)
    # Pairs: d(s0,s1)=2, d(s0,s3)=4, d(s1,s3)=2 -> {2, 4, 2}
    assert sorted(pair_vals.tolist()) == [2, 2, 4]
    assert pair_vals.min() == 2
    assert pair_vals.mean() == pytest.approx(8 / 3)


def test_compute_pairwise_nll_is_centered_and_direction_averaged():
    """compute_pairwise_nll subtracts the row diagonal, then averages the two
    directions.

    The stub matrix has a NON-ZERO diagonal, which is what discriminates the
    centered form from the uncentered one: for `AsymmetricNLL` the raw metric
    has d(x, x) > 0, so a symmetrization alone leaves a composition-dependent
    offset that is not a distance.

    Under both the mean and the min reduction, the target is distinct from all
    three rejected alternatives.
    """
    from duet.benchmark.baseline_objectives import (
        _pair_values,
        compute_pairwise_nll,
    )

    # D has a non-zero, non-constant diagonal [1, 2, 3].
    #
    #   D            D.T          row-centered D'   S = (D' + D'.T)/2
    #   [1 2 6]      [1 3 5]      [0 1 5]           [0   1   3.5]
    #   [3 2 3]      [2 2 4]      [1 0 1]           [1   0   1  ]
    #   [5 4 3]      [6 3 3]      [2 1 0]           [3.5 1   0  ]
    #
    # Upper-triangle pairs over codebook [0, 1, 2] and their reductions:
    #   target  S = (D'+D'.T)/2 : [1.0, 3.5, 1.0]  -> mean 5.5/3, min 1.0
    #   raw D                   : [2.0, 6.0, 3.0]  -> mean 11/3,  min 2.0
    #   min(D, D.T) uncentered  : [2.0, 5.0, 3.0]  -> mean 10/3,  min 2.0
    #   (D + D.T)/2 uncentered  : [2.5, 5.5, 3.5]  -> mean 11.5/3, min 2.5
    D = np.array([
        [1.0, 2.0, 6.0],
        [3.0, 2.0, 3.0],
        [5.0, 4.0, 3.0],
    ])

    class StubDecodingMetric:
        def compute(self, a, b):
            return D

    sequences = np.zeros((3, 4), dtype=int)  # shape only matters downstream
    nll_sym = compute_pairwise_nll(StubDecodingMetric(), sequences)

    expected = np.array([
        [0.0, 1.0, 3.5],
        [1.0, 0.0, 1.0],
        [3.5, 1.0, 0.0],
    ])
    assert nll_sym.shape == (3, 3)
    assert np.allclose(nll_sym, expected)

    # A distance: symmetric, and zero where the codewords agree.
    assert np.allclose(nll_sym, nll_sym.T)
    assert np.allclose(np.diag(nll_sym), 0.0)

    codebook = np.array([0, 1, 2])
    pair_vals = _pair_values(nll_sym, codebook)

    # Mean reduction pins the convention against all three alternatives.
    assert pair_vals.mean() == pytest.approx(5.5 / 3)
    assert pair_vals.mean() != pytest.approx(11 / 3)    # not raw D
    assert pair_vals.mean() != pytest.approx(10 / 3)    # not min(D, D.T)
    assert pair_vals.mean() != pytest.approx(11.5 / 3)  # not uncentered average

    # Min reduction pins it independently.
    assert pair_vals.min() == pytest.approx(1.0)
    assert pair_vals.min() != pytest.approx(2.0)  # not raw D, not min(D, D.T)
    assert pair_vals.min() != pytest.approx(2.5)  # not uncentered average


def test_pairwise_nll_is_weighted_hamming_on_binary_channels():
    """On a binary alphabet the centered, direction-averaged NLL is exactly a
    per-position weighted Hamming distance.

    Weights are recovered from the objective itself (w_p = distance from the
    all-zeros codeword to the unit vector at position p) rather than
    re-deriving them from the channel parameters, so the test cannot pass
    vacuously by duplicating the implementation. Two hand-derived numeric
    anchors pin the recovery.

    Consequence, asserted below: the symmetric and asymmetric channels give
    uniform weights, so the objective is Hamming distance rescaled and cannot
    rank codebooks differently. Only the two position-varying channels can.
    """
    import itertools

    from duet.benchmark.baseline_objectives import (
        _create_noise_and_decoding_metric,
        compute_pairwise_nll,
    )
    from duet.codebook_evaluator import HammingDistance

    L = 8
    Q = 2
    ERROR_RATE = 0.2

    # All 256 binary sequences of length 8 — the full space, so the identity is
    # checked at every Hamming distance and every 0/1 composition.
    sequences = np.array(list(itertools.product([0, 1], repeat=L)), dtype=int)
    hamming = HammingDistance().compute(sequences, sequences)

    zeros_idx = 0                      # (0, 0, ..., 0) is first in product order
    unit_idx = [                       # index of the sequence with a single 1 at p
        int(np.flatnonzero((sequences.sum(axis=1) == 1) & (sequences[:, p] == 1))[0])
        for p in range(L)
    ]

    # `rtol=0` throughout: np.allclose's default rtol=1e-5 would dominate atol
    # at these magnitudes (values reach ~18.7 nats, so the effective tolerance
    # would be ~1.9e-4, not 1e-12). The true additivity error is ~5e-15.
    S_by_channel = {}
    weights_by_channel = {}
    for noise_type in [
        "symmetric",
        "position_varying",
        "asymmetric",
        "position_varying_asymmetric",
    ]:
        _noise, decoding_metric = _create_noise_and_decoding_metric(
            noise_type, ERROR_RATE, L, Q,
        )
        S = compute_pairwise_nll(decoding_metric, sequences)

        # Recover the per-position weight, then rebuild the whole matrix from it.
        w = np.array([S[zeros_idx, unit_idx[p]] for p in range(L)])
        mismatch = sequences[:, None, :] != sequences[None, :, :]   # (256, 256, 8)
        rebuilt = mismatch @ w

        assert np.allclose(S, rebuilt, rtol=0, atol=1e-12), (
            f"{noise_type}: objective is not additive over positions"
        )
        S_by_channel[noise_type] = S
        weights_by_channel[noise_type] = w

    # Uniform weights <=> Hamming distance rescaled <=> identical rankings.
    for noise_type in ["symmetric", "asymmetric"]:
        w = weights_by_channel[noise_type]
        assert np.allclose(w, w[0], rtol=0, atol=1e-12), (
            f"{noise_type}: weights should be uniform"
        )
        assert np.allclose(
            S_by_channel[noise_type], w[0] * hamming, rtol=0, atol=1e-12
        )

    # Non-uniform, and monotonically decreasing, wherever the channel varies by
    # position (both use a ramp from 0.5x to 1.5x across positions).
    for noise_type in ["position_varying", "position_varying_asymmetric"]:
        w = weights_by_channel[noise_type]
        assert not np.allclose(w, w[0], rtol=0, atol=1e-12), (
            f"{noise_type}: weights should vary by position"
        )
        assert np.all(np.diff(w) < 0), f"{noise_type}: weights should decrease"

    # Hand-derived anchors, from the channel definitions at error_rate = 0.2:
    #   symmetric:  e0 = e1 = 0.2       -> w = 0.5 * 2 * log(0.8 / 0.2) = log 4
    #   asymmetric: e0 = 0.1, e1 = 0.3  -> w = 0.5 * [log(0.7 / 0.1) + log(0.9 / 0.3)]
    #                                        = 0.5 * log 21
    assert weights_by_channel["symmetric"][0] == pytest.approx(np.log(4.0), abs=1e-12)
    assert weights_by_channel["asymmetric"][0] == pytest.approx(
        0.5 * np.log(21.0), abs=1e-12
    )

    # The position-varying channels get endpoint anchors too: they are separate
    # implementations from the two above, and they are the only channels whose
    # weights vary — i.e. the only ones where this objective can rank codebooks
    # differently from Hamming distance. Structural checks alone (non-uniform,
    # decreasing) would not catch a scale or shape error confined to them.
    #   position_varying, eps ramps 0.1 -> 0.3, e0 = e1 = eps:
    #       w_0 = log(0.9 / 0.1) = log 9        w_7 = log(0.7 / 0.3) = log(7/3)
    assert weights_by_channel["position_varying"][0] == pytest.approx(
        np.log(9.0), abs=1e-12
    )
    assert weights_by_channel["position_varying"][-1] == pytest.approx(
        np.log(7.0 / 3.0), abs=1e-12
    )
    #   position_varying_asymmetric, eps[p, c] = position_factor[p] * class_factor[c] * x:
    #       p=0 -> e0 = 0.05, e1 = 0.15;  p=7 -> e0 = 0.15, e1 = 0.45
    assert weights_by_channel["position_varying_asymmetric"][0] == pytest.approx(
        0.5 * (np.log(0.85 / 0.05) + np.log(0.95 / 0.15)), abs=1e-12
    )
    assert weights_by_channel["position_varying_asymmetric"][-1] == pytest.approx(
        0.5 * (np.log(0.55 / 0.15) + np.log(0.85 / 0.45)), abs=1e-12
    )


def test_config_from_yaml(tmp_path: Path):
    """BaselineObjectivesConfig.from_yaml parses the experiment config shape."""
    from duet.benchmark.baseline_objectives import (
        BaselineObjectivesConfig,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent("""\
        outdir: results/benchmark/05-19-2026/duet_vs_baseline_objectives
        seed: 42
        trials: 5

        pool:
          num_groups: 1
          candidates_per_group: 15
          seq_length: 8
          alphabet_size: 2
          quota: 5

        evaluator:
          num_samples: 10000
          num_cpus: 20

        noise_channels:
          - symmetric
          - position_varying
          - asymmetric
          - position_varying_asymmetric
        error_rate: 0.2
    """))

    cfg = BaselineObjectivesConfig.from_yaml(str(config_path))

    assert cfg.outdir == "results/benchmark/05-19-2026/duet_vs_baseline_objectives"
    assert cfg.seed == 42
    assert cfg.trials == 5
    assert cfg.pool.candidates_per_group == 15
    assert cfg.pool.quota == 5
    assert cfg.evaluator.num_samples == 10000
    assert cfg.evaluator.num_cpus == 20
    assert cfg.noise_channels == ["symmetric", "position_varying", "asymmetric", "position_varying_asymmetric"]
    assert cfg.error_rate == 0.2


def test_runner_smoke(tmp_path: Path):
    """run_baseline_objectives produces a parquet with the expected schema."""
    from duet.benchmark.baseline_objectives import (
        BaselineObjectivesConfig,
        EvaluatorRunConfig,
        PoolConfig,
    )
    from scripts.benchmark.synthetic.run_duet_vs_baseline_objectives import (
        run_baseline_objectives,
    )
    from duet.benchmark.exhaustive import compute_total_codebooks
    from duet.benchmark.synthetic_pool import create_2d_synthetic_pool

    outdir = tmp_path / "run"
    noise_channels = ["symmetric", "position_varying", "asymmetric", "position_varying_asymmetric"]
    config = BaselineObjectivesConfig(
        outdir=str(outdir),
        seed=7,
        trials=2,
        pool=PoolConfig(
            num_groups=1,
            candidates_per_group=6,
            seq_length=6,
            alphabet_size=2,
            quota=3,
        ),
        evaluator=EvaluatorRunConfig(num_samples=200, num_cpus=1),
        noise_channels=noise_channels,
        error_rate=0.2,
    )

    df = run_baseline_objectives(config)

    pool = create_2d_synthetic_pool(
        num_groups=1, candidates_per_group=6, seq_length=6,
        alphabet_size=2, quota=3, seed=7,
    )
    per_trial_cbs = compute_total_codebooks(pool.group_to_candidates, pool.quotas)
    assert per_trial_cbs == 20
    # 2 trials × 4 noise channels × 20 codebooks = 160 rows.
    assert len(df) == config.trials * len(noise_channels) * per_trial_cbs

    assert set(df.columns) == {
        "trial", "seed", "noise_channel", "error_rate",
        "codebook_index", "codebook_members",
        "decode_accuracy", "duet_objective",
        "min_pairwise_hamming", "mean_pairwise_hamming",
        "mean_pairwise_nll", "min_pairwise_nll",
    }
    assert sorted(df["noise_channel"].unique().tolist()) == sorted(noise_channels)
    assert sorted(df["trial"].unique().tolist()) == [1, 2]
    assert df["decode_accuracy"].between(0.0, 1.0).all()
    # duet_objective: bounded above by 1, can go negative; just require finite.
    assert (df["duet_objective"] <= 1.0).all()
    assert np.isfinite(df["duet_objective"]).all()
    # Hamming reductions: integer min, float mean; both in [0, seq_length].
    assert df["min_pairwise_hamming"].between(0, 6).all()
    assert df["mean_pairwise_hamming"].between(0, 6).all()
    # NLL reductions: non-negative and finite.
    assert (df["mean_pairwise_nll"] >= 0).all()
    assert np.isfinite(df["mean_pairwise_nll"]).all()
    assert (df["min_pairwise_nll"] >= 0).all()
    assert np.isfinite(df["min_pairwise_nll"]).all()
    # min ≤ mean over the same set of upper-tri pairs.
    assert (df["min_pairwise_nll"] <= df["mean_pairwise_nll"] + 1e-12).all()
    assert (outdir / "objective_correlation.parquet").exists()


def test_visualizer_smoke(tmp_path: Path):
    """render_figure writes one trial's SVG with 4 noise-channel column
    titles, 5 objective row labels, and 20 per-cell Spearman annotations
    (one per cell, single trial)."""
    from scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives import (
        render_figure,
    )
    from duet.plotting import apply_style

    rng = np.random.default_rng(2)
    noise_channels = ["symmetric", "position_varying", "asymmetric", "position_varying_asymmetric"]
    rows = []
    # 3 trials × 4 noise channels × 50 codebooks = 600 rows.
    for noise in noise_channels:
        for trial in range(1, 4):
            # Force a non-trivial correlation by linking each objective to
            # decode_accuracy plus noise, so Spearman is well-defined and
            # not exactly 0.
            acc = rng.uniform(0.3, 0.95, size=50)
            for idx in range(50):
                base = acc[idx]
                rows.append({
                    "trial": trial,
                    "noise_channel": noise,
                    "decode_accuracy": float(base),
                    "duet_objective": float(base + rng.normal(0, 0.05)),
                    "mean_pairwise_nll": float(base * 2 + rng.normal(0, 0.1)),
                    "min_pairwise_nll": float(base * 1.5 + rng.normal(0, 0.1)),
                    "mean_pairwise_hamming": float(base * 4 + rng.normal(0, 0.5)),
                    "min_pairwise_hamming": int(round(base * 3 + rng.normal(0, 0.3))),
                })
    df = pd.DataFrame(rows)

    # render_figure now takes a single-trial DataFrame, the trial id, and
    # an explicit out_path. The caller owns directory creation and the
    # prior apply_style() call.
    trial_df = df[df["trial"] == 1]
    out_path = tmp_path / "trial_01.svg"
    apply_style()
    render_figure(trial_df, 1, out_path)

    assert out_path.exists()
    # Guards against fig-saved-but-half-empty bugs.
    assert out_path.stat().st_size >= 10_000

    svg = out_path.read_text()

    # Row labels: DUET objective is single-line; the four others are
    # line-broken before the trailing noun and render as separate
    # <text> elements per line. Check the per-line components.
    expected_label_lines = [
        "DUET objective",
        "Mean negative",
        "Minimum negative",
        "log-likelihood",     # appears in both NLL labels
        "Mean Hamming",
        "Minimum Hamming",
        "distance",           # appears in both Hamming labels
    ]
    for line in expected_label_lines:
        assert line in svg, f"missing row-label line: {line}"

    # Noise column titles (4). Display labels hyphenate "position-varying"
    # and break the fourth title onto two lines
    # ("position_varying_asymmetric" -> "position-varying,\nasymmetric"),
    # which matplotlib renders as one <text> element per line. So the
    # expected per-line strings are counted with multiplicity: "asymmetric"
    # appears twice (third title, and second line of the fourth).
    noise_display_labels = [
        "symmetric",
        "position-varying",
        "asymmetric",
        "position-varying,\nasymmetric",
    ]
    expected_title_lines: dict[str, int] = {}
    for label in noise_display_labels:
        for line in label.split("\n"):
            expected_title_lines[line] = expected_title_lines.get(line, 0) + 1
    assert expected_title_lines == {
        "symmetric": 1, "position-varying": 1, "asymmetric": 2,
        "position-varying,": 1,
    }
    primary_ok = all(
        svg.count(f">{line}<") >= n for line, n in expected_title_lines.items()
    )
    if not primary_ok:
        # Fallback: longest-first removal. The substring bug is still
        # present ("symmetric" in "asymmetric", "position-varying" in
        # "position-varying,"), so remove ALL occurrences of the longer
        # line before checking the shorter one, otherwise leftover
        # `asymmetric` substrings would satisfy a `symmetric in s` check.
        # This branch fires whenever matplotlib emits titles without a
        # bare `>name<` boundary (e.g., tspan-wrapped text, entity
        # escapes, or extra attributes in the tag).
        s = svg
        for line in sorted(expected_title_lines, key=len, reverse=True):
            n = expected_title_lines[line]
            assert s.count(line) >= n, (
                f"missing column title line: {line!r} (expected {n}x)"
            )
            s = s.replace(line, "")

    # X-label (bottom row only).
    assert "Decode accuracy" in svg

    # 5 objective rows × 4 noise columns = 20 cells; each cell gets one
    # `ρ = ...` annotation (single trial, no across-trial summary).
    # Accept either the literal ρ or the entity-encoded form.
    assert svg.count("ρ = ") == 20 or svg.count("&#961; = ") == 20
