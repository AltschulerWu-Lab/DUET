"""Unit tests for the baseline-summary visualizer."""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def _make_minimal_enum_parquet(tmp_path: Path) -> Path:
    """Create a tiny enumeration-shape parquet inside the given tmp_path."""
    rng = np.random.default_rng(0)
    rows = []
    for trial in range(3):
        for noise_channel in ["symmetric"]:
            for cb_idx in range(10):
                acc = rng.uniform(0.4, 0.9)
                # All five objectives are constructed POSITIVELY correlated
                # with accuracy — small additive noise gives a strong but
                # imperfect rank correlation in each. This matches the
                # actual semantics: `compute_duet_objective` returns
                # `1 - error_bound` (max-direction), the NLL/Hamming
                # baselines are max-direction confusability scores.
                rows.append({
                    "trial": trial + 1,
                    "seed": 42 + trial,
                    "noise_channel": noise_channel,
                    "error_rate": 0.2,
                    "codebook_index": cb_idx,
                    "codebook_members": "0,1",
                    "decode_accuracy": acc,
                    "duet_objective": acc + rng.uniform(-0.01, 0.01),
                    "min_pairwise_hamming": int(round(acc * 8 + rng.uniform(-0.5, 0.5))),
                    "mean_pairwise_hamming": acc * 4 + rng.uniform(-0.1, 0.1),
                    "mean_pairwise_nll": acc + rng.uniform(-0.05, 0.05),
                    "min_pairwise_nll": acc + rng.uniform(-0.05, 0.05),
                })
    df = pd.DataFrame(rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "objective_correlation.parquet"
    df.to_parquet(p, index=False)
    return p


def test_spearman_extraction_preserves_raw_sign(tmp_path):
    """All five objectives are max-direction, so the visualizer does
    NOT sign-flip the raw Spearman. With a fixture where every
    objective is positively correlated with accuracy, every row in
    the output frame has positive Spearman."""
    from duet.benchmark.baseline_objectives import OBJECTIVES
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        compute_per_trial_spearman,
    )

    p = _make_minimal_enum_parquet(tmp_path)
    df = pd.read_parquet(p)
    spear = compute_per_trial_spearman(df)

    # The fixture builds every objective positively correlated with
    # accuracy. With all-max-direction objectives and no sign flip, every
    # row in the output frame must have positive Spearman.
    for col, _label, _direction in OBJECTIVES:
        rows = spear[spear["objective"] == col]
        assert (rows["spearman"] > 0).all(), (
            f"{col}: expected positive spearman, got "
            f"{rows['spearman'].tolist()}"
        )


def test_spearman_figure_yticks_are_method_labels(tmp_path):
    """The y-tick labels on the rendered Spearman figure are the
    spelled-out objective names from OBJECTIVES — not numeric tick
    positions. This regressed once because matplotlib's auto-formatter
    overrode `set_yticklabels` at draw time; the fix is to call
    `set_yticks(positions, labels=...)` atomically."""
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        render_spearman_figure,
    )
    from duet.benchmark.baseline_objectives import OBJECTIVES

    p = _make_minimal_enum_parquet(tmp_path)
    df = pd.read_parquet(p)

    figdir = tmp_path / "figures"
    figdir.mkdir()
    fig_path = figdir / "summary_spearman.svg"
    fig = render_spearman_figure(df, fig_path)

    ax0 = fig.axes[0]
    label_texts = [t.get_text() for t in ax0.get_yticklabels()]
    expected_labels = [label for _col, label, _direction in OBJECTIVES]
    assert label_texts == expected_labels, (
        f"y-tick labels mismatch:\n  got={label_texts}\n  expected={expected_labels}"
    )

    # No (−ρ) suffix on any label (all five objectives are max-direction).
    for lbl in label_texts:
        assert "(−ρ)" not in lbl, f"unexpected (−ρ) suffix in {lbl!r}"

    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_argmax_regret_direction_and_arithmetic():
    """Construct a fixture where each objective's argmax/argmin codebook
    is different from the decode-accuracy-optimal codebook, with
    hand-known regret values."""
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        compute_argmax_regret,
    )

    # 1 trial, 1 noise channel, 6 codebooks. Build accuracy so the
    # optimum is at index 5, and arrange each objective's argmax to
    # land on a different index with a known regret.
    rows = []
    accuracies = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    # duet_objective (argmax): max at index 3, regret = 1.00 - 0.80 = 0.20.
    duet_obj = [0.10, 0.20, 0.30, 0.95, 0.40, 0.50]
    # mean_pairwise_nll: argmax at index 1, regret = 1.00 - 0.60 = 0.40.
    mean_nll = [0.2, 1.0, 0.3, 0.4, 0.5, 0.6]
    # min_pairwise_nll: argmax at index 2, regret = 1.00 - 0.70 = 0.30.
    min_nll = [0.1, 0.2, 1.0, 0.3, 0.4, 0.5]
    # mean_pairwise_hamming: argmax at index 4, regret = 1.00 - 0.90 = 0.10.
    mean_h = [1.0, 1.5, 2.0, 2.5, 3.0, 2.8]
    # min_pairwise_hamming: argmax at index 0, regret = 1.00 - 0.50 = 0.50.
    min_h = [4, 3, 2, 1, 0, 2]

    for i in range(6):
        rows.append({
            "trial": 1, "seed": 42, "noise_channel": "symmetric", "error_rate": 0.2,
            "codebook_index": i, "codebook_members": str(i),
            "decode_accuracy": accuracies[i],
            "duet_objective": duet_obj[i],
            "mean_pairwise_nll": mean_nll[i],
            "min_pairwise_nll": min_nll[i],
            "mean_pairwise_hamming": mean_h[i],
            "min_pairwise_hamming": min_h[i],
        })
    df = pd.DataFrame(rows)

    regret = compute_argmax_regret(df)
    regret_lookup = {
        (row["objective"], row["trial"], row["noise_channel"]): row
        for _, row in regret.iterrows()
    }

    cases = [
        ("duet_objective",        0.20),
        ("mean_pairwise_nll",     0.40),
        ("min_pairwise_nll",      0.30),
        ("mean_pairwise_hamming", 0.10),
        ("min_pairwise_hamming", 0.50),
    ]
    for obj, expected in cases:
        row = regret_lookup[(obj, 1, "symmetric")]
        assert row["regret"] == pytest.approx(expected, abs=1e-9), (
            f"{obj}: expected regret {expected}, got {row['regret']}"
        )
        assert row["n_tied"] == 1


def test_argmax_regret_tie_mean():
    """When an objective has multiple argmax codebooks, regret is the
    mean of the tied codebooks' individual regrets, and n_tied reflects
    the multiplicity."""
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        compute_argmax_regret,
    )

    # 1 trial. Three codebooks tied at min_pairwise_hamming = 4 (the
    # maximum); their accuracies are 0.8, 0.6, 0.4 → individual regrets
    # are 0.0, 0.2, 0.4 vs the optimum at 0.8. Mean = 0.2.
    rows = [
        {"trial": 1, "seed": 42, "noise_channel": "symmetric", "error_rate": 0.2,
         "codebook_index": 0, "codebook_members": "0",
         "decode_accuracy": 0.8, "duet_objective": 0.1,
         "mean_pairwise_nll": 0.5, "min_pairwise_nll": 0.2,
         "mean_pairwise_hamming": 3.0, "min_pairwise_hamming": 4},
        {"trial": 1, "seed": 42, "noise_channel": "symmetric", "error_rate": 0.2,
         "codebook_index": 1, "codebook_members": "1",
         "decode_accuracy": 0.6, "duet_objective": 0.2,
         "mean_pairwise_nll": 0.4, "min_pairwise_nll": 0.1,
         "mean_pairwise_hamming": 2.5, "min_pairwise_hamming": 4},
        {"trial": 1, "seed": 42, "noise_channel": "symmetric", "error_rate": 0.2,
         "codebook_index": 2, "codebook_members": "2",
         "decode_accuracy": 0.4, "duet_objective": 0.3,
         "mean_pairwise_nll": 0.3, "min_pairwise_nll": 0.05,
         "mean_pairwise_hamming": 2.0, "min_pairwise_hamming": 4},
        {"trial": 1, "seed": 42, "noise_channel": "symmetric", "error_rate": 0.2,
         "codebook_index": 3, "codebook_members": "3",
         "decode_accuracy": 0.5, "duet_objective": 0.4,
         "mean_pairwise_nll": 0.2, "min_pairwise_nll": 0.01,
         "mean_pairwise_hamming": 1.5, "min_pairwise_hamming": 2},
    ]
    df = pd.DataFrame(rows)

    regret = compute_argmax_regret(df)
    minh_row = regret[regret["objective"] == "min_pairwise_hamming"].iloc[0]
    expected_mean = (0.0 + 0.2 + 0.4) / 3.0
    assert minh_row["regret"] == pytest.approx(expected_mean)
    assert minh_row["n_tied"] == 3


def test_regret_support_detection_by_column_name(tmp_path):
    """Enumeration parquet (has `codebook_index`) → regret figure
    emitted. Sampled parquet (has `sample_index`) → only the Spearman
    figure."""
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        main as viz_main,
    )

    # Make an enumeration-shape parquet inside enum_dir.
    enum_dir = tmp_path / "enum"
    _make_minimal_enum_parquet(enum_dir)

    # Run the visualizer.
    import sys
    argv_backup = sys.argv
    sys.argv = ["prog", "--indir", str(enum_dir)]
    try:
        viz_main()
    finally:
        sys.argv = argv_backup

    assert (enum_dir / "figures" / "summary_spearman.svg").exists()
    assert (enum_dir / "figures" / "summary_regret.svg").exists()

    # Now a sampled-shape parquet.
    sampled_dir = tmp_path / "sampled"
    sampled_dir.mkdir()
    df = pd.read_parquet(enum_dir / "objective_correlation.parquet")
    df = df.rename(columns={"codebook_index": "sample_index"})
    df.to_parquet(sampled_dir / "objective_correlation_sampled.parquet", index=False)

    sys.argv = ["prog", "--indir", str(sampled_dir)]
    try:
        viz_main()
    finally:
        sys.argv = argv_backup

    assert (sampled_dir / "figures" / "summary_spearman.svg").exists()
    assert not (sampled_dir / "figures" / "summary_regret.svg").exists()


# log((q-1)(1-eps)/eps) at q=2, eps=0.2 — the symmetric channel's NLL weight,
# so mean_pairwise_nll is exactly _WEIGHT * mean_pairwise_hamming.
_WEIGHT = np.log(4.0)


def _pseudo_tie_rows(n_codebooks: int, seed: int) -> list:
    """Build enumeration-shaped rows that reproduce the float pseudo-tie artifact.

    Mirrors the real pipeline: each codebook has `C(5, 2) = 10` integer pairwise
    distances; `mean_pairwise_hamming` is their mean and `mean_pairwise_nll` is
    the mean of the same values scaled by the symmetric channel's weight. Two
    codebooks whose distances sum equally have a bit-identical Hamming mean (one
    integer sum, one division) but can differ in the last bits of the NLL mean,
    because that sums ten distinct floats whose values depend on the multiset,
    not just its total. This is exactly the 12-tie-groups-into-24-values
    splitting seen in the shipped figure.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cb_idx in range(n_codebooks):
        pair_distances = rng.integers(1, 9, size=10).astype(float)
        mean_ham = float(pair_distances.mean())
        mean_nll = float((_WEIGHT * pair_distances).mean())
        rows.append({
            "trial": 1,
            "seed": 42,
            "noise_channel": "symmetric",
            "error_rate": 0.2,
            "codebook_index": cb_idx,
            "codebook_members": "0,1",
            # Accuracy correlates strongly but imperfectly with distance, so the
            # Spearman is non-degenerate and sensitive to how ties are ranked.
            "decode_accuracy": 0.5 + mean_ham * 0.05 + float(rng.uniform(-0.01, 0.01)),
            "duet_objective": 0.5 + mean_ham * 0.04,
            "min_pairwise_hamming": int(pair_distances.min()),
            "mean_pairwise_hamming": mean_ham,
            "mean_pairwise_nll": mean_nll,
            "min_pairwise_nll": float(_WEIGHT * pair_distances.min()),
        })
    return rows


def test_pseudo_ties_do_not_split_spearman():
    """Two objectives equal up to float summation error must give identical rho.

    On the symmetric channel `mean_pairwise_nll` is exactly
    `log(4) * mean_pairwise_hamming`, so their Spearman correlations are
    mathematically forced to be equal. Unrounded they are not, because
    `spearmanr` assigns distinct ranks to values that should have been tied and
    rank-averaged — the artifact worth up to 0.018 in rho in the shipped figure.
    """
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        compute_per_trial_spearman,
    )

    df = pd.DataFrame(_pseudo_tie_rows(300, seed=0))

    # Guard the fixture itself: if these counts ever match, the construction has
    # stopped reproducing the artifact and the test below passes vacuously.
    assert (
        df["mean_pairwise_nll"].nunique() > df["mean_pairwise_hamming"].nunique()
    ), "fixture no longer produces pseudo-ties"

    spear = compute_per_trial_spearman(df)
    rho_ham = spear[spear["objective"] == "mean_pairwise_hamming"]["spearman"].iloc[0]
    rho_nll = spear[spear["objective"] == "mean_pairwise_nll"]["spearman"].iloc[0]

    # Bit-exact, not approximate: they are the same objective up to a scale.
    assert rho_nll == rho_ham


def test_pseudo_ties_do_not_split_argmax_ties():
    """An objective's argmax tie set must not be split by float summation error.

    `compute_argmax_regret` selects the argmax with an exact `==`, so an
    unrounded column reports a smaller `n_tied` than the true tie group and
    returns one arbitrary member's regret instead of the mean over ties.
    """
    from scripts.benchmark.synthetic.visualize_baseline_summary import (
        compute_argmax_regret,
    )

    # Three codebooks whose pair distances sum to 35 via different multisets:
    # identical means mathematically, last-bit-distinct once scaled and averaged.
    multisets = [
        [2, 3, 4, 5, 2, 3, 4, 5, 3, 4],
        [4, 4, 4, 3, 3, 3, 5, 5, 2, 2],
        [5, 5, 5, 5, 5, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # sums to 10 — strictly below the tie group
    ]
    accuracies = [0.9, 0.7, 0.7, 0.4]

    rows = []
    for cb_idx, (members, acc) in enumerate(zip(multisets, accuracies)):
        pair_distances = np.array(members, dtype=float)
        rows.append({
            "trial": 1,
            "seed": 42,
            "noise_channel": "symmetric",
            "error_rate": 0.2,
            "codebook_index": cb_idx,
            "codebook_members": "0,1",
            "decode_accuracy": acc,
            "duet_objective": 0.5,
            "min_pairwise_hamming": int(pair_distances.min()),
            "mean_pairwise_hamming": float(pair_distances.mean()),
            "mean_pairwise_nll": float((_WEIGHT * pair_distances).mean()),
            "min_pairwise_nll": float(_WEIGHT * pair_distances.min()),
        })
    df = pd.DataFrame(rows)

    # Guard the fixture: the three tie-group members must be last-bit-distinct.
    top3 = df["mean_pairwise_nll"].to_numpy()[:3]
    assert np.unique(top3).size > 1, "fixture no longer produces pseudo-ties"
    assert np.allclose(top3, top3[0], rtol=0, atol=1e-12)

    regret = compute_argmax_regret(df)
    row = regret[regret["objective"] == "mean_pairwise_nll"].iloc[0]

    # All three top codebooks are tied; regret is their mean, not one member's.
    assert int(row["n_tied"]) == 3
    assert row["regret"] == pytest.approx((0.0 + 0.2 + 0.2) / 3)


def test_cell_spearman_pseudo_ties_bit_identical():
    """`_cell_spearman` (the per-cell Spearman reducer behind the shipped
    Panel 2A per-trial scatter, `scripts.benchmark.synthetic.
    visualize_duet_vs_baseline_objectives`) must not let the float
    pseudo-tie artifact split two objectives that are mathematically equal
    up to a scale — the same property `test_pseudo_ties_do_not_split_spearman`
    above pins for `compute_per_trial_spearman`.

    This is the one `round_for_ranking` call site
    (`visualize_duet_vs_baseline_objectives.py:69`) with no prior regression
    test, so it is exercised directly here rather than through
    `compute_per_trial_spearman`.
    """
    from scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives import (
        _cell_spearman,
    )

    df = pd.DataFrame(_pseudo_tie_rows(300, seed=0))

    # Guard the fixture itself: if these counts ever match, the construction
    # has stopped reproducing the artifact and the assertion below passes
    # vacuously.
    assert (
        df["mean_pairwise_nll"].nunique() > df["mean_pairwise_hamming"].nunique()
    ), "fixture no longer produces pseudo-ties"

    rho_ham = _cell_spearman(df, "mean_pairwise_hamming", trial=1, noise_key="symmetric")
    rho_nll = _cell_spearman(df, "mean_pairwise_nll", trial=1, noise_key="symmetric")

    # Bit-exact, not approximate: on the symmetric channel these are the same
    # objective up to a scale (mean_pairwise_nll == log(4) * mean_pairwise_hamming).
    assert rho_nll == rho_ham
