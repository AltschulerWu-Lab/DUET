"""OPS parity tests: byte-identical for no-multi-targeting; empirical for multi."""
import numpy as np
import pandas as pd
import pytest

from duet.candidate_pool import CandidatePool
from duet.runner import DuetOptimizerConfig, run_duet_ops
from duet.evaluator_config import EvaluatorConfig


def test_ops_no_multitargeting_dataframe_round_trips_unchanged():
    """A DataFrame with all-unique sequences should produce one candidate per row."""
    df = pd.DataFrame({
        "Group":    ["G1", "G1", "G2", "G2"],
        "Sequence": ["ATCG", "GGCA", "TTAA", "TTAC"],   # all distinct
        "Score":    [0.9, 0.8, 0.95, 0.7],
        "Quota":    [1, 1, 1, 1],
    })
    pool = CandidatePool.from_dataframe(df)
    assert pool.pool_size == 4
    assert pool.group_to_candidates == {"G1": [0, 1], "G2": [2, 3]}
    # Each candidate is in exactly one group
    for c in range(4):
        assert len(pool.candidate_to_groups[c]) == 1


def test_ops_no_multitargeting_pep_dedup_is_noop():
    """With all distinct sequences, U == pool_size and candidate_to_sequence_idx is identity."""
    df = pd.DataFrame({
        "Group":    ["G1", "G1", "G2", "G2"],
        "Sequence": ["ATCG", "GGCA", "TTAA", "TTAC"],
        "Score":    [0.9, 0.8, 0.95, 0.7],
        "Quota":    [1, 1, 1, 1],
    })
    pool = CandidatePool.from_dataframe(df)
    assert len(pool.unique_sequences) == pool.pool_size
    np.testing.assert_array_equal(
        pool.candidate_to_sequence_idx, np.arange(pool.pool_size, dtype=np.int32),
    )


def test_ops_multitargeting_pareto_frontier_within_seed_noise():
    """OPS with multi-targeting sgRNAs: empirical Pareto-frontier sanity check.

    Under the per-row design a sgRNA that targets two genes becomes two
    candidates (one per gene), each with its own activity score, that share a
    single PEP row. This test verifies cross-seed agreement (not vs. a frozen
    baseline)."""
    # 5 genes, 4 sgRNAs each; sgRNAs M0/M1/M2 each target two genes.
    multi = ["ATCGATCG", "GGGCAAAC", "TTACATGA"]
    rng = np.random.default_rng(0)
    rows = []
    for g in range(5):
        sgrnas = ["".join(rng.choice(list("ACGT"), 8)) for _ in range(2)]
        if g < 3:
            sgrnas.append(multi[g])
            sgrnas.append(multi[(g + 1) % len(multi)])
        else:
            sgrnas.extend(["".join(rng.choice(list("ACGT"), 8)) for _ in range(2)])
        for s in sgrnas:
            # Per-row independent scores — distinct scores for a shared sequence
            # are now allowed (the point of the per-row design).
            rows.append({
                "Group": f"gene_{g}", "Sequence": s,
                "Score": float(rng.uniform(0.5, 1.0)), "Quota": 1,
            })
    df = pd.DataFrame(rows)
    candidates = CandidatePool.from_dataframe(df)

    pep_cfg = EvaluatorConfig.from_dict({
        "noise_channel": {"type": "symmetric", "epsilon": 0.1},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 200,
        "num_cpus": 1,
    })
    opt_cfg = DuetOptimizerConfig(
        lambda_=[0.0, 0.5, 1.0], max_iter=200, max_patience=50, num_cpus=1,
    )

    final_objectives_per_seed = []
    for seed in (0, 1, 2):
        init = candidates.sample_initial_selection(strategy="random", seed=seed)
        result = run_duet_ops(
            candidates=candidates, pep_config=pep_cfg, optimizer_config=opt_cfg,
            init=init, alphabet_size=4, seed=seed,
        )
        finals = [h["decode"][-1] for h in result["histories"]]
        final_objectives_per_seed.append(np.array(finals, dtype=float))

    stacked = np.stack(final_objectives_per_seed, axis=0)
    per_lambda_spread = stacked.max(axis=0) - stacked.min(axis=0)
    assert (per_lambda_spread < 0.05).all(), (
        f"Cross-seed objective spread {per_lambda_spread} exceeds 0.05 cap."
    )


def test_ops_lambda0_has_no_within_group_duplicates():
    """Regression: at lambda=0.0 the decode-objective weight is 0, so the diagonal
    duplicate penalty vanishes. Before the within-group duplicate guard, the
    optimizer filled a group's quota with copies of the single highest-activity
    candidate (mean activity above the true per-group maximum). The guard must
    keep every group's selection free of repeated candidate indices.
    """
    # Two quota-2 groups with clearly separated scores so an activity-only
    # objective is tempted to stack the top guide into both slots.
    df = pd.DataFrame({
        "Group":    ["G1", "G1", "G1", "G2", "G2", "G2"],
        "Sequence": ["AAAA", "CCCC", "GGGG", "ATAT", "CGCG", "TATA"],
        "Score":    [0.95, 0.50, 0.30, 0.92, 0.55, 0.25],
        "Quota":    [2, 2, 2, 2, 2, 2],
    })
    candidates = CandidatePool.from_dataframe(df)
    assert candidates.quotas == {"G1": 2, "G2": 2}

    pep_cfg = EvaluatorConfig.from_dict({
        "noise_channel": {"type": "symmetric", "epsilon": 0.1},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 100,
        "num_cpus": 1,
    })
    opt_cfg = DuetOptimizerConfig(
        lambda_=[0.0], max_iter=500, max_patience=100, num_cpus=1,
        avoid_duplicates=True, duplicate_penalty=100.0,
    )
    init = candidates.sample_initial_selection(strategy="random", seed=0)
    result = run_duet_ops(
        candidates=candidates, pep_config=pep_cfg, optimizer_config=opt_cfg,
        init=init, alphabet_size=4, seed=0,
    )
    best = np.asarray(result["best_indices"][result["lambdas"].index(0.0)])

    # No candidate index may appear more than once within a single group.
    ctg = candidates.codeword_to_group
    by_group: dict[str, list[int]] = {}
    for pos, c in enumerate(best):
        by_group.setdefault(ctg[pos], []).append(int(c))
    for group, sel in by_group.items():
        assert len(sel) == len(set(sel)), (
            f"group {group} selected a duplicate candidate: {sel}"
        )

    # Mean activity must not exceed the honest per-group top-quota maximum
    # (which duplicate stacking would have beaten).
    scores = candidates.scores
    max_activity = (
        sorted(scores[candidates.group_to_candidates["G1"]], reverse=True)[:2]
        + sorted(scores[candidates.group_to_candidates["G2"]], reverse=True)[:2]
    )
    assert scores[best].mean() <= np.mean(max_activity) + 1e-9
