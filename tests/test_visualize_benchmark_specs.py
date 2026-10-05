"""Tests for the guide-level comparison spec builder.

Covers `build_comparison_specs`, which decides -- for one trial -- which
baseline is compared against which DUET lambda, under three reference policies:
activity-matched, and one per entry in DUET_REFERENCE_FRACTIONS.

Pure function over the aggregated results frame: no plotting, no IO.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/benchmark/ is not on sys.path by default -- visualize_benchmark.py is
# a script, not a package module. Add it here so the tests can import it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmark"))

from visualize_benchmark import (  # noqa: E402
    DUET_REFERENCE_FRACTIONS,
    MAX_ACTIVITY_METHOD,
    build_comparison_specs,
    build_overlay_specs,
)

FELDMAN = ["Feldman et al. (ED=1)"]
SIVANANDAN = ["Sivanandan et al. (ED=1)", "Sivanandan et al. (ED=2)"]
BASELINES = FELDMAN + SIVANANDAN


def _agg(rows, trial: int = 1) -> pd.DataFrame:
    """Build a minimal aggregated frame: (method, decode, activity) rows."""
    return pd.DataFrame(
        [
            {
                "Method": method,
                "Method group": (
                    "DUET" if method.startswith("DUET") else MAX_ACTIVITY_METHOD
                    if method == MAX_ACTIVITY_METHOD
                    else "Feldman et al." if method.startswith("Feldman")
                    else "Sivanandan et al."
                ),
                "Trial": trial,
                "Mean decode accuracy": decode,
                "Mean activity score": activity,
            }
            for method, decode, activity in rows
        ]
    )


def _default_frame(trial: int = 1) -> pd.DataFrame:
    """A frontier shaped like the real benchmark.

    Max activity is 0.90, so the 97.5% threshold is 0.8775 and the 95%
    threshold is 0.855. Decode accuracy falls monotonically as activity rises,
    which is what makes `select_duet_lambda`'s "highest decode above the
    threshold" pick the *highest* qualifying lambda (lambda weights decoding).

    Sivanandan ED=1 sits at the activity ceiling on purpose: its matched policy
    can only resolve to lambda=0.00, reproducing the degeneracy that motivated
    the fraction policies.
    """
    return _agg(
        [
            ("DUET (lambda=0.20)", 0.79, 0.860),  # clears 95%, not 97.5%
            ("DUET (lambda=0.10)", 0.78, 0.880),  # clears 97.5%
            ("DUET (lambda=0.05)", 0.77, 0.895),
            ("DUET (lambda=0.00)", 0.73, 0.900),  # activity-only extreme
            (MAX_ACTIVITY_METHOD, 0.73, 0.900),
            ("Feldman et al. (ED=1)", 0.74, 0.890),
            ("Sivanandan et al. (ED=1)", 0.74, 0.899),  # at the ceiling
            ("Sivanandan et al. (ED=2)", 0.75, 0.870),
        ],
        trial=trial,
    )


def _by_policy(specs, policy):
    return [s for s in specs if s.policy == policy]


def test_fraction_group_shares_one_duet_arm():
    """Every panel at a given fraction compares against the same DUET codebook."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)

    for suffix in DUET_REFERENCE_FRACTIONS:
        group = _by_policy(specs, suffix)
        assert group, f"no specs emitted for fraction {suffix}"
        assert len({s.duet for s in group}) == 1, (
            f"fraction {suffix} used more than one DUET arm: "
            f"{sorted({s.duet for s in group})}"
        )
        # The max-activity panel is part of the group, not a separate reference.
        assert MAX_ACTIVITY_METHOD in {s.baseline for s in group}


def test_fraction_arms_are_the_expected_lambdas():
    """The 97.5% and 95% thresholds select distinct, correctly-ranked lambdas."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)

    arm_97p5 = {s.duet for s in _by_policy(specs, "97p5pct")}.pop()
    arm_95 = {s.duet for s in _by_policy(specs, "95pct")}.pop()

    # 0.880 >= 0.8775 clears 97.5%; 0.860 does not, but clears 0.855.
    assert arm_97p5 == "DUET (lambda=0.10)"
    assert arm_95 == "DUET (lambda=0.20)"


def test_matched_policy_reproduces_per_baseline_selection():
    """Matched policy keeps today's behavior, including the ceiling degeneracy."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    matched = {s.baseline: s.duet for s in _by_policy(specs, "matched")}

    assert matched == {
        # activity >= 0.890 -> lambda=0.05 (0.895) beats lambda=0.00 on decode
        "Feldman et al. (ED=1)": "DUET (lambda=0.05)",
        # activity >= 0.899 -> only lambda=0.00 qualifies
        "Sivanandan et al. (ED=1)": "DUET (lambda=0.00)",
        # activity >= 0.870 -> lambda=0.10 (0.880) is the highest-decode qualifier
        "Sivanandan et al. (ED=2)": "DUET (lambda=0.10)",
    }


def test_max_activity_has_no_matched_panel():
    """"DUET with activity >= max activity" is a near-copy of max activity."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    matched_baselines = {s.baseline for s in _by_policy(specs, "matched")}
    assert MAX_ACTIVITY_METHOD not in matched_baselines


def test_existing_prefixes_are_unchanged():
    """Pre-existing panels keep the filenames earlier runs produced."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    prefixes = {s.prefix for s in specs}

    assert "max_activity_97p5pct" in prefixes
    assert "max_activity_95pct" in prefixes
    assert "Feldman_et_al_ED1" in prefixes
    assert "Sivanandan_et_al_ED1" in prefixes
    # New panels are suffixed, so they cannot collide with the above.
    assert "Feldman_et_al_ED1_vs_duet_97p5pct" in prefixes
    assert "Feldman_et_al_ED1_vs_duet_95pct" in prefixes


def test_prefixes_are_unique():
    """No two specs write to the same file."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    prefixes = [s.prefix for s in specs]
    assert len(prefixes) == len(set(prefixes))


def test_max_activity_fractions_precede_matched_group():
    """Ordering the error-metrics legacy bar chart depends on."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    first_matched = next(i for i, s in enumerate(specs) if s.policy == "matched")
    max_act_indices = [
        i for i, s in enumerate(specs) if s.baseline == MAX_ACTIVITY_METHOD
    ]
    assert max(max_act_indices) < first_matched


def test_absent_baseline_is_skipped_in_every_policy():
    """A baseline with no rows this trial produces no specs at all."""
    frame = _default_frame()
    frame = frame[frame["Method"] != "Sivanandan et al. (ED=2)"]

    specs = build_comparison_specs(frame, 1, FELDMAN, SIVANANDAN)

    assert "Sivanandan et al. (ED=2)" not in {s.baseline for s in specs}
    # The other baselines are unaffected.
    assert "Sivanandan et al. (ED=1)" in {s.baseline for s in specs}


def test_missing_max_activity_returns_no_specs():
    """Without max activity there is no reference to anchor to."""
    frame = _default_frame()
    frame = frame[frame["Method"] != MAX_ACTIVITY_METHOD]

    assert build_comparison_specs(frame, 1, FELDMAN, SIVANANDAN) == []


def test_unresolvable_fraction_drops_only_its_own_group():
    """A fraction with no qualifying DUET leaves the other policies intact.

    Every DUET lambda is dropped except one below the 97.5% threshold but above
    the 95% one, so `97p5pct` cannot resolve while `95pct` still can.
    """
    frame = _agg(
        [
            ("DUET (lambda=0.20)", 0.79, 0.860),  # 0.855 <= 0.860 < 0.8775
            (MAX_ACTIVITY_METHOD, 0.73, 0.900),
            ("Feldman et al. (ED=1)", 0.74, 0.850),
        ]
    )

    specs = build_comparison_specs(frame, 1, FELDMAN, [])

    assert _by_policy(specs, "97p5pct") == []
    assert {s.duet for s in _by_policy(specs, "95pct")} == {"DUET (lambda=0.20)"}
    assert {s.duet for s in _by_policy(specs, "matched")} == {"DUET (lambda=0.20)"}


def test_specs_are_scoped_to_the_requested_trial():
    """Rows from another trial never leak into a trial's specs."""
    trial_1 = _default_frame(trial=1)
    trial_2 = _default_frame(trial=2)
    # Give trial 2 a different frontier so a leak would be visible.
    trial_2.loc[trial_2["Method"] == "DUET (lambda=0.10)", "Mean activity score"] = 0.99

    combined = pd.concat([trial_1, trial_2], ignore_index=True)

    from_combined = build_comparison_specs(combined, 1, FELDMAN, SIVANANDAN)
    from_isolated = build_comparison_specs(trial_1, 1, FELDMAN, SIVANANDAN)

    assert from_combined == from_isolated


@pytest.mark.parametrize("policy", ["matched", *DUET_REFERENCE_FRACTIONS])
def test_baseline_arm_is_never_a_duet_method(policy):
    """The two arms of a comparison are always a baseline and a DUET point."""
    specs = build_comparison_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    for spec in _by_policy(specs, policy):
        assert not spec.baseline.startswith("DUET")
        assert spec.duet.startswith("DUET")


# ---------------------------------------------------------------------------
# Overlay specs: every baseline against one DUET arm, one panel per fraction.
# ---------------------------------------------------------------------------


def test_overlay_specs_one_per_fraction_with_every_baseline():
    specs = build_overlay_specs(_default_frame(), 1, FELDMAN, SIVANANDAN)
    assert [spec.policy for spec in specs] == list(DUET_REFERENCE_FRACTIONS)
    for spec in specs:
        assert spec.baselines == (MAX_ACTIVITY_METHOD, *FELDMAN, *SIVANANDAN)
    assert [spec.prefix for spec in specs] == [
        "all_baselines_vs_duet_97p5pct",
        "all_baselines_vs_duet_95pct",
    ]


def test_overlay_duet_arm_is_the_fraction_group_arm():
    frame = _default_frame()
    pairwise = {
        spec.policy: spec.duet
        for spec in build_comparison_specs(frame, 1, FELDMAN, SIVANANDAN)
        if spec.policy in DUET_REFERENCE_FRACTIONS
    }
    overlay = {
        spec.policy: spec.duet
        for spec in build_overlay_specs(frame, 1, FELDMAN, SIVANANDAN)
    }
    assert overlay == pairwise


def test_overlay_skips_baselines_absent_from_the_trial():
    frame = _default_frame()
    frame = frame[frame["Method"] != "Feldman et al. (ED=1)"]
    specs = build_overlay_specs(frame, 1, FELDMAN, SIVANANDAN)
    assert specs
    for spec in specs:
        assert "Feldman et al. (ED=1)" not in spec.baselines
        assert MAX_ACTIVITY_METHOD in spec.baselines


def test_overlay_without_max_activity_returns_no_specs():
    frame = _default_frame()
    frame = frame[frame["Method"] != MAX_ACTIVITY_METHOD]
    assert build_overlay_specs(frame, 1, FELDMAN, SIVANANDAN) == []
