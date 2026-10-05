#!/usr/bin/env python3
"""Render the 18 configs of the Fig 3c sequencing-rounds x error-rate sweep.

Migrated copy of scripts/benchmark/archive/08-20-2026/robustness_sweep/make_configs.py,
the generator of the 2026-09-02 run (RESULTS_DATE "09-02-2026", 31-value lambda
grid) that produced the Fig 3c replacement data. Changes, nothing else:

* `duet.optimizer.lambda` is the as-run list rewritten as 1 - x, sorted ascending
  and written as exact decimals (docs/adr/0001-lambda-weights-decoding-accuracy.md).
  `LAMBDAS` below is checked against `AS_RUN_LAMBDA_OLD` on every render.
* `evaluator.seed: 42` and `duet.pep.seed: 43` are added (both unset as run), as
  in experiments/ops_crispri_symmetric.
* `outdir`, `cache_dir` and `scratch_dir` are config-relative paths under this
  repo's results/ (RESULTS_DATE and the absolute lab-share paths are gone).
* Each YAML opens with a migration block; the archived header follows verbatim.
  The lambda and path comments inside the YAML were rewritten to match.
* New modes: `--check` (verify the committed YAMLs) and `--smoke` (write the
  three smoke variants, which are tracked too).

Every other key is as run, including `duet.optimizer.num_cpus: 31`.

The template is held here as a string rather than read from another config so
this generator pins its own parameters.

Usage (from any directory; files are written beside this script):
    python make_configs.py            # (re)write the 18 committed YAMLs
    python make_configs.py --check    # render into a temp dir, fail unless every
                                      # committed YAML is byte-identical
    python make_configs.py --smoke    # (re)write the three <stem>.smoke.yaml files
"""
from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_BUNDLE = "scripts/benchmark/archive/08-20-2026/robustness_sweep"

SEQ_ROUNDS = [8, 9, 10, 11, 12, 13]
EPSILONS = [0.03, 0.05, 0.10]

# (rounds, epsilon) for all 18 points, ordered so stems sort into grid order.
GRID: list[tuple[int, float]] = [(r, e) for r in SEQ_ROUNDS for e in EPSILONS]

# The as-run grid (old convention, lambda_old = 1 - lambda), kept to check the flip.
AS_RUN_LAMBDA_OLD = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
    0.8, 0.825, 0.85, 0.875,
    0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0,
]
# 1 - x as exact literals, ascending. Computing 1 - x would give 0.17500000000000004
# (labelled "0.18") and 0.09999999999999998 (seed offset 99 instead of 100).
LAMBDAS = [
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    0.125, 0.15, 0.175, 0.2, 0.25,
    0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    1.0,
]

# Smoke variants (VARIANT=smoke; tracked, and rewritten by `--smoke`).
# The shrink matches experiments/ops_crispri_symmetric/config.smoke.yaml
# (trials 1, 50 groups, 20 controls, 200 samples for evaluator and PEP, seeds
# 42/43), so visualize_sweep.py's exact reference check can run on the smoke:
# sr10_eps0.10 is that config with a lambda superset. The extra lambda (0.9)
# changes the evaluation union, as the 13 extra real values do, but sits below
# the 97.5% activity floor, so the 97.5% arm matches the reference's and the
# Tier B comparison is like for like. sr10_eps0.05 exercises the
# epsilon-independent checks and sr08_eps0.03 a second rounds column.
SMOKE_POINTS: list[tuple[int, float]] = [(8, 0.03), (10, 0.05), (10, 0.10)]
SMOKE_LAMBDAS = [0.0, 0.9, 1.0]
SMOKE_TRIALS = 1
SMOKE_NUM_GROUPS = 50
SMOKE_NUM_CONTROLS = 20
SMOKE_NUM_SAMPLES = 200


def stem_for(rounds: int, eps: float) -> str:
    """Stem naming the YAML, the results subdirectory and the log pair.

    Rounds are zero-padded to two digits and epsilon to two decimals so the 18
    stems sort lexicographically into grid order.
    """
    return f"sr{rounds:02d}_eps{eps:.2f}"


def check_lambda_flip() -> None:
    """Fail unless LAMBDAS is the ascending 1 - x image of the as-run grid.

    Also checks that every literal gives an exact per-lambda seed offset
    (`int(lambda * 1000)`, src/duet/runner/core.py) and a distinct `:.2f` label.
    """
    flipped = sorted(round(1.0 - x, 10) for x in AS_RUN_LAMBDA_OLD)
    if LAMBDAS != sorted(LAMBDAS) or LAMBDAS != flipped:
        raise SystemExit(f"LAMBDAS is not the sorted 1 - x flip of AS_RUN_LAMBDA_OLD: {flipped}")
    for lam in LAMBDAS:
        if int(lam * 1000) != round(lam * 1000):
            raise SystemExit(f"lambda {lam!r}: seed offset int(lam*1000) is not exact")
    labels = [f"{lam:.2f}" for lam in LAMBDAS]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"lambda labels collide under :.2f: {labels}")


def _lambda_yaml() -> str:
    """The lambda list as a YAML flow sequence, one band per line."""
    bands = [LAMBDAS[:11], LAMBDAS[11:16], LAMBDAS[16:30], LAMBDAS[30:]]
    lines = [", ".join(repr(x) for x in band) for band in bands]
    return "[" + (",\n             ").join(lines) + "]"


CONFIG_TEMPLATE = """\
# =============================================================================
# MIGRATED COPY of {source}/{stem}.yaml
# for experiments/ops_crispri_rounds_error_sweep (Fig 3c), the grid as run on
# 2026-09-02 (RESULTS_DATE 09-02-2026, 31-value lambda grid). GENERATED FILE:
# edit make_configs.py in this folder, then run `python make_configs.py`.
# Changes, nothing else:
#   * duet.optimizer.lambda rewritten as 1 - x (lambda now weights decoding
#     accuracy; docs/adr/0001-lambda-weights-decoding-accuracy.md), sorted
#     ascending, exact decimals; its comment rewritten in the new convention.
#   * outdir, cache_dir, scratch_dir moved under this repo's results/; the path
#     comment rewritten.
#   * evaluator.seed: 42 and duet.pep.seed: 43 added (both unset as run), as in
#     experiments/ops_crispri_symmetric: the PEP gets its own seed so its Monte
#     Carlo streams are independent of the ground truth's (both spawn one child
#     stream per index from their root seed). The top-level seed only sets the
#     trial seeds.
#   * duet.optimizer.num_cpus comment extended (value kept as run).
# The archived header follows unchanged. Its make_configs.py is now this
# folder's copy; the spec and template it names are the as-run ones.
# =============================================================================

# =============================================================================
# OPS robustness sweep -- {rounds} sequencing rounds, symmetric epsilon = {eps_g}
# =============================================================================
# GENERATED FILE -- do not edit by hand.
# Regenerate with:  python make_configs.py
# Edits belong in make_configs.py (CONFIG_TEMPLATE / GRID).
#
# One point of the rounds x epsilon robustness sweep. Derived from the config
# of the original ops_crispri_symmetric run (experiments/ops_crispri_symmetric/
# config.yaml is its copy); only outdir, candidate_pool.seq_rounds and the two
# noise_channel epsilons vary across the 18 points.
#
# Scalar epsilon applied identically at every position and symbol, paired with
# the plain Hamming decoding metric: under a symmetric channel Hamming distance
# is monotone-equivalent to the symmetric NLL, so unique_minimum decoding gives
# identical decisions at lower cost.
# =============================================================================

# Paths: OpsBenchmarkConfig takes outdir, cache_dir and scratch_dir verbatim,
# relative to the working directory. run.sh cd's into this folder first, so
# they resolve relative to this file. scratch_dir must already exist
# (tempfile.mkdtemp does not create it); run.sh creates it. All 18 points share
# cache_dir; each PEP's fingerprint covers its library and channel.
outdir: {outdir}
cache_dir: {cache_dir}
scratch_dir: {scratch_dir}
trials: 5
seed: 42
device: "gpu:all"
mem_budget_gb: 64.0

candidate_pool:
  source: WeissmanCRISPRi
  seq_rounds: {rounds}
  quota: 2
  num_controls: 200
  num_groups: 1000
  min_rank: 10

evaluator:
  noise_channel:
    type: symmetric
    epsilon: {eps_g}
  decoding_metric:
    type: hamming
  decoding_rule:
    type: unique_minimum
  num_samples: 2000
  num_cpus: 30
  seed: 42

duet:
  use_mmap: true
  force_rebuild: false
  initialization:
    type: random
  optimizer:
    # As run (old convention, lambda_old = 1 - lambda): [0.0, 0.05, ..., 0.75,
    # 0.8, 0.825, 0.85, 0.875, 0.9, 0.91, ..., 0.99, 1.0]. Now 1 - x, sorted
    # ascending, exact decimals.
    # 31 values, a strict superset of the 18-value Fig 3b grid that the
    # 2026-08-20 run used (nothing dropped). The 13 additions fill (0.25, 0.5)
    # and (0.5, 1.0), which that run left unsampled: activity falls as lambda
    # rises, and at low epsilon / long reads the 97.5% activity floor falls
    # inside that band. With nothing there, 10 of 18 points were forced to
    # lambda=0.5 and overshot the floor by up to 0.02 activity, understating
    # DUET's gain and invalidating the Sivanandan ED=3 comparison. The 0.01
    # spacing across 0.05-0.10 is kept: that is where the high-epsilon /
    # short-read points cross the floor.
    lambda: {lambdas}
    temperature: 0.0
    max_iter: 100000
    max_patience: 3000
    avoid_duplicates: true
    duplicate_penalty: 100.0
    # One scheduling wave: must be >= the number of lambda values above.
    # Kept as run. Speed only (each lambda job reseeds itself); 31 workers
    # oversubscribe the 20 physical cores of the as-run host.
    num_cpus: 31
  pep:
    noise_channel:
      type: symmetric
      epsilon: {eps_g}
    decoding_metric:
      type: hamming
    decoding_rule:
      type: unique_minimum
    num_samples: 5000
    num_cpus: 30
    seed: 43

sivanandan:
  edit_distances: [1, 2, 3]
  num_cpus: 8

feldman:
  conda_env: ops
  edit_distances: [1, 2]
"""


def render_config(rounds: int, eps: float) -> str:
    """Render one point's YAML text.

    `eps_g` is the shortest round-trip form of the float (0.1, not 0.10), as in
    the as-run configs; the two-decimal form appears only in the stem.
    """
    check_lambda_flip()
    stem = stem_for(rounds, eps)
    return CONFIG_TEMPLATE.format(
        source=SOURCE_BUNDLE,
        stem=stem,
        rounds=rounds,
        eps_g=f"{eps:g}",
        outdir=f"../../results/experiments/ops_crispri_rounds_error_sweep/{stem}",
        cache_dir="../../results/cache/ops_crispri_rounds_error_sweep",
        scratch_dir="../../results/scratch/ops_crispri_rounds_error_sweep",
        lambdas=_lambda_yaml(),
    )


def render_smoke_config(rounds: int, eps: float) -> str:
    """Render one point's smoke YAML: the real config, shrunk."""
    import yaml  # only the smoke mode needs it

    stem = stem_for(rounds, eps)
    cfg = yaml.safe_load(render_config(rounds, eps))
    cfg["outdir"] = f"../../results/smoke/ops_crispri_rounds_error_sweep/{stem}"
    cfg["cache_dir"] = "../../results/smoke/cache/ops_crispri_rounds_error_sweep"
    cfg["scratch_dir"] = "../../results/smoke/scratch/ops_crispri_rounds_error_sweep"
    cfg["trials"] = SMOKE_TRIALS
    cfg["candidate_pool"]["num_groups"] = SMOKE_NUM_GROUPS
    cfg["candidate_pool"]["num_controls"] = SMOKE_NUM_CONTROLS
    cfg["evaluator"]["num_samples"] = SMOKE_NUM_SAMPLES
    cfg["duet"]["pep"]["num_samples"] = SMOKE_NUM_SAMPLES
    cfg["duet"]["optimizer"]["lambda"] = list(SMOKE_LAMBDAS)
    header = (
        "# Smoke config (VARIANT=smoke), written by `python make_configs.py --smoke`\n"
        f"# from {stem}.yaml: trials {SMOKE_TRIALS}, num_groups {SMOKE_NUM_GROUPS}, "
        f"num_controls {SMOKE_NUM_CONTROLS},\n"
        f"# evaluator/pep num_samples {SMOKE_NUM_SAMPLES}, lambda {SMOKE_LAMBDAS}. "
        "Outputs, caches and scratch under\n"
        "# results/smoke/. Same device and code path as the real config.\n"
    )
    return header + yaml.safe_dump(cfg, sort_keys=False)


def write_all(outdir: Path = HERE) -> list[Path]:
    """Write all 18 configs into `outdir`. Idempotent."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for rounds, eps in GRID:
        path = outdir / f"{stem_for(rounds, eps)}.yaml"
        path.write_text(render_config(rounds, eps))
        written.append(path)
    return written


def write_smoke(outdir: Path = HERE) -> list[Path]:
    """Write the smoke variants into `outdir`."""
    written = []
    for rounds, eps in SMOKE_POINTS:
        path = Path(outdir) / f"{stem_for(rounds, eps)}.smoke.yaml"
        path.write_text(render_smoke_config(rounds, eps))
        written.append(path)
    return written


def check(committed_dir: Path = HERE) -> list[str]:
    """Re-render into a temp dir and compare with the committed YAMLs byte for byte.

    Returns problem descriptions; empty means every committed YAML matches and
    no stray point config (sr*.yaml outside GRID) sits beside them.
    """
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for fresh in write_all(Path(tmp)):
            committed = Path(committed_dir) / fresh.name
            if not committed.is_file():
                problems.append(f"missing: {committed}")
                continue
            a, b = committed.read_bytes(), fresh.read_bytes()
            if a != b:
                diff = difflib.unified_diff(
                    a.decode().splitlines(), b.decode().splitlines(),
                    f"committed/{fresh.name}", f"rendered/{fresh.name}", lineterm="", n=1,
                )
                problems.append(f"differs: {committed}\n" + "\n".join(list(diff)[:40]))
    expected = {f"{stem_for(r, e)}.yaml" for r, e in GRID}
    for path in sorted(Path(committed_dir).glob("sr*.yaml")):
        if path.name not in expected and not path.name.endswith(".smoke.yaml"):
            problems.append(f"stray point config not produced by the generator: {path}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="verify the committed YAMLs are byte-identical to a fresh render")
    mode.add_argument("--smoke", action="store_true",
                      help="(re)write the three <stem>.smoke.yaml variants")
    args = parser.parse_args(argv)

    if args.check:
        problems = check()
        if problems:
            print("CHECK FAILED:\n" + "\n".join(problems))
            return 1
        print(f"CHECK PASSED: all {len(GRID)} configs in {HERE} match a fresh render byte for byte")
        return 0
    if args.smoke:
        paths = write_smoke()
        print(f"Wrote {len(paths)} smoke configs to {HERE}: " + ", ".join(p.name for p in paths))
        return 0
    paths = write_all()
    print(f"Wrote {len(paths)} configs to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
