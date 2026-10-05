#!/usr/bin/env bash
# =============================================================================
# Supp Fig S1: 2-D synthetic HVR benchmark, small (enumerable) regime. DUET,
# Greedy-NLL-MO and Greedy-Hamming-MO over a 101-value lambda grid, scored
# against the exhaustive Pareto front (4 channels x 3 error rates x 5 trials).
# Then the figures (aggregate_hvr.svg is the panel).
#
# Source bundle: scripts/benchmark/archive/09-08-2026/2d_synthetic_dense_lambda/
#
# Activate an env whose `duet` resolves to this repo first (checked below).
# Works from any cwd.
#   bash experiments/synthetic_hvr/run.sh
#   VARIANT=smoke bash experiments/synthetic_hvr/run.sh
#       (reads config.smoke.yaml beside config.yaml: 1 trial, one error rate)
#
# Outputs: the config's outdir (results/experiments/synthetic_hvr/), including
# the runner's hard-coded PEP cache under <outdir>/artifacts/pep_cache/;
# logs: <outdir>/logs/. TMPDIR (evaluator mmap scratch; the runner ignores
# evaluator.scratch_dir) goes to results/scratch/synthetic_hvr/.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
CONFIG="config${VARIANT:+.$VARIANT}.yaml"
NAME=synthetic_hvr
export PYTHONUNBUFFERED=1

clean_log() {
    sed 's/\r$//' "$1" | sed $'s/.*\r//' | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tail -n +2 | head -n -1 > "$2"
}
yaml_get() {  # $1 config, $2 dotted key; fails if the key is missing or null
    python -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d = d[k]
sys.exit(f"{sys.argv[2]} is null in {sys.argv[1]}") if d is None else print(d)' "$1" "$2"
}
check_duet() {  # the active env must import duet from this repo's src/duet/
    python -c 'import sys, pathlib, duet; p = pathlib.Path(duet.__file__).resolve(); r = pathlib.Path(sys.argv[1]).resolve()
sys.exit(0 if r in p.parents else f"ERROR: the active env imports duet from {p}, not from {r}")' "$REPO_ROOT/src/duet"
}

cd "$HERE"
[[ -f "$CONFIG" ]] || { echo "ERROR: $HERE/$CONFIG not found (VARIANT=smoke selects the smoke config; leave VARIANT unset for the real run)" >&2; exit 1; }
check_duet
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # for python -m scripts.benchmark...
OUTDIR="$(realpath -m "$(yaml_get "$CONFIG" outdir)")"
LOGDIR="$OUTDIR/logs"
SCRATCH="$REPO_ROOT/results/${VARIANT:+$VARIANT/}scratch/$NAME"
mkdir -p "$LOGDIR" "$SCRATCH"
export TMPDIR="$SCRATCH"

echo "=== synthetic_hvr ($CONFIG) === started $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable; config requests device=gpu:all"

echo "=== Stage 1/2: benchmark === $(date)"
script -qefc "python -m scripts.benchmark.synthetic.run_2d_synthetic_benchmark --config \"$CONFIG\" -v" "$LOGDIR/run.raw.log"
clean_log "$LOGDIR/run.raw.log" "$LOGDIR/run.log"

echo "=== Stage 2/2: figures === $(date)"
python -m scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark --indir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/visualize.log"

echo "=== synthetic_hvr === finished $(date)"
echo "Panel source: $OUTDIR/aggregate_hvr.svg"
