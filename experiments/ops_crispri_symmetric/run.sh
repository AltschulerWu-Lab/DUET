#!/usr/bin/env bash
# =============================================================================
# Fig 3b and Supp Fig S2a-g: Weissman CRISPRi under a symmetric channel.
# run_benchmark.py (DUET sweep + Feldman + Sivanandan + max activity, 5 trials)
# then visualize_benchmark.py (Pareto fronts, hypervolume, guide-level panels).
#
# Source bundle: scripts/benchmark/archive/08-18-2026/symmetric/
#
# Activate an env whose `duet` resolves to this repo first (checked below). The
# Feldman baseline shells out to the conda env named by `feldman.conda_env` in
# the config (`ops`). Works from any cwd.
#   bash experiments/ops_crispri_symmetric/run.sh
#   VARIANT=smoke bash experiments/ops_crispri_symmetric/run.sh
#       (reads config.smoke.yaml beside config.yaml: a small pool, 1 trial)
#
# Outputs: the config's outdir (results/experiments/ops_crispri_symmetric/);
# logs: <outdir>/logs/; PEP cache and scratch: the config's cache_dir and
# scratch_dir. TMPDIR is pointed at <scratch_dir>/tmp so the Feldman runner's
# temporary files stay under results/ too.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
CONFIG="config${VARIANT:+.$VARIANT}.yaml"
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
OUTDIR="$(realpath -m "$(yaml_get "$CONFIG" outdir)")"
SCRATCH="$(realpath -m "$(yaml_get "$CONFIG" scratch_dir)")"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR" "$SCRATCH/tmp"
export TMPDIR="$SCRATCH/tmp"

echo "=== ops_crispri_symmetric ($CONFIG) === started $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable; config requests device=gpu:all"

script -qefc "python \"$REPO_ROOT/scripts/benchmark/run_benchmark.py\" --config \"$CONFIG\" -vv" "$LOGDIR/run.raw.log"
clean_log "$LOGDIR/run.raw.log" "$LOGDIR/run.log"

echo "=== Visualization === $(date)"
python "$REPO_ROOT/scripts/benchmark/visualize_benchmark.py" --config "$CONFIG" \
    2>&1 | tee "$LOGDIR/visualize.log"

echo "=== ops_crispri_symmetric === finished $(date)"
echo "Outputs: $OUTDIR"
