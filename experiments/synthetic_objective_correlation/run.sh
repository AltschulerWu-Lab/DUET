#!/usr/bin/env bash
# =============================================================================
# Fig 2a,b: synthetic objective-correlation enumeration (20 trials x 4 noise
# channels x 3,003 codebooks), then the per-trial scatter and summary figures.
#
# Source bundle: scripts/benchmark/archive/08-05-2026/duet_vs_baseline_objectives_centered_nll/
# run_rtb.py is that bundle's runner with its docstrings updated (package name,
# usage line); the code is unchanged.
#
# Activate an env whose `duet` resolves to this repo first (checked below).
# Works from any cwd.
#   bash experiments/synthetic_objective_correlation/run.sh
#   VARIANT=smoke bash experiments/synthetic_objective_correlation/run.sh
#       (reads config.smoke.yaml beside config.yaml: 1 trial, 200 samples)
#
# Outputs: the config's outdir (results/experiments/synthetic_objective_correlation/);
# logs: <outdir>/logs/.
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
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # for python -m scripts.benchmark...
OUTDIR="$(realpath -m "$(yaml_get "$CONFIG" outdir)")"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

echo "=== synthetic_objective_correlation ($CONFIG) === started $(date)"
script -qefc "python run_rtb.py --config \"$CONFIG\" -v" "$LOGDIR/run.raw.log"
clean_log "$LOGDIR/run.raw.log" "$LOGDIR/run.log"

echo "=== Scatter visualizer === $(date)"
python -m scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives --indir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/visualize_scatter.log"

echo "=== Summary visualizer === $(date)"
python -m scripts.benchmark.synthetic.visualize_baseline_summary --indir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/visualize_summary.log"

echo "=== synthetic_objective_correlation === finished $(date)"
echo "Figures: $OUTDIR/figures/"
