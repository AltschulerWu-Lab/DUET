#!/usr/bin/env bash
# =============================================================================
# Fig 4c,d,e: MERFISH, 2,000 random whole-brain genes, 32-bit MHD4 (HW4/HW5).
# DUET warm-starts from the Bostrom HW5 panel and trades decode accuracy
# against optical crowding over a 9-value lambda sweep; baselines Bostrom HW5
# and MERFISH MHD4 HW5 (random assignment). Then the Fig 4d,e figures.
#
# Source bundle: scripts/benchmark/archive/06-28-2026/2000_genes/ + 06-28-2026/make_figures.py
#
# GPU (config: device gpu:all for the PEP build and the union evaluation).
# Every config is a cold PEP build (~100k candidates at 30,000 samples):
# hours of GPU time and ~40 GB of cache under results/cache/merfish_2000_genes/.
#
# Activate an env whose `duet` resolves to this repo first (checked below).
# Works from any cwd.
#   bash experiments/merfish_2000_genes/run.sh
#   VARIANT=smoke bash experiments/merfish_2000_genes/run.sh
#       (reads config.smoke.yaml beside config.yaml: a small pool, two lambdas)
# Input not in git, checked before any GPU work: the whole-brain expression
# table (expression.path; build it with scripts/wmb10x/build_whole_brain_cpm.py).
#
# Outputs: the config's outdir (results/experiments/merfish_2000_genes/), figures
# under <outdir>/figures/; logs: <outdir>/logs/. PEP cache and scratch: the
# config's cache_dir and evaluator.scratch_dir.
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
EXPRESSION="$(realpath -m "$(yaml_get "$CONFIG" expression.path)")"
[[ -f "$EXPRESSION" ]] || { echo "ERROR: expression.path $EXPRESSION not found (WMB-10X atlas values, not" \
    "shipped). Build it with python scripts/wmb10x/build_whole_brain_cpm.py in the scanpy environment." >&2; exit 1; }
# MerfishBenchmarkConfig resolves outdir against this folder; the cwd is this folder.
OUTDIR="$(realpath -m "$(yaml_get "$CONFIG" outdir)")"
SCRATCH="$(realpath -m "$(yaml_get "$CONFIG" evaluator.scratch_dir)")"
PEP_SCRATCH="$(realpath -m "$(yaml_get "$CONFIG" duet.pep.scratch_dir)")"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR" "$SCRATCH/tmp" "$PEP_SCRATCH"
export TMPDIR="$SCRATCH/tmp"

echo "=== merfish_2000_genes ($CONFIG) === started $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable; config requests device=gpu:all"

script -qefc "python \"$REPO_ROOT/scripts/benchmark/run_merfish.py\" --config \"$CONFIG\" -vv" "$LOGDIR/run.raw.log"
clean_log "$LOGDIR/run.raw.log" "$LOGDIR/run.log"

echo "=== Visualization === $(date)"
python "$REPO_ROOT/scripts/benchmark/visualize_merfish.py" --config "$CONFIG" \
    2>&1 | tee "$LOGDIR/visualize.log"

echo "=== Fig 4d,e (make_figures.py, lambda=0.90) === $(date)"
python make_figures.py --config "$CONFIG" --outdir "$OUTDIR/figures/fig4de" \
    2>&1 | tee "$LOGDIR/make_figures.log"

echo "=== merfish_2000_genes === finished $(date)"
echo "Fig 4c: $OUTDIR/figures/crowding_pareto_front.svg + figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg"
echo "Fig 4d,e: $OUTDIR/figures/fig4de/{round_expression_uniformity_y0_k2000,expr_by_hw_violin_k2000}.svg"
