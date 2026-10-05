#!/usr/bin/env bash
# =============================================================================
# Fig 4b: MERFISH, Zhang et al. 2023 panel (v2, 1,147 genes), whole-brain
# expression, 32-bit MHD4 (HW4/HW5). DUET warm-starts from the published Zhang
# v2 codebook; baselines Bostrom HW4, Bostrom HW5 and the Zhang v2 codebook.
#
# Source bundle: scripts/benchmark/archive/06-28-2026/zhang_2023/ (zhang_v2.yaml only)
#
# GPU (config: device gpu:all for the PEP build and the union evaluation).
# Every config is a cold PEP build (~100k candidates at 30,000 samples):
# hours of GPU time and ~40 GB of cache under results/cache/merfish_zhang2023_v2/.
#
# Activate an env whose `duet` resolves to this repo first (checked below).
# Works from any cwd.
#   bash experiments/merfish_zhang2023_v2/run.sh
#   VARIANT=smoke bash experiments/merfish_zhang2023_v2/run.sh
#       (reads config.smoke.yaml beside config.yaml: a small pool, two lambdas)
# Inputs not in git, checked before any GPU work: the converted Zhang et al.
# codebook (genes.path; build it with
# scripts/data_processing/build_zhang2023_codebook.py) and the whole-brain
# expression table (expression.path; build it with
# scripts/wmb10x/build_whole_brain_cpm.py).
#
# Outputs: the config's outdir (results/experiments/merfish_zhang2023_v2/), figures
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
GENES="$(realpath -m "$(yaml_get "$CONFIG" genes.path)")"
[[ -f "$GENES" ]] || { echo "ERROR: genes.path $GENES not found (the Zhang et al. 2023 codebook, not in git)." \
    "Build it with python scripts/data_processing/build_zhang2023_codebook.py --download" >&2; exit 1; }
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

echo "=== merfish_zhang2023_v2 ($CONFIG) === started $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable; config requests device=gpu:all"

script -qefc "python \"$REPO_ROOT/scripts/benchmark/run_merfish.py\" --config \"$CONFIG\" -vv" "$LOGDIR/run.raw.log"
clean_log "$LOGDIR/run.raw.log" "$LOGDIR/run.log"

echo "=== Visualization === $(date)"
python "$REPO_ROOT/scripts/benchmark/visualize_merfish.py" --config "$CONFIG" \
    2>&1 | tee "$LOGDIR/visualize.log"


echo "=== merfish_zhang2023_v2 === finished $(date)"
echo "Fig 4b: $OUTDIR/figures/crowding_pareto_front.svg + figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg"
