#!/usr/bin/env bash
# =============================================================================
# Supp Fig S3b: Weissman CRISPRi cross-evaluation under NIS-seq HeLa learned
# error rates.
#   Phase 1: DUET (+ Feldman, Sivanandan, max activity) optimized under each of
#            four assumed noise channels: run_benchmark.py + visualize_benchmark.py.
#   Phase 2: every phase-1 codebook re-scored under the position-varying
#            asymmetric channel: compare_benchmarks.py + visualize_comparison.py.
#
# Source bundle: scripts/benchmark/archive/08-05-2026/Weissman_cross_eval/
# (the 4 phase-1 configs and cross_eval/eval_position_varying_asymmetric.yaml;
# the other three cross-evals are not in the paper and are not carried over).
#
# Activate an env whose `duet` resolves to this repo first (checked below). The
# Feldman baseline shells out to the conda env named by `feldman.conda_env`
# (`ops`). Works from any cwd.
#   bash experiments/ops_crispri_cross_eval/run.sh
#   VARIANT=smoke bash experiments/ops_crispri_cross_eval/run.sh
#       (reads the <name>.smoke.yaml files beside the configs: short runs)
#
# Inputs: three of the four arms and the cross-evaluation read the NIS-seq
# channel matrices in noise_matrices/. They are included only with the
# permission of the NIS-seq authors; the driver checks for them before any GPU
# work.
#
# Outputs: each config's outdir under results/experiments/ops_crispri_cross_eval/
# (<model>/ for phase 1, cross_eval/eval_position_varying_asymmetric/ for
# phase 2); logs: results/experiments/ops_crispri_cross_eval/logs/.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
SFX="${VARIANT:+.$VARIANT}"
MODELS=(symmetric position_varying asymmetric position_varying_asymmetric)
EVAL=eval_position_varying_asymmetric
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
channel_files() {  # $1.. configs: every channel matrix file they name, resolved against the cwd
    python -c 'import os, sys, yaml
keys = {"channel_matrix_path", "epsilon_path", "channel_matrices_path"}
def walk(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in keys and isinstance(v, str): yield v
            else: yield from walk(v)
    elif isinstance(d, list):
        for v in d: yield from walk(v)
paths = {os.path.realpath(p) for c in sys.argv[1:] for p in walk(yaml.safe_load(open(c)))}
print(*sorted(paths), sep="\n")' "$@"
}

cd "$HERE"
for c in "${MODELS[@]/%/$SFX.yaml}" "$EVAL$SFX.yaml"; do
    [[ -f "$c" ]] || { echo "ERROR: $HERE/$c not found (VARIANT=smoke selects the smoke configs; leave VARIANT unset for the real run)" >&2; exit 1; }
done
check_duet
# The NIS-seq channel matrices the configs read (noise_matrices/nisseq_channel.npy,
# nisseq_positional_10rounds.npy and nisseq_positional_channel_10rounds.npy):
# check them before any GPU work, since the first arm that needs one comes second.
mapfile -t MATRICES < <(channel_files "${MODELS[@]/%/$SFX.yaml}" "$EVAL$SFX.yaml")
(( ${#MATRICES[@]} )) || { echo "ERROR: could not read the channel matrix paths from the configs" >&2; exit 1; }
MISSING=()
for f in "${MATRICES[@]}"; do [[ -f "$f" ]] || MISSING+=("$f"); done
if (( ${#MISSING[@]} )); then
    printf 'ERROR: NIS-seq channel matrix not found: %s\n' "${MISSING[@]}" >&2
    cat >&2 <<'MSG'
The NIS-seq channel matrices are fitted from spot-level base calls that the
authors of Fandrey et al. (2025) shared on request, and are included only with
their permission. If one is missing from your copy, rebuild it with
experiments/nisseq_error_analysis from the spot calls, which are available from
those authors on request. Run it with SUBSAMPLE_NS=50000: nisseq_channel.npy is
its nisseq_hela_channel_pctl50_subsample50000.npy, and the two *_10rounds.npy
files are the first 10 rounds of its positional and positional_channel matrices.
MSG
    exit 1
fi
# Logs go beside the phase-1 outdirs: <experiment results dir>/logs/.
SYM_OUT="$(realpath -m "$(yaml_get "symmetric$SFX.yaml" outdir)")"
EXP_OUT="$(dirname "$SYM_OUT")"
SCRATCH="$(realpath -m "$(yaml_get "symmetric$SFX.yaml" scratch_dir)")"
EVAL_SCRATCH="$(realpath -m "$(yaml_get "$EVAL$SFX.yaml" evaluator.scratch_dir)")"
LOGDIR="$EXP_OUT/logs"
mkdir -p "$LOGDIR" "$SCRATCH/tmp" "$EVAL_SCRATCH"
export TMPDIR="$SCRATCH/tmp"

echo "=== ops_crispri_cross_eval (suffix '${SFX}') === started $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable; configs request device=gpu:all"

for model in "${MODELS[@]}"; do
    CONFIG="$model$SFX.yaml"
    echo "=== Phase 1: $model === started $(date)"
    script -qefc "python \"$REPO_ROOT/scripts/benchmark/run_benchmark.py\" --config \"$CONFIG\" -vv" "$LOGDIR/$model.raw.log"
    clean_log "$LOGDIR/$model.raw.log" "$LOGDIR/$model.log"
    echo "=== Phase 1 visualization: $model === $(date)"
    python "$REPO_ROOT/scripts/benchmark/visualize_benchmark.py" --config "$CONFIG" \
        2>&1 | tee "$LOGDIR/$model.visualize.log"
done

# compare_benchmarks.py writes CSVs only; visualize_comparison.py reads the
# aggregated CSV (same YAML) and renders the Pareto-front / hypervolume plots.
# Both use a bare `from comparison import`, so they run as `python <path>`.
CONFIG="$EVAL$SFX.yaml"
echo "=== Phase 2: $EVAL === started $(date)"
script -qefc "python \"$REPO_ROOT/scripts/benchmark/compare_benchmarks.py\" --config \"$CONFIG\" -vv" "$LOGDIR/$EVAL.raw.log"
clean_log "$LOGDIR/$EVAL.raw.log" "$LOGDIR/$EVAL.log"
echo "=== Phase 2 visualization: $EVAL === $(date)"
python "$REPO_ROOT/scripts/benchmark/visualize_comparison.py" --config "$CONFIG" \
    2>&1 | tee "$LOGDIR/$EVAL.visualize.log"

echo "=== ops_crispri_cross_eval === finished $(date)"
echo "S3b panel: $(realpath -m "$(yaml_get "$CONFIG" outdir)")/pareto_fronts/Mean_decode_accuracy_aggregated.svg"
