#!/usr/bin/env bash
# =============================================================================
# Supp Fig S4b: optical-crowding robustness to a mismatched expression prior,
# over resampled 2,000-gene panels. Steps, in order:
#   build      build_per_class_expression.py   (scanpy_env: h5py, no duet)
#              ABC WMB-10X precomputed stats -> 34 per-class CPM profiles
#   panels     make_panels.py                  gene sets, baseline panels, per-panel configs
#   duet       check_pep_cache.py, then run_merfish.py once per panel (lambda 0.80), in
#              parallel; the check stops the step if any panel would rebuild the PEP
#   crowding   evaluate_crowding_by_cell_type.py once per panel, in parallel
#   aggregate  aggregate_panels.py             tables and the S4b figure
#   check      check_pep_cache.py alone (not a default step)
#
# Depends on experiments/merfish_2000_genes: every panel reuses its PEP cache and
# config. Run that first.
#
# Activate an env whose `duet` resolves to this repo first (checked below).
# Works from any cwd.
#   bash experiments/merfish_expression_prior/run.sh                    # every step
#   bash experiments/merfish_expression_prior/run.sh crowding aggregate # named steps, in order
#   VARIANT=smoke bash experiments/merfish_expression_prior/run.sh
#       (reads config.smoke.yaml beside config.yaml: 2 panels, chained on the
#       merfish_2000_genes smoke run)
# Inputs not in git: the atlas files abc_stats_h5 and abc_gene_csv (build step;
# download them into data/raw/WMB-10X/) and the whole-brain table prior_csv
# (build it with scripts/wmb10x/build_whole_brain_cpm.py). Checked up front.
# Environment:
#   MAX_JOBS       panels run at once in the duet and crowding steps (default 10;
#                  each DUET run's OpenBLAS threads spread over every core)
#   SKIP_COMPLETE  1 = skip panels whose duet or crowding step already finished
#                  (duet/summary.yaml or crowding/per_class_metrics.csv exists). A
#                  panel that fails has that file renamed to *.failed, so it reruns.
#   SCANPY_ENV     conda env for the build step (default scanpy_env)
#   SCANPY_PYTHON  interpreter for the build step (default: that env's python)
#
# Outputs: the config's outdir (results/experiments/merfish_expression_prior/);
# logs: <outdir>/logs/ and <outdir>/<panel>/{duet,crowding}.log.
# Cost (20 panels, MAX_JOBS 10): about 4-5 h of CPU wall clock, no GPU time beyond
# a minute per panel. Each panel is ~1h40m of DUET plus ~17 min of crowding.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
CONFIG="config${VARIANT:+.$VARIANT}.yaml"
SCANPY_ENV="${SCANPY_ENV:-scanpy_env}"
# A relative SCANPY_PYTHON means relative to the caller's cwd: resolve before cd.
if [[ -n "${SCANPY_PYTHON:-}" ]]; then SCANPY_PYTHON="$(readlink -f "$(command -v "$SCANPY_PYTHON")")"; fi
export PYTHONUNBUFFERED=1

clean_log() {
    sed 's/\r$//' "$1" | sed $'s/.*\r//' | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tail -n +2 | head -n -1 > "$2"
}
cfg_path() {  # $1 config, $2 key of a path: printed resolved against this folder
    python -c 'import sys, os, yaml; d = yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]
sys.exit(f"{sys.argv[2]} is null in {sys.argv[1]}") if d is None else print(os.path.realpath(d))' "$1" "$2"
}
check_duet() {  # the active env must import duet from this repo's src/duet/
    python -c 'import sys, pathlib, duet; p = pathlib.Path(duet.__file__).resolve(); r = pathlib.Path(sys.argv[1]).resolve()
sys.exit(0 if r in p.parents else f"ERROR: the active env imports duet from {p}, not from {r}")' "$REPO_ROOT/src/duet"
}
# $1 step, $2 output that marks a panel done, $3... command (@PANEL@ becomes the panel id).
# A failed panel's $2 is renamed to $2.failed (the crowding step writes it before its
# anchor check), so SKIP_COMPLETE reruns that panel instead of counting it done.
run_panels() {
    local step="$1" done_file="$2"; shift 2
    local pids=() ids=() failed=() f
    for id in "${PANELS[@]}"; do
        if [[ "${SKIP_COMPLETE:-0}" == 1 && -e "$OUTDIR/$id/$done_file" ]]; then
            echo "  $id: $done_file exists, skipped"; continue
        fi
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do wait -n || true; done
        (
            export TMPDIR="$SCRATCH/$id/tmp"; mkdir -p "$TMPDIR"
            "${@//@PANEL@/$id}" > "$OUTDIR/$id/$step.log" 2>&1
        ) &
        pids+=($!); ids+=("$id")
        echo "  $id: started (pid $!, log $OUTDIR/$id/$step.log)"
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            failed+=("${ids[$i]}")
            f="$OUTDIR/${ids[$i]}/$done_file"
            if [[ -e "$f" ]]; then mv "$f" "$f.failed"; fi
        fi
    done
    if (( ${#failed[@]} )); then
        echo "ERROR: $step failed for ${failed[*]} (see their $step.log)" >&2; exit 1
    fi
}

cd "$HERE"
[[ -f "$CONFIG" ]] || { echo "ERROR: $HERE/$CONFIG not found (VARIANT=smoke selects the smoke config; leave VARIANT unset for the real run)" >&2; exit 1; }
check_duet
OUTDIR="$(cfg_path "$CONFIG" outdir)"
SCRATCH="$(cfg_path "$CONFIG" scratch_dir)"
LOGDIR="$OUTDIR/logs"
mapfile -t PANELS < <(python -c 'import sys, yaml; [print(p["id"]) for p in yaml.safe_load(open(sys.argv[1]))["panels"]]' "$CONFIG")
MAX_JOBS="${MAX_JOBS:-10}"
STEPS=("$@"); (( ${#STEPS[@]} )) || STEPS=(build panels duet crowding aggregate)
for s in "${STEPS[@]}"; do
    [[ " build panels check duet crowding aggregate " == *" $s "* ]] || { echo "ERROR: unknown step $s" >&2; exit 1; }
done

if [[ " ${STEPS[*]} " == *" build "* ]]; then
    for key in abc_stats_h5 abc_gene_csv; do
        f="$(cfg_path "$CONFIG" "$key")"
        [[ -e "$f" ]] || { echo "ERROR: $key $f not found (not in git). Download the atlas files into" \
            "data/raw/WMB-10X/ with the commands in docs/reproducing_the_paper.md (Inputs not in the" \
            "repository, Atlas files); experiments/INPUTS.md lists their sha256." >&2; exit 1; }
    done
    SCANPY_PYTHON="${SCANPY_PYTHON:-$("${CONDA_EXE:-conda}" run -n "$SCANPY_ENV" python -c 'import sys; print(sys.executable)')}"
fi
if [[ " ${STEPS[*]} " =~ \ (build|panels|duet|crowding)\  ]]; then
    f="$(cfg_path "$CONFIG" prior_csv)"
    [[ -e "$f" ]] || { echo "ERROR: prior_csv $f not found (WMB-10X atlas values, not shipped). Build it" \
        "with python scripts/wmb10x/build_whole_brain_cpm.py in the scanpy environment." >&2; exit 1; }
fi
TEMPLATE="$(cfg_path "$CONFIG" duet_template)"
[[ -f "$TEMPLATE" ]] || { echo "ERROR: duet_template $TEMPLATE not found" >&2; exit 1; }
mkdir -p "$LOGDIR" "$SCRATCH"

echo "=== merfish_expression_prior ($CONFIG): ${#PANELS[@]} panels, steps ${STEPS[*]}, MAX_JOBS $MAX_JOBS === started $(date)"
for step in "${STEPS[@]}"; do
    echo "=== $step === $(date)"
    case "$step" in
        build)     script -qefc "\"$SCANPY_PYTHON\" build_per_class_expression.py --config \"$CONFIG\"" "$LOGDIR/build.raw.log"
                   clean_log "$LOGDIR/build.raw.log" "$LOGDIR/build.log" ;;
        panels)    python make_panels.py --config "$CONFIG" 2>&1 | tee "$LOGDIR/panels.log" ;;
        check)     python check_pep_cache.py --config "$CONFIG" 2>&1 | tee "$LOGDIR/check.log" ;;
        duet)      python check_pep_cache.py --config "$CONFIG" > "$LOGDIR/check.log" 2>&1 \
                       || { cat "$LOGDIR/check.log" >&2; exit 1; }
                   nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable"
                   run_panels duet duet/summary.yaml \
                       python "$REPO_ROOT/scripts/benchmark/run_merfish.py" \
                       --config "$OUTDIR/@PANEL@/run_config.yaml" -v ;;
        crowding)  run_panels crowding crowding/per_class_metrics.csv \
                       python evaluate_crowding_by_cell_type.py \
                       --config "$OUTDIR/@PANEL@/crowding_config.yaml" ;;
        aggregate) python aggregate_panels.py --config "$CONFIG" 2>&1 | tee "$LOGDIR/aggregate.log" ;;
    esac
done
echo "=== merfish_expression_prior === finished $(date)"
echo "S4b panel: $OUTDIR/crowding cell type robustness.svg"
