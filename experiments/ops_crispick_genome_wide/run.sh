#!/usr/bin/env bash
# =============================================================================
# Fig 3d-f and Supp Fig S3c: genome-wide CRISPick CRISPR-ko libraries (20,114
# genes x 3 guides + 1,000 controls = 61,342 guides from 305,202 candidates,
# 14 rounds) under NIS-seq HeLa noise channels.
#   Four arms (one run each): DUET (+ Feldman, Sivanandan, max activity)
#   optimized under the symmetric, position-varying, asymmetric and
#   position-varying asymmetric channel: run_benchmark.py only.
#   cross_eval: every arm's DUET codebooks, plus Feldman et al. and maximum
#   activity, re-scored under the position-varying asymmetric channel:
#   compare_benchmarks.py.
#   visualize: the re-visualization pass (visualize_benchmark.py per arm,
#   then visualize_comparison.py for the cross-eval).
#
# Source bundles: scripts/benchmark/archive/04-02-2026/ (nisseq_* configs,
# run_all.sh, cross_eval/eval_positional_channel.yaml) and
# scripts/benchmark/archive/04-24-2026/nisseq_revisualization/
# (rerun_visualization.sh, cross_eval/eval_positional_channel.yaml).
#
# Activate an env whose `duet` resolves to this repo first (checked below). The
# Feldman baseline shells out to the conda env named by `feldman.conda_env`
# (`ops`). Works from any cwd.
#   bash experiments/ops_crispick_genome_wide/run.sh              # all steps
#   bash experiments/ops_crispick_genome_wide/run.sh cross_eval visualize
#   GW_STEPS="symmetric visualize" bash experiments/ops_crispick_genome_wide/run.sh
#   bash experiments/ops_crispick_genome_wide/run.sh check        # guards only
#   VARIANT=smoke bash experiments/ops_crispick_genome_wide/run.sh
#       (reads the <name>.smoke.yaml files beside the configs: short runs)
#
# Inputs not in git, checked before any work (also by `check`): the CRISPick
# candidate table (candidate_pool.csv_path; build it with
# scripts/data_processing/build_crispick_candidates.py) and, for each arm and
# the cross-evaluation, the NIS-seq channel matrix its config reads (included
# only with the permission of the NIS-seq authors).
#
# Steps, always run in this order: position_varying_asymmetric symmetric
# position_varying asymmetric cross_eval visualize. Name steps on the command
# line or in GW_STEPS (command line wins; default "all"). An unknown step exits
# 2 before any work. `check` runs every guard for the selected steps, prints
# the resolved paths and exits. The arm and cross_eval steps stop the script at
# the first failure; visualize renders everything it can and exits non-zero at
# the end if an input was missing or a visualizer failed.
#
# Home-drive guard: the arm and cross_eval steps refuse to start if a
# cache_dir or scratch_dir resolves under $HOME, because one arm writes about
# 0.5 TB of PEP cache and up to about 0.5 TB of evaluation scratch. Point
# those paths (or the results/cache and results/scratch/<name> symlinks) at a
# large data volume, or set ALLOW_HOME_SCRATCH=1. VARIANT=smoke is exempt.
#
# Outputs: each config's outdir under results/experiments/ops_crispick_genome_wide/
# (<arm>/ and cross_eval/eval_position_varying_asymmetric/); logs:
# results/experiments/ops_crispick_genome_wide/logs/.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
SFX="${VARIANT:+.$VARIANT}"
ARMS=(position_varying_asymmetric symmetric position_varying asymmetric)
EVAL=eval_position_varying_asymmetric
ALL_STEPS=("${ARMS[@]}" cross_eval visualize)
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
not_on_home() {  # $1 description, $2 path: fails if the path resolves under $HOME
    local p h
    p="$(realpath -m "$2")"; h="$(realpath -m "$HOME")"
    if [[ $p == "$h" || $p == "$h"/* ]]; then
        echo "ERROR: $1 resolves to $p, under \$HOME ($h). The genome-wide steps write" \
             "hundreds of GB there (per arm about 0.5 TB of PEP cache and up to about 0.5 TB of" \
             "evaluation scratch); point it at a large data volume, or set ALLOW_HOME_SCRATCH=1" \
             "to run anyway." >&2
        return 1
    fi
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
MISSING_CSV=() MISSING_NISSEQ=()
declare -A SEEN_CSV=() SEEN_NISSEQ=()
check_matrices() {  # $1 config: record each channel matrix it names that is missing
    local f found=0
    while IFS= read -r f; do
        [[ -n $f ]] || continue
        found=1
        if [[ ! -f $f && -z ${SEEN_NISSEQ[$f]:-} ]]; then SEEN_NISSEQ[$f]=1; MISSING_NISSEQ+=("$f ($1)"); fi
    done < <(channel_files "$1")
    (( found )) || { echo "ERROR: could not read the channel matrix paths from $1" >&2; exit 1; }
}
check_sources() {  # $1 eval config: every source results.csv exists; every source pool equals reference_dir's
    python -c 'import filecmp, pathlib, sys, yaml
d = yaml.safe_load(open(sys.argv[1])); ref = pathlib.Path(d["reference_dir"])
refs = sorted(ref.glob("trial_*/guides.csv")); errs = []
if not refs: errs.append(f"no trial_*/guides.csv in reference_dir {ref.resolve()}")
for p in dict.fromkeys(pathlib.Path(m["path"]) for m in d["methods"]):
    if not p.is_file():
        errs.append(f"missing {p.resolve()}"); continue
    for r in refs:
        g = p.parent / r.parent.name / "guides.csv"
        if not g.is_file() or not filecmp.cmp(g, r, shallow=False):
            errs.append(f"{g.resolve()} is missing or differs from {r.resolve()} (source pools must equal the reference pool)")
sys.exit("ERROR: cross_eval inputs not ready:\n  " + "\n  ".join(errs) if errs else 0)' "$1"
}

# ---- step selection ---------------------------------------------------------
if (( $# )); then REQ=("$@"); else read -r -a REQ <<< "${GW_STEPS:-all}"; fi
CHECK_ONLY=0
declare -A WANT=()
for s in "${REQ[@]}"; do
    case " all check ${ALL_STEPS[*]} " in
        *" $s "*) ;;
        *) echo "ERROR: unknown step '$s' (steps: ${ALL_STEPS[*]}; also all, check)" >&2; exit 2 ;;
    esac
    if [[ $s == all ]]; then for a in "${ALL_STEPS[@]}"; do WANT[$a]=1; done
    elif [[ $s == check ]]; then CHECK_ONLY=1
    else WANT[$s]=1; fi
done
if (( ${#WANT[@]} == 0 )); then   # `check` alone checks all steps
    for a in "${ALL_STEPS[@]}"; do WANT[$a]=1; done
fi
STEPS=()
for a in "${ALL_STEPS[@]}"; do
    if [[ -n ${WANT[$a]:-} ]]; then STEPS+=("$a"); fi
done

# ---- guards (before any work) ----------------------------------------------
cd "$HERE"
for c in "${ARMS[@]/%/$SFX.yaml}" "$EVAL$SFX.yaml"; do
    [[ -f "$c" ]] || { echo "ERROR: $HERE/$c not found (VARIANT=smoke selects the smoke configs; leave VARIANT unset for the real run)" >&2; exit 1; }
done
check_duet
# Logs go beside the arm outdirs: <experiment results dir>/logs/.
EXP_OUT="$(dirname "$(realpath -m "$(yaml_get "${ARMS[0]}$SFX.yaml" outdir)")")"
LOGDIR="$EXP_OUT/logs"
EVAL_OUT="$(realpath -m "$(yaml_get "$EVAL$SFX.yaml" outdir)")"
GUARD_HOME=1
if [[ ${VARIANT:-} == smoke ]]; then GUARD_HOME=0
elif [[ ${ALLOW_HOME_SCRATCH:-0} == 1 ]]; then GUARD_HOME=0; echo "WARNING: ALLOW_HOME_SCRATCH=1: the home-drive guard is off" >&2
fi
HEAVY=0
for s in "${STEPS[@]}"; do
    case $s in
        cross_eval)
            HEAVY=1
            check_matrices "$EVAL$SFX.yaml"
            [[ $GUARD_HOME == 0 ]] || not_on_home "$EVAL$SFX.yaml evaluator.scratch_dir" "$(yaml_get "$EVAL$SFX.yaml" evaluator.scratch_dir)" ;;
        visualize) ;;
        *)
            HEAVY=1
            csv="$(yaml_get "$s$SFX.yaml" candidate_pool.csv_path)"
            csv="$(realpath -m "$csv")"
            if [[ ! -f $csv && -z ${SEEN_CSV[$csv]:-} ]]; then SEEN_CSV[$csv]=1; MISSING_CSV+=("$csv ($s$SFX.yaml candidate_pool.csv_path)"); fi
            check_matrices "$s$SFX.yaml"
            if [[ $GUARD_HOME == 1 ]]; then
                not_on_home "$s$SFX.yaml cache_dir" "$(yaml_get "$s$SFX.yaml" cache_dir)"
                not_on_home "$s$SFX.yaml scratch_dir" "$(yaml_get "$s$SFX.yaml" scratch_dir)"
            fi
            ops_env="$(yaml_get "$s$SFX.yaml" feldman.conda_env)" ;;
    esac
done
if (( ${#MISSING_CSV[@]} )); then
    printf 'ERROR: CRISPick candidate table not found: %s\n' "${MISSING_CSV[@]}" >&2
    cat >&2 <<'MSG'
The table is not shipped with DUET (Broad Institute GPP terms of use). Build it
from the three CRISPick downloads in data/raw/CRISPick/ with
    python scripts/data_processing/build_crispick_candidates.py --raw-dir data/raw/CRISPick
(see docs/reproducing_the_paper.md, Inputs not in the repository).
MSG
fi
if (( ${#MISSING_NISSEQ[@]} )); then
    printf 'ERROR: NIS-seq channel matrix not found: %s\n' "${MISSING_NISSEQ[@]}" >&2
    cat >&2 <<'MSG'
The NIS-seq channel matrices are fitted from spot-level base calls that the
authors of Fandrey et al. (2025) shared on request, and are included only with
their permission. If one is missing from your copy, rebuild it with
experiments/nisseq_error_analysis from the spot calls, which are available from
those authors on request. Run it with SUBSAMPLE_NS=100000: each
channels/archive/nisseq_hela_<kind>_subsample.npy is its
nisseq_hela_<kind>_pctl50_subsample100000.npy.
MSG
fi
(( ${#MISSING_CSV[@]} + ${#MISSING_NISSEQ[@]} == 0 )) || exit 1
if [[ -n ${ops_env:-} ]]; then  # Feldman runs after DUET (hours in); fail now if its env is broken
    conda run -n "$ops_env" python -c 'import ops.pool_design' \
        || { echo "ERROR: conda env '$ops_env' (feldman.conda_env) cannot import ops.pool_design" >&2; exit 1; }
fi

if (( CHECK_ONLY )); then
    echo "=== ops_crispick_genome_wide check (suffix '${SFX}'): steps ${STEPS[*]} ==="
    for s in "${STEPS[@]}"; do
        case $s in
            cross_eval) echo "cross_eval: outdir $EVAL_OUT; scratch $(realpath -m "$(yaml_get "$EVAL$SFX.yaml" evaluator.scratch_dir)")" ;;
            visualize) echo "visualize: arm outdirs under $EXP_OUT; cross-eval outdir $EVAL_OUT" ;;
            *) echo "$s: outdir $(realpath -m "$(yaml_get "$s$SFX.yaml" outdir)"); cache $(realpath -m "$(yaml_get "$s$SFX.yaml" cache_dir)"); scratch $(realpath -m "$(yaml_get "$s$SFX.yaml" scratch_dir)")" ;;
        esac
    done
    echo "logs: $LOGDIR; all guards passed"
    exit 0
fi

mkdir -p "$LOGDIR"
echo "=== ops_crispick_genome_wide (suffix '${SFX}'): steps ${STEPS[*]} === started $(date)"
if (( HEAVY )); then
    nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "WARNING: nvidia-smi unavailable; configs request device=gpu:all"
fi

# ---- steps -------------------------------------------------------------------
run_arm() {  # run_benchmark.py only; visualization is the visualize step
    local arm="$1" cfg="$1$SFX.yaml" scratch
    scratch="$(realpath -m "$(yaml_get "$cfg" scratch_dir)")"
    mkdir -p "$scratch/tmp"
    export TMPDIR="$scratch/tmp"
    echo "=== $arm === started $(date)"
    script -qefc "python \"$REPO_ROOT/scripts/benchmark/run_benchmark.py\" --config \"$cfg\" -vv" "$LOGDIR/$arm.raw.log"
    clean_log "$LOGDIR/$arm.raw.log" "$LOGDIR/$arm.log"
    echo "=== $arm === finished $(date)"
}

# compare_benchmarks.py writes CSVs only (visualize_comparison.py plots them).
# Both use a bare `from comparison import`, so they run as `python <path>`.
run_cross_eval() {
    local cfg="$EVAL$SFX.yaml" scratch
    check_sources "$cfg"
    scratch="$(realpath -m "$(yaml_get "$cfg" evaluator.scratch_dir)")"
    mkdir -p "$scratch/tmp"
    export TMPDIR="$scratch/tmp"
    echo "=== cross_eval: $EVAL === started $(date)"
    script -qefc "python \"$REPO_ROOT/scripts/benchmark/compare_benchmarks.py\" --config \"$cfg\" -vv" "$LOGDIR/$EVAL.raw.log"
    clean_log "$LOGDIR/$EVAL.raw.log" "$LOGDIR/$EVAL.log"
    echo "=== cross_eval: $EVAL === finished $(date)"
}

# Re-visualization pass (as rerun_visualization.sh): every arm whose results.csv
# exists, then the cross-eval panels if its aggregated_metrics.csv exists.
VIS_RC=0
run_visualize() {
    local arm cfg out problems=()
    for arm in "${ARMS[@]}"; do
        cfg="$arm$SFX.yaml"
        out="$(realpath -m "$(yaml_get "$cfg" outdir)")"
        if [[ ! -f "$out/results.csv" ]]; then problems+=("missing $out/results.csv"); continue; fi
        echo "=== visualize: $arm === $(date)"
        python "$REPO_ROOT/scripts/benchmark/visualize_benchmark.py" --config "$cfg" \
            2>&1 | tee "$LOGDIR/$arm.visualize.log" || problems+=("visualize_benchmark.py failed for $arm (see $LOGDIR/$arm.visualize.log)")
    done
    if [[ -f "$EVAL_OUT/aggregated_metrics.csv" ]]; then
        echo "=== visualize: $EVAL === $(date)"
        python "$REPO_ROOT/scripts/benchmark/visualize_comparison.py" --config "$EVAL$SFX.yaml" \
            2>&1 | tee "$LOGDIR/$EVAL.visualize.log" || problems+=("visualize_comparison.py failed (see $LOGDIR/$EVAL.visualize.log)")
    else
        problems+=("missing $EVAL_OUT/aggregated_metrics.csv")
    fi
    if (( ${#problems[@]} )); then
        printf 'ERROR: visualize: %s\n' "${problems[@]}" >&2
        VIS_RC=1
    fi
}

for s in "${STEPS[@]}"; do
    case $s in
        cross_eval) run_cross_eval ;;
        visualize) run_visualize ;;
        *) run_arm "$s" ;;
    esac
done

echo "=== ops_crispick_genome_wide === finished $(date)"
PVA_OUT="$(realpath -m "$(yaml_get "${ARMS[0]}$SFX.yaml" outdir)")"
for panel in \
    "Fig 3d:$PVA_OUT/pareto_fronts/Mean_decode_accuracy_aggregated.svg" \
    "Fig 3e:$PVA_OUT/pareto_fronts/10th_percentile_decode_accuracy_aggregated.svg" \
    "Fig 3f:$PVA_OUT/pareto_fronts/95th_over_5th_percentile_decode_accuracy_aggregated.svg" \
    "S3c:$EVAL_OUT/pareto_fronts/Mean_decode_accuracy_aggregated.svg"; do
    f="${panel#*:}"
    echo "${panel%%:*} panel: $f$([[ -f $f ]] || echo '  (not rendered yet)')"
done
exit "$VIS_RC"
