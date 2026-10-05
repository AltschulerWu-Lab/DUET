#!/usr/bin/env bash
# =============================================================================
# Run every paper experiment in dependency order. Each experiment's run.sh is
# also runnable on its own. ops_crispick_genome_wide runs as one entry per step
# of its run.sh (`ops_crispick_genome_wide:<step>`, run as `run.sh <step>`), so
# a failed arm costs that arm, not the other arms.
#
# Activate an env whose `duet` resolves to this repo first (see README.md for
# the other two envs, `ops` and `scanpy_env`). Works from any cwd. Checks the
# inputs that are not in git before starting anything.
#   bash experiments/run_all.sh                        # every entry, in order
#   bash experiments/run_all.sh synthetic_hvr ...      # a subset, still in order
#   bash experiments/run_all.sh ops_crispick_genome_wide:cross_eval    # one step
#   VARIANT=smoke bash experiments/run_all.sh          # the smoke configs (short runs)
#   UNATTENDED=1 bash experiments/run_all.sh ...       # carry on past failures (below)
#   DRY_RUN=1 bash experiments/run_all.sh ...          # preflight and plan only; runs nothing
# A subset argument is an experiment (all of its entries) or one entry
# (experiment:step). Unknown names are rejected. Entries always run in the
# order below, whatever order they are named in.
#
# Entries: synthetic_objective_correlation, synthetic_hvr, ops_crispri_symmetric,
# ops_crispri_cross_eval, merfish_zhang2023_v2, merfish_2000_genes,
# merfish_expression_prior, wmb10x_landscape, nisseq_error_analysis,
# pep_grid_glyph, ops_crispri_rounds_error_sweep, then
# ops_crispick_genome_wide:{position_varying_asymmetric, symmetric,
# position_varying, asymmetric, cross_eval, visualize}.
#
# Dependencies. Hard (the dependent needs the other's outputs):
#   merfish_expression_prior           <- merfish_2000_genes
#   ops_crispick_genome_wide:cross_eval <- the four ops_crispick_genome_wide arms
# Soft (never block; their state is noted in the status file):
#   ops_crispri_rounds_error_sweep     <- ops_crispri_symmetric (its reference
#                                         check; SKIPPED if the outputs are absent)
#   ops_crispick_genome_wide:visualize <- the four arms and cross_eval (renders
#                                         what exists, exits non-zero at the end
#                                         if anything is missing)
# Dependencies count only among the entries selected in this invocation: an
# unselected one is assumed done, and each run.sh checks its own inputs.
#
# Default mode stops at the first failure and exits with that entry's code.
# Unattended mode (UNATTENDED=1) carries on: an entry whose hard dependency
# failed, was skipped or was interrupted is marked skipped-dependency, and
# every other entry still runs. It exits 0 only if every selected entry is ok.
#
# Status and logs (both modes): results/experiments/run_all/<id>/, or
# results/<VARIANT>/run_all/<id>/ with VARIANT set; <id> is RUN_ALL_ID or the
# start time, and run_all/latest points at the newest. The path is printed
# at start.
#   status.tsv    one row per selected entry: state (pending, running, ok,
#                 failed, skipped-dependency, interrupted), exit code, start and
#                 end time, elapsed, note, dependencies and log path. The `#`
#                 lines give the driver's pid, host, mode, HEAD and overall
#                 state. Rewritten atomically (mktemp + mv) on every change,
#                 so `cat` it at any time. Readable view:
#                   grep -v '^#' status.tsv | cut -f1-7 | column -t -s $'\t'
#   <entry>.log   the entry's stdout and stderr (':' becomes '.'), also shown on
#                 the terminal. Each experiment keeps its own logs as well.
#
# Stopping: SIGINT, SIGTERM or SIGHUP (Ctrl-C in the terminal, `kill <pid>`
# with the pid from the status file, `tmux kill-session`) stops the running
# entry and every process under it, marks it interrupted and exits 128+signal.
# The entry's process tree is recorded first; it gets SIGTERM, up to 30 s to
# exit, then SIGKILL for whatever remains (named in the driver note if even
# that fails).
#
# Cost: about 4 days for all twelve on the reference hardware, most of it the
# genome-wide runs (about 80 h); see README.md (Cost).
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$HERE")"
SFX="${VARIANT:+.$VARIANT}"
UNATTENDED="${UNATTENDED:-0}"
DRY_RUN="${DRY_RUN:-0}"
[[ $UNATTENDED == 0 || $UNATTENDED == 1 ]] || { echo "ERROR: UNATTENDED must be 0 or 1, not '$UNATTENDED'" >&2; exit 1; }
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || { echo "ERROR: DRY_RUN must be 0 or 1, not '$DRY_RUN'" >&2; exit 1; }

# ---- Entries in run order, with dependencies ---------------------------------
# entry <name> [hard deps] [soft deps]: space-separated entry names.
ORDER=()
declare -A HARD=() SOFT=()
entry() { ORDER+=("$1"); HARD[$1]="${2:-}"; SOFT[$1]="${3:-}"; }

entry synthetic_objective_correlation
entry synthetic_hvr
entry ops_crispri_symmetric
entry ops_crispri_cross_eval
entry merfish_zhang2023_v2
entry merfish_2000_genes
entry merfish_expression_prior merfish_2000_genes     # reuses its PEP cache and config
entry wmb10x_landscape
entry nisseq_error_analysis                           # Supp S3a (CPU, about 20 min)
entry pep_grid_glyph                                  # Fig 1b glyph (seconds)
entry ops_crispri_rounds_error_sweep "" ops_crispri_symmetric   # Fig 3c; checks against Fig 3b
GW=ops_crispick_genome_wide                           # Fig 3d-f, Supp S3c: one entry per step
GW_ARMS=(position_varying_asymmetric symmetric position_varying asymmetric)
GW_ARM_ENTRIES="${GW_ARMS[*]/#/$GW:}"
for a in "${GW_ARMS[@]}"; do entry "$GW:$a"; done
entry "$GW:cross_eval" "$GW_ARM_ENTRIES"
entry "$GW:visualize" "" "$GW_ARM_ENTRIES $GW:cross_eval"

exp_of() { printf '%s\n' "${1%%:*}"; }
step_of() { if [[ $1 == *:* ]]; then printf '%s\n' "${1#*:}"; fi; }

# ---- Selection ---------------------------------------------------------------
ARGS=("$@")
for want in "${ARGS[@]}"; do
    known=0
    for e in "${ORDER[@]}"; do
        if [[ $want == "$e" || $want == "$(exp_of "$e")" ]]; then known=1; break; fi
    done
    if [[ $known == 0 ]]; then
        echo "unknown experiment: $want" >&2
        echo "(an experiment folder, or one of: ${ORDER[*]})" >&2
        exit 1
    fi
done
entry_selected() {   # $1 entry; no arguments = all
    [[ ${#ARGS[@]} -eq 0 ]] && return 0
    local a
    for a in "${ARGS[@]}"; do [[ $a == "$1" || $a == "$(exp_of "$1")" ]] && return 0; done
    return 1
}
selected() {         # $1 experiment: true if any of its entries is selected
    local e
    for e in "${ORDER[@]}"; do [[ $(exp_of "$e") == "$1" ]] && entry_selected "$e" && return 0; done
    return 1
}
RUN=()
for e in "${ORDER[@]}"; do if entry_selected "$e"; then RUN+=("$e"); fi; done
in_run() { local x; for x in "${RUN[@]}"; do [[ $x == "$1" ]] && return 0; done; return 1; }

# ---- Status file and logs (not written with DRY_RUN=1) -----------------------
declare -A STATE=() T0=() T1=() RC=() NOTE=()
for e in "${RUN[@]}"; do STATE[$e]=pending; done
MODE="$([[ $UNATTENDED == 1 ]] && echo unattended || echo stop-on-first-failure)"
DRIVER_STATE=starting DRIVER_RC=- DRIVER_T0="$(date +%s)" DRIVER_T1=- DRIVER_NOTE=-
STATUS_FILE= STATUS_TMP= RUN_DIR=
entry_log() { printf '%s/%s.log\n' "$RUN_DIR" "${1//:/.}"; }
iso() { if [[ $1 == - ]]; then echo -; else date -d "@$1" +%FT%T%z; fi; }
dur() {
    if [[ $1 == - || $2 == - ]]; then echo -; return 0; fi
    local s=$(($2 - $1))
    printf '%d:%02d:%02d\n' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}
deps_of() {          # "hard: a b; soft: c", with the entry's own experiment prefix shortened to ':'
    local own out="" d h="" s=""
    own="$(exp_of "$1")"
    for d in ${HARD[$1]}; do h+=" ${d/#$own:/:}"; done
    for d in ${SOFT[$1]}; do s+=" ${d/#$own:/:}"; done
    [[ -n $h ]] && out="hard:$h"
    [[ -n $s ]] && out="${out:+$out; }soft:$s"
    printf '%s\n' "${out:--}"
}
counts() {
    local e n_ok=0 n_failed=0 n_skipped=0 n_int=0 n_running=0 n_pending=0
    for e in "${RUN[@]}"; do
        case ${STATE[$e]} in
            ok) n_ok=$((n_ok + 1)) ;;
            failed) n_failed=$((n_failed + 1)) ;;
            skipped-dependency) n_skipped=$((n_skipped + 1)) ;;
            interrupted) n_int=$((n_int + 1)) ;;
            running) n_running=$((n_running + 1)) ;;
            *) n_pending=$((n_pending + 1)) ;;
        esac
    done
    printf '%d ok, %d failed, %d skipped-dependency, %d interrupted, %d running, %d pending\n' \
        "$n_ok" "$n_failed" "$n_skipped" "$n_int" "$n_running" "$n_pending"
}
write_status() {     # the whole file to a temp file in the same folder, then rename: atomic
    [[ -n $STATUS_FILE ]] || return 0
    local e
    STATUS_TMP="$(mktemp "$STATUS_FILE.XXXXXX")" || return 1
    {
        printf '# run_all status: %s (rewritten atomically on every change)\n' "$STATUS_FILE"
        printf '# driver: %s, exit %s, pid %s on %s, started %s, finished %s; %s\n' "$DRIVER_STATE" "$DRIVER_RC" \
            "$$" "$(hostname)" "$(iso "$DRIVER_T0")" "$(iso "$DRIVER_T1")" "$(counts)"
        printf '# driver note: %s\n' "$DRIVER_NOTE"
        printf '# mode %s, variant %s, HEAD %s; args: %s; updated %s\n' "$MODE" "${VARIANT:--}" "$HEAD_REV" \
            "${ARGS[*]:-(all)}" "$(date +%FT%T%z)"
        printf 'entry\tstate\texit\tstarted\tfinished\telapsed\tnote\tdepends\tlog\n'
        for e in "${RUN[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$e" "${STATE[$e]}" "${RC[$e]:--}" \
                "$(iso "${T0[$e]:--}")" "$(iso "${T1[$e]:--}")" "$(dur "${T0[$e]:--}" "${T1[$e]:--}")" \
                "${NOTE[$e]:--}" "$(deps_of "$e")" "$(entry_log "$e")"
        done
    } > "$STATUS_TMP" && chmod 644 "$STATUS_TMP" && mv -f "$STATUS_TMP" "$STATUS_FILE" && STATUS_TMP= && return 0
    rm -f "$STATUS_TMP"; STATUS_TMP=
    return 1
}
status() { write_status || echo "WARNING: could not update $STATUS_FILE" >&2 || true; }
finish() {           # $1 driver state, $2 exit code, $3 note for entries that did not run
    local e
    DRIVER_STATE="$1"; DRIVER_RC="$2"; DRIVER_T1="$(date +%s)"
    for e in "${RUN[@]}"; do
        if [[ ${STATE[$e]} == pending ]]; then NOTE[$e]="${3:-not run}"; fi
    done
    status
}

if [[ $DRY_RUN == 0 ]]; then
    RUN_ALL_BASE="$REPO_ROOT/results/${VARIANT:-experiments}/run_all"
    mkdir -p "$RUN_ALL_BASE"
    if [[ -n ${RUN_ALL_ID:-} ]]; then
        [[ $RUN_ALL_ID =~ ^[A-Za-z0-9._-]+$ && $RUN_ALL_ID != latest && $RUN_ALL_ID != . && $RUN_ALL_ID != .. ]] \
            || { echo "ERROR: RUN_ALL_ID must be a plain name (letters, digits, . _ -), not '$RUN_ALL_ID'" >&2; exit 1; }
        RUN_DIR="$RUN_ALL_BASE/$RUN_ALL_ID"
        mkdir "$RUN_DIR" 2>/dev/null || { echo "ERROR: $RUN_DIR already exists; choose another RUN_ALL_ID" >&2; exit 1; }
    else
        id="$(date +%Y%m%d-%H%M%S)"; n=1
        RUN_DIR="$RUN_ALL_BASE/$id"
        until mkdir "$RUN_DIR" 2>/dev/null; do n=$((n + 1)); RUN_DIR="$RUN_ALL_BASE/$id-$n"; done
    fi
    ln -sfn "$(basename "$RUN_DIR")" "$RUN_ALL_BASE/latest"
    STATUS_FILE="$RUN_DIR/status.tsv"
    HEAD_REV="$(git -C "$REPO_ROOT" log --oneline -1 2>/dev/null | cut -c1-60 || true)"
    HEAD_REV="${HEAD_REV:-unknown}"
    DRIVER_NOTE="preflight"
    status
    echo "######## run_all ($MODE${VARIANT:+, VARIANT=$VARIANT}): ${#RUN[@]} entries; pid $$; status $STATUS_FILE ########"
fi

# ---- Stopping on a signal ----------------------------------------------------
CHILD= CURRENT=
descendants() { local c; for c in $(pgrep -P "$1" || true); do descendants "$c"; echo "$c"; done; }
kill_tree() {        # $1 signal, $2 pid: the pid and all its descendants
    local pids
    pids="$(descendants "$2")"
    # shellcheck disable=SC2086
    kill -s "$1" $pids "$2" 2> /dev/null || true
}
proc_id() {          # "<pid>:<start time>" of a live, non-zombie process; fails otherwise
    local st rest
    st="$(cat "/proc/$1/stat" 2> /dev/null)" || return 1
    rest="${st##*) }"                        # fields 3.. of /proc/<pid>/stat (the name may hold spaces)
    # shellcheck disable=SC2086
    set -- "$1" $rest                        # $2 = state, $21 = start time (field 22)
    [[ $2 != Z && $2 != X ]] || return 1
    printf '%s:%s\n' "$1" "${21}"
}
still_alive() {      # $@ "<pid>:<start time>" ids: those whose process still runs (not a reused pid)
    local id
    for id in "$@"; do [[ $(proc_id "${id%%:*}" 2> /dev/null) == "$id" ]] && printf '%s\n' "$id"; done
    return 0
}
stop_tree() {        # $1 pid: TERM its whole tree, 30 s grace, then KILL; prints the pids still alive
    local p i ids=() alive=()
    # Record the tree first: the ( run.sh | tee ) subshell dies at once on TERM,
    # and its descendants are then reparented out of reach of a later walk.
    for p in $(descendants "$1") "$1"; do
        if id="$(proc_id "$p")"; then ids+=("$id"); fi
    done
    [[ ${#ids[@]} -gt 0 ]] || return 0
    for id in "${ids[@]}"; do kill -s TERM "${id%%:*}" 2> /dev/null || true; done
    for i in $(seq 30); do
        mapfile -t alive < <(still_alive "${ids[@]}")
        [[ ${#alive[@]} -gt 0 ]] || return 0
        sleep 1
    done
    # KILL the survivors and anything they started since. The log tee ignores
    # TERM by design and exits on its own once its writers are gone, so it gets
    # 5 s to drain after the others die before it is killed too.
    for id in "${alive[@]}"; do
        p="${id%%:*}"
        [[ $(cat "/proc/$p/comm" 2> /dev/null) == tee ]] || kill_tree KILL "$p"
    done
    for i in 1 2 3 4 5; do
        mapfile -t alive < <(still_alive "${alive[@]}")
        [[ ${#alive[@]} -gt 0 ]] || return 0
        sleep 1
    done
    for id in "${alive[@]}"; do kill_tree KILL "${id%%:*}"; done
    sleep 1
    mapfile -t alive < <(still_alive "${alive[@]}")
    [[ ${#alive[@]} -eq 0 ]] || printf '%s\n' "${alive[@]%%:*}"
}
on_signal() {        # status first: the terminal or log pipe may already be gone
    trap '' INT TERM HUP PIPE
    local code=$((128 + $(kill -l "$1"))) left
    [[ -n $STATUS_TMP ]] && rm -f "$STATUS_TMP"
    if [[ -n $CHILD ]]; then
        STATE[$CURRENT]=interrupted; T1[$CURRENT]="$(date +%s)"; RC[$CURRENT]=$code
        NOTE[$CURRENT]="SIG$1 to run_all; stopping the entry"
        status
        left="$(stop_tree "$CHILD" | tr '\n' ' ')"
        wait "$CHILD" 2> /dev/null || true
        NOTE[$CURRENT]="SIG$1 to run_all"
        echo "######## $CURRENT ######## interrupted by SIG$1 $(date)" 2> /dev/null || true
    fi
    DRIVER_NOTE="interrupted by SIG$1${CURRENT:+ during $CURRENT}${left:+; still alive after SIGKILL: pid ${left% }}"
    finish interrupted "$code" "not run: interrupted"
    exit "$code"
}
on_exit() {          # any other exit while the driver is still marked starting or running
    local rc=$?
    if [[ $DRIVER_STATE == starting || $DRIVER_STATE == running ]]; then
        if [[ -n $CURRENT && ${STATE[$CURRENT]} == running ]]; then
            STATE[$CURRENT]=interrupted; T1[$CURRENT]="$(date +%s)"; NOTE[$CURRENT]="run_all stopped unexpectedly"
        fi
        DRIVER_NOTE="run_all stopped unexpectedly (exit $rc)${CURRENT:+ during $CURRENT}"
        finish failed "$rc" "not run: run_all stopped"
    fi
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
trap on_exit EXIT

# ---- Preflight ---------------------------------------------------------------
# Inputs that are not in git (see INPUTS.md), checked up front so a missing
# one does not surface hours into the run. Each check mirrors the one the
# experiment's own run.sh makes. need <path> <hint> records a missing input
# once, with how to get it; docs/reproducing_the_paper.md (Inputs not in the
# repository) has the details.
missing=()
declare -A seen_missing=()
need() {
    [[ -e $1 || -n ${seen_missing[$1]:-} ]] && return 0
    seen_missing[$1]=1
    missing+=("$1"$'\n'"       $2")
}
cfg_path() {         # $1 experiment folder, $2 config, $3 dotted key: the path it names, resolved against the folder
    (cd "$HERE/$1" && python -c 'import os, sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d = d[k]
print(os.path.realpath(d))' "$2" "$3")
}
cfg_matrices() {     # $1 experiment folder, $2.. configs: every channel matrix file they name, resolved
    (cd "$HERE/$1" && shift && python -c 'import os, sys, yaml
keys = {"channel_matrix_path", "epsilon_path", "channel_matrices_path"}
def walk(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in keys and isinstance(v, str): yield v
            else: yield from walk(v)
    elif isinstance(d, list):
        for v in d: yield from walk(v)
paths = {os.path.realpath(p) for c in sys.argv[1:] for p in walk(yaml.safe_load(open(c)))}
print(*sorted(paths), sep="\n")' "$@")
}
WMB_HINT="atlas values, not shipped: build it with python scripts/wmb10x/build_whole_brain_cpm.py (scanpy environment)"
ATLAS_HINT="download it into data/raw/WMB-10X/ (docs/reproducing_the_paper.md, Atlas files)"
NISSEQ_HINT="NIS-seq channel matrix, included only with the permission of the NIS-seq authors: rebuild it with experiments/nisseq_error_analysis (see below)"
nisseq_needed=0
# (A missing config is left to the experiment's run.sh to report.)
if selected merfish_zhang2023_v2 && [[ -f "$HERE/merfish_zhang2023_v2/config$SFX.yaml" ]]; then
    f="$(cfg_path merfish_zhang2023_v2 "config$SFX.yaml" genes.path)"
    need "$f" "build it with python scripts/data_processing/build_zhang2023_codebook.py --download"
    f="$(cfg_path merfish_zhang2023_v2 "config$SFX.yaml" expression.path)"
    need "$f" "$WMB_HINT"
fi
if selected merfish_2000_genes && [[ -f "$HERE/merfish_2000_genes/config$SFX.yaml" ]]; then
    f="$(cfg_path merfish_2000_genes "config$SFX.yaml" expression.path)"
    need "$f" "$WMB_HINT"
fi
if selected merfish_expression_prior && [[ -f "$HERE/merfish_expression_prior/config$SFX.yaml" ]]; then
    for key in abc_stats_h5 abc_gene_csv; do   # the paths its config names
        f="$(cfg_path merfish_expression_prior "config$SFX.yaml" "$key")"
        need "$f" "$ATLAS_HINT"
    done
    f="$(cfg_path merfish_expression_prior "config$SFX.yaml" prior_csv)"
    need "$f" "$WMB_HINT"
fi
if selected wmb10x_landscape && [[ " ${WMB10X_STEPS:-2} " == *" 2 "* ]]; then   # what step 2 reads
    for n in wmb_precomputed_stats.h5 wmb_gene.csv; do
        need "$REPO_ROOT/data/raw/WMB-10X/$n" "$ATLAS_HINT"
    done
fi
# The 8 NIS-seq spot files and the Brunello table, reached through the
# gitignored data/raw/NISseq_HeLa_IL1b link: every input_sha256 key of the
# config, with run.sh's NISSEQ_DATA_DIR / NISSEQ_LIBRARY_PATH overrides
# (relative to the caller's cwd). run.sh also checks their sha256.
if selected nisseq_error_analysis && [[ -f "$HERE/nisseq_error_analysis/config$SFX.yaml" ]]; then
    list="$(python -c 'import os, sys, yaml
here, cfg = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(os.path.join(here, cfg)))
data = os.environ.get("NISSEQ_DATA_DIR") or os.path.join(here, d["data_dir"])
lib = os.environ.get("NISSEQ_LIBRARY_PATH") or os.path.join(here, d["library_path"])
for k in d["input_sha256"]:
    print("library" if k == "library" else "spots", os.path.realpath(lib if k == "library" else os.path.join(data, k)), sep="\t")' \
        "$HERE/nisseq_error_analysis" "config$SFX.yaml")" \
        || { echo "ERROR: cannot read the inputs from nisseq_error_analysis/config$SFX.yaml" >&2; exit 1; }
    while IFS=$'\t' read -r kind f; do
        [[ -n $f ]] || continue
        if [[ $kind == library ]]; then
            need "$f" "the Brunello table: docs/reproducing_the_paper.md (NIS-seq data)"
        else
            need "$f" "NIS-seq spot-level base calls, available from the authors of Fandrey et al. (2025) on request"
        fi
    done <<< "$list"
fi
# The NIS-seq channel matrices the cross-evaluation configs read.
if selected ops_crispri_cross_eval; then
    cx=(symmetric position_varying asymmetric position_varying_asymmetric eval_position_varying_asymmetric)
    cx=("${cx[@]/%/$SFX.yaml}")
    if (cd "$HERE/ops_crispri_cross_eval" && ls "${cx[@]}" > /dev/null 2>&1); then
        while IFS= read -r f; do
            [[ -z $f ]] || { [[ -e $f ]] || nisseq_needed=1; need "$f" "$NISSEQ_HINT"; }
        done < <(cfg_matrices ops_crispri_cross_eval "${cx[@]}")
    fi
fi
# The genome-wide arms' CRISPick table and NIS-seq matrices are reported by
# `$GW/run.sh check` below, which also names the rebuild steps.
problems=()
if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'ERROR: input not found (not in git): %s\n' "${missing[@]}" >&2
    echo "See docs/reproducing_the_paper.md (Inputs not in the repository) and experiments/INPUTS.md." >&2
    if [[ $nisseq_needed == 1 ]]; then
        cat >&2 <<'MSG'
The NIS-seq channel matrices are fitted from spot-level base calls that the
authors of Fandrey et al. (2025) shared on request, and are included only with
their permission. If one is missing from your copy, rebuild it with
experiments/nisseq_error_analysis from the spot calls, which are available from
those authors on request (the run.sh of ops_crispri_cross_eval and of
ops_crispick_genome_wide says which subsample size gives which file).
MSG
    fi
    problems+=("${#missing[@]} input(s) not found")
fi
# The Feldman baseline shells out to the conda env its configs name
# (feldman.conda_env, `ops`) hours into a run: check that env imports the OPS code.
feldman_dirs=()
for x in ops_crispri_symmetric ops_crispri_cross_eval ops_crispri_rounds_error_sweep; do
    if selected "$x"; then feldman_dirs+=("$HERE/$x"); fi
done
for a in "${GW_ARMS[@]}"; do
    if in_run "$GW:$a"; then feldman_dirs+=("$HERE/$GW"); break; fi
done
if [[ ${#feldman_dirs[@]} -gt 0 ]]; then
    envs="$(python -c 'import glob, os, sys, yaml
sfx, envs = sys.argv[1], set()
for folder in sys.argv[2:]:
    for f in glob.glob(os.path.join(folder, "*.yaml")):
        if (sfx and not f.endswith(sfx + ".yaml")) or (not sfx and f.endswith(".smoke.yaml")):
            continue
        d = yaml.safe_load(open(f))
        e = (d.get("feldman") or {}).get("conda_env") if isinstance(d, dict) else None
        if e:
            envs.add(e)
print(*sorted(envs), sep="\n")' "$SFX" "${feldman_dirs[@]}")"
    while IFS= read -r env; do
        [[ -n $env ]] || continue
        if ! conda run -n "$env" python -c 'import ops.pool_design' < /dev/null > /dev/null 2>&1; then
            echo "ERROR: conda env '$env' (feldman.conda_env) cannot import ops.pool_design; see experiments/INPUTS.md" >&2
            problems+=("Feldman env '$env'")
        fi
    done <<< "$envs"
fi
# The genome-wide driver's own guards for the selected steps (configs, duet,
# candidate table, Feldman env, home-drive guard), without running anything.
gw_steps=()
for e in "${RUN[@]}"; do if [[ $(exp_of "$e") == "$GW" ]]; then gw_steps+=("$(step_of "$e")"); fi; done
if [[ ${#gw_steps[@]} -gt 0 ]]; then
    if ! out="$(bash "$HERE/$GW/run.sh" check "${gw_steps[@]}" 2>&1 < /dev/null)"; then
        printf '%s\n' "$out" >&2
        echo "ERROR: $GW/run.sh check ${gw_steps[*]} failed (above)" >&2
        problems+=("$GW guards")
    fi
fi
if [[ ${#problems[@]} -gt 0 ]]; then
    finish failed 1 "not run: preflight failed"
    DRIVER_NOTE="preflight failed: ${problems[*]}"; status
    exit 1
fi

if [[ $DRY_RUN == 1 ]]; then
    echo "run_all plan ($MODE${VARIANT:+, VARIANT=$VARIANT}): ${#RUN[@]} entries; preflight passed"
    i=0
    for e in "${RUN[@]}"; do
        i=$((i + 1))
        printf '%3d. %-52s %s\n' "$i" "$e" "$(deps_of "$e" | sed 's/^-$//')"
    done
    echo "DRY_RUN=1: nothing was run and no status file was written"
    exit 0
fi

# ---- Run ---------------------------------------------------------------------
not_ok() {           # $1 space-separated deps: "<dep>=<state>" for each selected one that is not ok
    local d
    for d in $1; do
        if in_run "$d" && [[ ${STATE[$d]} != ok ]]; then printf '%s=%s\n' "$d" "${STATE[$d]}"; fi
    done
    return 0
}
DRIVER_STATE=running DRIVER_NOTE=-
status
not_ok_count=0
for name in "${RUN[@]}"; do
    mapfile -t blocked < <(not_ok "${HARD[$name]}")
    mapfile -t soft < <(not_ok "${SOFT[$name]}")
    if [[ ${#blocked[@]} -gt 0 ]]; then
        STATE[$name]=skipped-dependency; NOTE[$name]="needs ${blocked[*]}"
        not_ok_count=$((not_ok_count + 1)); status
        echo "######## $name ######## skipped: needs ${blocked[*]} $(date)"
        continue
    fi
    if [[ ${#soft[@]} -gt 0 ]]; then NOTE[$name]="soft dependency not ok: ${soft[*]}"; fi
    exp="$(exp_of "$name")"; step="$(step_of "$name")"
    args=()
    if [[ -n $step ]]; then args=("$step"); fi
    log="$(entry_log "$name")"
    STATE[$name]=running; T0[$name]="$(date +%s)"; status
    echo "######## $name ######## started $(date)"
    CURRENT="$name"
    # In the background (with `wait`) so that a signal is handled at once; stdin
    # from /dev/null so nothing typed in the terminal reaches the entry; the log
    # tee ignores signals and drains until the entry's output closes.
    ( bash "$HERE/$exp/run.sh" "${args[@]}" < /dev/null 2>&1 \
        | { trap '' INT TERM HUP; exec tee -p -a "$log"; } ) &
    CHILD=$!
    rc=0; wait "$CHILD" || rc=$?
    CHILD=; CURRENT=; T1[$name]="$(date +%s)"; RC[$name]=$rc
    if [[ $rc -eq 0 ]]; then
        STATE[$name]=ok; status
        echo "######## $name ######## finished $(date)"
        continue
    fi
    STATE[$name]=failed; not_ok_count=$((not_ok_count + 1)); status
    echo "######## $name ######## FAILED (exit $rc) $(date); log: $log"
    if [[ $UNATTENDED != 1 ]]; then
        DRIVER_NOTE="stopped at the first failure ($name, exit $rc); UNATTENDED=1 carries on"
        finish failed "$rc" "not run: stopped at $name"
        exit "$rc"
    fi
done

if [[ $not_ok_count -eq 0 ]]; then
    finish ok 0
else
    DRIVER_NOTE="$not_ok_count of ${#RUN[@]} entries not ok"
    finish failed 1
fi
echo "######## run_all: $DRIVER_STATE, $(counts), $(date) ########"
grep -v '^#' "$STATUS_FILE" | cut -f1-7 | column -t -s $'\t' || true
exit "$DRIVER_RC"
