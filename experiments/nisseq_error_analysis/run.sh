#!/usr/bin/env bash
# =============================================================================
# Supp Fig S3a: NIS-seq HeLa IL-1b error analysis, and the provenance of the
# NIS-seq noise matrices behind Supp S3b and Fig 3d-f / Supp S3c.
# For each subsample size N (reads per file, from the config or SUBSAMPLE_NS):
#   1. 02_error_analysis.ipynb run headless (nbconvert --execute), CPU only:
#      matches the first N reads of each of the 8 files to Brunello, fits the
#      four channels at the 50th-percentile intensity threshold, saves the
#      figures and the four .npy matrices;
#   2. check_provenance.py: the four matrices must equal the reference ones for
#      that N byte for byte. Sizes the config does not list are skipped, and so
#      is each reference file missing from your copy (the NIS-seq channel
#      matrices are included only with the permission of the NIS-seq authors).
# A notebook failure stops the script. A provenance failure does not: every
# size still runs, and the script exits 1 at the end, naming the failed sizes.
#
# The notebook here is a parameterized copy of the original analysis notebook
# (cell 1 only). Its helper module error_utils.py sits beside it.
#
# Activate an env whose `duet` resolves to this repo first (checked below); the
# notebook kernel runs that env's python. Works from any cwd.
#   bash experiments/nisseq_error_analysis/run.sh
#   VARIANT=smoke bash experiments/nisseq_error_analysis/run.sh
#       (reads config.smoke.yaml beside config.yaml: a short check of the pipeline)
# Environment overrides:
#   SUBSAMPLE_NS         sizes to run, e.g. "5000" (default: the config's
#                        subsample_ns, "5000 50000 100000"); "none" = all reads
#                        (untested; hours and a very large dict of ties)
#   NISSEQ_DATA_DIR      data folder (default: the config's data_dir)
#   NISSEQ_LIBRARY_PATH  Brunello file (default: the config's library_path)
#   CHECK_INPUT_HASHES   0 skips the input sha256 check (existence is still checked)
#   NISSEQ_CELL_TIMEOUT  per-cell nbconvert timeout in seconds (default 7200; the
#                        slowest cell takes about 610 s at N = 100,000; -1 = none),
#                        so a stuck kernel cannot block an unattended launch
#
# Outputs: <outdir>/subsample<N>/ (figures/, noise_matrices/ and the executed
# notebook), outdir = results/experiments/nisseq_error_analysis/; logs:
# <outdir>/logs/. S3a panel: subsample5000/figures/error_lines_4_pctl50_subsample5000.svg.
# CPU only, about 10 GB peak RSS; see README.md for the run time.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
CONFIG="config${VARIANT:+.$VARIANT}.yaml"
NOTEBOOK=02_error_analysis.ipynb
EXECUTED=02_error_analysis.executed.ipynb
# Relative overrides are relative to the caller's cwd: resolve them before cd.
if [[ -n "${NISSEQ_DATA_DIR:-}" ]]; then NISSEQ_DATA_DIR="$(realpath -m "$NISSEQ_DATA_DIR")"; fi
if [[ -n "${NISSEQ_LIBRARY_PATH:-}" ]]; then NISSEQ_LIBRARY_PATH="$(realpath -m "$NISSEQ_LIBRARY_PATH")"; fi
export PYTHONUNBUFFERED=1
# The kernel imports error_utils.py from this folder: leave no __pycache__ here.
export PYTHONDONTWRITEBYTECODE=1

clean_log() {
    sed 's/\r$//' "$1" | sed $'s/.*\r//' | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tail -n +2 | head -n -1 > "$2"
}
yaml_get() {  # $1 config, $2 dotted key; fails if the key is missing or null
    python -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d = d[k]
sys.exit(f"{sys.argv[2]} is null in {sys.argv[1]}") if d is None else print(d)' "$1" "$2"
}
yaml_words() {  # $1 config, $2 key of a list: its items, space-separated
    python -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]
sys.exit(f"{sys.argv[2]} is empty in {sys.argv[1]}") if not d else print(*d)' "$1" "$2"
}
check_duet() {  # the active env must import duet from this repo's src/duet/
    python -c 'import sys, pathlib, duet; p = pathlib.Path(duet.__file__).resolve(); r = pathlib.Path(sys.argv[1]).resolve()
sys.exit(0 if r in p.parents else f"ERROR: the active env imports duet from {p}, not from {r}")' "$REPO_ROOT/src/duet"
}
input_manifest() {  # "<sha256>  <path>" per input (sha256sum -c format): the library, then the spot files
    python -c 'import sys, os, yaml; d = yaml.safe_load(open(sys.argv[1]))["input_sha256"]
for k, h in d.items(): print(h, sys.argv[3] if k == "library" else os.path.join(sys.argv[2], k), sep="  ")' \
        "$CONFIG" "$DATA_DIR" "$LIBRARY_PATH"
}
nb_outputs() {  # $1 executed notebook: what each code cell printed (stdout, results), with its run time
    python -c 'import sys, nbformat, datetime as dt
nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    t = c.metadata.get("execution", {})
    a, b = t.get("iopub.execute_input"), t.get("shell.execute_reply")
    secs = f"{(dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).total_seconds():.1f} s" if a and b else "?"
    txt = [o.text for o in c.outputs if o.output_type == "stream" and o.name == "stdout"]
    txt += [o.data["text/plain"] for o in c.outputs if o.output_type == "execute_result" and "text/plain" in o.data]
    print(f"--- cell {i} ({secs}) ---")
    if txt: print("".join(txt).rstrip())' "$1"
}

cd "$HERE"
[[ -f "$CONFIG" ]] || { echo "ERROR: $HERE/$CONFIG not found (VARIANT=smoke selects the smoke config; leave VARIANT unset for the real run)" >&2; exit 1; }
check_duet
OUTDIR="$(realpath -m "$(yaml_get "$CONFIG" outdir)")"
DATA_DIR="${NISSEQ_DATA_DIR:-$(realpath -m "$(yaml_get "$CONFIG" data_dir)")}"
LIBRARY_PATH="${NISSEQ_LIBRARY_PATH:-$(realpath -m "$(yaml_get "$CONFIG" library_path)")}"
SUBSAMPLE_NS="${SUBSAMPLE_NS:-$(yaml_words "$CONFIG" subsample_ns)}"
CELL_TIMEOUT="${NISSEQ_CELL_TIMEOUT:-7200}"
[[ "$CELL_TIMEOUT" == -1 || "$CELL_TIMEOUT" =~ ^[1-9][0-9]*$ ]] \
    || { echo "ERROR: NISSEQ_CELL_TIMEOUT must be a positive number of seconds or -1, not '$CELL_TIMEOUT'" >&2; exit 1; }
LOGDIR="$OUTDIR/logs"
for N in $SUBSAMPLE_NS; do
    [[ "$N" == none || "$N" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: bad subsample size '$N' in SUBSAMPLE_NS" >&2; exit 1; }
done

# Inputs are not in git (see README.md): check they exist, then their sha256.
MANIFEST="$(input_manifest)"
missing=0
while IFS= read -r line; do
    f="${line#*  }"
    [[ -f "$f" ]] || { echo "ERROR: input $f not found (not in git; see $HERE/README.md, Inputs)" >&2; missing=1; }
done <<< "$MANIFEST"
[[ $missing -eq 0 ]] || exit 1
mkdir -p "$LOGDIR"

echo "=== nisseq_error_analysis ($CONFIG; N = $SUBSAMPLE_NS) === started $(date)"
if [[ "${CHECK_INPUT_HASHES:-1}" != 0 ]]; then
    echo "=== Input sha256 === $(date)"
    sha256sum -c - <<< "$MANIFEST" 2>&1 | tee "$LOGDIR/inputs.log" \
        || { echo "ERROR: an input differs from the sha256 in $CONFIG (CHECK_INPUT_HASHES=0 skips this check)" >&2; exit 1; }
fi

export NISSEQ_DATA_DIR="$DATA_DIR" NISSEQ_LIBRARY_PATH="$LIBRARY_PATH"
export NISSEQ_ERROR_UTILS_DIR="$HERE"
export NISSEQ_SAVE_NOISE_MODEL=1 NISSEQ_SAVE_FIGURES=1
PROV_FAILED=()   # sizes whose provenance check failed; reported (exit 1) after every size has run
for N in $SUBSAMPLE_NS; do
    if [[ "$N" == none ]]; then TAG=full; else TAG="subsample$N"; fi
    NDIR="$OUTDIR/$TAG"
    mkdir -p "$NDIR"
    export NISSEQ_SUBSAMPLE_N="$N" NISSEQ_OUTDIR="$NDIR"
    START=$SECONDS
    echo "=== $TAG: notebook === started $(date)"
    # nbconvert starts the kernel in this folder (the notebook's), with this env's python.
    script -qefc "python -m nbconvert --to notebook --execute \"$NOTEBOOK\" --ExecutePreprocessor.timeout=$CELL_TIMEOUT --output-dir \"$NDIR\" --output \"$EXECUTED\"" "$LOGDIR/$TAG.raw.log"
    clean_log "$LOGDIR/$TAG.raw.log" "$LOGDIR/$TAG.log"
    nb_outputs "$NDIR/$EXECUTED" > "$LOGDIR/$TAG.outputs.log"
    echo "=== $TAG: notebook finished in $(( (SECONDS - START) / 60 ))m$(( (SECONDS - START) % 60 ))s; cell outputs in $LOGDIR/$TAG.outputs.log"
    echo "=== $TAG: provenance check === $(date)"
    if [[ "$N" == none ]]; then
        echo "PROVENANCE CHECK SKIPPED: no reference matrices come from all reads" | tee "$LOGDIR/$TAG.provenance.log"
    else
        python check_provenance.py --config "$CONFIG" --n "$N" --matrices-dir "$NDIR/noise_matrices" \
            2>&1 | tee "$LOGDIR/$TAG.provenance.log" \
            || { PROV_FAILED+=("$N"); echo "=== $TAG: provenance check FAILED; carrying on with the other sizes"; }
    fi
done

echo "=== nisseq_error_analysis === finished $(date)"
echo "Outputs: $OUTDIR"
if [[ " $SUBSAMPLE_NS " == *" 5000 "* ]]; then
    echo "S3a panel: $OUTDIR/subsample5000/figures/error_lines_4_pctl50_subsample5000.svg"
fi
if [[ ${#PROV_FAILED[@]} -gt 0 ]]; then
    echo "ERROR: PROVENANCE CHECK FAILED for N = ${PROV_FAILED[*]} (see $LOGDIR/subsample<N>.provenance.log)" >&2
    exit 1
fi
echo "Provenance: no size failed its check (see $LOGDIR/subsample<N>.provenance.log)"
