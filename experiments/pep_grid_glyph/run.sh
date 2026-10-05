#!/usr/bin/env bash
# =============================================================================
# Fig 1b: the pairwise-error-probability (PEP) grid glyph. Illustrative values:
# no data, no noise channel, no seed, no randomness.
#
# generate_pep_grid.py is a copy of the script that drew the paper's glyph (as
# run 2026-07-08), changed only to import duet.plotting and to write to the
# config's outdir instead of next to itself. Its default output should be
# byte-identical to the paper's glyph: the last step compares its sha256 with
# SHIPPED_SHA256 below and only warns if they differ.
#
# Activate an env whose `duet` resolves to this repo first (checked below).
# Works from any cwd.
#   bash experiments/pep_grid_glyph/run.sh
#   VARIANT=smoke bash experiments/pep_grid_glyph/run.sh
#       (reads config.smoke.yaml beside config.yaml: outputs under results/smoke/,
#       plus the corner-radius variants)
#
# Outputs: the config's outdir (results/experiments/pep_grid_glyph/);
# logs: <outdir>/logs/. CPU, about a second (mostly Python start-up). No PEP
# cache, no scratch.
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
CONFIG="config${VARIANT:+.$VARIANT}.yaml"
# sha256 of the glyph as drawn for the paper (pairwise_error_final.svg).
SHIPPED_SHA256=13b0473c6c7384308f939d9a16accd60cd0995032c556446aa0689266d7733a2
export PYTHONUNBUFFERED=1

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
RX_VARIANTS="$(yaml_get "$CONFIG" rx_variants)"
OUT_FILES=(pairwise_error_final.svg)
case "$RX_VARIANTS" in
    True)  RX_FLAG=(--rx-variants); OUT_FILES+=(pairwise_error_final_rx{6,4,3,2,0}.svg) ;;
    False) RX_FLAG=() ;;
    *) echo "ERROR: rx_variants in $CONFIG must be true or false, not '$RX_VARIANTS'" >&2; exit 1 ;;
esac
LOGDIR="$OUTDIR/logs"
LOG="$LOGDIR/generate.log"
mkdir -p "$LOGDIR"

echo "=== pep_grid_glyph ($CONFIG) === started $(date)"
python generate_pep_grid.py --outdir "$OUTDIR" ${RX_FLAG[@]+"${RX_FLAG[@]}"} 2>&1 | tee "$LOG"

# Provenance check: the default output should equal the paper's glyph byte for
# byte (sha256 SHIPPED_SHA256). A difference is reported, not fatal.
OUT="$OUTDIR/pairwise_error_final.svg"
(cd "$OUTDIR" && sha256sum "${OUT_FILES[@]}") | tee -a "$LOG"
OUT_SHA256="$(sha256sum "$OUT" | cut -d ' ' -f 1)"
if [[ "$OUT_SHA256" == "$SHIPPED_SHA256" ]]; then
    echo "OK: pairwise_error_final.svg is byte-identical to the paper's glyph (sha256 $SHIPPED_SHA256)" | tee -a "$LOG"
else
    echo "WARNING: pairwise_error_final.svg (sha256 $OUT_SHA256) differs from the paper's glyph (sha256 $SHIPPED_SHA256)" | tee -a "$LOG"
fi

echo "=== pep_grid_glyph === finished $(date)"
echo "PEP grid glyph: $OUT"
