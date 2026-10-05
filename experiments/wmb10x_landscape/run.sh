#!/usr/bin/env bash
# =============================================================================
# Supp Fig S4a: WMB-10X cell-type landscape (PCA + cosine UMAP of the 5,322
# Allen whole-mouse-brain 10x cluster centroids).
#
# Runs the four live scripts in scripts/wmb10x/ UNMODIFIED, in scanpy_env:
#   1. download_wmb_metadata.py  ~16 MB of ABC metadata -> data/raw/WMB-10X/ (skips files present)
#   2. build_centroid_matrix.py  h5 + metadata -> data/processed/WMB-10X/ (687 MB centroids.npy)
#   3. embed_centroids.py        PCA + UMAP + purity -> results/wmb10x/ (~6 min, CPU)
#   4. plot_embeddings.py        figures + captions -> results/wmb10x/figures/, captions.md
# The scripts derive every path from their own location and write to those
# fixed, gitignored repo paths (not results/experiments/). Only the logs go to
# results/experiments/wmb10x_landscape/logs/.
#
# Large inputs (never downloaded by this driver): wmb_precomputed_stats.h5
# (1.38 GB) and wmb_gene.csv, both in data/raw/WMB-10X/ (gitignored).
# scripts/wmb10x/download_wmb_metadata.py's resolve_large_input() looks for
# each there and otherwise DOWNLOADS it. When step 2 is requested, this driver
# refuses to start unless both files are in data/raw/WMB-10X/ (a dangling
# symlink counts as missing). merfish_expression_prior reads the same two
# files. Download them with the commands in docs/reproducing_the_paper.md
# (Inputs not in the repository, Atlas files); ../INPUTS.md lists their sha256.
#
# Works from any cwd; no conda env needs to be active.
#   bash experiments/wmb10x_landscape/run.sh
#   VARIANT=smoke bash experiments/wmb10x_landscape/run.sh        # steps 1-2 only
#   WMB10X_STEPS="3 4" bash experiments/wmb10x_landscape/run.sh   # any subset of steps
# Environment overrides:
#   SCANPY_ENV     conda env to run in (default scanpy_env; needs umap-learn)
#   SCANPY_PYTHON  interpreter (default: that env's python, called by path, not
#                  through `conda run`, per scripts/wmb10x/README.md)
#   WMB10X_STEPS   steps to run (default "1 2 3 4"; "1 2" when VARIANT=smoke)
#   VARIANT        if set, logs go to results/$VARIANT/wmb10x_landscape/logs/.
#                  There is no smoke config: the scripts take no parameters, so
#                  VARIANT=smoke runs steps 1-2 only (steps 3-4 are the full run).
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
NAME=wmb10x_landscape
SCANPY_ENV="${SCANPY_ENV:-scanpy_env}"
if [[ "${VARIANT:-}" == smoke ]]; then DEFAULT_STEPS="1 2"; else DEFAULT_STEPS="1 2 3 4"; fi
STEPS="${WMB10X_STEPS:-$DEFAULT_STEPS}"
# A relative SCANPY_PYTHON means relative to the caller's cwd: resolve before cd.
if [[ -n "${SCANPY_PYTHON:-}" ]]; then SCANPY_PYTHON="$(readlink -f "$(command -v "$SCANPY_PYTHON")")"; fi

clean_log() {
    sed 's/\r$//' "$1" | sed $'s/.*\r//' | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tail -n +2 | head -n -1 > "$2"
}

cd "$HERE"
LOGDIR="$REPO_ROOT/results/${VARIANT:-experiments}/$NAME/logs"
SCRATCH="$REPO_ROOT/results/${VARIANT:+$VARIANT/}scratch/$NAME"
CACHE="$REPO_ROOT/results/${VARIANT:+$VARIANT/}cache/$NAME"
mkdir -p "$LOGDIR" "$SCRATCH/tmp" "$SCRATCH/mpl" "$CACHE/numba"

SCANPY_PYTHON="${SCANPY_PYTHON:-$("${CONDA_EXE:-conda}" run -n "$SCANPY_ENV" python -c 'import sys; print(sys.executable)')}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"   # duet.plotting for step 4 (duet is not installed in scanpy_env)
export MPLCONFIGDIR="$SCRATCH/mpl" TMPDIR="$SCRATCH/tmp" NUMBA_CACHE_DIR="$CACHE/numba"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Refuse to let build_centroid_matrix.py (step 2) fall through to its 1.38 GB download.
if [[ " $STEPS " == *" 2 "* ]]; then
    for f in wmb_precomputed_stats.h5 wmb_gene.csv; do
        if [[ ! -e "$REPO_ROOT/data/raw/WMB-10X/$f" ]]; then
            echo "ERROR: $f not found in data/raw/WMB-10X/ (not in git)." >&2
            echo "       Download it with the commands in docs/reproducing_the_paper.md (Inputs not in" >&2
            echo "       the repository, Atlas files), or link the file there." >&2
            exit 1
        fi
    done
fi

S="$REPO_ROOT/scripts/wmb10x"
STEP_NAMES=([1]=download_wmb_metadata [2]=build_centroid_matrix [3]=embed_centroids [4]=plot_embeddings)
echo "=== $NAME (steps: $STEPS; $SCANPY_PYTHON) === started $(date)"
for i in $STEPS; do
    step="${STEP_NAMES[$i]}"
    echo "=== Step $i/4: $step === $(date)"
    script -qefc "\"$SCANPY_PYTHON\" \"$S/$step.py\"" "$LOGDIR/$i-$step.raw.log"
    clean_log "$LOGDIR/$i-$step.raw.log" "$LOGDIR/$i-$step.log"
done

echo "=== $NAME === finished $(date)"
echo "S4a panel: $REPO_ROOT/results/wmb10x/figures/wmb_umap_panel_nokey.svg + wmb_umap_key.svg (caption: results/wmb10x/captions.md)"
