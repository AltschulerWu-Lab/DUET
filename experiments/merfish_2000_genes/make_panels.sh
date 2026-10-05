#!/usr/bin/env bash
# =============================================================================
# Rebuild the two Fig 4c-e baseline panels from the 2,000-gene list
# panels/k2000_genes.csv. Genes are sorted by whole-brain WMB-10X expression.
# The Bostrom HW5 panel pairs them with the HW5 codewords in that order; the
# MERFISH MHD4 panel uses the same codewords in a random pairing (seed 42).
# Both panels hold Gene and Sequence only (--no-expression-col), because the
# atlas values are not redistributed. The committed panels/ are the output of
# this script; see panels/README.md.
#
# One input is not in git:
#   examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv
#       built by scripts/wmb10x/build_whole_brain_cpm.py
#
# Expected sha256 prefixes:
#   panels/k2000_bostrom_hw5_panel.csv         cf4df7e22cc9
#   panels/k2000_merfish_mhd4_hw5_panel.csv    b237b8d9d991
#   panels/k2000_genes.csv (input, unchanged)  4a185367482a
#
# Activate an env whose `duet` resolves to this repo first. Works from any cwd.
# Usage: bash experiments/merfish_2000_genes/make_panels.sh
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$HERE"

REPO=../..
EXPR=$REPO/examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv
HW5=$REPO/examples/data/bostrom/32Bit_HW5_HD4_finalsize6579Binary_reordered557.csv
G=panels/k2000_genes.csv
C=(--expression "$EXPR" --gene-col gene_symbol --expression-col mean_cpm --genes-col Gene
   --no-expression-col)

python "$REPO/scripts/benchmark/make_bostrom_panel.py" --codebook "$HW5" "${C[@]}" --genes-csv "$G" \
    --out panels/k2000_bostrom_hw5_panel.csv
python "$REPO/scripts/benchmark/make_bostrom_panel.py" --codebook "$HW5" "${C[@]}" --genes-csv "$G" \
    --shuffle-assignment --shuffle-seed 42 --out panels/k2000_merfish_mhd4_hw5_panel.csv

sha256sum panels/k2000_genes.csv panels/k2000_bostrom_hw5_panel.csv panels/k2000_merfish_mhd4_hw5_panel.csv
