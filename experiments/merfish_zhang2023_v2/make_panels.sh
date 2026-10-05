#!/usr/bin/env bash
# =============================================================================
# Rebuild the two Fig 4b Bostrom baseline panels (HW4, HW5) over the Zhang et al.
# 2023 v2 gene list. Genes are sorted by whole-brain WMB-10X expression and
# paired with the Bostrom codewords in that order. The panels hold Gene and
# Sequence only (--no-expression-col), because the atlas values are not
# redistributed. The committed panels/ are the output of this script; see
# panels/README.md for their sources and licence.
#
# Two inputs are not in git:
#   data/processed/BIL_MERFISH/codebooks/zhang_2023_v2_processed.csv
#       built by scripts/data_processing/build_zhang2023_codebook.py
#   examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv
#       built by scripts/wmb10x/build_whole_brain_cpm.py
#
# Expected sha256 prefixes of the output (= the committed panels/):
#   panels/zhang_v2_bostrom_hw4_panel.csv   e73e4a096cd5
#   panels/zhang_v2_bostrom_hw5_panel.csv   c3a49c0f840d
#
# Activate an env whose `duet` resolves to this repo first. Works from any cwd.
# Usage: bash experiments/merfish_zhang2023_v2/make_panels.sh
# =============================================================================
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$HERE"

REPO=../..
EXPR=$REPO/examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv
HW4=$REPO/examples/data/bostrom/32Bit_HW4_HD4_finalsize1240Binary_reordered40.csv
HW5=$REPO/examples/data/bostrom/32Bit_HW5_HD4_finalsize6579Binary_reordered557.csv
ZV2=$REPO/data/processed/BIL_MERFISH/codebooks/zhang_2023_v2_processed.csv
C=(--expression "$EXPR" --gene-col gene_symbol --expression-col mean_cpm --genes-col Gene
   --no-expression-col)

python "$REPO/scripts/benchmark/make_bostrom_panel.py" --codebook "$HW4" "${C[@]}" --genes-csv "$ZV2" \
    --out panels/zhang_v2_bostrom_hw4_panel.csv
python "$REPO/scripts/benchmark/make_bostrom_panel.py" --codebook "$HW5" "${C[@]}" --genes-csv "$ZV2" \
    --out panels/zhang_v2_bostrom_hw5_panel.csv

sha256sum panels/zhang_v2_bostrom_hw4_panel.csv panels/zhang_v2_bostrom_hw5_panel.csv
