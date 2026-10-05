# Boström et al. codebooks

Constant-Hamming-weight MERFISH codebooks with minimum Hamming distance 4 (SECDED),
from Boström J et al. (2025), *Science Advances* 11:eadr4026,
<https://doi.org/10.1126/sciadv.adr4026>.

**Source and licence.** The codebooks come from the authors' Dryad deposit,
<https://doi.org/10.5061/dryad.zkh1893m5>, published under CC0 1.0. They are the same
codebooks as the article's Data S3 to S5. The article and its supplement are licensed
CC BY-NC 4.0, so the Dryad deposit is the source cited here.

This folder holds two kinds of files:

- **Codebooks** (`*Bit_HW*_HD4_finalsize*.csv`), as distributed in the Dryad
  deposit (CC0).
- **Generated panels** (`bostrom_*_panel.csv`), with the columns
  `{Gene, Sequence, mean_raw_counts}`. They are made by
  `scripts/benchmark/make_bostrom_panel.py`, which assigns the codebook's barcodes to
  genes in order of expression; the third column is named after `--expression-col`.

## Codebooks

- `10Bit_HW4_HD4_finalsize30{Binary,Set}.csv`: a tiny CI fixture (30 codewords, in
  both encodings) used by `tests/test_bostrom_codebook.py` and
  `tests/test_make_bostrom_panel.py`.
- `16Bit_HW4_HD4_finalsize140Binary_reordered20.csv`: 16-bit HW4, 140 codewords.
  Input of `bostrom_hw4_16bit_panel.csv`.
- `16Bit_HW5_HD4_finalsize315Binary_reordered93.csv`: 16-bit HW5, 315 codewords.
  Input of `bostrom_hw5_16bit_panel.csv`.
- `32Bit_HW4_HD4_finalsize1240Binary_reordered40.csv`: 32-bit HW4, 1240 codewords, as
  published (reordered to M=40, see below).
  `experiments/merfish_zhang2023_v2/make_panels.sh` builds the HW4 baseline panel
  from this file.
- `32Bit_HW5_HD4_finalsize6579Binary_reordered557.csv`: 32-bit HW5, 6579 codewords,
  reordered to M=557. Input of the HW5 baseline panels of
  `experiments/merfish_zhang2023_v2`, `experiments/merfish_2000_genes` and
  `experiments/merfish_expression_prior`.

## The `_reordered<M>` suffix

`M` is emitted by the authors' own `CodebookHDSorter.R`: it is **how many codewords
the sorter reordered before stopping**. Their sorter front-loads maximally-separated
codewords so that, under their descending-expression gene assignment, the
highest-expressed genes get the most-separated codes. It stops once the best remaining
codeword is only average-separated from those already placed, then appends the rest
**in lexicographic construction order**. For the 32-bit HW5 book the stop came at 557
of 6579 codewords, and for the 32-bit HW4 book at 40 of 1240.

## Generated panels (`{Gene, Sequence, mean_raw_counts}`)

Both 16-bit panels hold the top 140 genes (`ERCC-` controls left out) of the
Kastriti et al. 2022 sensory-neuron table
(`../processed/Kastriki_et_al_2022/sensory_neuron_mean_expression.csv`), with its
mean counts in `mean_raw_counts` (CC BY 4.0; see [`../README.md`](../README.md)):

- `bostrom_hw4_16bit_panel.csv`: the 16-bit HW4 codebook. Starting codebook of
  `examples/quickstart_merfish.py` and of `docs/design_merfish.md`.
- `bostrom_hw5_16bit_panel.csv`: the 16-bit HW5 codebook. Reference codebook in
  `docs/design_merfish.md`.

## Generating a panel

The Boström codebooks carry no gene assignment. Turn one into a panel; run from the
repository root, this command rebuilds `bostrom_hw4_16bit_panel.csv` byte for byte:

    python scripts/benchmark/make_bostrom_panel.py \
      --codebook examples/data/bostrom/16Bit_HW4_HD4_finalsize140Binary_reordered20.csv \
      --expression examples/data/processed/Kastriki_et_al_2022/sensory_neuron_mean_expression.csv \
      --top-k 140 \
      --out bostrom_hw4_16bit_panel.csv

Use `16Bit_HW5_HD4_finalsize315Binary_reordered93.csv` for the HW5 panel. Add
`--no-expression-col` to write only `Gene` and `Sequence`.

A MERFISH benchmark config reads the panel in both `genes:` and `initialization:` and
must use the `expression:` file the panel was built from; see
`experiments/merfish_2000_genes/config.yaml` and `make_panels.sh`.
