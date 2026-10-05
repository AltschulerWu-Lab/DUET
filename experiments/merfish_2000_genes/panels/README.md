# 2,000-gene panel and its baseline codebooks

| File | What | Read by `../config.yaml` as |
|---|---|---|
| `k2000_genes.csv` | 2,000 mouse gene symbols, one column `Gene`. | input of `../make_panels.sh` only |
| `k2000_bostrom_hw5_panel.csv` | The 2,000 genes paired with codewords of the Boström et al. 32-bit HW5 codebook (`32Bit_HW5_HD4_finalsize6579Binary_reordered557.csv`). | `genes.path`, `initialization.path` (warm start) and the baseline "Bostrom et al. (Hamming weight 5)" |
| `k2000_merfish_mhd4_hw5_panel.csv` | The same genes and the same 2,000 codewords in a random pairing (seed 42), as in standard MERFISH. | the baseline "MERFISH MHD4 (Hamming weight 5)" |

The two panels have the columns `Gene` (gene symbol) and `Sequence` (32-character
0/1 barcode). The MERFISH runner reads only these two columns.

## How they are made

- `k2000_genes.csv` was drawn at random (seed 42) from the genes with non-zero
  expression (excluding `ERCC-` controls) in an earlier build of the WMB-10X
  whole-brain table, in which 22 gene symbols had more than one row. A new draw
  from the current table (one row per symbol) gives a different list, so the file is
  kept as it is. It holds names only, no expression values.
- `../make_panels.sh` builds both panels with
  `scripts/benchmark/make_bostrom_panel.py`. The genes are sorted by descending
  whole-brain expression in the WMB-10X table. In the Boström panel the first gene
  gets the first codeword, the second gene the second, and so on. The MERFISH MHD4
  panel keeps the same row order and the same codewords but permutes the pairing.
- No expression values are stored in these files. The WMB-10X table is not
  redistributed; build it with `scripts/wmb10x/build_whole_brain_cpm.py` before you
  run `../make_panels.sh`.

## Sources and licences

- **Barcodes.** Boström J et al. (2025), *Science Advances* 11:eadr4026,
  <https://doi.org/10.1126/sciadv.adr4026>. The codebooks come from the Dryad
  deposit <https://doi.org/10.5061/dryad.zkh1893m5> (CC0 1.0). See
  [the Boström codebook notes](../../../examples/data/bostrom/README.md).
- **Gene choice and row order.** Whole-brain mean CPM per gene from the Allen Brain
  Cell Atlas WMB-10X data (Yao Z et al. (2023), *Nature*,
  <https://doi.org/10.1038/s41586-023-06812-z>), which is CC BY-NC 4.0. The atlas
  was used to choose the genes and to order the rows. No atlas values are shipped.
