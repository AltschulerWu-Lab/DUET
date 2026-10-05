# Bostrom baseline panels over the Zhang et al. 2023 v2 genes

Two baseline codebooks of `../config.yaml` (Fig 4b). Each pairs the 1,147 genes of
the Zhang et al. 2023 v2 codebook with codewords of a Boström et al. 32-bit codebook.

| File | Codewords from | Baseline name in `../config.yaml` |
|---|---|---|
| `zhang_v2_bostrom_hw4_panel.csv` | `32Bit_HW4_HD4_finalsize1240Binary_reordered40.csv` (HW4, MHD4) | "Bostrom et al. (Hamming weight 4)" |
| `zhang_v2_bostrom_hw5_panel.csv` | `32Bit_HW5_HD4_finalsize6579Binary_reordered557.csv` (HW5, MHD4) | "Bostrom et al. (Hamming weight 5)" |

Columns: `Gene` (gene symbol) and `Sequence` (32-character 0/1 barcode). The
MERFISH runner reads only these two columns.

## How they are made

`../make_panels.sh` builds both files with `scripts/benchmark/make_bostrom_panel.py`.
The genes are sorted by descending whole-brain expression in the WMB-10X table, and
the first gene gets the first codeword of the Boström codebook, the second gene the
second, and so on. 11 of the 1,147 genes are not in the WMB-10X table; they count as
zero expression and come last, in codebook order.

No expression values are stored in these files. The WMB-10X table is not
redistributed (see below); build it with `scripts/wmb10x/build_whole_brain_cpm.py`
before you run `../make_panels.sh`. The Zhang v2 codebook is built by
`scripts/data_processing/build_zhang2023_codebook.py`.

## Sources and licences

- **Gene list.** From the codebook `codebook_32bit_v2.csv` of the dataset Zhuang X,
  Jung W, Zhang M (2023), Brain Image Library, <https://doi.org/10.35077/act-bag>,
  licensed CC BY-SA 4.0. The article is Zhang M et al. (2023), *Nature*,
  <https://doi.org/10.1038/s41586-023-06808-9>. Changes: the gene names were taken
  from that codebook (its `blank-*` rows dropped), sorted by WMB-10X expression and
  paired with Boström codewords. These two files are adapted material and are
  licensed under CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>).
  The MIT licence of this repository covers the code only.

  This work used data from the Brain Image Library (RRID:SCR_017272), which is supported by the
  National Institutes of Mental Health of the National Institutes of Health under
  award number R24-MH-114793.
- **Barcodes.** Boström J et al. (2025), *Science Advances* 11:eadr4026,
  <https://doi.org/10.1126/sciadv.adr4026>. The codebooks come from the Dryad
  deposit <https://doi.org/10.5061/dryad.zkh1893m5> (CC0 1.0). See
  [the Boström codebook notes](../../../examples/data/bostrom/README.md).
- **Row order.** Whole-brain mean CPM per gene from the Allen Brain Cell Atlas
  WMB-10X data (Yao Z et al. (2023), *Nature*,
  <https://doi.org/10.1038/s41586-023-06812-z>), which is CC BY-NC 4.0. Only the
  order is used. No atlas values are shipped.
