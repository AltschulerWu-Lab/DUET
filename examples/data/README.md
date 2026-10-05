# Example data

The data files used by the examples, the tests and the regression fixtures, with their
sources and licences. The MIT licence of this repository covers the code, not these
data files.

## `quickstart_ops.csv`

A small subset of the human CRISPRi library hCRISPRi-v2.1 for
`examples/quickstart_ops.py`: 550 rows, one sgRNA per row.

- **Source.** Horlbeck MA, Gilbert LA, Villalta JE, et al. Compact and highly
  active next-generation libraries for CRISPR-mediated gene repression and
  activation. *eLife* 5:e19760 (2016). <https://doi.org/10.7554/eLife.19760>.
  Supplementary table of library contents (sheet `hCRISPRi-v2.1`).
- **License.** eLife articles and their supplementary files are published
  under the Creative Commons Attribution 4.0 License (CC BY 4.0). This subset
  is a derivative, redistributed under the same license (<https://creativecommons.org/licenses/by/4.0/>) with
  attribution to the authors above.
- **Processing.** Taken from `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv`
  in this repository, which was built from the supplementary table (the
  leading G of each spacer removed, so sequences are 19 nt; the activity score
  is the predicted score, filled with the empirical score where missing; see
  [`data/processed/README.md`](../../data/processed/README.md)).
  Then:
  - 50 genes (the table's `Gene` column; one entry per gene TSS) drawn with
    `numpy.random.default_rng(0)` from the genes whose ten top-ranked sgRNAs
    (`Rank` <= 10) all have activity scores between 0 and 1, with all ten
    sgRNAs of each;
  - the first 50 non-targeting controls (`negative_control`). They have no
    activity score in the source; they get 1.0 here, as in the paper's pool.

  Rows keep the source table's order. `make_quickstart_ops.py` rebuilds the
  file from a source checkout.

| Column | Meaning |
|---|---|
| `sgID` | sgRNA identifier from the source |
| `gene` | target (gene and TSS), or `negative_control` |
| `gene_name` | gene symbol |
| `sequence` | 19-nt spacer, 5' to 3' |
| `activity` | predicted activity score (1.0 for controls) |

## `bostrom/`: Boström et al. codebooks and two 16-bit panels

- **Codebooks** (`*Bit_HW*_HD4_finalsize*.csv`). Constant-weight MERFISH
  codebooks with minimum Hamming distance 4, from Boström J et al.,
  *Science Advances* 11:eadr4026 (2025), <https://doi.org/10.1126/sciadv.adr4026>.
  The files come from the authors' Dryad deposit,
  <https://doi.org/10.5061/dryad.zkh1893m5>, which Dryad publishes under CC0 1.0
  (public domain dedication).
- **Panels** (`bostrom_hw4_16bit_panel.csv`, `bostrom_hw5_16bit_panel.csv`). The
  140 most expressed genes of the Kastriti et al. 2022 table below (`ERCC-` controls
  left out), paired with the codewords of the 16-bit HW4 and HW5 codebooks. The
  `mean_raw_counts` column holds that table's values (CC BY 4.0, see below).
  `examples/quickstart_merfish.py` uses only the gene names and barcodes of
  `bostrom_hw4_16bit_panel.csv`; its expression is synthetic, and the panel's
  measured counts are not used.

See [`bostrom/README.md`](bostrom/README.md) for each file and the command that
rebuilds the panels.

## `processed/MERFISH/codebook_1.csv`

The 16-bit MERFISH codebook of the 140-gene library of Chen et al. 2015: 140
codewords of Hamming weight 4 with minimum Hamming distance 4 (130 genes plus 5
`notarget` and 5 `blank` control words), in the columns `Gene` and `Sequence`.

- **Source.** Chen KH, Boettiger AN, Moffitt JR, Wang S, Zhuang X. Spatially
  resolved, highly multiplexed RNA profiling in single cells. *Science*
  348:aaa6090 (2015). <https://doi.org/10.1126/science.aaa6090>. Supplementary
  Table S1.
- **License.** The article has no open license. This file reproduces only the
  factual gene-to-barcode assignment of Table S1, with attribution to the authors
  above.
- **Used by** `tests/test_api_parity.py` and the regression fixture
  `scripts/benchmark/regression/merfish_asymmetric/`.

## `processed/Kastriki_et_al_2022/sensory_neuron_mean_expression.csv`

Mean raw counts per gene over the cells labelled "sens. neu." (sensory neurons):
24,581 rows in the columns `gene_name` and `mean_raw_counts`, including 92 `ERCC-`
spike-in controls. The folder name keeps an old misspelling of the first author's
name, because configs and tests use the path.

- **Source.** Kastriti ME, Faure L, Von Ahsen D, et al. Schwann cell precursors
  represent a neural crest-like state with biased multipotency. *EMBO Journal*
  41:e108780 (2022). <https://doi.org/10.15252/embj.2021108780>. Data: GEO
  accession GSE201257.
- **License.** The article is published under CC BY 4.0. This table is a
  derivative (per-gene means computed by the DUET authors), redistributed under
  CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>) with attribution to the authors above.
- **Used by** the regression fixture
  `scripts/benchmark/regression/merfish_asymmetric/`, to rebuild the two 16-bit
  panels in `bostrom/`, and by `tests/test_optical_crowding_bostrom.py` (its tests of
  the published codebooks, which need `DUET_BOSTROM_DATA`).

## Not redistributed

- **WMB-10X whole-brain expression.** The MERFISH experiments in `experiments/` read
  `processed/WMB-10X/whole_brain_per_gene_cpm.csv`, a whole-brain mean CPM per gene
  from the Allen Brain Cell Atlas WMB-10X data (Yao Z et al. 2023, *Nature*,
  <https://doi.org/10.1038/s41586-023-06812-z>). The atlas is licensed CC BY-NC 4.0,
  so the table is not part of this repository. Build it with
  `scripts/wmb10x/build_whole_brain_cpm.py` (see `scripts/wmb10x/README.md`), which
  writes it to that path.
- **Xia et al. 2019.** The MERFISH data of Xia C, Fan J, Emanuel G, Hao J, Zhuang X,
  *PNAS* 116:19490 (2019), <https://doi.org/10.1073/pnas.1912459116>, are published
  under CC BY-NC-ND 4.0 and are not included. No example or experiment uses them.
