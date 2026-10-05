# Processed data tables

Reference tables that the candidate-pool factories and the experiments read. Only
`Horlbeck_2016/CRISPRi_v2_1.csv` is in the repository; the others are built by the
scripts or steps named below. The MIT licence of this repository covers the code, not these
tables.

## `Horlbeck_2016/CRISPRi_v2_1.csv`

The human CRISPRi library hCRISPRi-v2.1: 207,175 sgRNAs, one per row. It is the
default table of the `WeissmanCRISPRi` candidate pool, read by the OPS CRISPRi
experiments, `examples/scaling_ops.py`, the `ops_uniform` regression fixture, and
`examples/data/make_quickstart_ops.py`, which cuts the demo subset
`examples/data/quickstart_ops.csv` from it.

- **Source.** Horlbeck MA, Gilbert LA, Villalta JE, et al. Compact and highly
  active next-generation libraries for CRISPR-mediated gene repression and
  activation. *eLife* 5:e19760 (2016). <https://doi.org/10.7554/eLife.19760>.
  Supplementary file `TableS3_hCRISPRiv2_libraries.xlsx`: sheet `hCRISPRi-v2.1`,
  plus the non-targeting controls of sheet `hCRISPRi-v2`.
- **Processing.**
  1. Read sheet `hCRISPRi-v2.1` and rename its columns (`gene` to `Gene`,
     `transcript` to `TSS`, `protospacer sequence` to `Sequence`, `selection rank`
     to `Rank`, `predicted score` to `Predicted score`, `empirical score` to
     `Empirical score`, `off-target stringency` to `Off-target stringency`).
  2. Drop its `negative_control` rows. Add instead the `negative_control` rows of
     sheet `hCRISPRi-v2` whose `Sublibrary half` is `Top5`, with `TSS` set to `P1P2`.
  3. `Activity score` is the predicted score, or the empirical score where the
     predicted score is missing.
  4. `Gene name` keeps the gene symbol. `Gene` becomes the gene and TSS joined by
     `_` (for example `A1BG_P1`), so that separate TSSs are separate targets; a
     `P1P2` TSS adds no suffix, so the controls stay `negative_control`.
  5. `Full sequence` keeps the 20-nt protospacer. `Sequence` is the same spacer in
     upper case without its leading G (19 nt), because that base is always G.
  6. `ID` is `Gene` and `Rank` joined by `_` (the `sgID` for controls), and
     `Complementary full sequence` is the base-wise complement of `Full sequence`.
- **sha256** `2aa08896d77b79a6225055bff59d324d8e73d93f795bfb3a90c1403690a73494`
  (34,028,926 bytes).
- **License.** eLife publishes the article and its supplementary files under the
  Creative Commons Attribution 4.0 License (CC BY 4.0). This table is a derivative,
  redistributed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>) with attribution to the authors above.

## Not in the repository

- **`Horlbeck_2016/CRISPRa.csv` and `Horlbeck_2016/CRISPRi_v2.csv`.** The hCRISPRa-v2
  library (supplementary file `TableS5_hCRISPRav2_libraries.xlsx`) and the
  hCRISPRi-v2 library (sheet `hCRISPRi-v2` of `TableS3_hCRISPRiv2_libraries.xlsx`) of
  the same article, CC BY 4.0. No experiment, example or test reads them, so they are
  not included. The `WeissmanCRISPRa` candidate pool expects `Horlbeck_2016/CRISPRa.csv`;
  build it from the supplementary file with the steps above, taking its own
  `negative_control` rows with `Sublibrary half` `Top5`.
- **`CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv`.** The genome-wide
  candidate table of `experiments/ops_crispick_genome_wide`. It comes from CRISPick, of
  the Genetic Perturbation Platform of the Broad Institute (DeWeirdt PC et al. 2022,
  *Nature Communications* 13:5255, <https://doi.org/10.1038/s41467-022-33024-2>).
  CRISPick output is for research use under the Broad Institute's terms and is not
  redistributed here. Download the three CRISPick files into `data/raw/CRISPick/` and
  build the table with

  ```bash
  python scripts/data_processing/build_crispick_candidates.py
  ```

  ([Reproducing the paper](../../docs/reproducing_the_paper.md#crispick-table) lists
  the files and their hashes).
- **`BIL_MERFISH/codebooks/zhang_2023_v2_processed.csv`.** The 1,147-gene codebook of
  Zhang M et al. 2023 (*Nature*, <https://doi.org/10.1038/s41586-023-06808-9>) used by
  `experiments/merfish_zhang2023_v2`. It is converted from the Brain Image Library
  file `codebook_32bit_v2.csv` of the dataset Zhuang X, Jung W, Zhang M (2023),
  <https://doi.org/10.35077/act-bag>, licensed CC BY-SA 4.0. Build it with

  ```bash
  python scripts/data_processing/build_zhang2023_codebook.py --download
  ```

  which fetches the file into `data/raw/BIL_MERFISH/additional_files/` if it is not
  there and writes the codebook here
  ([Reproducing the paper](../../docs/reproducing_the_paper.md#zhang-et-al-codebook)
  has the details). This work used data from the Brain Image Library (RRID:SCR_017272), which is
  supported by the National Institutes of Mental Health of the National Institutes of
  Health under award number R24-MH-114793.
- **`WMB-10X/`.** The cluster-centroid matrix and its tables, written by
  `scripts/wmb10x/build_centroid_matrix.py` from the Allen Brain Cell Atlas
  (CC BY-NC 4.0); see `scripts/wmb10x/README.md`.
