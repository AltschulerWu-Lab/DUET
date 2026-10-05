# Reproducing the paper

`experiments/` holds one folder per analysis: a `run.sh` driver, its config and a README (settings,
outputs, values to compare); [experiments/README.md](../experiments/README.md) maps folders to figure
panels. The drivers run the benchmark runners behind the paper's numbers, which evaluate DUET with the
baselines. The public API accepts the paper's settings ([OPS](design_ops.md#settings-used-in-the-paper),
[MERFISH](design_merfish.md#settings-used-in-the-paper)) but derives its own seeds, so its results
match the paper's statistically, not number for number.

## Hardware and time

Reference machine: Ubuntu 20.04, 2 x Intel Xeon E5-2640 v4 (40 threads), 220 GB RAM, 4 x NVIDIA
TITAN X (Pascal, 12 GB), CUDA 12.7 driver. Eight experiments need NVIDIA GPUs; do not switch them
to CPU (far slower, and decoding can differ: [Reproducibility](design.md#reproducibility)). All
twelve took about 104 h (4.3 days): 19 h for the first eight in run order, 20 min for the next two,
9 h and 76 h for the last two. Most need under 15 GB of disk and a few GB of RAM;
[Experiments](#experiments) lists the exceptions.

## Setup

The drivers are Linux bash scripts; they call `conda run` and stop unless Python imports `duet` from
this clone (`python -c "import duet; print(duet.__file__)"`). Install DUET editable with the
`benchmark` and `gpu` extras ([other CUDA versions](installation.md#choosing-a-cupy-package)) and the
`ops` and `scanpy_env` environments, keeping their names ([environments/README.md](../environments/README.md)):

<!-- docs-test: skip (creates conda environments and installs packages) -->
```bash
git clone https://github.com/AltschulerWu-Lab/DUET.git
cd DUET
conda env create -f environment.yml   # env "duet": editable install, benchmark and test extras
conda activate duet
pip install -e ".[benchmark,gpu]"
conda env create -f environments/ops.yml      # env "ops": the Feldman et al. baseline
conda env create -f environments/scanpy.yml   # env "scanpy_env": the atlas steps
pip install -e ".[benchmark,notebooks]"       # notebook tools, for nisseq_error_analysis only
```

## Inputs not in the repository

Six experiments need files you download, build or request (the Needs column of
[Experiments](#experiments)); [experiments/INPUTS.md](../experiments/INPUTS.md) lists every input
with its size and sha256. `ops_crispri_cross_eval` and `ops_crispick_genome_wide` also need the
NIS-seq channel matrices, which are included only with permission ([NIS-seq data](#nis-seq-data)).

| Input | Used by | Source | Where it goes |
|---|---|---|---|
| Zhang et al. 2023 MERFISH codebook v2 | `merfish_zhang2023_v2` | Brain Image Library, [doi:10.35077/act-bag](https://doi.org/10.35077/act-bag), [converted](#zhang-et-al-codebook) | `data/raw/BIL_MERFISH/additional_files/`; converted to `data/processed/BIL_MERFISH/codebooks/` |
| Yao et al. (2023) whole-mouse-brain 10x atlas: cluster statistics (1.4 GB) and gene table | `wmb10x_landscape`, `merfish_expression_prior` and the expression table | Allen Brain Cell Atlas, [commands below](#atlas-files) | `data/raw/WMB-10X/` |
| Whole-brain expression table (mean CPM per gene) | `merfish_zhang2023_v2`, `merfish_2000_genes`, `merfish_expression_prior` | built from the atlas files, [below](#atlas-files) | `examples/data/processed/WMB-10X/` |
| NIS-seq spot-level base calls (8 files, 1.5 GB) and the Brunello table | `nisseq_error_analysis` | spot files from the authors of Fandrey et al. 2025 ([doi:10.1038/s41587-024-02516-5](https://doi.org/10.1038/s41587-024-02516-5)) on request; [Brunello below](#nis-seq-data) | `data/raw/NISseq_HeLa_IL1b/` |
| CRISPick candidate table | `ops_crispick_genome_wide` | built from three [Broad GPP CRISPick](https://portals.broadinstitute.org/gppx/crispick/public) downloads, [below](#crispick-table) | `data/raw/CRISPick/`; built into `data/processed/CRISPick/` |

### Zhang et al. codebook

`scripts/data_processing/build_zhang2023_codebook.py` converts the Brain Image Library file
`codebook_32bit_v2.csv` (92,715 bytes, sha256
`7e45c01cab5bfa398d86031748ff9769ad3829906ef57c16c5e287842e222667`). With `--download` it first
fetches the [file](https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/additional_files/codebook_32bit_v2.csv)
into `data/raw/BIL_MERFISH/additional_files/` if it is not there. The conversion drops the
`blank-*` rows and joins the 32 readout bits (the `RS*` columns, in file order) into one barcode
per gene. It writes `data/processed/BIL_MERFISH/codebooks/zhang_2023_v2_processed.csv`: 1,147
genes, sha256 `3a6159f8ce074c78db13dbce00cb422f716611764ea9409683ff015a29dde779`, as in the paper.

<!-- docs-test: skip (downloads from the Brain Image Library) -->
```bash
python scripts/data_processing/build_zhang2023_codebook.py --download
```

### Atlas files

Download the two atlas files as below (sha256 `b21ca985652fb25f9608f99005139a40757133a76fbe845ae5b175c5c26a447b`
and `ec5a211c23fab5cce8a06f01bb0a68af087fb979b647ab42d635201f98de2f69`), then build the
whole-brain expression table from them in the `scanpy_env` environment, from the repository root.
The build writes `examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv`, sha256
`a8314155531fefb3e7abfb4993f1f4c184ece9cdbf541fdf38cc2503584fd8e4`, as in the paper. The atlas
values are licensed CC BY-NC 4.0, so the table is not in the repository.

<!-- docs-test: skip (downloads 1.4 GB) -->
```bash
ABC=https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com
mkdir -p data/raw/WMB-10X
curl -L -o data/raw/WMB-10X/wmb_precomputed_stats.h5 \
    "$ABC/mapmycells/WMB-10X/20240831/precomputed_stats_ABC_revision_230821.h5"
curl -L -o data/raw/WMB-10X/wmb_gene.csv "$ABC/metadata/WMB-10X/20241115/gene.csv"
sha256sum data/raw/WMB-10X/wmb_precomputed_stats.h5 data/raw/WMB-10X/wmb_gene.csv
conda run -n scanpy_env python scripts/wmb10x/build_whole_brain_cpm.py
```

### NIS-seq data

Only `nisseq_error_analysis` needs the spot files. Put them in
`data/raw/NISseq_HeLa_IL1b/NIS-seq_HeLa_IL1b/` and the Brunello table at
`data/raw/NISseq_HeLa_IL1b/Brunello_sgRNAs/Brunello_sgRNAs.txt`, which you can rebuild from the
[Addgene](https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/) table (gene and
spacer per row, tab-separated, no header, same order; then set `CHECK_INPUT_HASHES=0`).

`ops_crispri_cross_eval` and `ops_crispick_genome_wide` read the NIS-seq channel matrices that
`nisseq_error_analysis` fits ([Channels in the repository](noise_channels.md#channels-in-the-repository)
lists them). The NIS-seq channel matrices are fitted from spot-level base calls that the authors of
Fandrey et al. (2025) shared on request, and are included only with their permission. If one is
missing from your copy, rebuild it with `experiments/nisseq_error_analysis` from the spot calls,
which are available from those authors on request.

### CRISPick table

The candidates come from CRISPick, of the Genetic Perturbation Platform of the Broad Institute
(DeWeirdt et al. 2022, [doi:10.1038/s41467-022-33024-2](https://doi.org/10.1038/s41467-022-33024-2)).
CRISPick output is for research use and is not redistributed here, so build the table from three
files you download from the [CRISPick portal](https://portals.broadinstitute.org/gppx/crispick/public)
into `data/raw/CRISPick/`:

| File | Size (bytes) | sha256 |
|---|---|---|
| `sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_RS3seq-Chen2013+RS3target_Ensembl_aggrCFD_20251121.txt` | 2,116,333,376 | `5dd4da45440dc095e14755680a2d0ac9051467eef92aa613725ba2ba12195a24` |
| `sgrna-negcontrols-nosite-9606-GRCh38-SpyoCas9-2000.txt` | 42,000 | `e86fce65335216b4bf0606a512494f2d5229610fb5f1cf96b218ce75589fbb85` |
| `sgrna-negcontrols-onesite-9606-GRCh38-SpyoCas9-2000.txt` | 42,000 | `dd350b75d89c341e67387229129b7b77256946ccbc13f341750fef2d3a62d3d6` |

<!-- docs-test: skip (needs the CRISPick downloads) -->
```bash
python scripts/data_processing/build_crispick_candidates.py   # about a minute, 2 GB of RAM
```

The build keeps the top 20 sgRNAs per gene by `Pick Order` from the design file and appends the
2,000 `NO_SITE` and 2,000 `ONE_SITE_INTERGENIC` controls. It writes
`data/processed/CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv`: 405,059 rows with the
columns `Target Gene ID`, `Target Gene Symbol`, `sgRNA Sequence`, `Aggregate CFD Score`,
`On-Target Efficacy Score`, `Pick Order`, `Picking Round` and `Source`. From the 2025-11-21 design
file above, the table is byte-identical to the paper's (sha256
`e49d92641f932c2610e8f8817428532386a0f56425d306e50141d56fb531a484`). Row order seeds the reads, so a
table built from another CRISPick release gives other designs.

## Experiments

In the order `run_all.sh` runs them; each folder's README has the details.

| Folder | What it reproduces | Device | Time | Needs |
|---|---|---|---|---|
| `synthetic_objective_correlation` | How well codebook objectives rank codebooks by true decoding accuracy | CPU | about 10 min | |
| `synthetic_hvr` | How much of an enumerable Pareto front the λ sweep recovers | GPUs | about 20 min | |
| `ops_crispri_symmetric` | Medium-scale CRISPRi benchmark (1,000 genes, 10 rounds, symmetric channel) against the baselines | GPUs | about 30 min | `ops` env |
| `ops_crispri_cross_eval` | The same design optimized under four NIS-seq channels, evaluated under the measured one | GPUs | about 2 h | `ops` env, [NIS-seq channel matrices](#nis-seq-data) |
| `merfish_zhang2023_v2` | MERFISH, 1,147 genes, 32 bits, from the published codebook | GPUs | about 3.3 h | Zhang et al. codebook, expression table; 60 GB disk and about 40 GB RAM (memory-mapped PEP) |
| `merfish_2000_genes` | MERFISH, 2,000 brain genes, from a Boström et al. 2025 codebook | GPUs | about 6 h | expression table; disk and RAM as above |
| `merfish_expression_prior` | Crowding under 34 cell-type profiles, over 20 resampled 2,000-gene panels, each with its own DUET codebook and baselines | CPU, and GPUs for the decoding evaluation | about 4 to 6 h | the `merfish_2000_genes` PEP cache, atlas files, expression table, `scanpy_env` |
| `wmb10x_landscape` | PCA and UMAP of the atlas cluster centroids | CPU | about 6 min | atlas files, `scanpy_env`, network, 4 GB RAM |
| `nisseq_error_analysis` | NIS-seq error rates and the four fitted channels | CPU | about 21 min | NIS-seq data, `notebooks` extra, 10 GB RAM |
| `pep_grid_glyph` | The illustrative PEP grid of the method schematic | CPU | seconds | |
| `ops_crispri_rounds_error_sweep` | The CRISPRi benchmark at 8 to 13 rounds and ε = 0.03, 0.05, 0.10 | GPUs | about 9 h | `ops` env, 47 GB disk |
| `ops_crispick_genome_wide` | Genome-wide knockout library (20,114 genes, 14 rounds) under four NIS-seq channels | GPUs | about 76 h (16 to 19 h per arm) | CRISPick table, [NIS-seq channel matrices](#nis-seq-data), `ops` env, 1.7 TB disk at peak (334 GB of PEP cache per arm, kept for re-runs, plus up to 0.5 TB of scratch), 64 GB RAM |

**What to compare.** `pep_grid_glyph` and `synthetic_objective_correlation` reproduce the paper
exactly, and `nisseq_error_analysis` rebuilds the NIS-seq channel matrices byte for byte (it checks
them against the repository's copies when your copy includes them); `synthetic_hvr` reproduced the
paper on the reference GPUs. The OPS and MERFISH designs were first run with draws the configs do
not repeat, so re-runs agree statistically: codebooks can differ and an operating point can move
to a neighboring λ.

## Running

<!-- docs-test: skip (runs experiments for hours) -->
```bash
bash experiments/ops_crispri_symmetric/run.sh   # one experiment, from any directory
bash experiments/run_all.sh                     # all twelve in table order, or name some
```

Each `run.sh` checks its setup, logs every step and draws the figures. `run_all.sh` first checks the
inputs, the `ops` environment and the genome-wide disk guard; `DRY_RUN=1` stops there:

```bash
DRY_RUN=1 bash experiments/run_all.sh synthetic_objective_correlation pep_grid_glyph
```

```text
run_all plan (stop-on-first-failure): 2 entries; preflight passed
  1. synthetic_objective_correlation
  2. pep_grid_glyph
DRY_RUN=1: nothing was run and no status file was written
```

- **Long runs.** `UNATTENDED=1` carries on past a failure, skipping only its dependents
  (`merfish_expression_prior` needs `merfish_2000_genes`; `cross_eval` needs the genome-wide arms).
  Progress is in `results/experiments/run_all/latest/status.tsv`.
- **Outputs** go to `results/experiments/<name>/`, PEP caches (reused on re-runs) to
  `results/cache/<name>/`; `wmb10x_landscape` also writes to `data/` and `results/wmb10x/`.
- **Smoke tests.** Set `VARIANT=smoke` to run the reduced configs tracked next to the real ones
  (`config.smoke.yaml` or `<name>.smoke.yaml`), for one driver or for `run_all.sh`. They use the
  same devices and write under `results/smoke/`, except `wmb10x_landscape`, whose smoke run
  (steps 1 and 2) writes its usual `data/` files.

**The genome-wide design** runs as six steps (four arms, `cross_eval`, `visualize`). Its cache and
scratch folders must not resolve under your home directory (or set `ALLOW_HOME_SCRATCH=1`): link
them to a large volume, then check with `bash experiments/ops_crispick_genome_wide/run.sh check`.

<!-- docs-test: skip (example paths) -->
```bash
BIG=/path/to/large/volume/duet
mkdir -p "$BIG/cache" "$BIG/scratch" results/cache results/scratch
ln -sT "$BIG/cache" results/cache/ops_crispick_genome_wide
ln -sT "$BIG/scratch" results/scratch/ops_crispick_genome_wide
```

## A quick check without GPUs

1. The two demos ([Demo](../README.md#demo)) and the tests ([Running the tests](../README.md#running-the-tests)).
2. `bash experiments/pep_grid_glyph/run.sh` (seconds) writes `pairwise_error_final.svg` with sha256
   `13b0473c6c7384308f939d9a16accd60cd0995032c556446aa0689266d7733a2` on every machine, and prints
   `OK` when the file matches the paper's.
3. `bash experiments/synthetic_objective_correlation/run.sh` (about 10 min, one core) reproduces the
   paper exactly ([README](../experiments/synthetic_objective_correlation/README.md)).

## Regression checks

`scripts/benchmark/regression/run_regression.py` re-runs two seeded GPU benchmarks against recorded
references ([README](../scripts/benchmark/regression/README.md)). Its two fixtures write their
outputs, PEP cache and scratch files under `results/`. Move old outputs in
`results/benchmark/regression/` aside before a check (`--check`, the default): files left by an
earlier run can make it fail. `ops_uniform` should pass on any GPU; `merfish_asymmetric` may drift
on other GPU models. Its comparison code and the runners' parity tests run on CPU (20 to 30 s,
`benchmark` extra):

<!-- docs-test: requires yaml -->
```bash
python -m pytest tests/regression scripts/benchmark/regression/tests -m ""
```
