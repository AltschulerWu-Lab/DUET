# merfish_expression_prior: Supp Fig S4b (crowding under cell-type-specific expression)

DUET optimizes a MERFISH codebook against whole-brain expression. This experiment asks whether its optical-crowding advantage holds when the cells being imaged follow one cell type's expression instead. Each codebook is evaluated under the expression profiles of the 34 WMB-10X major cell types, at matched panel-spot density. Nothing is re-optimized for a cell type.

The spread is taken over **resampled gene panels**:
- There are 20 panels, each its own set of 2,000 genes.
- Each panel gets its own Boström and MHD4 baselines and its own DUET codebook (λ = 0.80 only).
- The figure shows, per cell type and method, the mean over panels ± its standard error.
- Every panel is a uniform draw with its own gene seed (0-19). None is the Fig 4c gene set.

**Why panels.** Error bars over 100 simulated cells (SE 0.002-0.003) measure Monte Carlo precision, not how much the result depends on the codebook. Three different DUET codebooks for the Fig 4c genes gave OPC-Oligo values of 0.759, 0.776 and 0.788: a swing about ten times that error bar.

## Run

```bash
conda activate duet        # the DUET environment (environment.yml) with the benchmark and gpu extras (run.sh checks it imports this clone's duet)
bash experiments/merfish_expression_prior/run.sh                      # every step
bash experiments/merfish_expression_prior/run.sh crowding aggregate   # named steps only, in order
```

- **Run `merfish_2000_genes` first.** Every panel reuses its PEP cache and config. `bash experiments/run_all.sh merfish_2000_genes merfish_expression_prior` runs just the two, in order.
- **Steps.**

  | Step | Script | Env | What it does |
  |---|---|---|---|
  | `build` | `build_per_class_expression.py` | `scanpy_env` (override with `SCANPY_ENV` or `SCANPY_PYTHON`) | Rolls the ABC atlas up to 34 per-class CPM profiles |
  | `panels` | `make_panels.py` | active | Gene sets, baseline panels and per-panel configs |
  | `duet` | `check_pep_cache.py`, then `scripts/benchmark/run_merfish.py` once per panel | active | One DUET run per panel, in parallel |
  | `crowding` | `evaluate_crowding_by_cell_type.py` once per panel | active | Cell-type crowding, in parallel |
  | `aggregate` | `aggregate_panels.py` | active | Tables and the S4b figure |

  `check` runs the PEP-cache pre-flight on its own.
- **Knobs.**
  - `MAX_JOBS` caps the panels running at once (default 10). Each DUET run's OpenBLAS threads spread over every core, so concurrent runs share the machine (see [Cost](#cost)).
  - `SKIP_COMPLETE=1` skips panels whose `duet/summary.yaml` or `crowding/per_class_metrics.csv` exists. When a panel fails, `run.sh` renames that file to `*.failed`, so a resume reruns the panel. (The crowding step writes `per_class_metrics.csv` before its anchor check.) It does not detect stale outputs: after changing a config, move the affected panels' outputs aside first.
- **Large inputs.** Three are not in git (see [Inputs](#inputs)). `run.sh` checks for the two atlas files before the build step, and for the whole-brain table before the `build`, `panels`, `duet` and `crowding` steps.
- **Hardware.** DUET runs on the CPU, one process per panel. Each panel's decode evaluation (about a minute) runs on one GPU, round robin over `gpus`. The cell-type crowding is single-core CPU.

## Inputs

Not in git:
- **Atlas files** `data/raw/WMB-10X/wmb_precomputed_stats.h5` (1,376,366,584 B, sha256 `b21ca985652f…`) and `data/raw/WMB-10X/wmb_gene.csv` (2,297,493 B, `ec5a211c23fa…`), from the Allen Brain Cell Atlas (Yao et al. 2023, CC BY-NC 4.0). The build step reads them (config keys `abc_stats_h5`, `abc_gene_csv`). Download them with the commands in `docs/reproducing_the_paper.md` (Atlas files).
- **Whole-brain expression table** `examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv` (config key `prior_csv`; 30,599 rows, sha256 `a8314155…`). The panel gene sets are drawn from it, and it must be the same file as the template's `expression.path`. Build it in the scanpy environment; the script downloads the atlas files to `data/raw/WMB-10X/` if they are absent:

  ```bash
  conda run -n scanpy_env python scripts/wmb10x/build_whole_brain_cpm.py
  ```

From `merfish_2000_genes`:
- its `config.yaml` (`duet_template`) and its PEP cache under `results/cache/merfish_2000_genes/`;
- `panels/k2000_bostrom_hw5_panel.csv` (`panel_csv`), for the build step's capture check (only its `Gene` column is read).

Shipped: the Boström HW5 codebook `examples/data/bostrom/32Bit_HW5_HD4_finalsize6579Binary_reordered557.csv` (`bostrom_codebook`). `experiments/INPUTS.md` lists every input with its sha256.

## Design

- **Gene sets.**
  - p01-p20 are uniform draws without replacement over the non-zero, non-ERCC genes of the de-duplicated WMB-10X table, with `gene_seed` 0-19. The Fig 4c genes were drawn from the same kind of universe (seed 42, but from an earlier version of the table), so no `gene_seed` reproduces them.
  - `make_panels.py` rejects repeated gene seeds.
  - Any two panels share 94-155 genes.
  - `make_panels.py` checks that every gene is in the per-class tables.
- **Baselines.**
  - They are made with `make_bostrom_panel.py` in the same way as `merfish_2000_genes/make_panels.sh` makes its panels: Boström assigns codewords in expression order, and MHD4 permutes the same codewords with `mhd4_shuffle_seed` 42. (These per-panel files, under `results/`, also keep an expression column.)
  - `make_panels.py` checks that `prior_csv` is the template's `expression.path` file, which `make_bostrom_panel.py` requires.
- **DUET.** Each panel's `run_config.yaml` is `merfish_2000_genes/config.yaml` with only these changes:
  - absolute paths;
  - the panel's own scratch dir;
  - `duet.lambda: [0.80]`;
  - one GPU;
  - `evaluator.num_cpus: 8` and `duet.num_cpus: 1`.

  So `seed` (42), the per-λ optimizer seed (842), the crowding physics and the 1,000 whole-brain crowding trials all match Fig 4c.
- **Cell-type crowding.** Each panel's `crowding_config.yaml` runs the evaluation below on that panel's three codebooks. The settings are 100 trials, seed 42 and matched density. Its anchor check compares each whole-brain row with that panel's own `metrics.csv` (tolerance 0.02) and each codebook with that panel's `results.csv` rows. The check runs after `per_class_metrics.csv` is written.
- **Simulation settings.** `cell_size_um` 12.0, `wavelength_nm` 500, `numerical_aperture` 1.4, `expansion_factor` 1.0, `abbe`, `box`; `BASE_TOTAL_READS` 8700 and a 200,000 read cap. Each class profile is an equal-weight roll-up of its clusters; the build asserts 34 classes and a capture within 25% of the prior.

### Why every panel reuses the Fig 4c PEP

The PEP cache key hashes the candidate pool **in order**, together with the `duet.pep` block (including its verbatim `channel_matrix_path` strings).
- **How the pool is built.** The pool is built position by position (`duet.merfish_benchmark.runner.build_candidates`). Position *i* is anchored by the warm-start codeword in row *i* and by each baseline's codeword for gene *i*. The factory then draws candidates per position with `sample_seed`, and pool order is first occurrence.
- **Why the anchors do not depend on the genes.** `make_bostrom_panel.py` sorts genes by expression and always uses the first 2,000 codewords of the Boström file in file order. The MHD4 shuffle is `default_rng(seed).permutation(2000)` over those rows. So each position's anchors depend only on the row index and the shuffle seed, never on the genes.
- **Result.** With `mhd4_shuffle_seed: 42` for every panel, every panel gets `merfish_2000_genes`' pool, and its PEP (fingerprint `a0559c0796d02bcb`, 38 GB) is reused. `tests/test_merfish_benchmark_build_candidates.py` covers both properties.
- **The trade-off.** The MHD4 baseline uses the same rank→codeword permutation in every panel. Its genes still differ, so its assignment is still expression-agnostic. But the assignment's own randomness is not resampled.
- **Before any DUET run.** `check_pep_cache.py` builds each panel's pool with `build_candidates` and stops unless `PEPMatrixProvider.cache_status` reports a symmetrized-cache hit. A miss would cost about 4 h on 4 GPUs and about 38 GB per panel.
- **Working directory.** The channel paths in the fingerprint are relative to the working directory. So `run.sh` runs from this folder, which sits at the same depth as `merfish_2000_genes`.

## Outputs

Everything goes under `results/experiments/merfish_expression_prior/`.

| File | Contents |
|---|---|
| `crowding cell type robustness.{svg,pdf,png}` | **Supp S4b**: per class and method, mean ± SEM over panels; dashed lines are each method's whole-brain value, averaged over panels |
| `per_class_cpm_wide.csv`, `class_summary.csv` | build step: 32,285 genes × 34 class CPM columns; clusters per class |
| `<panel>/genes.csv`, `bostrom_hw5_panel.csv`, `mhd4_hw5_panel.csv` | the gene set and its baseline panels |
| `<panel>/run_config.yaml`, `<panel>/duet/` | the DUET run (`results.csv`, `metrics.csv`, `selected_codewords_lambda0.80.csv`, ...); log `<panel>/duet.log` |
| `<panel>/crowding_config.yaml`, `<panel>/crowding/per_class_metrics.csv` | 105 rows per panel (3 whole-brain references + 3 methods × 34 classes); the build tables are symlinked in; log `<panel>/crowding.log` |
| `per_panel_class_metrics.csv` | all panels' per-class rows, with a `panel` column |
| `panel_summary.csv` | per panel: whole-brain identified fraction and decode accuracy per method, DUET wins, min and median gap |
| `class_across_panels.csv` | per class: mean, SD and SEM over panels for each method and for DUET minus each baseline, plus DUET's win count |
| `logs/` | `build{,.raw}.log`, `panels.log`, `check.log`, `aggregate.log` |

Disk: about 92 MB per panel, mostly `duet/candidates.csv`; 1.9 GB in all.

The legend reads `duet_label: "DUET (λ=0.80)"`. The column and method names are the `METHOD_PALETTE` keys.

## Compare after a re-run

Values of the reference 20-panel run (p02-p20 on 2026-09-30, p01 on 2026-10-01):

- **DUET beats both baselines in all 680 (panel, class) pairs**, and wins every class in every panel.
- **Gap vs the better baseline.**
  - Median over classes, per panel: 0.043-0.133; the median of these is 0.063.
  - Median over classes of the gap's mean over panels: 0.069. The whole-brain gap, averaged over panels, is also 0.069.
  - Smallest single gap: 0.023 (p04, Immune). p01's smallest gap is 0.064.
  - Smallest class mean: OEC +0.052 (SD 0.018).
- **p01 (`gene_seed` 0).** Whole-brain identified fraction: DUET 0.8185, Boström 0.7521, MHD4 0.7319. Its anchor check passed (largest difference 0.003, tolerance 0.02).
- **Error bars.** Median over classes of the SEM over panels: DUET 0.0062, Boström 0.0108, MHD4 0.0128.
- **Malat1 panels.** Three panels (p06, p16, p18) contain *Malat1*, the most expressed gene of the whole-brain table (p18 also has *Meg3*, the second). That one gene carries 19-28% of the panel's expression (most panels' top gene carries 3-10%).
  - In those panels the baselines fall to 0.58-0.71 whole-brain identified fraction. DUET holds 0.77-0.82, so the gap is 0.11-0.13.
  - They roughly double the baselines' panel-to-panel SD. Without them (17 panels): median SEM DUET 0.0057, Boström 0.0075, MHD4 0.0080.
  - They come from the same gene universe as every other panel (and as the Fig 4c genes), so they are kept.
- **Decode accuracy barely depends on the panel.** DUET is 0.9337-0.9340 and both baselines are 0.925 in every panel (same codeword pool and PEP).
- **p01's genes.** It shares 120 of its 2,000 genes with the Fig 4c set. Its top gene (*mt-Co1*) carries 7.6% of panel expression, within the 3-10% of most panels.

## Cost

Measured on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs on 20 physical cores) on 2026-09-30.

| Step | Time |
|---|---|
| `build` | 32-34 s, 4.1 GB peak RSS (re-measured 2026-09-23) |
| `panels` + pre-flight check | about 8 min for 20 panels (the check builds each panel's pool) |
| DUET, per panel, one process | 1h36m-1h42m with 5 panels at once; 3h24m-3h37m with 15 at once; 50m19s alone (p01, 2026-10-01) |
| decode evaluation + whole-brain crowding at 1,000 trials | about 4 min per panel |
| cell-type crowding, per panel, one core | about 17 min |

- **Wall clock.**
  - The 5-panel pilot (p01-p05) took 2h06m.
  - p06-p20, 15 at once, took 4h12m, while other jobs also ran on the machine.
  - With the default `MAX_JOBS` of 10, a full 20-panel run should take about 4-5 h (extrapolated).
  - Re-running only p01 took 1h18m in all: `panels` 30 s, pre-flight 7 min, its `run_merfish.py` 56 min, crowding 15 min, aggregate 4 s.
- **CPU use.** DUET's swap search is one process with no GPU path. Its numpy calls go through OpenBLAS, which starts one thread per logical CPU (80 threads on the reference machine); no thread cap is set.
  - Alone (p01, 2026-10-01) the run used about 25-28 cores, and its DUET step took 50m19s.
  - With 5 runs at once each got about 2 cores and took about 1h40m. With 15 at once the threads oversubscribed the machine and ran at half that speed.
  - So 14 times the cores bought a 2x speed-up. A cap such as `OPENBLAS_NUM_THREADS=4` would likely keep most of it on a shared machine. It is untested: check on the smoke test that it leaves the codebooks unchanged before using it for paper runs.
- **GPU.** Under a minute per panel. No PEP is built.

## Determinism

- **Knobs that change the numbers.**
  - The gene seeds and `mhd4_shuffle_seed`.
  - Everything the DUET template fixes (see `merfish_2000_genes/README.md`, Determinism).
  - `n_trials` and `seed` of the cell-type evaluation (trial t is seeded `seed + t`).
  - The simulation settings, and the expression tables including their row order.
  - Do not de-duplicate `per_class_cpm_wide.csv`. Its 40 extra rows for repeated gene symbols are summed by the simulator, and removing them changes the RNG stream.
- **Knobs that change the cost only.**
  - `MAX_JOBS`, `evaluator_num_cpus` and `gpus` affect speed only. The decode evaluation runs on a GPU. On CPU with NLL decoding metrics, `num_cpus` 1 versus 2 or more can change PEP counts; values of 2 or more agree ([Numerical determinism](../README.md#numerical-determinism)).
  - The per-panel device (one GPU instead of `gpu:all`) does not change the evaluation. With the Fig 4c genes as a panel, its baselines scored 0.9249566 decode accuracy, as in Fig 4c.
  - `classes` restricts the evaluation. Each evaluation is seeded independently, so a subset gives the same values for the classes it keeps.
- **Single-λ runs.** A panel run at λ = 0.80 alone gives the sweep's λ = 0.80 codebook. With the Fig 4c genes as a panel, the per-panel pipeline (`build_candidates`, the per-panel configs, one λ, one GPU) rebuilt `merfish_2000_genes`' baseline panels (the same genes and codewords, in the same order), and its λ = 0.80 codebook equalled `merfish_2000_genes`' `selected_codewords_lambda0.80.csv` (2000/2000 (Gene, Sequence) pairs). Its DUET decode accuracy was 0.9339 vs Fig 4c's 0.9338 with the same codebook: its DUET-only codewords take different union rows (and Monte Carlo streams) when only one λ is evaluated.

## Smoke test

`VARIANT=smoke bash experiments/merfish_expression_prior/run.sh` reads the tracked `config.smoke.yaml`. It is chained on the `merfish_2000_genes` smoke run:
- `duet_template` is `../merfish_2000_genes/config.smoke.yaml`, so each panel inherits its small pool, sample counts, λ budget and smoke PEP cache;
- panels p01 and p02 (`gene_seed` 0 and 1);
- cell-type `n_trials` 2 on two classes (`01 IT-ET Glut`, `31 OPC-Oligo`);
- anchor tolerance 1e-9;
- outputs under `results/smoke/`.

It needs the same inputs as the real run. Expected, with `VARIANT=smoke bash experiments/merfish_expression_prior/run.sh panels duet crowding aggregate` on the existing `merfish_2000_genes` smoke outputs:
- It ran in 1m35s and exited 0.
- Both panels hit the smoke PEP cache (`af3edd820b240197`).
- Every whole-brain anchor and codebook check matched exactly (tolerance 1e-9).

`VARIANT=smoke bash experiments/run_all.sh merfish_2000_genes merfish_expression_prior` runs the whole chain, including the build step.
