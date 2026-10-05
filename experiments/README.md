# Paper experiments

Re-runnable configs for the panels of the DUET paper, one sub-folder per experiment.
Folder names describe content, not figure numbers, because figure numbers change in revision.

Each folder holds a `run.sh` driver, its configs and a `README.md`. The README says what the
experiment makes, how to run it, its inputs and outputs, the values to compare after a run, its cost,
its determinism and its smoke test. Setup, downloads and the conda environments are described in
[docs/reproducing_the_paper.md](../docs/reproducing_the_paper.md). Every input file, with its size,
sha256 and source, is listed in [INPUTS.md](INPUTS.md).

## Panels and experiments

| Panel(s) | Folder | Needs, beyond the repository |
|---|---|---|
| Fig 1b (PEP grid glyph only) | [`pep_grid_glyph/`](pep_grid_glyph/README.md) | nothing |
| Fig 2a,b | [`synthetic_objective_correlation/`](synthetic_objective_correlation/README.md) | nothing |
| Supp S1 | [`synthetic_hvr/`](synthetic_hvr/README.md) | GPUs |
| Fig 3b (both panels), Supp S2a-g | [`ops_crispri_symmetric/`](ops_crispri_symmetric/README.md) | GPUs, `ops` env |
| Fig 3c | [`ops_crispri_rounds_error_sweep/`](ops_crispri_rounds_error_sweep/README.md) | GPUs, `ops` env |
| Fig 3d-f, Supp S3c | [`ops_crispick_genome_wide/`](ops_crispick_genome_wide/README.md) | GPUs, `ops` env, the CRISPick table, the NIS-seq channel matrices (if not in your copy), 2.5 TB of disk |
| Supp S3a | [`nisseq_error_analysis/`](nisseq_error_analysis/README.md) | the `notebooks` extra, the NIS-seq spot-level base calls and the Brunello table |
| Supp S3b | [`ops_crispri_cross_eval/`](ops_crispri_cross_eval/README.md) | GPUs, `ops` env, the NIS-seq channel matrices (if not in your copy) |
| Fig 4b | [`merfish_zhang2023_v2/`](merfish_zhang2023_v2/README.md) | GPUs, the Zhang et al. codebook, the WMB-10X expression table |
| Fig 4c,d,e | [`merfish_2000_genes/`](merfish_2000_genes/README.md) | GPUs, the WMB-10X expression table |
| Supp S4a | [`wmb10x_landscape/`](wmb10x_landscape/README.md) | `scanpy_env`, the WMB-10X atlas files, network |
| Supp S4b | [`merfish_expression_prior/`](merfish_expression_prior/README.md) | GPUs, `scanpy_env`, the WMB-10X atlas files and expression table, the `merfish_2000_genes` outputs |

[Inputs](#inputs) below says how to get each of these.

- **Fig 1b.** Fig 1 is a schematic. The glyph's values are illustrative (three Hamming-distance
  tiers, no numbers), and the published glyph was edited at figure assembly, so regenerating it
  does not rebuild Fig 1b.
- **S3a / S3b / Fig 3d-f.** Three different NIS-seq fits feed these panels: the first 5,000,
  50,000 and 100,000 reads of each file (S3a, S3b and Fig 3d-f/S3c). `nisseq_error_analysis`
  regenerates all three and checks them against the matrices in the repository, when your copy
  includes them.
- **S4b.** `merfish_expression_prior` evaluates 20 resampled 2,000-gene panels, each with its own
  baselines and DUET codebook, and S4b's error bars are the standard error over panels. Every
  panel is a seeded uniform gene draw (seeds 0-19). Every panel reuses `merfish_2000_genes`' PEP
  cache, so it needs no GPU time beyond a minute per panel, but about 4-5 h of CPU.

## Running

```bash
conda activate duet                             # the env from environment.yml (see Environments)
bash experiments/<name>/run.sh                  # one experiment, from any cwd
bash experiments/run_all.sh                     # every entry in order; stops at the first failure
bash experiments/run_all.sh synthetic_hvr ...   # a subset, still in order; unknown names are rejected
bash experiments/run_all.sh ops_crispick_genome_wide:cross_eval   # one step of the genome-wide set
UNATTENDED=1 bash experiments/run_all.sh ...    # carry on past failures (see Unattended runs)
DRY_RUN=1 bash experiments/run_all.sh ...       # preflight and plan only; runs nothing
VARIANT=smoke bash experiments/<name>/run.sh    # the smoke config instead of the real one
VARIANT=smoke bash experiments/run_all.sh       # the whole smoke suite (see Smoke variant)
```

**Entries and order.** `run_all.sh` runs these entries, in this order, whatever order they are
named in: `synthetic_objective_correlation`, `synthetic_hvr`, `ops_crispri_symmetric`,
`ops_crispri_cross_eval`, `merfish_zhang2023_v2`, `merfish_2000_genes`, `merfish_expression_prior`,
`wmb10x_landscape`, `nisseq_error_analysis`, `pep_grid_glyph`, `ops_crispri_rounds_error_sweep`, then
`ops_crispick_genome_wide:position_varying_asymmetric`, `:symmetric`, `:position_varying`,
`:asymmetric`, `:cross_eval` and `:visualize`.
- The genome-wide experiment is one entry per step of its `run.sh`, run as `run.sh <step>`. So one
  failed arm (16-19 h each) costs that arm, not the others, and each has its own status row.
- A subset argument is an experiment folder (all of its entries) or one entry (`folder:step`).
- The Fig 3d-f arm runs first among the four arms, so it finishes earliest.

**Dependencies.** Hard dependencies (the dependent reads the other's outputs):

```text
merfish_2000_genes ──────────────────────────────▶ merfish_expression_prior
ops_crispick_genome_wide:position_varying_asymmetric ┐
ops_crispick_genome_wide:symmetric                   ├─▶ ops_crispick_genome_wide:cross_eval
ops_crispick_genome_wide:position_varying            │
ops_crispick_genome_wide:asymmetric                  ┘
```

Soft dependencies never block; their state is noted in the status file:
- `ops_crispri_rounds_error_sweep` ← `ops_crispri_symmetric`. The sweep's reference check compares
  its rounds-10 points with Fig 3b's outputs, and prints `SKIPPED` if those are absent.
- `ops_crispick_genome_wide:visualize` ← the four arms and `:cross_eval`. It renders whatever exists,
  then exits non-zero if anything was missing. So Fig 3d-f still render if the cross-eval fails.

Dependencies count only among the entries selected in one invocation: an unselected dependency is
assumed done, and each `run.sh` checks its own inputs. For example, `merfish_expression_prior/run.sh`
stops before its DUET step unless every panel hits `merfish_2000_genes`' PEP cache, and `:cross_eval`
stops unless all four arms' `results.csv` exist and share one pool.

**Unattended runs (`UNATTENDED=1`).**
- **Default mode** (`UNATTENDED` unset or 0) stops at the first failure and exits with that entry's
  code.
- **Unattended mode** carries on. An entry whose hard dependency failed, was skipped or was
  interrupted becomes `skipped-dependency`, and every other entry still runs. The exit status is 0
  only if every selected entry is ok.
- **Status file, in both modes.** `results/experiments/run_all/<id>/status.tsv`, or
  `results/<VARIANT>/run_all/<id>/` with a variant. `<id>` is `RUN_ALL_ID` or the start time, and
  `run_all/latest` points at the newest. The path is printed at start.
  - One row per selected entry: `entry`, `state`, `exit`, `started`, `finished`, `elapsed`, `note`,
    `depends` and `log`. States are `pending`, `running`, `ok`, `failed`, `skipped-dependency` and
    `interrupted`.
  - The `#` lines give the driver's state, exit code and pid, the host, mode, variant, HEAD and
    arguments.
  - The file is rewritten atomically (temp file plus `mv`) on every change, so `cat` it at any time,
    for example over ssh. Readable view: `grep -v '^#' status.tsv | cut -f1-7 | column -t -s $'\t'`.
  - Entries that did not run stay `pending`, with a note saying why: stopped at a failure,
    interrupted, or preflight failed.
- **Per-entry logs.** Each entry's stdout and stderr also go to `run_all/<id>/<entry>.log` (`:`
  becomes `.`). Every experiment still writes its own logs under its outdir.
- **Stopping.** SIGINT, SIGTERM or SIGHUP stops the running entry and every process under it, marks it
  `interrupted` and exits 128 + the signal number. That covers Ctrl-C, `kill <pid>` with the pid on
  the status file's `driver:` line (not a `conda run` pid), and closing the tmux session. The driver
  records the entry's whole process tree before signalling, sends it SIGTERM, waits up to 30 s for
  every recorded process to exit, then sends SIGKILL to any that remain. A process that survives
  even that is named in the status file's driver note.
- **How entries run.** Each entry runs in the background with stdin from `/dev/null`, and `run_all.sh`
  waits for it. So a signal is handled at once, and keystrokes in the terminal never reach a run.
- **Resuming.** After a failure, re-run the failed entries and their dependents in a new invocation,
  for example `UNATTENDED=1 bash experiments/run_all.sh ops_crispick_genome_wide:symmetric
  ops_crispick_genome_wide:cross_eval ops_crispick_genome_wide:visualize`. PEP caches are reused
  (`force_rebuild: false`). `SKIP_COMPLETE=1` makes the Fig 3c sweep skip points that already
  finished.

**Guards.** Guards run before any work. Each config-driven `run.sh` stops if its config is
missing, and every `run.sh` except `wmb10x_landscape`'s stops unless the active env imports `duet`
from this repo's `src/`. Reading a config key that is missing or null also stops the driver.
`run_all.sh` first checks the inputs that are not in the repository for the selected experiments
(the Zhang v2 codebook, the WMB-10X expression table, the WMB-10X atlas h5 and gene table), so a
missing one fails at once rather than hours into the run, and prints how to get it.
`merfish_expression_prior/run.sh` checks the same two atlas files.
`wmb10x_landscape/run.sh` refuses to start a run that includes step 2 unless both are in
`data/raw/WMB-10X/`, so it never starts the 1.38 GB download.

`run_all.sh`'s preflight also checks the following, each mirroring the experiment's own `run.sh`:
- **S3a inputs.** The 8 NIS-seq spot files and `Brunello_sgRNAs.txt`, as listed in
  `nisseq_error_analysis/config.yaml` (`input_sha256`), with its `NISSEQ_DATA_DIR` /
  `NISSEQ_LIBRARY_PATH` overrides. The experiment's `run.sh` also checks their sha256.
- **The CRISPick table** named by each selected genome-wide arm.
- **The NIS-seq channel matrices** named by the selected cross-eval and genome-wide configs.
- **The Feldman env.** The conda env named by `feldman.conda_env` (`ops`) must import
  `ops.pool_design`. This is checked whenever an experiment that runs Feldman is selected
  (`ops_crispri_symmetric`, `ops_crispri_cross_eval`, the sweep, or a genome-wide arm), because
  Feldman runs hours into each of them.
- **The genome-wide guards.** `ops_crispick_genome_wide/run.sh check <steps>` runs that driver's own
  guards for the selected steps: configs, `duet`, the candidate table, the Feldman env, and the
  home-drive guard, which refuses caches or scratch under `$HOME`.

A preflight failure stops the run in both modes, and is recorded in the status file.

**Smoke variant.** With `VARIANT=smoke`, each `run.sh` reads `config.smoke.yaml` beside the real
config, for example `VARIANT=smoke bash experiments/synthetic_hvr/run.sh`. The cross-eval and
genome-wide configs use `<name>.smoke.yaml`, and the Fig 3c sweep uses `<stem>.smoke.yaml` (three
points; `python make_configs.py --smoke` writes them). The smoke configs are in the repository. They
are shrink-copies of the real configs (fewer trials, samples and λ values, smaller pools) that write
under `results/smoke/` and keep the code path the real run takes. For example, the
`merfish_2000_genes` smoke λ list contains 0.80 and 0.90 (read by `make_figures.py` and S4b) and keeps
all 2,000 genes (S4b's panel check). A smoke run reads the same input files as the real run.
`wmb10x_landscape` has no config: `VARIANT=smoke` runs steps 1-2 only and moves its logs, scratch and
numba cache under `results/smoke/`; `WMB10X_STEPS` picks steps (default `"1 2 3 4"`). On the
reference hardware the smoke suite took about 10 min for the first eight experiments. The last four
smokes took about 1 min (S3a), 3 s (glyph), 1 min (sweep) and 30 min (genome-wide) when each was run
on its own. The sweep smoke's reference check reads the `ops_crispri_symmetric` smoke outputs, which
run earlier in the suite.

**Environments.**

| Env | Used by | How it is selected |
|---|---|---|
| `duet` from `environment.yml`, or any env with this clone installed editable with the `benchmark` extra; add the `gpu` extra for the GPU experiments and the `notebooks` extra (`pip install -e ".[benchmark,notebooks]"`) for `nisseq_error_analysis` | every runner, visualizer and script that imports `duet`; also the S3a notebook's kernel | activate it first; `run.sh` calls the active `python`. S3a runs `python -m nbconvert --execute`, whose `python3` kernel resolves to the same env |
| `ops` (`environments/ops.yml`: Python 3.7, Feldman et al. `ops` package) | Feldman baseline in the four OPS experiments | `feldman.conda_env: ops` in the OPS configs (`conda run -n ops`) |
| `scanpy_env` (`environments/scanpy.yml`: h5py, umap-learn 0.5.11; no `duet`) | all four `wmb10x_landscape` steps; step 1 of `merfish_expression_prior`; `scripts/wmb10x/build_whole_brain_cpm.py` | `SCANPY_ENV` (env name, default `scanpy_env`) or `SCANPY_PYTHON` (interpreter path; a relative one is taken from the caller's cwd) |

Keep the names `ops` and `scanpy_env`: the configs and drivers use them
([environments/README.md](../environments/README.md)).

`wmb10x_landscape/run.sh` needs no active env. It puts `src/` on `PYTHONPATH` so step 4 can import
`duet.plotting`. The GPU configs (`device: "gpu:all"`) are `synthetic_hvr`, the four OPS experiments
and both MERFISH experiments. Their `run.sh` prints `nvidia-smi` output first, to the terminal only
(it runs outside `script(1)`, so it is not in `<outdir>/logs/`). `nisseq_error_analysis` and
`pep_grid_glyph` are CPU only.

## Cost on the reference hardware

Reference hardware: 4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs (2 Intel Xeon E5-2640
v4), 220 GB RAM, shared with other jobs. The wall clock below is from full runs of these drivers
and configs. The notes break it down where a log or an earlier run of the same experiment
measured the parts.

| Experiment | Hardware | Wall clock | Notes |
|---|---|---|---|
| synthetic_objective_correlation | CPU, one process | 7m26s | About 1 min of it is figures. |
| synthetic_hvr | 4 GPUs; swap search on CPU | 17m06s | About 60% is exhaustive-front scoring (including DUET setup). |
| ops_crispri_symmetric | 4 GPUs; DUET swaps and baselines on CPU | 31m16s | 5 trials of about 4 min, then about 8-10 min of figures. |
| ops_crispri_cross_eval | 4 GPUs; CPU as above | 2h06m | 4 phase-1 runs of about 20 min plus about 4 min of figures each, then the cross-eval, about 6 min. |
| merfish_zhang2023_v2 | 4 GPUs | 3h19m | Mostly the PEP: in a 2h56m run, PEP 2h26m (GPU pass 2h07m, host transpose and symmetrize 19m) and DUET 9-λ sweep 29m. |
| merfish_2000_genes | 4 GPUs | 6h04m | Mostly the PEP: in a 4h01m run, PEP 2h33m and DUET 9-λ sweep 1h25m. |
| merfish_expression_prior | CPU; under a minute of GPU per panel | about 4-5 h (extrapolated) | 20 panels with `MAX_JOBS` 10: each panel about 1h40m of DUET with 5 at once (50 min alone) and about 17 min of cell-type crowding. A 5-panel pilot took 2h06m; 15 panels at once took 4h12m. See its README. |
| wmb10x_landscape | CPU | 5m56s | Step 3 (`embed_centroids.py`, 7 UMAP fits) dominates. |
| nisseq_error_analysis | CPU, one kernel process (multithreaded BLAS) | 34m59s | Another run took 21m06s for all three sizes (N = 5,000 / 50,000 / 100,000 per file: 1m11s / 7m34s / 12m09s). Needs about 10 GB of free RAM. |
| pep_grid_glyph | CPU | 2 s | Almost all Python start-up. |
| ops_crispri_rounds_error_sweep | 4 GPUs; DUET swaps (31 workers) and baselines on CPU | 8h50m03s | 18 points, run one after another, all 90 PEPs built cold. The DUET sweep took 379 min. PEP cache about 46.5 GB. |
| ops_crispick_genome_wide | 4 GPUs; DUET swaps, Sivanandan and Feldman on CPU | 75h50m | Four arms 68h36m (15h33m-18h51m each), cross-eval 5h48m, visualize 1h26m. Disk below. |

All twelve take about 4 days, most of it the genome-wide arms.

The genome-wide PEP covers 289,113 unique 14-mers. Per arm, a full run measured:
- PEP 5h45m-7h14m. The host transpose and symmetrize are I/O-bound and vary.
- DUET 18-λ sweep 3h34m-5h45m.
- Sivanandan 2h46m-2h59m and Feldman 19 min.
- Evaluation 2h38m-4h24m, the symmetric arm longest.

The S3c cross-eval took 5h47m41s. Its re-scoring of 74 codebooks with a batched matrix multiply
took 1h46m of that. Details: [`ops_crispick_genome_wide/README.md`](ops_crispick_genome_wide/README.md).

Two genome-wide cost notes:
- **Disk.** The PEP cache is 334 GB per arm (two 167.2 GB U × U uint16 files, measured), plus 167 GB
  transient during symmetrization, and 1.34 TB for the four arms. Evaluation scratch peaked at
  239-513 GB per arm and 368 GB for the cross-eval (from the logged nonzero counts), and is removed
  at exit. The cumulative peak on the data volume is about 1.71 TB, during the cross-eval. Keep at
  least 2.5 TB free: a killed evaluation can leave up to about 0.5 TB of scratch behind.
  - `results/cache` and `results/scratch/ops_crispick_genome_wide` must point at a large data volume.
    Both are gitignored, so they can be links to it ([docs/reproducing_the_paper.md](../docs/reproducing_the_paper.md)
    shows the commands). The driver's home-drive guard refuses caches or scratch under `$HOME`
    unless `ALLOW_HOME_SCRATCH=1` is set.
  - Details: `ops_crispick_genome_wide/README.md`, Disk and memory.
- **Memory.** The reference machine has 220 GB and is shared. RSS reached 149-192 GB, almost all of
  it file-backed (the mmap'd PEP and CSR), with no step short of memory (the `[mem]` snapshots, taken
  between stages, logged at most 14.9 GB anonymous). Anonymous memory:
  - the PEP symmetrize batches up to 64 GB by design (`mem_budget_gb: 64.0`);
  - the DUET λ workers need little (about 0.6 GB each) beyond the shared page cache;
  - the evaluation (each arm's last stage, and the cross-eval) peaks at about 1.5-2.5× its batch
    budget plus a 6.5-8.6 GB copy of the CSR row pointers. At a budget of 64 that would be about
    130-145 GB per evaluation, so the arms set `evaluator.mem_budget_gb: 20.0` and the cross-eval
    `eval_mem_budget_gb: 20.0` (the key's default): about 35-55 GB. The budget changes memory only,
    not results.

Two MERFISH cost notes:

- Each MERFISH config builds its PEP cold on its first run. Budget about 40 GB of disk per config
  under `results/cache/<name>/` (38 GB measured), and about 60 GB transiently during symmetrization.
- MERFISH `crowding.n_trials` is 1000. The crowding stage took about 35 s at 100 trials; at 1000 it
  takes longer, by an amount not measured separately.

## Output files and panels

Paths are under `results/experiments/` unless noted. This table is the panel manifest: one row per
file that goes into the paper. The last column says why a default run writes the file (see Figure
tiers below):
- **aggregated**: a figure over all trials (for MERFISH, over the whole λ sweep). The visualizer
  writes it on every run.
- **`visualization.paper_panels` in `<config>`**: a per-trial or per-λ file, a debug plot by tier,
  that the config lists so it is written by default too.
- **hard-coded in `<script>`**: the same, for a script that exists for one experiment and names its
  panels in code.
- **untiered**: the experiment has no tiers and writes every figure on every run.

| Panel | File | Kept by default as |
|---|---|---|
| Fig 1b glyph (before the hand edits at assembly) | `pep_grid_glyph/pairwise_error_final.svg` | untiered |
| Fig 2a | `synthetic_objective_correlation/figures/baseline_comparison_per_trial/trial_01.svg` | hard-coded in `scripts/benchmark/synthetic/visualize_duet_vs_baseline_objectives.py` (`PAPER_PANELS`) |
| Fig 2b | `synthetic_objective_correlation/figures/summary_spearman.svg` | aggregated |
| Supp S1 | `synthetic_hvr/aggregate_hvr.svg` | aggregated |
| Fig 3b left | `ops_crispri_symmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | aggregated |
| Fig 3b right | `ops_crispri_symmetric/guide_level_comparisons/all_baselines_vs_duet_97p5pct_trial_3_distributions.svg` | `visualization.paper_panels` in `ops_crispri_symmetric/config.yaml` |
| Fig 3c top | `ops_crispri_rounds_error_sweep/sweep_figures/heatmap_maximum_activity_mean_decode_accuracy_10th_percentile.svg` | hard-coded in `ops_crispri_rounds_error_sweep/visualize_sweep.py` (`FIG3C_PANELS`) |
| Fig 3c bottom | `ops_crispri_rounds_error_sweep/sweep_figures/heatmap_maximum_activity_standard_deviation_decode_accuracy.svg` | hard-coded in `ops_crispri_rounds_error_sweep/visualize_sweep.py` (`FIG3C_PANELS`) |
| Fig 3d | `ops_crispick_genome_wide/position_varying_asymmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | aggregated |
| Fig 3e | `ops_crispick_genome_wide/position_varying_asymmetric/pareto_fronts/10th_percentile_decode_accuracy_aggregated.svg` | aggregated |
| Fig 3f | `ops_crispick_genome_wide/position_varying_asymmetric/pareto_fronts/95th_over_5th_percentile_decode_accuracy_aggregated.svg` | aggregated |
| Supp S2a | `ops_crispri_symmetric/hypervolume/normalized_hypervolume_by_method_group.svg` | aggregated |
| Supp S2b | `ops_crispri_symmetric/pareto_fronts/10th_percentile_decode_accuracy_aggregated.svg` | aggregated |
| Supp S2c | `ops_crispri_symmetric/pareto_fronts/95th_over_5th_percentile_decode_accuracy_aggregated.svg` | aggregated |
| Supp S2d | `ops_crispri_symmetric/guide_level_comparisons/max_activity_97p5pct_trial_5_jointplot.svg` | `visualization.paper_panels` in `ops_crispri_symmetric/config.yaml` |
| Supp S2e | `ops_crispri_symmetric/guide_level_comparisons/Feldman_et_al_ED1_vs_duet_97p5pct_trial_5_jointplot.svg` | `visualization.paper_panels` in `ops_crispri_symmetric/config.yaml` |
| Supp S2f | `ops_crispri_symmetric/guide_level_comparisons/Sivanandan_et_al_ED1_vs_duet_97p5pct_trial_5_jointplot.svg` | `visualization.paper_panels` in `ops_crispri_symmetric/config.yaml` |
| Supp S2g | `ops_crispri_symmetric/guide_level_comparisons/Sivanandan_et_al_ED2_vs_duet_97p5pct_trial_5_jointplot.svg` | `visualization.paper_panels` in `ops_crispri_symmetric/config.yaml` |
| Supp S3a | `nisseq_error_analysis/subsample5000/figures/error_lines_4_pctl50_subsample5000.svg` | untiered |
| Supp S3b | `ops_crispri_cross_eval/cross_eval/eval_position_varying_asymmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | aggregated |
| Supp S3c | `ops_crispick_genome_wide/cross_eval/eval_position_varying_asymmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | aggregated |
| Fig 4b left | `merfish_zhang2023_v2/figures/crowding_pareto_front.svg` | aggregated |
| Fig 4b right (inset) | `merfish_zhang2023_v2/figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg` | `visualization.paper_panels` in `merfish_zhang2023_v2/config.yaml` |
| Fig 4c left | `merfish_2000_genes/figures/crowding_pareto_front.svg` | aggregated |
| Fig 4c right (inset) | `merfish_2000_genes/figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg` | `visualization.paper_panels` in `merfish_2000_genes/config.yaml` |
| Fig 4d | `merfish_2000_genes/figures/fig4de/round_expression_uniformity_y0_k2000.svg` | hard-coded in `merfish_2000_genes/make_figures.py` |
| Fig 4e | `merfish_2000_genes/figures/fig4de/expr_by_hw_violin_k2000.svg` | hard-coded in `merfish_2000_genes/make_figures.py` |
| Supp S4a panel | `results/wmb10x/figures/wmb_umap_panel_nokey.svg` (fixed path; caption text in `results/wmb10x/captions.md`) | untiered |
| Supp S4a key | `results/wmb10x/figures/wmb_umap_key.svg` (fixed path) | untiered |
| Supp S4b | `merfish_expression_prior/crowding cell type robustness.svg` | untiered |

- Fig 3b right and S2d-g use the 97.5%-activity DUET arm. The expected pick is λ = 0.125, labelled
  `DUET (λ=0.12)` because labels use `:.2f`. The per-λ seeds differ from those of the paper's
  runs, so this is an expectation, not a guarantee.
- The Fig 4b,c insets are the λ = 0.80 sweep files. They are the only jointplot-sweep files a
  default run writes. Do not use the canonical `figures/decode_vs_identified_jointplot.svg`
  (written only with `--debug-plots`): its matched rule picked λ = 0.50 on Zhang v2.
- Fig 4d,e use λ = 0.90. S4b uses DUET λ = 0.80.
- S4a uses the `_nokey` panel plus the separate key, not `wmb_umap_panel.svg`.
- **Fig 3c** is the bottom-decile mean ("mean decode accuracy ≤ 10th percentile"), as a gain over
  maximum activity (top), and the relative change in the inter-guide SD (bottom).
  - Each cell uses one DUET arm per point and trial: the highest mean decode accuracy at ≥ 97.5% of
    maximum activity.
  - `sweep_figures/` also holds `sweep_summary.csv` and six diagnostic heatmaps: mean decode
    accuracy against each baseline. The other 52 heatmaps are written only with `--debug-plots`.
  - The sweep's visualizer ends with its reference check against `ops_crispri_symmetric`. A Tier A
    mismatch exits 1 after the figures are written.
- **Fig 3d-f** use the `_aggregated` panels of the position-varying asymmetric arm, and the 97.5%
  rule should pick λ = 0.07.
  - Feldman et al. ED=1 appears as a point. ED=2 is one guide short at this scale, so it is invalid
    and dropped.
  - With one trial, the panels draw 4 pt points with no error bars, and square markers in the
    legend only.
  - Sivanandan et al. is invalid at every HD. The panels draw its legend row with "N/A"
    themselves. In Fig 3f, maximum activity and Feldman ED=1 also read "N/A": their 5th percentile
    is 0, so the ratio is undefined.
- **S3c** legend labels come from the config: "DUET (Uniform)", "DUET (Positional)", "DUET
  (Channel-wise)", "DUET (Positional, channel-wise)" and "Feldman et al.".
  - With one trial, S3c also uses 4 pt points and square legend markers.
  - Sivanandan et al. is not a cross-eval source (see `ops_crispick_genome_wide/README.md`, S3c
    baselines), so its "N/A" row, if wanted, is an assembly edit.
  - The Feldman source also carries the maximum-activity anchor, as in S3b.
- **S3a** is the N = 5,000 fit. The run also writes the 50,000 and 100,000 panels and matrices, as
  provenance checks for S3b and for Fig 3d-f / S3c; see `nisseq_error_analysis/README.md`.
- **Fig 1b**: the regenerated glyph is the panel before the edits made at figure assembly (codeword
  glyphs, colour bar, 8 pt labels and titles).
- `run_all.sh` writes its status file and per-entry logs under `run_all/<id>/` here (see Running).

## Figure tiers

The visualizers sort their figures into three tiers (`src/duet/plotting/tiers.py`):

1. **Paper panels**: the files in the manifest above and in each experiment's Outputs table. Always
   written, at the same path and under the same name as before.
2. **Diagnostics**: the trial-aggregated figures you need to choose λ, such as the Pareto front of
   every metric, the hypervolume bar charts and the λ-labelled MERFISH crowding front. Written by
   default.
3. **Debug plots**: per-trial copies, per-λ sweeps and everything else. Written only with
   `--debug-plots`.

- **The drivers never pass `--debug-plots`.** `run.sh` and `run_all.sh` write paper panels and
  diagnostics only. For the debug plots, re-render by hand with the flag: add it to the re-render
  command in the experiment's README. For example, from the repo root:

  ```bash
  (cd experiments/ops_crispri_symmetric && python ../../scripts/benchmark/visualize_benchmark.py --config config.yaml --debug-plots)
  (cd experiments/ops_crispri_cross_eval && python ../../scripts/benchmark/visualize_comparison.py --config eval_position_varying_asymmetric.yaml --debug-plots)
  (cd experiments/merfish_2000_genes && python ../../scripts/benchmark/visualize_merfish.py --config config.yaml --debug-plots)
  python experiments/ops_crispri_rounds_error_sweep/visualize_sweep.py --debug-plots
  PYTHONPATH=. python -m scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark --indir results/experiments/synthetic_hvr --debug-plots
  ```

- **Which scripts take the flag.** `visualize_benchmark.py`, `visualize_comparison.py`,
  `visualize_merfish.py`, `merfish_2000_genes/make_figures.py`,
  `ops_crispri_rounds_error_sweep/visualize_sweep.py`, and the synthetic
  `visualize_2d_synthetic_benchmark.py` and `visualize_duet_vs_baseline_objectives.py`.
  `visualize_baseline_summary.py` accepts it and ignores it: both its figures are aggregated.
- **Each run says what it skipped.** A tiered visualizer prints how many debug plots it skipped and
  names `--debug-plots`. A default run creates a folder only when it writes into it, so it leaves
  no empty `error_metrics/` or `*_sweep/` folders.
- **`visualization.paper_panels`.** A paper panel that is a debug-tier file (one trial or one λ,
  picked by hand) is listed in the experiment config:

  ```yaml
  visualization:
    paper_panels:
      - guide_level_comparisons/max_activity_97p5pct_trial_5_jointplot
  ```

  - Paths are relative to the visualizer's output dir, without the extension.
  - `visualize_benchmark.py`, `visualize_comparison.py` and `visualize_merfish.py` read the block
    from `--config` and write the listed files by default. At the end of a run they print a
    `WARNING` for each listed file that was not written (a typo, or a trial or λ the run did not
    produce). `visualize_benchmark.py --input` has no config, so it writes no listed files.
  - The runners ignore the block, and no cache fingerprint includes it.
  - Used by `ops_crispri_symmetric/config.yaml` (Fig 3b right, S2d-g) and both MERFISH configs (the
    λ = 0.80 inset).
- **Hard-coded panels.** Scripts that exist for one experiment name their panels in code:
  `visualize_sweep.py` (`FIG3C_PANELS`), `merfish_2000_genes/make_figures.py` (Fig 4d,e) and
  `visualize_duet_vs_baseline_objectives.py` (`PAPER_PANELS`, Fig 2a; it takes no config).
- **Untiered.** These take no flag and write every figure on every run:
  - `nisseq_error_analysis`: its notebook is a copy that changes only cell 1, so gating the save
    cells would break that rule, and `nbconvert` passes no arguments. It has no per-trial figures.
  - `pep_grid_glyph`: it writes only the panel. Its `rx` variants are already opt-in
    (`rx_variants` in the config).
  - `wmb10x_landscape`: it runs the live scripts unmodified, and `captions.md` captions all six
    figures.
  - `merfish_expression_prior` is not tiered either.
- **Old files stay.** The visualizers write in place and delete nothing. A default re-render into a
  folder that already holds debug plots (from a `--debug-plots` render, or from an older version of
  the visualizers) leaves them there. Render into an empty folder, or delete the debug folders, to
  see the default set alone.

Figures written per experiment, full configs:

| Experiment | Default | With `--debug-plots` |
|---|---|---|
| synthetic_objective_correlation | 3 | 22 |
| synthetic_hvr | 5 | 15 |
| ops_crispri_symmetric | 13 (8 aggregated, 5 listed panels) | 459 |
| ops_crispri_cross_eval | 40 (8 per arm, 8 for the cross-eval) | 1,884 |
| ops_crispick_genome_wide | 40 (8 per arm, 8 for the cross-eval) | 244 |
| ops_crispri_rounds_error_sweep | 8 heatmaps, plus `sweep_summary.csv` | 60 heatmaps |
| merfish_zhang2023_v2 | 9, each as SVG + PNG | 43 |
| merfish_2000_genes | 11 (9, plus 2 in `fig4de/`), each as SVG + PNG | 48 |

## Output layout

- **Outputs, logs, caches and scratch.** Outputs go to `results/experiments/<name>/` (the config's
  `outdir`) and logs to `<outdir>/logs/`. The cross-eval logs go to
  `results/experiments/ops_crispri_cross_eval/logs/`, and the wmb10x logs to
  `results/experiments/wmb10x_landscape/logs/`. PEP caches go to `results/cache/<name>/`. Scratch
  goes to `results/scratch/<name>/`. The GPU drivers and wmb10x point `TMPDIR` there (OPS and
  MERFISH use `<scratch_dir>/tmp`). The Fig 2, S4b, S3a and Fig 1b drivers set no `TMPDIR`; S3a
  and Fig 1b write no scratch.
- **Logs.** Every runner, visualizer and figure script tees its output to `<outdir>/logs/*.log`.
  The runners, the S4b build and evaluate steps and the four wmb10x steps run under `script(1)`.
  Each `script(1)` step leaves a verbatim `*.raw.log` and a cleaned `*.log`; if the step fails, the
  driver stops before writing the cleaned copy.
- **Exceptions to the layout.**
  - S1's PEP cache is hard-coded by the runner at `<outdir>/artifacts/pep_cache/`, not under
    `results/cache/`. The runner also ignores `evaluator.scratch_dir`, so `run.sh` sets `TMPDIR`
    instead.
  - `wmb10x_landscape` runs the scripts unmodified, and they write to fixed, gitignored repo paths:
    `data/raw/WMB-10X/`, `data/processed/WMB-10X/` (687 MB `centroids.npy`) and `results/wmb10x/`.
    Only its logs go under `results/experiments/`.
  - The MERFISH runner copies the YAML verbatim to `<outdir>/config.yaml`. The relative paths in that
    copy are relative to the experiment folder, not to `<outdir>`. The OPS runners do the same.
  - `nisseq_error_analysis` reads its inputs from the gitignored `data/raw/NISseq_HeLa_IL1b/` (a
    folder, or a link to one), and imports `error_utils.py` from its own folder. It sets
    `PYTHONDONTWRITEBYTECODE=1`, so it leaves no `__pycache__` there.
- **λ convention** ([ADR 0001](../docs/adr/0001-lambda-weights-decoding-accuracy.md)). The configs use
  J = λ·decode + (1-λ)·secondary, so λ = 1 is decode-only. Every λ is written as a literal decimal.
  Some config comments quote λ values of the paper's original runs, which used λ_old = 1 − λ.
- **Config headers.** Each config opens with a block listing how it differs from the configuration
  of the paper's original run. Some configs then keep that run's header. Its references to
  `README.md` and sibling configs mean files of the original run, which are not in this repository,
  and the S1 header's prose uses the old λ convention. Likewise, the `MIGRATED COPY of
  scripts/benchmark/archive/...` and `Source bundle:` lines at the top of configs, drivers and
  scripts name the as-run bundle each file was copied from; that bundle is not in this repository.

## Path convention

Every relative path in a config is relative to that config's folder. Each `run.sh` finds its folder
with `readlink -f` (so a symlinked driver works too) and `cd`s into it before calling anything.
Loaders that resolve paths against the cwd and loaders that resolve against the config folder
therefore agree, and the drivers work from any cwd.

- **Resolved against the config folder:** in MERFISH configs, `outdir`, `cache_dir`, `duet.cache_dir`,
  `expression.path`, `genes.path`, `initialization.path` and `baselines[].path`. Also every key the
  `merfish_expression_prior` scripts read (its per-panel configs carry absolute paths), the S3a keys `data_dir`, `library_path` and `outdir`
  (resolved by its `run.sh`), and the glyph's `outdir`. S3a's `provenance` paths are opened relative
  to the cwd by `check_provenance.py`, which `run.sh` runs from the folder.
- **Taken verbatim, relative to the cwd:** Fig 2 and S1 `outdir`; OPS `outdir`, `cache_dir`,
  `scratch_dir`, `candidate_pool.csv_path` (genome-wide) and noise-matrix `*_path` keys (all four OPS
  experiments); the comparison config's `outdir`, `reference_dir`,
  `methods[].path`, `evaluator.scratch_dir` and matrix paths; MERFISH `channel_matrix_path` and every
  `scratch_dir`.
- `scratch_dir` must exist before a run because `tempfile.mkdtemp` does not create it. `run.sh`
  creates every one its configs name: OPS `scratch_dir`, the comparison config's
  `evaluator.scratch_dir`, and MERFISH `evaluator.scratch_dir` and `duet.pep.scratch_dir`.
- The `WeissmanCRISPRi` pool CSV is located through the installed package, not the config. It is
  `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv`, so it needs the editable install.
- Unknown keys are ignored silently: an extra `lambdas:` key does nothing. The key is `lambda`, and it
  is required (a config with only `lambdas:` fails with `KeyError`). The exceptions are the strict
  `pool` and `evaluator` blocks of the synthetic configs, which reject unknown keys. Path strings
  enter the PEP cache fingerprint.

## Numerical determinism

The configs keep the machine-specific knobs (`device`, `num_cpus`, memory budgets) of the paper's
runs, with one exception: the genome-wide evaluation budget is 20 GB instead of 64 (memory only; see
the cost notes).

- **`device` changes numbers for NLL metrics.** The DUET PEP and the ground-truth evaluator draw
  identical random streams on CPU and GPU, but the arithmetic differs. The PEP is float32 on both
  devices, summed by different BLAS libraries (CPU BLAS vs cuBLAS). The ground truth is float64 on
  CPU and a float32 matrix multiply on GPU. Near-ties can flip in both. Measured on CPU by applying
  both kinds of ground-truth arithmetic to the same draws: 22,276 of 3.1M competitor comparisons flip
  for the synthetic asymmetric channel, and 41 of 14.4M for NIS-seq asymmetric. Measured CPU vs GPU
  (2026-09-23; 250 random codewords, K = 300, two probes with different codebooks): OPS NIS-seq
  asymmetric NLL 23 and 32 of 62,500 PEP cells differ, position-varying NLL 1 and 0,
  position-varying asymmetric NLL and Hamming 0. MERFISH (Zhang channel, asymmetric NLL, HW4 32-bit
  codewords): 4,375 and 4,495 of 62,500 PEP cells differ (by up to 21 of 300 counts), and 219 and 228
  of 250 codeword accuracies (by up to 0.12). A Hamming control on the Zhang channel matched exactly,
  and repeats on the same GPU were identical. Hamming costs are exact. This matters for S1, the NLL
  cross-eval configs and both MERFISH experiments. Keep the config's `device` when comparing numbers.
- **`num_cpus` never changes random draws, but on CPU it can change NLL arithmetic.** With NLL
  decoding metrics on CPU, `num_cpus` 1 and `num_cpus` 2 or more can give different PEP counts: one
  process uses multithreaded BLAS, while worker processes pin BLAS to one thread each. Values of 2 or
  more agree with each other. GPU runs are unaffected. The PEP config's `num_cpus` is also part of
  the PEP cache fingerprint: `duet.pep.num_cpus` for OPS and MERFISH, and `evaluator.num_cpus` for S1
  (its runner builds the PEP from the `evaluator` block). Changing it forces a rebuild, which is a
  new draw when unseeded.
- **Memory budgets and batch sizes affect speed and memory, not results.** `num_samples` is numerical.
- **OPS is seeded (ground truth 42, PEP 43), but differs from the paper's draws.** The paper's OPS
  runs left `evaluator.seed` and `duet.pep.seed` unset, so the ground truth and the PEP drew fresh OS
  entropy. They are now 42 and 43 (the cross-eval comparison config has only `evaluator.seed`; its
  top-level `seed` is required but unused), so a re-run on the same device should repeat the previous
  one byte for byte, as the `ops_uniform` regression fixture in `scripts/benchmark/regression/` does.
  On the `ops_crispri_symmetric` smoke, a second run of `config.smoke.yaml` with its own, cold PEP
  cache repeated `results.csv`, `trial_01/guides.csv` and the PEP counts byte for byte. Pools,
  initialization, Sivanandan and Feldman are deterministic. A run is still a new draw relative to the
  paper's unseeded runs: expect agreement with the paper's means within about ±0.005.
- **MERFISH is seeded (ground truth 42, PEP 43; the paper's runs used 42 for both), but its union
  evaluation is not stable across λ lists.** Each codeword's stream is indexed by its position in
  the union of all evaluated codewords (initialization, then DUET in ascending λ, then baselines).
  Changing the λ list or the baselines therefore shifts every stream.
- **Re-runs are new draws, not relabelings.** The per-λ optimizer seed is `seed + int(λ·1000)`
  (`src/duet/runner/core.py`). At temperature 0 this seed only breaks exact ties among maximal swap
  deltas.
- **Synthetic experiments reproduce bit for bit.** The Fig 2 parquet (240,240 rows x 12 columns)
  repeats the paper's bit for bit; its file hash differs only through parquet-writer metadata. All
  240 S1 per-cell HVR values repeat the paper's (median DUET 0.9531, Greedy NLL 0.7372, Greedy
  Hamming 0.7471). Every S1 pick at λ equals the paper's pick at λ_old = 1 − λ.
- **S3a** has no randomness: `head(N)` of tile-sorted files, and exact integer distances. Its
  matrices repeat the paper's byte for byte. Its SVGs are not byte-reproducible (matplotlib writes a
  date and random ids), so compare panels by coordinates.
- **The Fig 1b glyph** is written as text and repeats byte for byte.
- **Fig 3c** uses Hamming costs, which are integers, so CPU and GPU agree.
  - Its PEPs are rebuilt, not shared with Fig 3b: the `cache_dir` differs, but the fingerprint does
    not include the path. On the smoke, the rebuilt PEP equalled Fig 3b's byte for byte.
  - Decode statistics still differ from Fig 3b's at the same point. Ground-truth draws are indexed by
    position in the per-trial union of evaluated codebooks, and the sweep evaluates 13 more DUET arms.
- **The genome-wide arms and cross-eval use NLL metrics**, so `device` matters, as above. A second
  smoke run with a cold PEP repeated `results.csv`, the pool and the PEP counts byte for byte.
  - Sivanandan is deterministic. In the paper's runs it was not, because the pool's group order
    then followed the per-process string-hash seed. Expect it to stay invalid, but its counts will
    not match the paper's.
  - Feldman ED=1 is deterministic, and ED=2 is fixed by `PYTHONHASHSEED=0`.

## Inputs

[INPUTS.md](INPUTS.md) lists every input file the experiments read, with its size, sha256 and
source. Some inputs are not in the repository, or may be missing from your copy.
[docs/reproducing_the_paper.md](../docs/reproducing_the_paper.md) (Inputs not in the repository) and
[INPUTS.md](INPUTS.md#inputs-not-in-the-repository) give the download and build steps.
- **Zhang et al. 2023 MERFISH codebook v2 (Fig 4b).** Run
  `python scripts/data_processing/build_zhang2023_codebook.py --download`. It downloads
  `codebook_32bit_v2.csv` from the Brain Image Library into `data/raw/BIL_MERFISH/additional_files/`
  and writes `data/processed/BIL_MERFISH/codebooks/zhang_2023_v2_processed.csv`.
- **WMB-10X atlas files (S4a, S4b).** `wmb_precomputed_stats.h5` (1.38 GB) and `wmb_gene.csv` from
  the Allen Brain Cell Atlas, in `data/raw/WMB-10X/`.
- **WMB-10X expression table (Fig 4b-e, S4b).** The atlas values are CC BY-NC 4.0, so the table is
  not shipped. Build it with `conda run -n scanpy_env python scripts/wmb10x/build_whole_brain_cpm.py`.
  It downloads the two atlas files into `data/raw/WMB-10X/` if they are absent, and writes
  `examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv`.
- **WMB-10X metadata (S4a).** Step 1 of `wmb10x_landscape` downloads it.
- **NIS-seq spot-level base calls (S3a).** The 8 spot-level NIS-seq files (1.53 GB) and the Brunello
  table, in `data/raw/NISseq_HeLa_IL1b/`. The spot files are not in the public NIS-seq Zenodo record;
  they are available from the authors of Fandrey et al. (2025) on request. The Brunello table can be
  rebuilt from the public Broad GPP table.
- **The CRISPick table (Fig 3d-f, S3c).** Derived from Broad GPP CRISPick output, which is not
  redistributed. Download the three CRISPick files into `data/raw/CRISPick/`, then run
  `python scripts/data_processing/build_crispick_candidates.py --raw-dir data/raw/CRISPick`. From the
  2025-11-21 design file it rebuilds the paper's table byte for byte.
- **The NIS-seq channel matrices (S3b, Fig 3d-f, S3c, and S3a's provenance check).** They are fitted
  from spot-level base calls that the authors of Fandrey et al. (2025) shared on request, and are
  included only with their permission. If one is missing from your copy, rebuild it with
  `experiments/nisseq_error_analysis` from the spot calls, which are available from those authors on
  request. [INPUTS.md](INPUTS.md#nis-seq-channel-matrices) says where each rebuilt file goes.
- **The Feldman et al. OPS code** (the Feldman baseline of the four OPS experiments): the `ops` env
  from `environments/ops.yml`.
