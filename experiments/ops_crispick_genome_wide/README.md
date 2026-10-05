# ops_crispick_genome_wide: Fig 3d-f and Supp Fig S3c

The genome-wide CRISPR-ko design: 3 guides for each of 20,114 genes plus 100 no-site and 900 one-site-intergenic controls, 61,342 guides chosen from 305,202 CRISPick candidates, 14 sequencing rounds, one trial. DUET's 18-point λ sweep is compared with Feldman et al. (ED 1, 2), Sivanandan et al. (HD 1, 2, 3) and maximum activity. The design is optimized four times (the four arms), once under each NIS-seq HeLa optimization channel. Each arm is evaluated under its own channel.

- **Fig 3d-f** are the aggregated Pareto panels of the `position_varying_asymmetric` arm, which uses the measured position-varying asymmetric channel for both optimization and evaluation.
- **Supp Fig S3c** is the cross-evaluation: every arm's DUET codebooks, plus Feldman et al. and maximum activity, re-scored under the position-varying asymmetric channel.
- S3a (the NIS-seq error notebook) and S3b (the 1,000-gene CRISPRi cross-eval, `ops_crispri_cross_eval/`) are not made here.

This is the largest experiment: about 76 h on 4 GPUs, up to about 1.7 TB of disk and about 64 GB of free RAM (see Cost and Disk and memory). It needs the CRISPick candidate table, which you build yourself, and the NIS-seq channel matrices (see Inputs).

## Run

<!-- docs-test: skip (runs the experiment) -->
```bash
conda activate duet          # any env whose `duet` imports from this clone (run.sh checks)
bash experiments/ops_crispick_genome_wide/run.sh                    # every step, in order
bash experiments/ops_crispick_genome_wide/run.sh cross_eval visualize   # some steps
GW_STEPS="symmetric visualize" bash experiments/ops_crispick_genome_wide/run.sh
bash experiments/ops_crispick_genome_wide/run.sh check              # guards only, then exit
```

**Environment.** Use the DUET environment from `environment.yml` (this clone installed editable with the `benchmark` extra), with the `gpu` extra added for the GPUs: `pip install -e ".[benchmark,gpu]"`. See [Reproducing the paper](../../docs/reproducing_the_paper.md#setup). The Feldman baseline also needs the `ops` environment from `environments/ops.yml` ([environments/README.md](../../environments/README.md)).

**Steps**, always run in this order:

| Step | What it runs | Needs |
|---|---|---|
| `position_varying_asymmetric` | `scripts/benchmark/run_benchmark.py --config position_varying_asymmetric.yaml` (Fig 3d-f arm) | nothing |
| `symmetric` | the same for `symmetric.yaml` | nothing |
| `position_varying` | the same for `position_varying.yaml` | nothing |
| `asymmetric` | the same for `asymmetric.yaml` | nothing |
| `cross_eval` | `scripts/benchmark/compare_benchmarks.py --config eval_position_varying_asymmetric.yaml` (S3c evaluation) | all four arms' `results.csv` |
| `visualize` | the re-visualization pass: `visualize_benchmark.py --config <arm>.yaml` for every arm with a `results.csv`, then `visualize_comparison.py --config eval_position_varying_asymmetric.yaml` if the cross-eval's `aggregated_metrics.csv` exists | whatever exists |

- **Selecting steps.** Name steps on the command line or in `GW_STEPS` (space-separated). The command line wins, and the default is `all`. Steps always run in the canonical order above, whatever order you name them in. An unknown step exits with status 2 before any work. A launcher can therefore register each step as its own unit (`run.sh <step>`).
- **Failure behavior.** An arm or `cross_eval` failure stops the script with that step's exit code. `visualize` renders everything it can, then exits 1 if an input was missing or a visualizer failed, and lists what went wrong.
- **Visualization is its own step.** The arm steps run only the optimizer and baselines. A plotting failure therefore cannot mark a 15-hour run as failed or block the cross-eval. `ops_crispri_cross_eval/run.sh` plots inside each arm; this driver does not.
- **Guards before any work.** run.sh stops before any work if:
  - any of the five configs (or `<name>.$VARIANT.yaml`) is missing;
  - the active env's `duet` is not this repo's `src/duet`;
  - a key it reads is missing or null;
  - an arm step's `candidate_pool.csv_path` does not exist (see Inputs);
  - a NIS-seq channel matrix that an arm or the cross-eval reads does not exist (see Noise matrices);
  - an arm step's Feldman env (`feldman.conda_env`, `ops`) cannot `import ops.pool_design`. Feldman runs after DUET, hours into the run, so this is checked first;
  - the home-drive guard fails (next bullet).

  `cross_eval` also checks, before it starts, that every source's `results.csv` exists and that every source run's `trial_01/guides.csv` is byte-identical to the `reference_dir` pool. It prints each missing or differing file.
- **Home-drive guard.** The arm and `cross_eval` steps refuse to start if the resolved (symlinks followed) `cache_dir` or `scratch_dir` of an arm, or the cross-eval's `evaluator.scratch_dir`, is under `$HOME`. Each arm writes about 0.5 TB of PEP cache and up to about 0.5 TB of evaluation scratch (see Disk and memory), which would fill a home drive. Link `results/cache/ops_crispick_genome_wide` and `results/scratch/ops_crispick_genome_wide` to a large data volume first ([Reproducing the paper](../../docs/reproducing_the_paper.md#running) shows the commands), then run `run.sh check`. `ALLOW_HOME_SCRATCH=1` turns the guard off (with a warning). `VARIANT=smoke` is exempt, because the smoke configs write under `results/smoke/` on purpose (under 1 GB: 744 MB of PEP caches and 103 MB of outputs).
- **`check`.** `run.sh check [steps]` runs every guard for the named steps (default all), prints the resolved outdir, cache and scratch of each, and exits 0. Nothing is written.
- **Other details.**
  - Works from any cwd. run.sh finds its folder with `readlink -f` and cd's there, because the OPS and Comparison loaders take paths relative to the cwd.
  - `TMPDIR` is `<scratch_dir>/tmp` for every heavy step, so Feldman's temporary CSVs and the `conda run` wrappers stay off `/tmp`.
  - The GPU steps print `nvidia-smi` first. That goes to the terminal only (outside `script(1)`).
  - `VARIANT=smoke` reads the tracked `<name>.smoke.yaml` files (see Smoke test).
  - Logs: `results/experiments/ops_crispick_genome_wide/logs/<arm>.{raw.log,log}`, `logs/eval_position_varying_asymmetric.{raw.log,log}`, and the visualizer output in `logs/<arm>.visualize.log` and `logs/eval_position_varying_asymmetric.visualize.log`.
- **Re-render only.** `bash run.sh visualize` redoes every panel without re-evaluating. For one panel, run `python ../../scripts/benchmark/visualize_comparison.py --config eval_position_varying_asymmetric.yaml` or `python ../../scripts/benchmark/visualize_benchmark.py --config position_varying_asymmetric.yaml` from this folder. Both need the cwd, and the comparison scripts must run as `python <path>`, not `-m` (bare `from comparison import`). `run.sh visualize` never passes `--debug-plots`, so it writes no per-trial figures (see Outputs); add the flag to either command to write them.

The Feldman baseline shells out to `conda run -n ops`. All five configs set `device: "gpu:all"`: the PEP pass and the ground-truth evaluation use the GPUs, while the DUET swap search, Sivanandan and Feldman run on CPU.

## Inputs

**CRISPick candidate table.** `candidate_pool.csv_path` names `../../data/processed/CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv`. It is not in the repository: CRISPick output is provided by the Broad Institute Genetic Perturbation Platform under its terms of use, for research use, and DUET does not redistribute it or tables built from it. Cite CRISPick as DeWeirdt et al. 2022, Nat Commun 13:5255. Build the table yourself:

1. Download three files from the [CRISPick portal](https://portals.broadinstitute.org/gppx/crispick/public) into `data/raw/CRISPick/`:
   - `sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_RS3seq-Chen2013+RS3target_Ensembl_aggrCFD_20251121.txt`, the precomputed human (GRCh38) SpCas9 knockout design with Ensembl gene IDs and aggregate CFD scores, released 2025-11-21 (2,116,333,376 bytes, sha256 `5dd4da45440dc095e14755680a2d0ac9051467eef92aa613725ba2ba12195a24`);
   - `sgrna-negcontrols-nosite-9606-GRCh38-SpyoCas9-2000.txt` and `sgrna-negcontrols-onesite-9606-GRCh38-SpyoCas9-2000.txt`, 2,000 control guides each.
2. From the repository root, run:

   <!-- docs-test: skip (needs the CRISPick downloads) -->
   ```bash
   python scripts/data_processing/build_crispick_candidates.py --raw-dir data/raw/CRISPick
   ```

   It writes the table to its default `--out`, the path above (about 1 min, 1.7 GB of RAM).

From the 2025-11-21 files the table is byte-identical to the paper's: 405,059 rows (the first 20 CRISPick picks per gene, then 2,000 no-site and 2,000 one-site intergenic controls), 27,708,495 bytes, sha256 `e49d92641f932c2610e8f8817428532386a0f56425d306e50141d56fb531a484`. The script compares the sha256 values and warns on a mismatch. Row order seeds the pool, so a table built from another CRISPick release gives another pool, and the results then agree with the paper's only statistically. Keep the built table out of public repositories. The script's docstring describes each step.

**NIS-seq channel matrices.** The arms read the 100,000-reads-per-file fit in `scripts/benchmark/noise_model_matrices/channels/archive/`; see [Noise matrices](#noise-matrices) below.

**The `ops` environment** for the Feldman baseline (see Run).

## Settings

- **The pool.** The arms build the same pool, which is the pool of the paper's run:
  - `trial_01/guides.csv` sha256 `3db7cb4c2d0ce24978ba7dfa4dbb0e2e613ae1dfafbe0c14ac6f907fa9b9e747`, 13,792,308 B;
  - 305,202 candidates (`min_rank: 15`); 20,114 genes (2 dropped with fewer than 3 candidates) plus the 2 control groups (2,000 candidates each, quotas 100 / 900);
  - 61,342 slots, 289,113 distinct 14-mers, trial seed 191664963.

  If your `guides.csv` matches this hash, your CRISPick table is the paper's.
- **The four arms.** They differ only in the channel, the metric, the matrix file and `outdir`:

  | Arm (config) | Channel and metric | S3c legend |
  |---|---|---|
  | `symmetric.yaml` | `asymmetric` + `asymmetric_nll`, with a symmetric matrix | DUET (Symmetric error) |
  | `position_varying.yaml` | `position_varying` + `position_varying_nll` | DUET (Position-varying error) |
  | `asymmetric.yaml` | `asymmetric` + `asymmetric_nll` | DUET (Asymmetric error) |
  | `position_varying_asymmetric.yaml` | `position_varying_asymmetric` + `position_varying_asymmetric_nll` | DUET (Position-varying asymmetric error) |

  The `symmetric` arm is a 4×4 matrix with equal off-diagonals: diagonal 0.857202, off-diagonal 0.047599, so ε = 0.142798. It runs through the general `asymmetric` channel and `asymmetric_nll` metric, not through the native `symmetric` / `symmetric_nll` types. The model is the same. S3b's `ops_crispri_cross_eval/symmetric.yaml` uses the native types (with Hamming) instead.
- **λ:** `[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 1.0]` in every arm, ascending exact literals. λ weights decoding accuracy, so λ = 1 is decode-only ([ADR 0001](../../docs/adr/0001-lambda-weights-decoding-accuracy.md)). Labels use `:.2f`, so λ = 0.125 and 0.175 label as 0.12 and 0.17.
  - The 97.5%-activity rule in `visualize_benchmark.py` (`select_duet_lambda`) picks the highlighted codebook; expect λ = 0.07. The cross-eval takes every λ (`regex: "DUET.*"`).
- **Other keys.** `trials: 1`, `seed: 42`, `seq_rounds: 14`, `quota: 3`, `num_groups: all`, `control_quotas {NO_SITE: 100, ONE_SITE_INTERGENIC: 900}`; evaluator and `duet.pep` `num_samples: 5000`, `num_cpus: 20`; `max_iter: 100000`, `max_patience: 100`, `temperature: 0.0`, random initialization, optimizer `num_cpus: 18`; `avoid_duplicates: true` with `duplicate_penalty: 100.0`; Sivanandan HD [1, 2, 3] with 8 CPUs; Feldman ED [1, 2].
- **Paths.** `outdir` is `../../results/experiments/ops_crispick_genome_wide/<arm>` and `.../cross_eval/eval_position_varying_asymmetric`. `cache_dir` is `../../results/cache/ops_crispick_genome_wide`, shared by the four arms. `scratch_dir` is `../../results/scratch/ops_crispick_genome_wide`, a top-level key in the arms and `evaluator.scratch_dir` in the cross-eval. run.sh creates `<scratch>/tmp`, because `mkdtemp` needs the directory to exist.
- **Memory keys.** Top-level `device: "gpu:all"` and `mem_budget_gb: 64.0`, inherited by the DUET step (sym 64). The evaluation's budget is `evaluator.mem_budget_gb: 20.0` in the arms and `eval_mem_budget_gb: 20.0` in the cross-eval. They set memory only (see Disk and memory).
- **Seeds.** `evaluator.seed: 42` and `duet.pep.seed: 43` in the four arms, and `evaluator.seed: 42` in the cross-eval. This is the seed policy of the other OPS experiments (see [ops_crispri_cross_eval](../ops_crispri_cross_eval/README.md), Settings).
- **Validity.** A codebook is valid only with exact cardinality, distinct indices, and every group at exactly its quota (`_is_valid_codebook`, `src/duet/ops_benchmark/runner.py`).
- **Feldman et al. runs at this scale.** The wrapper bypasses the OPS library's unfinished >80k ED=2 branch (`src/duet/benchmark/external_ops/ops_wrapper.py`) and gives each CRISPick control group its own quota (100 / 900). A full-pool check gave:
  - **ED=1:** valid, 61,342 of 61,342 guides, 3,345 of them sharing a 14-mer codeword, controls 100 / 900, 5.1 s.
  - **ED=2:** 61,341 guides, one short (ENSG00000281593 gets 2 of 3), so **invalid**; 3,396 sharing; controls 100 / 900; 21.6 min on CPU; peak RSS 0.48 GB.

  Expect the same in every arm. The Feldman selection does not depend on the channel, so each arm re-selects the same Feldman codebooks (about 20 min of CPU per arm).
- **Sivanandan et al.** (HD 1, 2, 3, 8 CPUs) is invalid at every HD.
- **The cross-evaluation.**
  - `reference_dir` is the `position_varying_asymmetric` run. It only supplies `trial_01/guides.csv`, the pool used to map every source's pool indices to sequences and scores. run.sh refuses to start the cross-eval unless every source's `guides.csv` equals the reference's.
  - Legend labels, order and the blue colors are set in the config. The order is `ops_crispri_cross_eval`'s, dark to light (`#08306b`, `#2171b5`, `#6baed6`, `#9ecae1`), and the fronts are drawn in that order, so the lightest is on top. Feldman takes `METHOD_PALETTE["Feldman et al."]`. "DUET (Position-varying error)" and "DUET (Position-varying asymmetric error)" wrap onto two legend lines: `visualize_comparison.py` wraps at 24 characters (`LEGEND_WRAP_WIDTH`, no config key).
  - The labels are the `ops_crispri_cross_eval` wording since 2026-10-04. Cross-eval outputs made before then carry the earlier labels in the `Label` column of `aggregated_metrics.csv`, and `visualize_comparison.py` draws only rows whose `Label` matches the config, so such outputs need their labels renamed (see PROVENANCE.md) or a cross-eval re-run.
  - The flat `ComparisonConfig` schema: top-level `device` and `eval_mem_budget_gb`, `scratch_dir` inside `evaluator`. `cache_dir` and the top-level `seed` are accepted but unused by `compare_benchmarks.py`.

## Noise matrices

The configs use the NIS-seq fit on the first 100,000 reads per file (800,000 matched, 400,000 kept at the per-screen 50th-percentile intensity threshold), the files the paper's runs read. Each config writes the path once, as the YAML anchor `&noise_matrix`. The other uses (evaluator metric, `duet.pep` channel and metric) are `*noise_matrix` aliases.

| Arm | File in `scripts/benchmark/noise_model_matrices/channels/archive/` | sha256 |
|---|---|---|
| symmetric | `nisseq_hela_uniform_subsample.npy` | `b899d43997dddb7e077ff79c6ffbcd9d918e0e2a8292c13d6707f46a98c87068` |
| position_varying | `nisseq_hela_positional_subsample.npy` | `a66a64fe5aa4f9cb3a2c1fdb166a2f2be34ba8ebda3b486f6c2e5c0304f584cf` |
| asymmetric | `nisseq_hela_channel_subsample.npy` | `27fcda8b163ec02bff141313fd7b606a61589adb4e47c761799fe8b530a6ddf4` |
| position_varying_asymmetric (and the cross-eval) | `nisseq_hela_positional_channel_subsample.npy` | `0f2a4d93d671066f39d686a2235ac57b0e58e9674319076581630893474def50` |

- **Availability.** The NIS-seq channel matrices are fitted from spot-level base calls that the authors of Fandrey et al. (2025) shared on request, and are included only with their permission. If one is missing from your copy, rebuild it with `experiments/nisseq_error_analysis` from the spot calls, which are available from those authors on request. Its N = 100,000 run (`SUBSAMPLE_NS="100000"`) writes `results/experiments/nisseq_error_analysis/subsample100000/noise_matrices/nisseq_hela_<kind>_pctl50_subsample100000.npy`, byte for byte the file above of the same `<kind>` (uniform, positional, channel, positional_channel). Copy each one to the name above, creating the folder first if your copy lacks it (`mkdir -p scripts/benchmark/noise_model_matrices/channels/archive`).
- **The S3b fit is different.** `ops_crispri_cross_eval` uses the 50,000-reads-per-file fit (400,000 matched, 200,000 kept). Largest absolute differences between the two fits: symmetric ε 0.142798 vs 0.144656; position-varying 4.9e-3; asymmetric 5.9e-3; position-varying asymmetric 1.38e-2 (at position 11).

## S3c baselines

S3c shows Feldman et al. ED=1 with the maximum-activity anchor, like S3b.

- **In the config: Feldman with the maximum-activity anchor.** Regex `["Feldman et al.*", "Maximum activity"]`, label "Feldman et al.". This is the S3b pattern.
  - `visualize_comparison.py` anchors every label's connector line, DUET's included, at the "Maximum activity" point. It draws that point in Feldman's color.
  - The Maximum-activity row also protects the cross-eval: if Feldman produced no row at all, the source still matches something.
  - Feldman ED=2 rows exist but are invalid at this scale; `compare_benchmarks.py` drops them with the `Valid` filter.
- **Without the anchor.** Delete the `- "Maximum activity"` line from the Feldman source. Connectors would then run through the DUET points only, with Feldman ED=1 as a lone point, and `load_method_results` would raise `ValueError` if the arm produced no Feldman row.
- **Sivanandan is left out.**
  - Every Sivanandan codebook is invalid at this scale, and `compare_benchmarks.py` keeps only `Valid` rows, so it could not plot anything.
  - A Sivanandan-only source also crashes the group check, which runs before the `Valid` filter. `["Sivanandan et al.*"]` alone gives `ValueError: Group set mismatch`, because the Sivanandan union misses 25-30 genes.
  - Adding "Maximum activity" to its regex avoids the crash but yields a "Sivanandan et al." entry that plots only the maximum-activity point.
  - So a Sivanandan "N/A" legend entry cannot come from a config key; the S3c panel has none. (`visualize_comparison.py` does write "N/A" for a configured label with no finite point, such as Feldman et al. in the 95th/5th panel, which is not a paper panel.)

In Fig 3d-f, `visualize_benchmark.py` draws the fixed groups Maximum activity, DUET, Feldman et al. and Sivanandan et al. Feldman ED=1 appears as a point; ED=2 is dropped as invalid. Sivanandan keeps its legend row with no points, and the panel writes "N/A" in it. In Fig 3f, maximum activity and Feldman ED=1 read "N/A" too, because their 95th/5th ratio is undefined. With one trial, the panels draw 4 pt points with no error bars and square markers in the legend only.

## Outputs

Under `results/experiments/ops_crispick_genome_wide/`:

| File | Use |
|---|---|
| `position_varying_asymmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | **Fig 3d** |
| `position_varying_asymmetric/pareto_fronts/10th_percentile_decode_accuracy_aggregated.svg` | **Fig 3e** (the empirical 10th percentile, not the bottom-decile mean) |
| `position_varying_asymmetric/pareto_fronts/95th_over_5th_percentile_decode_accuracy_aggregated.svg` | **Fig 3f** |
| `cross_eval/eval_position_varying_asymmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | **Supp S3c** |
| `<arm>/results.csv` | per-guide rows for every method (24, Feldman ED 1, 2 included), with `Valid`; about 132 MB |
| `<arm>/trial_01/guides.csv` | the pool (about 13.8 MB); the pva copy is the cross-eval's `reference_dir` |
| `<arm>/config.yaml` | verbatim copy of the config (relative paths are relative to this folder) |
| `<arm>/hypervolume/normalized_hypervolume_by_method_group.svg`, `<arm>/pareto_fronts/<metric>_aggregated.svg` | the other phase-1 figures from `visualize`, 8 per arm with the panels above (not in the paper) |
| `<arm>/pareto_fronts/<metric>_trial_1.svg`, `<arm>/{hypervolume_visualization,guide_level_comparisons,error_metrics}/` | phase-1 per-trial figures (not in the paper); debug plots, written only with `--debug-plots` |
| `cross_eval/eval_position_varying_asymmetric/aggregated_metrics.csv` | per (Label, Method, Trial) metrics behind S3c |
| `cross_eval/eval_position_varying_asymmetric/reevaluated_results.csv` | per-guide re-scores (about 1.0 GB) |
| `cross_eval/eval_position_varying_asymmetric/hypervolume/hypervolume_barplot.svg`, the other `pareto_fronts/<metric>_aggregated.svg`, `compare_config.yaml` | other panels (not in the paper; the bar chart has horizontal bars); config copy |
| `cross_eval/eval_position_varying_asymmetric/hypervolume/hypervolume_trial_1.svg`, `pareto_fronts/<metric>_trial_1.svg` | per-trial copies (not in the paper); debug plots, written only with `--debug-plots` |
| `logs/` | `<arm>.{raw.log,log,visualize.log}`, `eval_position_varying_asymmetric.{raw.log,log,visualize.log}` |

- A default `visualize` writes 40 figures: 8 per arm and 8 for the cross-eval. With `--debug-plots` it writes all 244.
- Fig 3d-f use the `_aggregated` variant. With one trial it carries the same values as `trial_1` (written only with `--debug-plots`) but different axis limits and legend, so do not mix the two.
- The PEP caches are in `results/cache/ops_crispick_genome_wide/`. Evaluation scratch goes to `results/scratch/ops_crispick_genome_wide/`, whose `duet_cache_*` directories are removed at exit.
- run.sh prints the four panel paths at the end, marking any not yet rendered.

## Compare after a re-run

The paper's values come from its run's `results.csv` and `aggregated_metrics.csv`. A full re-run with these configs on the reference machine (2026-09-26) reproduced them: every accuracy and activity is within 0.0005 of the paper's value. The tables give the re-run values, with the paper run's in parentheses. Values were computed with the visualizer's definitions (pandas linear quantiles).

**Fig 3d-f** (`position_varying_asymmetric` arm), re-run (paper run):

| Quantity | Maximum activity | DUET λ = 0.07 |
|---|---|---|
| Mean decode accuracy | 0.637972 (0.638016) | 0.728198 (0.728108) |
| 10th percentile | 0.541200 (0.541000) | 0.691600 (0.691400) |
| 5th percentile | 0.0 (0.0) | 0.676800 (0.676800) |
| 95th/5th ratio | undefined | 1.150414 (1.150709) |
| Mean activity | 0.979228 (0.979228) | 0.955759 (0.956220) |

- **Operating point:** λ = 0.07 again (activity 0.955759, above the 0.9547475 threshold); λ = 0.08 misses it at 0.951608.
- The 97.5% threshold is 0.975 × 0.9792282 = 0.9547475. In the paper run too, λ = 0.08 missed it (at 0.952016) and λ = 0.07 cleared it. A new draw can move the pick to a neighbor.
- **Maximum-activity activity (0.979228) should match to every digit.** The pool and the max-activity selection are deterministic, and activity is not Monte Carlo.
- **95th/5th ratio:** 1.503794 at λ = 0 (1.500189) and 1.131334-1.176275 at the other 17 λ (1.131993-1.176222).
- **Codeword-sharing guides:** 213-217 at every λ > 0 (211-213).
- **Feldman et al.:** ED=1 valid, mean accuracy 0.643052, p10 0.5490, p5 0, p95 0.7904, mean activity 0.978871 and 3,345 zero-accuracy guides, all as expected; its 95th/5th ratio is undefined ("N/A" in Fig 3f). ED=2 invalid (61,341 guides), mean accuracy 0.652954.
  - No library with all-distinct codewords can fill every quota in this pool (at most 61,239 of 61,342, by max-flow).
  - The paper's panels show no Feldman point: they were drawn from a run made before the Feldman baseline could run on pools above 80,000 sequences.
- **Sivanandan et al.:** invalid at every HD, with 61,203 / 61,136 / 60,846 guides at HD 1 / 2 / 3 (paper run 61,204 / 61,151 / 61,002). The counts can differ from the paper run's; see Determinism.

**S3c**, re-run (paper run), all under the position-varying asymmetric channel:

| Label | Mean accuracy, averaged over the 18 λ | At λ = 0.07 (accuracy @ activity) | Highest |
|---|---|---|---|
| DUET (Symmetric error) | 0.7211 (0.7211) | 0.724608 @ 0.937454 (0.724431 @ 0.937575) | 0.726365 at λ = 1 (0.726426) |
| DUET (Position-varying error) | 0.7196 (0.7197) | 0.722380 @ 0.957277 (0.722531 @ 0.957172) | 0.726230 at λ = 1 (0.726279) |
| DUET (Asymmetric error) | 0.7227 (0.7227) | 0.726058 @ 0.950236 (0.725833 @ 0.950065) | 0.729018 at λ = 0.5 (0.729015 at λ = 1) |
| DUET (Position-varying asymmetric error) | 0.7250 (0.7250) | 0.728173 @ 0.955759 (0.728068 @ 0.956220) | 0.732366 at λ = 1 (0.732294) |

- Feldman ED=1: 0.643114 at activity 0.978871. Maximum activity: 0.638030 at 0.979228.
- The order of the four DUET labels is unchanged, and the largest shift (0.000225, DUET (Asymmetric error) at λ = 0.07) is far inside the tolerance below.
- **Tolerance.** There is no measured tolerance at this scale. Two points bound it:
  - The Monte Carlo re-draw alone is small. The paper's cross-eval re-scored the pc arm's λ = 0.07 codebook at 0.728068, against 0.728108 in its own phase-1 evaluation (a different draw of the same codebook).
  - The codebooks themselves can change with a new PEP draw and new per-λ seeds. For the medium-scale OPS experiments, DUET means moved by up to about 0.005 between runs.

  Treat larger shifts, or a change in the order of the four DUET labels, as worth a look. The gaps between the DUET labels (about 0.001-0.004) are of that order, so their order is not guaranteed.
- Determinism was not checked at full scale: that needs a second full run.

## Cost

Measured in the 2026-09-26 re-run on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs, 220 GB RAM; step walls from the `run_all.sh` status file, stages from the `[time]` lines of each step's log):

| Step, in run order | Step wall | PEP | DUET 18-λ sweep | Sivanandan HD 1/2/3 | Feldman ED 1, 2 | Evaluation |
|---|---|---|---|---|---|---|
| position_varying_asymmetric | 18:50:31 | 7:14:00 | 5:45:09 | 41:47 / 45:38 / 1:18:22 | 19:03 | 2:37:55 |
| symmetric | 17:58:12 | 6:25:37 | 3:42:05 | 46:21 / 50:10 / 1:22:58 | 18:59 | 4:24:26 |
| position_varying | 15:32:58 | 5:44:59 | 3:49:00 | 42:52 / 46:49 / 1:19:17 | 18:53 | 2:44:02 |
| asymmetric | 16:14:32 | 6:25:37 | 3:34:04 | 46:30 / 50:09 / 1:22:50 | 18:39 | 2:49:11 |

- **Four arms: 68:36:13.** Per arm 15:33-18:51.
- **Cross-eval: 5:47:41**: load 58 s, evaluator init 3:58:31 (union of 205,290 codewords, nnz 29.6e9), re-evaluation 1:46:23 for 74 codebooks.
- **`visualize`:** 1:26:18 for all 244 figures (`--debug-plots`). A default run writes only the aggregated figures; the pva arm's 8 took 9 s.
- **In all: 75:50:12.** Allow about 75-92 h: the PEP transposes and symmetrizes are bound by disk I/O, and the DUET sweep and Sivanandan are CPU-bound, so both vary with the storage and the load on the machine.
- Each cold PEP is a GPU pass of about 2 h on 4 GPUs, then a host transpose and a symmetrize. The GPUs are busy only about 3.5 h per arm (PEP pass plus evaluation streaming); the rest is CPU and disk I/O.
- The Fig 3d-f arm runs first so that it finishes earliest.

## Disk and memory

- **PEP cache per arm.** Two U × U uint16 files (`<fp>.npy` raw counts, `<fp>.sym.dat` symmetrized), U = 289,113 unique 14-mers: 167.2 GB each, **334.3 GB per arm**. Symmetrization writes a transient `<fp>.sym.dat.T.tmp` of another 167.2 GB beside them (deleted in a `finally`), so the peak is **501.5 GB**.
  - The four arms keep **1.34 TB** under `results/cache/ops_crispick_genome_wide/` (`force_rebuild: false` keeps them for re-runs).
  - Delete the caches when done if space is needed.
- **Evaluation scratch** (the GPU evaluation cache in `results/scratch/ops_crispick_genome_wide/duet_cache_*`). Its steady size is 9·nnz + 13·rows bytes, and its peak (during index widening) 12·nnz + 13·rows. From the nnz logged in the re-run:

  | Evaluation | Peak (steady) |
  |---|---|
  | symmetric | 513 GB (387 GB; 41.8e9 nnz), the largest |
  | position_varying | 260 GB |
  | asymmetric | 295 GB |
  | position_varying_asymmetric | 239 GB (19.0e9 nnz) |
  | S3c cross-eval | 368 GB (29.6e9 nnz) |

  - It is removed at exit. A killed evaluation can leave up to about 0.5 TB behind in a `duet_cache_*` directory; delete it by hand.
  - **Why symmetric is the largest.** Its reads have about twice as many competitors. All four channels give about 2 errors per 14-mer read, but the symmetric channel's NLL cost depends only on Hamming distance, so every candidate as close to the read as the true guide ties, and unique-minimum decoding (cost ≤) records each tie. Under the measured channels only 20-36% of those equally distant candidates compete (checked 2026-10-01 with a CPU replica of the competitor count on the real pool).
- **Cumulative peak on the data volume**, running in order with caches kept: about 1.63 TB during the last arm's evaluation and 1.71 TB during the cross-eval (1.34 TB of PEPs plus the scratch). Budget 2.5 TB of free space to cover one leftover scratch directory. The home-drive guard keeps the runs off your home drive.
- **Memory.** In the re-run, RSS reached 149-192 GB, almost all of it file-backed (the mmap'd PEP and CSR, which the kernel can drop), with no step short of memory on a 220 GB machine. The `[mem]` snapshots (taken between stages, so not true peaks) logged at most 14.9 GB anonymous. What matters is anonymous memory. Per stage:
  - **Symmetrize** (PEP build): batches of up to 64 GB of anonymous memory by design at the top-level `mem_budget_gb: 64.0`.
  - **DUET:** the PEP rows come from the shared page cache, and the code streams 256-row batches (`build_shared_decode_state`), so each of the 18 λ workers needs well under 1 GB of anonymous memory.
  - **Evaluation** (each arm's last stage, and the cross-eval): one pass over the ground-truth CSR in batches sized by the eval budget (`_batch_matmul_core`, `src/duet/codebook_evaluator.py`). Its real peak is about 1.5-2.5× the budget, because the previous batch's arrays are still alive and each CSR block is copied, plus a full in-RAM copy of the CSR row pointers (union × 5,000 × 8 B: about 6.5 GB per arm, 8.6 GB for the cross-eval). Measured on scaled-down copies of the real batch layout:
    - at 64: about 130-145 GB of anonymous memory per evaluation;
    - at 20 (used here): about 35-55 GB per arm and 40-55 GB for the cross-eval.

    So the arms set `evaluator.mem_budget_gb: 20.0`, which overrides the top-level 64 for the evaluation only, and the cross-eval sets `eval_mem_budget_gb: 20.0`. 20 is the key's built-in default. **The budget changes memory only.** Each codeword's accuracy is computed from its own K rows, so batching cannot change a result. Budgets giving 1, 7, 64 and all codewords per batch produced byte-identical accuracies and error metrics on a small CPU evaluator.

## Determinism

- **Seeded Monte Carlo.** `evaluator.seed: 42` and `duet.pep.seed: 43` in the arms, and `evaluator.seed: 42` in the cross-eval. The paper run had none of them (fresh OS entropy). With the same code, envs, inputs, configs and device, a re-run should repeat each `results.csv` and the cross-eval CSVs. A second run of `position_varying_asymmetric.smoke.yaml` with its own, cold PEP cache repeated `results.csv`, `trial_01/guides.csv` and the PEP counts (`.npy`) byte for byte, Feldman and Sivanandan included. Not checked at full scale. A re-run is a new draw relative to the paper's unseeded run.
  - The ground-truth draws are indexed by each guide's position in the union of all selections, per run and per cross-eval. Adding Feldman, or any change in one method's selection, shifts every other method's draws.
  - The PEP fingerprint contains the seed, so a seeded run never reuses an unseeded cached PEP.
- **Per-λ optimizer seed.** It is still trial seed + int(λ·1000), so re-runs are new draws, not relabelings. At `temperature: 0.0` it only breaks exact ties among maximal swaps.
- **`device` changes numbers for NLL metrics.** Every arm and the cross-eval use an NLL metric. On GPU the ground truth is a float32 matmul; on CPU it is float64. The PEP is float32 on both, but summed by different BLAS libraries. Measured flips for the NIS-seq channels are rare (see [Numerical determinism](../README.md#numerical-determinism)), but keep `gpu:all` to compare numbers.
- **Speed and memory only, not results:** `num_cpus` (but `duet.pep.num_cpus` is in the PEP fingerprint, so changing it forces a rebuild with the same draws), `mem_budget_gb`, `evaluator.mem_budget_gb`, `eval_mem_budget_gb`. None of the memory budgets is in a fingerprint.
  - That holds on GPU, as configured. On CPU with these NLL metrics, `duet.pep.num_cpus` 1 versus 2 or more can change PEP counts; values of 2 or more agree ([Numerical determinism](../README.md#numerical-determinism)).
- **Changes numbers:** `num_samples`, the seeds, the λ list, the noise matrices.
- **Baselines.**
  - The pool, maximum activity and Feldman ED=1 are deterministic. Feldman ED=2 is fixed by the `PYTHONHASHSEED=0` pin.
  - **Sivanandan is deterministic, but its counts differ from the paper run's.** Sivanandan's greedy order sorts genes by candidate count, and most genes tie at 15. Ties fall back to the order of the pool's `quotas` dict, which now follows row order. In the paper run it followed `set(groups)`, whose iteration order depends on the per-process string-hash seed, so the four arms' Sivanandan codebooks differed although their pools were identical.
    - Checked on a 2,000-gene CRISPick subset (33,946 candidates, HD=1, current code): row-order quotas gave the identical selection under `PYTHONHASHSEED` 1 and 2. Quotas in `set()` order gave two different selections under the same two hash seeds.
    - So expect Sivanandan to repeat run to run, and to stay invalid, but not to match the paper run's counts.

## Smoke test

`VARIANT=smoke bash experiments/ops_crispick_genome_wide/run.sh` reads the `<name>.smoke.yaml` files, which are tracked beside the configs. They are the real configs with:
- `num_groups: 200`. Both control groups are always kept, at their real quotas 100 / 900.
- Evaluator and PEP `num_samples: 200`, λ `[0.0, 1.0]`.
- Outputs, cache and scratch under `results/smoke/ops_crispick_genome_wide/`, `results/smoke/cache/...` and `results/smoke/scratch/...`.

Everything else (channels, matrices, seeds, trials 1, `gpu:all`, Feldman ED 1, 2, Sivanandan HD 1, 2, 3, the five cross-eval sources) is as in the real configs. The smoke still needs the CRISPick table and the NIS-seq matrices. The smoke pool is far under 80,000 sequences, so it does not exercise the >80k Feldman path; the full-pool check above covers that.

- **Cost.** About 30 min on the reference machine (each arm about 3 to 6.5 min, `cross_eval` seconds, `visualize` about 9 min for every figure), with under 2 GB of RAM, 744 MB of PEP caches and about 100 MB of outputs.
- **Pool.** 6,991 candidates: 200 sampled genes plus both control groups (2,000 candidates each); 202 groups, 1,600 slots, 6,980 distinct 14-mers. `trial_01/guides.csv` is byte-identical across the four arms.
- **What it does not exercise.** At this pool size nothing is invalid: Feldman ED=1 picks exactly the maximum-activity set, and ED=2 is valid. The full-scale invalid paths (Feldman ED=2 dropped by the `Valid` filter, and Sivanandan's "N/A" legend row) therefore run only in the real configs.
- **Cross-eval.** 5 sources and 17,600 rows: 4 labels × 2 λ × 1,600, plus the Feldman source's ED=1, ED=2 and Maximum activity (3 × 1,600). All four panel files render.
