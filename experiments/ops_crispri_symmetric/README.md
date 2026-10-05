# ops_crispri_symmetric: Fig 3b (both panels) and Supp Fig S2a-g

The library is Weissman CRISPRi with 2,200 guides (1,000 genes × 2 guides + 200 non-targeting controls). The simulation uses 10 sequencing rounds, a symmetric channel (ε = 0.1), the Hamming metric and unique-minimum decoding, over 5 trials. DUET's 18-point λ sweep is compared with Feldman et al. (ED 1, 2), Sivanandan et al. (ED 1, 2, 3) and maximum activity.

## Run

<!-- docs-test: skip (runs the experiment) -->
```bash
conda activate duet        # any env whose `duet` imports from this clone (run.sh checks)
bash experiments/ops_crispri_symmetric/run.sh
```

- **Environment.** Use the DUET environment from `environment.yml` (this clone installed editable with the `benchmark` extra), with the `gpu` extra added for the GPUs: `pip install -e ".[benchmark,gpu]"`. See [Reproducing the paper](../../docs/reproducing_the_paper.md#setup).
- run.sh runs `scripts/benchmark/run_benchmark.py` (DUET sweep, baselines and ground-truth evaluation), then `scripts/benchmark/visualize_benchmark.py`. It works from any cwd, and stops before any work unless the config exists and the active env imports `duet` from this repo's `src/`.
- **Feldman baseline.** It shells out to `conda run -n ops` (`feldman.conda_env: ops`). Create that environment from `environments/ops.yml`: Python 3.7.12 and the `ops` package of feldman4/OpticalPooledScreens at commit 5417bf4 ([environments/README.md](../../environments/README.md)).
- To redo only the figures from an existing `results.csv`, run `cd experiments/ops_crispri_symmetric && python ../../scripts/benchmark/visualize_benchmark.py --config config.yaml`. The `cd` is required because the visualizer resolves `outdir` against the cwd. Add `--debug-plots` to also write the per-trial figures, which run.sh never writes (see Outputs).
- **Logs.** run.sh writes `run.raw.log`, `run.log` and `visualize.log` to `<outdir>/logs/`. It prints `nvidia-smi` first. It also stops if the config's `outdir` or `scratch_dir` is missing or null.

## Inputs

- **The pool.** It reads `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv` (hCRISPRi-v2.1, Horlbeck et al. 2016, eLife 5:e19760, CC BY 4.0; in the repository) as the `WeissmanCRISPRi` package default, not through a config key. The file's sha256 starts `2aa08896`.
- **The `ops` environment** for the Feldman baseline (above).
- Nothing to download.

## Settings

- **`duet.optimizer.lambda`:** `[0.0, 0.01, …, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 1.0]`, ascending exact literals. λ weights decoding accuracy, so λ = 1 is decode-only ([ADR 0001](../../docs/adr/0001-lambda-weights-decoding-accuracy.md)).
  - `0.175` must stay a literal so that it is labelled `0.17`. Computing `1 - 0.825` would give the label `0.18`.
- **Paths.** `outdir` is `../../results/experiments/ops_crispri_symmetric`, `cache_dir` is `../../results/cache/ops_crispri_symmetric` and `scratch_dir` is `../../results/scratch/ops_crispri_symmetric`. All three are cwd-relative, and run.sh `cd`s into this folder first.
- **Scratch and temp.** run.sh creates `scratch_dir`, which `mkdtemp` needs to exist. It also exports `TMPDIR=<scratch_dir>/tmp`, so the Feldman temp files and the `conda run` scripts stay under `results/`.
- **Seeds.** `evaluator.seed: 42` seeds the ground truth and `duet.pep.seed: 43` the DUET PEP. The PEP gets its own seed so that its Monte Carlo streams are independent of the ground truth's: both spawn one child stream per index from their root seed, so with one shared seed PEP batch j and ground-truth guide j would start from the same stream.
- **`visualization.paper_panels`** lists the five per-trial panel files (Fig 3b right, S2d-g), so `visualize_benchmark.py` writes them without `--debug-plots` (see Outputs). `run_benchmark.py` ignores the block, and no cache fingerprint includes it.
- **The 97.5% and 95% arms.** The rule takes the highest mean decode accuracy among DUET rows whose activity is at least 0.975 × (or 0.95 ×) the maximum-activity arm's. It is independent of the λ convention.
- **Other keys:** top-level `seed: 42`, `trials: 5`, `device: gpu:all`, `mem_budget_gb: 64.0`; the whole `candidate_pool` block (WeissmanCRISPRi, 10 rounds, quota 2, 200 controls, 1000 groups, `min_rank` 10); `evaluator` (symmetric ε 0.1, hamming, unique_minimum, `num_samples` 2000, `num_cpus` 30); `duet` (mmap, no force rebuild, random init, temperature 0.0, `max_iter` 100000, `max_patience` 3000, `avoid_duplicates` true with penalty 100.0, optimizer `num_cpus` 18; `pep` with the same channel, `num_samples` 5000, `num_cpus` 30); `sivanandan` (ED [1, 2, 3], `num_cpus` 8); and `feldman` (`conda_env: ops`, ED [1, 2]).

## Outputs

All outputs go under `results/experiments/ops_crispri_symmetric/`.

| Panel | File |
|---|---|
| Fig 3b left | `pareto_fronts/Mean_decode_accuracy_aggregated.svg` |
| Fig 3b right | `guide_level_comparisons/all_baselines_vs_duet_97p5pct_trial_3_distributions.svg` |
| S2a | `hypervolume/normalized_hypervolume_by_method_group.svg` |
| S2b | `pareto_fronts/10th_percentile_decode_accuracy_aggregated.svg` |
| S2c | `pareto_fronts/95th_over_5th_percentile_decode_accuracy_aggregated.svg` |
| S2d | `guide_level_comparisons/max_activity_97p5pct_trial_5_jointplot.svg` |
| S2e | `guide_level_comparisons/Feldman_et_al_ED1_vs_duet_97p5pct_trial_5_jointplot.svg` |
| S2f | `guide_level_comparisons/Sivanandan_et_al_ED1_vs_duet_97p5pct_trial_5_jointplot.svg` |
| S2g | `guide_level_comparisons/Sivanandan_et_al_ED2_vs_duet_97p5pct_trial_5_jointplot.svg` |

Figures written by default (13 of 459):
- the four aggregated panels above (Fig 3b left, S2a-c) and the other four `pareto_fronts/<metric>_aggregated.svg` panels, which are diagnostics;
- the five per-trial panels above (Fig 3b right, S2d-g), because `config.yaml` lists them under `visualization.paper_panels`.

Every other figure is a debug plot, written only with `--debug-plots`: the rest of `guide_level_comparisons/` (the 95% and activity-matched panels for every trial), `pareto_fronts/<metric>_trial_<k>.svg`, `hypervolume_visualization/` and `error_metrics/`. The Fig 3b right axis ranges are still taken over every comparison of its trial, with or without the flag.

Other outputs: `results.csv` (per-guide rows for every method and trial, with a `Valid` column), `trial_NN/guides.csv`, a config copy `config.yaml`, and `logs/`. The PEP cache is in `results/cache/ops_crispri_symmetric/`; scratch is in `results/scratch/ops_crispri_symmetric/`.

## Compare after a re-run

The paper's values, recomputed with this repo's statistics from the paper run's `results.csv`. Percentages are rounded to whole cells out of 500: `round(accuracy × 500)/500`.

| Quantity | Paper run |
|---|---|
| 97.5% arm | λ = 0.125 in all 5 trials (label `DUET (lambda=0.12)`) |
| 95% arm | λ = 0.20; trial 2: λ = 0.25 |
| DUET 97.5% mean decode | 0.78403 = 392 cells = 78.4% |
| Baseline mean decode | max activity 0.72933 (365, 73.0%); Feldman ED1 0.73805 (369); Sivanandan ED1 0.73780 (369), ED2 0.74906 (375, 75.0%) |
| Mean activity | DUET 0.88804; max activity 0.90849; Feldman ED1 0.89945; Sivanandan ED1 0.90788, ED2 0.89524 |
| 5th–95th band, cells | max activity 292–430 → DUET 348–427, 42.9% narrower |
| 10th percentile | max activity 0.62158 (311) → DUET 0.71808 (359), +15.5%; Feldman 314; Sivanandan 313–327 |
| 95th/5th ratio | DUET 1.2265; max activity 1.4731; Feldman ED1 1.4461; Sivanandan ED1 1.4445, ED2 1.3741 |
| S2a normalized HV, mean ± SE | DUET 0.8702 ± 0.0045; Feldman 0.1834 ± 0.0160; Sivanandan 0.3030 ± 0.0129 |
| Endpoints | λ = 1.0: decode 0.81027 at activity 0.72083; λ = 0: decode 0.72918 at activity 0.90849 |
| Zero-accuracy guides per trial (of 2,200) | max activity 45/28/32/14/26 (trial 3: 1.45%, printed 1.5%); Feldman ED1 6/0/0/6/0; DUET λ = 0 41/30/32/16/28; Sivanandan and DUET λ = 0.125: 0 |
| Trial-3 activity axis cap | 2.4 (pooled 99.5th percentile 2.347; maximum activity 9.90) |
| Invalid baselines | Feldman ED=2 (0 of 10,974 rows valid) and Sivanandan ED=3 (0 of 10,390) are dropped |

What to expect:
- **Not bit-identical to the paper run.** The paper run's ground truth and DUET PEP were unseeded Monte Carlo draws, so a re-run with the seeded config is a new draw. Every operating point also has a new per-λ seed, because the paper run used the opposite λ convention. Compare DUET statistics to about ±0.005: two earlier runs of this design differ by up to 0.005.
- **Labels.** `results.csv` reads `DUET (lambda=0.12)` and the figures `DUET (λ=0.12)` for λ = 0.125. Filenames carry no λ.
- **The 97.5% pick is fragile.** At λ = 0.125 the activity cleared the 0.975 × max threshold by only 0.0016–0.0033. The next λ (0.15) missed it, at 0.9706–0.9731 of max. A new draw may pick a neighbouring λ in some trials. Trial 2's 95% pick (0.25) cleared its threshold by only 0.0012.
- **Baselines.** The gene panels are seeded, and the max-activity, Feldman and Sivanandan selections are deterministic given the panel. Max-activity, Sivanandan and Feldman ED=1 mean activities should match the paper exactly, and only their decode accuracies should move. Feldman ED=2 now runs with a fixed string-hash seed, so its selection can differ from the paper run's. Expect Feldman ED=2 and Sivanandan ED=3 to be invalid again. The visualizer prints the dropped-row warning to `logs/visualize.log` (21,364 rows = 10,974 + 10,390 in the paper run); the summary table at the end of `logs/run.log` simply omits invalid methods.

## Cost

Measured on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs, 220 GB RAM):

| Stage | Wall clock | Device |
|---|---|---|
| Benchmark, 5 trials at 3m50s to 4m04s each | about 20 min | 4 GPUs + CPU |
| DUET PEP / 18-λ swap sweep (18 workers), summed over trials | 2.2 min / 12.3 min | GPU / CPU |
| Sivanandan ED 1–3 / Feldman ED 1–2, summed | 1.4 min / 1.0 min | CPU |
| Ground-truth evaluation, summed | 2.4 min (GPU cache 2.15 min) | GPU |
| Figures, all 459 (`--debug-plots`) | about 8 to 10.5 min | CPU |

- Budget about 30 min in total. A default render writes 13 figures and has not been timed.
- The first run builds a cold PEP cache, because `cache_dir` is new.

## Determinism

- **Seeded Monte Carlo.** `evaluator.seed: 42` and `duet.pep.seed: 43` seed the ground truth and the DUET PEP. They differ on purpose (see Seeds under Settings; the streams are spawned in `CodebookEvaluator.initialize_cache` and `gpu_cache.initialize_cache_gpu` for the ground truth, and in `CodebookEvaluator.compute_pep_matrix` and `gpu_pep.compute_pep_matrix_gpu` for the PEP). The paper run had neither and drew fresh OS entropy. They do not inherit the top-level `seed`; removing either makes that step unseeded again.
  - With the same code, envs, input and config, a re-run should repeat `results.csv` and every `trial_NN/guides.csv` byte for byte, on either device (see `device`). The `ops_uniform` regression fixture (`scripts/benchmark/regression/`), the same pipeline, did so in two fresh 4-GPU runs. A second run of `config.smoke.yaml` with its own, cold PEP cache repeated `results.csv`, `trial_01/guides.csv` and the PEP counts byte for byte. The full config has not been run twice. Logs, SVGs and the cache `*.meta.json` files always differ.
  - A re-run is still a new draw relative to the paper's unseeded run, so compare against the paper's numbers with the tolerance above.
  - The ground-truth draws are indexed by each guide's position in the per-trial union of all selections, so a change in any one method's selection shifts the draws for every method.
  - The PEP fingerprint contains the seed, so a seeded run never reuses an unseeded cached PEP, and a cached seeded PEP holds the counts a rebuild would give. The ground-truth evaluator is never cached.
- **Trial seeds.** `seed: 42` produces the 5 trial seeds, which drive the gene panel, the random initialization and DUET.
- **Per-λ seed.** The per-λ optimizer seed is still seed + int(λ·1000), so re-runs are new draws, not relabelings. At `temperature: 0.0` that seed only breaks exact ties among maximal swap deltas. Against the paper run, the DUET arms move anyway, because the PEP is a new draw.
- **`device`.** Changes speed only in this experiment. Hamming costs are integers, so CPU and GPU give identical results for both the PEP and the ground truth. This is not true for the NLL experiments.
- **`duet.pep.num_cpus`.** It does not change the random streams, but it is part of the PEP fingerprint. Changing it forces a PEP recompute, which gives the same counts: the seed is split per 100-codeword batch, not per worker.
- **Speed only.** `evaluator.num_cpus` (ground-truth workers; that evaluator is never cached), `duet.optimizer.num_cpus` (each λ job reseeds itself, independent of worker count and order), `sivanandan.num_cpus` and `mem_budget_gb`.
  - With an NLL metric on CPU, `num_cpus` 1 versus 2 or more can change PEP counts ([Numerical determinism](../README.md#numerical-determinism)). That does not apply here: Hamming costs are small integers, which floating-point sums give exactly in any order.
- **Changes numbers.** `num_samples` (evaluator 2000, PEP 5000), the λ list, `trials`, `seed`, `evaluator.seed` and `duet.pep.seed`.
- **Deterministic baselines.** Sivanandan and Feldman ED=1 are deterministic given the pool. Feldman ED=2 picks guides with `set.pop()` over gene-ID strings (`maxy_clique_groups` in the `ops` package), so it depends on Python's string hash seed. `src/duet/benchmark/feldman_runner.py` pins `PYTHONHASHSEED=0` for the `conda run -n ops` child. The `ops` package's `random_state=0` is in `add_barcodes`, which the wrapper never calls.

## Smoke test

`VARIANT=smoke bash experiments/ops_crispri_symmetric/run.sh` reads `config.smoke.yaml`, which is tracked beside `config.yaml`.
- **What is shrunk.** trials 1, num_groups 50, num_controls 20, evaluator and PEP num_samples 200, λ [0.0, 1.0]. Outputs, cache and scratch go under `results/smoke/`, with the same device (gpu:all) and code path. The seeds are the real ones.
- **Panels.** The smoke config has no `visualization.paper_panels` block: the panels `config.yaml` names come from trials 3 and 5, which a one-trial run does not write.
- **Result.** It takes about 2 min on the reference machine. All methods are valid at this pool size, including Feldman ED=2 and Sivanandan ED=3. A second run with its own, cold PEP cache repeats `results.csv`, `trial_01/guides.csv` and the PEP counts byte for byte.
