# ops_crispri_cross_eval: Supp Fig S3b

The 2,200-guide Weissman CRISPRi design (1,000 genes × 2 guides + 200 controls, 10 rounds, 5 trials) is optimized four times. Each run uses a different NIS-seq HeLa optimization channel: symmetric, position-varying, asymmetric, and position-varying asymmetric. This is phase 1. In phase 2, every phase-1 DUET codebook and the Feldman, Sivanandan and maximum-activity baselines are re-scored under the measured position-varying asymmetric channel. S3a (NIS-seq error notebook) and S3c (genome-scale cross-eval) are not produced here.

## Run

<!-- docs-test: skip (runs the experiment) -->
```bash
conda activate duet          # any env whose `duet` imports from this clone (run.sh checks)
bash experiments/ops_crispri_cross_eval/run.sh
```

- **Environment.** Use the DUET environment from `environment.yml` (this clone installed editable with the `benchmark` extra), with the `gpu` extra added for the GPUs: `pip install -e ".[benchmark,gpu]"`. See [Reproducing the paper](../../docs/reproducing_the_paper.md#setup).
- Works from any cwd. run.sh cd's into this folder once and stops before any work unless all five configs exist and the active env imports `duet` from this repo's `src/`. It then runs, in order:
  - For each of `symmetric`, `position_varying`, `asymmetric` and `position_varying_asymmetric`: `scripts/benchmark/run_benchmark.py`, then `visualize_benchmark.py`.
  - `compare_benchmarks.py` on `eval_position_varying_asymmetric.yaml`.
  - `visualize_comparison.py` on the same config.
- run.sh also stops before any work if a key it reads (`outdir`, `scratch_dir`, `evaluator.scratch_dir`) is missing or null, or if a NIS-seq channel matrix that a config names is missing (see Inputs).
- The Feldman baseline shells out to `conda run -n ops` (`feldman.conda_env`). Create that environment from `environments/ops.yml` ([environments/README.md](../../environments/README.md)).
- All configs set `device: "gpu:all"`. The DUET swap search, Sivanandan and Feldman run on CPU.
- Phase 2 reads phase-1 outputs, so all four phase-1 runs must finish first.
- To re-render the panel alone, run `python ../../scripts/benchmark/visualize_comparison.py --config eval_position_varying_asymmetric.yaml` from this folder. It must be `python <path>`, not `-m`: both comparison scripts use a bare `from comparison import`. For one arm's figures, run `python ../../scripts/benchmark/visualize_benchmark.py --config <model>.yaml` from this folder.
- run.sh never passes `--debug-plots`, so it writes no per-trial figures (see Outputs). Add the flag to either re-render command to write them.

## Inputs

- **The pool.** `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv` (hCRISPRi-v2.1, Horlbeck et al. 2016, eLife 5:e19760, CC BY 4.0; in the repository), read as the `WeissmanCRISPRi` package default.
- **The NIS-seq channels in `noise_matrices/`.** The three arms other than `symmetric` read them, and so does the phase-2 evaluator:
  - `nisseq_channel.npy` (4×4) is `nisseq_hela_channel_pctl50_subsample50000.npy`;
  - `nisseq_positional_10rounds.npy` (10,) and `nisseq_positional_channel_10rounds.npy` (10,4,4) are the first 10 of the 14 rounds of `nisseq_hela_positional_pctl50_subsample50000.npy` and `nisseq_hela_positional_channel_pctl50_subsample50000.npy`.

  These are the 50,000-reads-per-file fit of `experiments/nisseq_error_analysis`. The symmetric arm needs no file: its literal `epsilon: 0.14465571` equals 1 − diag(`nisseq_hela_uniform_pctl50_subsample50000.npy`).
- **Availability of the NIS-seq matrices.** The NIS-seq channel matrices are fitted from spot-level base calls that the authors of Fandrey et al. (2025) shared on request, and are included only with their permission. If the files in `noise_matrices/` are missing from your copy, rebuild them with `experiments/nisseq_error_analysis` from the spot calls, which are available from those authors on request. Its `SUBSAMPLE_NS="50000"` run writes the 14-round matrices to `results/experiments/nisseq_error_analysis/subsample50000/noise_matrices/`. Then, from the repository root:

  <!-- docs-test: skip (needs the nisseq_error_analysis outputs) -->
  ```python
  import os
  import numpy as np
  src = "results/experiments/nisseq_error_analysis/subsample50000/noise_matrices/nisseq_hela_{}_pctl50_subsample50000.npy"
  dst = "experiments/ops_crispri_cross_eval/noise_matrices/"
  os.makedirs(dst, exist_ok=True)
  np.save(dst + "nisseq_channel.npy", np.load(src.format("channel")))
  np.save(dst + "nisseq_positional_10rounds.npy", np.load(src.format("positional"))[:10])
  np.save(dst + "nisseq_positional_channel_10rounds.npy", np.load(src.format("positional_channel"))[:10])
  ```

  When the rebuilt 14-round matrices equal the paper's (the provenance check of `experiments/nisseq_error_analysis` tests this when your copy has the reference matrices; otherwise compare with the sha256 values in its README, Compare after a re-run), these three files are byte for byte the ones the paper used.

## Settings

- **λ** (`duet.optimizer.lambda`, all four phase-1 configs): `[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 1.0]`, ascending exact literals. λ weights decoding accuracy, so λ = 1 is decode-only ([ADR 0001](../../docs/adr/0001-lambda-weights-decoding-accuracy.md)). Labels use `:.2f`, so 0.125 reads `0.12` and 0.175 reads `0.17`.
- **No hard-coded λ.** The eval config selects `regex: "DUET.*"`, which takes every λ. `compare_benchmarks.py` and `visualize_comparison.py` have no λ logic (Pareto connectors sort by objective value).
- **Layout.** One folder: `<model>.yaml`, `eval_position_varying_asymmetric.yaml` and `noise_matrices/`. Every relative path in the configs resolves against this folder: the OPS and Comparison loaders take paths verbatim, relative to the cwd.
- **outdir:** `../../results/experiments/ops_crispri_cross_eval/<model>` for phase 1 and `.../cross_eval/eval_position_varying_asymmetric` for phase 2. `reference_dir` and all six `methods[].path` point at this experiment's phase-1 outdirs.
- **cache_dir:** `../../results/cache/ops_crispri_cross_eval`, shared by the four phase-1 runs. For the three `.npy`-based arms the PEP fingerprint also contains the path string. The eval config's `cache_dir` is parsed but unused.
- **scratch_dir:** `../../results/scratch/ops_crispri_cross_eval`. In phase 1 it is a top-level key; in the eval config it is `evaluator.scratch_dir`. run.sh reads both keys and `mkdir -p`s both directories (the same path here) because `mkdtemp` needs them to exist. `TMPDIR` is set to `<scratch_dir>/tmp`, which catches Feldman's temp directory and `conda run` temp files.
- **Seeds:** `evaluator.seed: 42` and `duet.pep.seed: 43` in the four phase-1 configs, and `evaluator.seed: 42` in the eval config. The PEP gets its own seed so that its Monte Carlo streams are independent of the ground truth's: both spawn one child stream per index from their root seed, so with one shared seed PEP batch j and ground-truth guide j would start from the same stream. `ComparisonConfig` reads `evaluator.seed` through `EvaluatorConfig.from_dict` and `compare_benchmarks.py` passes it to `create_evaluator`; the eval config's top-level `seed` never reaches the evaluator.
- **Other keys:**
  - `trials: 5`, top-level `seed: 42`, `device: "gpu:all"`, `mem_budget_gb: 64.0` and `eval_mem_budget_gb: 64.0`.
  - The candidate pool (`WeissmanCRISPRi`, `seq_rounds 10`, `quota 2`, `num_controls 200`, `num_groups 1000`, `min_rank 10`).
  - Evaluator `num_samples 2000` / `num_cpus 30`, and PEP `num_samples 5000` / `num_cpus 30`.
  - Optimizer: `temperature 0.0`, `max_iter 100000`, `max_patience 3000`, duplicate avoidance with penalty 100, `num_cpus 18`. Also `use_mmap`, `force_rebuild: false` and random initialization.
  - Sivanandan ED `[1, 2, 3]` with 8 CPUs, Feldman ED `[1, 2]` via `ops`, and the eval method labels and blue gradient colors.
  - The flat `ComparisonConfig` schema: top-level `device` and `eval_mem_budget_gb`, `scratch_dir` inside `evaluator`. The top-level `seed: 42` is required there but numerically unused.
- **Not shareable with `ops_crispri_symmetric`:** this experiment has the same pool, trials and seed as Fig 3b. However, its symmetric arm uses ε = 0.14465571, not Fig 3b's 0.1, so the two cannot share a run.

## Outputs

Under `results/experiments/ops_crispri_cross_eval/`:

| file | use |
|---|---|
| `cross_eval/eval_position_varying_asymmetric/pareto_fronts/Mean_decode_accuracy_aggregated.svg` | **Supp S3b** |
| `cross_eval/eval_position_varying_asymmetric/aggregated_metrics.csv` | per (Label, Method, Trial) metrics behind the panel |
| `cross_eval/.../reevaluated_results.csv` | per-guide re-scores (847k rows, about 236 MB) |
| `cross_eval/.../hypervolume/hypervolume_barplot.svg` | normalized HV, mean ± s.e.m. over trials, as horizontal bars (not in the paper) |
| `cross_eval/.../pareto_fronts/<metric>_aggregated.svg` (the other six metrics), `compare_config.yaml` | not in the paper; config copy |
| `cross_eval/.../hypervolume/hypervolume_trial_N.svg`, `cross_eval/.../pareto_fronts/<metric>_trial_N.svg` | per-trial copies (not in the paper); debug plots, written only with `--debug-plots` |
| `<model>/results.csv`, `<model>/trial_NN/guides.csv`, `<model>/config.yaml` | phase-1 results; the pva `trial_NN/guides.csv` files are phase 2's `reference_dir` |
| `<model>/hypervolume/normalized_hypervolume_by_method_group.svg`, `<model>/pareto_fronts/<metric>_aggregated.svg` | phase-1 figures, 8 per arm (not in the paper) |
| `<model>/pareto_fronts/<metric>_trial_N.svg`, `<model>/{hypervolume_visualization,guide_level_comparisons,error_metrics}/` | phase-1 per-trial figures (not in the paper); debug plots, written only with `--debug-plots` |
| `logs/` | `<model>.{raw.log,log,visualize.log}`, `eval_position_varying_asymmetric.{raw.log,log,visualize.log}` |

A default run writes 40 figures: 8 per arm and 8 for the cross-eval. With `--debug-plots` it writes all 1,884.

Caches go under `results/cache/ops_crispri_cross_eval/` and scratch under `results/scratch/ops_crispri_cross_eval/`.

## Compare after a re-run

The paper's values, recomputed from the paper run's `cross_eval/eval_position_varying_asymmetric/aggregated_metrics.csv`. All are means over 5 trials, re-scored under the pva channel.

| baseline | mean decode | mean activity |
|---|---|---|
| Maximum activity | 0.75006 | 0.90849 |
| Feldman ED=1 | 0.75899 | 0.89945 |
| Sivanandan ED=1 | 0.75899 | 0.90788 |
| Sivanandan ED=2 | 0.76817 | 0.89524 |

Normalized HV, mean ± SEM, from per-trial `compare_pareto_fronts` over all labels (`hypervolume_barplot.svg`):

| label | normalized HV |
|---|---|
| DUET (Position-varying asymmetric error) | 0.8878 ± 0.0026 |
| DUET (Position-varying error) | 0.8680 ± 0.0025 |
| DUET (Symmetric error) | 0.8595 ± 0.0046 |
| DUET (Asymmetric error) | 0.8334 ± 0.0033 |
| Sivanandan et al. | 0.3694 ± 0.0139 |
| Feldman et al. | 0.2305 ± 0.0185 |

- **Baseline activity** should match to every printed digit. Pools and baselines are deterministic, and activity is not Monte Carlo. The same four activities appear in the paper's Fig 3b run.
- **Everything else** is a new draw, not bit-identical: baseline and DUET decode, the DUET fronts and HV. The paper run's PEP and both evaluators were unseeded, so a re-run's seeded draws are new relative to them. Every DUET arm also has a new per-λ seed, because the paper run used the opposite λ convention.
- **Tolerance:** no S3b-specific tolerance was measured. For Fig 3b (same pool), expect about ±0.005 for DUET means: two earlier runs of that experiment differ by up to 0.005.
- **DUET HV order:** some gaps between DUET designs are only a few SEM (Position-varying 0.8680 ± 0.0025 vs Symmetric 0.8595 ± 0.0046), so their order is not guaranteed.

## Cost

Measured on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs, 220 GB RAM):

| step | wall clock |
|---|---|
| phase 1, per channel | sym 21m34s, pv 20m15s, asym 20m30s, pva 19m40s. Each: DUET 14.2-14.9 min (PEP 2.15-2.65 min on GPU, swap 12.0-12.6 min on CPU), Sivanandan 1.7-2.4 min, Feldman 1.0 min, evaluation 2.2-2.8 min |
| phase 1 figures, per channel | 3m50s-4m22s for every figure (now `--debug-plots`); a default render writes 8 per arm |
| phase 2 compare (`eval_position_varying_asymmetric`) | 5m50s |
| phase 2 visualization | about 15-17 s (not logged) |
| **this run.sh (4 runs + 1 eval)** | **about 1h44m** |

## Determinism

- **Seeded Monte Carlo:** `evaluator.seed: 42` and `duet.pep.seed: 43` (distinct on purpose; see Seeds under Settings) in the four phase-1 configs, and `evaluator.seed: 42` in the eval config. The paper run had none of them, so its phase-1 PEP, phase-1 evaluation and phase-2 re-evaluation drew fresh OS entropy. The keys do not inherit the top-level `seed`; removing one makes that step unseeded again.
  - With the same code, envs, inputs and configs, a re-run on the same device should repeat each phase-1 `results.csv` and `trial_NN/guides.csv` and the phase-2 CSVs byte for byte. For the NLL arms this also assumes the same GPU and BLAS stack (see `device` below). The `ops_uniform` regression fixture (`scripts/benchmark/regression/`), the same phase-1 pipeline with the Hamming metric, did so in two fresh 4-GPU runs. A second run of the `ops_crispri_symmetric` smoke config did so too; this experiment's configs were not run twice. Logs, SVGs and the cache `*.meta.json` files always differ.
  - A re-run is still a new draw relative to the paper's unseeded run, so compare against the paper's numbers with the tolerance above.
  - Phase 2 computes no PEP. Per trial, `compare_benchmarks.py` builds one evaluator over the union of the guides that the six method sources select, sorted by pool index, and seeds each guide by its position in that union. Every trial uses the same `evaluator.seed`: the per-trial seeds it derives from the top-level `seed` are only printed, and `cache_dir` is only printed. Phase-2 draws therefore repeat only for identical phase-1 outputs and method sources; a change to any selection shifts the draws of the guides after it in the union. Phase 1's ground truth indexes its own union the same way.
  - The PEP fingerprint contains the seed, so a seeded run never reuses an unseeded PEP from `results/cache/ops_crispri_cross_eval/`, and a cached seeded PEP holds the counts a rebuild on the same device would give.
- **Trial seeds:** trial seeds are drawn from `seed: 42`. They fix the pools, the random initialization and the DUET tie-break stream. Sivanandan and Feldman are deterministic; Feldman ED=2 only because `feldman_runner.py` pins `PYTHONHASHSEED=0` for the `ops` child (see [ops_crispri_symmetric](../ops_crispri_symmetric/README.md#determinism)).
- **The per-λ optimizer seed is still seed + int(λ·1000), so re-runs are new draws, not relabelings.** Here, seed is the trial seed.
- **device changes numbers for the NLL evaluators.** The ground-truth evaluator cache uses float64 on CPU and float32 matmul on GPU, so near-ties can flip.
  - Measured on CPU with the same draws, float64 vs float32 BLAS, NIS-seq 10-mers, 120 codewords, K = 1000 (2026-09-23): pva 0 / 14.4M competitor entries flip, asymmetric 41 / 14.4M, position-varying 2 / 14.4M.
  - The PEP is float32 on both devices; for NLL metrics it depends on BLAS summation order.
  - The Hamming metric (symmetric arm, phase 1 only) is exact on either device. Keep `gpu:all`.
- **Speed only:** `num_cpus` never changes the draws. However, `duet.pep.num_cpus` is in the PEP fingerprint, so changing it forces a PEP rebuild; the rebuild draws the same samples, because the seed is split per 100-codeword batch, not per worker. `mem_budget_gb` and `eval_mem_budget_gb` affect only batching.
  - On GPU, as configured, the worker count does not change the PEP. On CPU with an NLL metric (the three `.npy` arms), `duet.pep.num_cpus` 1 versus 2 or more can change PEP counts; values of 2 or more agree ([Numerical determinism](../README.md#numerical-determinism)).
- **Changes numbers:** `num_samples` (`duet.pep.num_samples` is also in the PEP fingerprint), `evaluator.seed` and `duet.pep.seed`.

## Smoke test

`VARIANT=smoke bash experiments/ops_crispri_cross_eval/run.sh` uses the `<name>.smoke.yaml` files, which are tracked beside the configs, with outputs under `results/smoke/`. The four phase-1 runs use `trials 1`, `num_groups 50`, `num_controls 20`, evaluator and PEP `num_samples 200`, λ `[0.0, 1.0]` and `gpu:all`; the eval reads their smoke outputs with `num_samples 200`.

It takes about 8 min on the reference machine. The compare step and `visualize_comparison` run, and all 6 labels are present. The smoke reads the same `noise_matrices/` files as the real run.
