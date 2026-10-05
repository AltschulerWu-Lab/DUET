# synthetic_hvr: Supplementary Fig. S1 (Pareto-front recovery, HVR on the enumerable 2-D synthetic benchmark)

DUET, Greedy NLL and Greedy Hamming each sweep a 101-value λ grid. Their fronts are scored against the exhaustive Pareto front of all 3,125 codebooks per cell. There are 60 cells: 4 noise channels × 3 error rates × 5 trials.

## Run

<!-- docs-test: skip (runs the experiment) -->
```bash
conda activate duet                    # any env whose `duet` imports from this clone (run.sh checks)
bash experiments/synthetic_hvr/run.sh  # works from any cwd
```

- **Environment.** Use the DUET environment from `environment.yml` (this clone installed editable with the `benchmark` extra), with the `gpu` extra added for the GPUs: `pip install -e ".[benchmark,gpu]"`. See [Reproducing the paper](../../docs/reproducing_the_paper.md#setup).
- run.sh `cd`s into this folder and puts the repo root on `PYTHONPATH`, because both stages run as `python -m scripts.benchmark.synthetic...`. It uses the active env's `python` and stops unless the config exists and that env imports `duet` from this repo's `src/`. No other env is used.
- Stage 1 is the live runner `scripts/benchmark/synthetic/run_2d_synthetic_benchmark.py`, wrapped in `script -qefc`. Stage 2 is `scripts/benchmark/synthetic/visualize_2d_synthetic_benchmark.py --indir <outdir>`.
- To redraw the figures only, run this from the repo root: `PYTHONPATH=. python -m scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark --indir results/experiments/synthetic_hvr`. Add `--debug-plots` for the per-trial folders `raw_scatter_per_trial/` and `pareto_per_trial/`, which run.sh never writes.
- The config requests `device: "gpu:all"`. If the GPUs are unavailable, do not switch to CPU: that changes the numbers (see Determinism).
- run.sh stops if the config's `outdir` is missing or null. It writes `run.raw.log`, `run.log` and `visualize.log` to `<outdir>/logs/` and exports `PYTHONUNBUFFERED=1`. `nvidia-smi` prints to the terminal only, so GPU names are not logged.

**Cost** on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs, 220 GB RAM): the runner takes about 18 to 19.5 min and the figures under 1 min. Summed over cells:
- Exhaustive ground-truth scoring plus DUET setup: 12.0 min (62%).
- DUET's 101-λ swap sweep on CPU: 5.1 min.
- Greedy baselines and HV bookkeeping: 1.9 min. PEP: 0.16 min. Evaluator-cache init: 0.17 min.

## Settings

- **λ lists** (`duet.optimizer.lambda`, `greedy_hamming_mo.lambda`, `greedy_nll_mo.lambda`): the 101-point grid 0.00…1.00. λ weights decoding accuracy, so λ = 1 is decode-only ([ADR 0001](../../docs/adr/0001-lambda-weights-decoding-accuracy.md)), and the greedy baselines blend λ·distance + (1−λ)·score. There is no other hard-coded λ in this pipeline.
- **Other keys:**
  - seed 42, trials 5, device `gpu:all`.
  - The `pool` block (5 groups × 5 candidates, length 8, binary, quota 1).
  - `evaluator.num_samples` 20000 and `evaluator.num_cpus` 20.
  - `noise_channels` (all four) and `error_rates` [0.05, 0.1, 0.2].
  - `duet.optimizer`: temperature 0.0, max_iter 100000, max_patience 3000, num_cpus 40.
  - This schema has no memory-budget keys.
- **`outdir`:** `../../results/experiments/synthetic_hvr`, relative to this folder, i.e. `results/experiments/synthetic_hvr/`.
- **PEP cache:** at `<outdir>/artifacts/pep_cache/<noise>_<err>_trial_NN/`. The runner hard-codes this path.
- **Scratch:** the runner ignores `evaluator.scratch_dir`, and the configs do not set it. The evaluator's mmap temp files go to `$TMPDIR`, which run.sh sets to `results/scratch/synthetic_hvr/` (smoke: `results/smoke/scratch/synthetic_hvr/`).
- **Inputs:** there is no expression table, codebook, noise-matrix file or other input. The runner reads only the config and builds the channels in memory.

## Outputs

Under `results/experiments/synthetic_hvr/`:

| File | Use |
|---|---|
| `aggregate_hvr.svg` | **Supp Fig. S1**: the whole figure, formerly panel A. It shows 4×3 boxplots over 5 trials, with the exhaustive method dropped. |
| `hv_summary.parquet` | Per-cell HV and HVR: 240 rows (60 cells × 4 methods). The text medians are `groupby('method').hvr.median()`; they are not drawn in the panel. |
| `results.parquet` | 18,738 rows: every recovered point plus the exhaustive front. |
| `aggregate_hv.svg`, `aggregate_igd.svg`, `recovered_fronts.svg`, `recovered_fronts_per_trial.svg` | Diagnostics, written on every run; not in the paper. `recovered_fronts_per_trial.svg` is one file with every trial overlaid, and the only default figure that draws the exhaustive front. |
| `raw_scatter_per_trial/`, `pareto_per_trial/` | Debug plots, one file per trial, written only with `--debug-plots`; not in the paper. |
| `artifacts/{pools,codebooks,exhaustive_fronts,pep_cache}/` | Codebook files are `small_<noise>_<err>_trial_NN.npz`, with keys `duet_lambda_{λ:.4f}`, `greedy_{hamming,nll}_lambda_{λ:.4f}` and `exhaustive_pf_NNN`. |
| `config.yaml`, `logs/` | The runner's copy of the config (an output, not an input) and the logs. |

## Compare after a re-run

The paper's values, recomputed from the paper run's `hv_summary.parquet`:

| Method | Median, 60 cells | Mean | Median at 5% | 10% | 20% |
|---|---|---|---|---|---|
| DUET | 0.9531 | 0.9313 | 0.9641 | 0.9573 | 0.9335 |
| Greedy NLL | 0.7372 | 0.6641 | 0.8075 | 0.7794 | 0.6878 |
| Greedy Hamming | 0.7471 | 0.6665 | 0.8075 | 0.7754 | 0.7066 |

- **DUET median by channel:** symmetric 0.9699, position-varying 0.9183, asymmetric 0.9425, position-varying asymmetric 0.9558. The 5-trial DUET medians per box range from 0.917 (position-varying, 20%) to 0.977 (symmetric, 5%).
- **The reference point** of each cell's hypervolume is taken over the union of all four fronts in the cell (DUET, both greedy baselines, exhaustive; `src/duet/benchmark/metrics.py`).

**Full re-run result (2026-09-23), checked against the paper run's outputs:**
- **HVR:** all 240 per-cell values are bit-identical.
  - Medians: DUET 0.9531, Greedy NLL 0.7372, Greedy Hamming 0.7471 (paper: 0.953 / 0.737 / 0.747).
  - Means 0.9313 / 0.6641 / 0.6665 and all per-error-rate medians are identical.
- **Picks:** every (cell, λ) pick of DUET, Greedy Hamming and Greedy NLL equals the paper run's pick at 1−λ, because the paper run used the opposite λ convention. That is 6,060 picks per method (60 cells × 101 λ). The grid is symmetric under λ → 1−λ, so the λ lists are the same.
- **Other artifacts:** pools (5) and exhaustive fronts (60) are identical.
- **Column names:** `hv_summary.parquet` has `decode_acc_at_lambda_one` and `duet_objective_at_lambda_one` (the decode-only endpoint). No visualizer reads them.
- **`results.parquet`:** same 18,738 rows and column names. Its `lambda` values and `codebook_id` strings name the mirrored points.
- **Expectation:** the same numbers as long as the device stays `gpu:all` and the seeds, `num_samples` and code are unchanged. The NLL float32 arithmetic also assumes the same GPU and BLAS stack (see Determinism).

## Determinism

- **`device` changes numbers.** Keep `gpu:all`.
  - The ground-truth evaluator cache draws the same random streams on CPU and GPU. The arithmetic differs: the CPU computes decoding costs as float64 gather-sums, the GPU as a float32 matrix multiply.
  - Competitor decisions use a zero margin, so near-ties flip. The ground truth here is NLL on all four channels.
  - Flips measured with identical draws (binary, L=8, 20% error, about 3.1M competitor entries): symmetric 0, position-varying 119, asymmetric 22,276, position-varying asymmetric 1,732 (measured on CPU, 2026-09-23).
  - DUET's PEP is float32 on both devices. For NLL costs, exact equality depends on the BLAS summation order.
  - The swap search always runs on CPU.
- **`num_cpus` affects speed only.** This covers `evaluator.num_cpus` 20 (unused on GPU) and `duet.optimizer.num_cpus` 40.
  - The draws do not depend on the worker count: one seed per codeword for the cache, one per batch index for the PEP, and a per-λ reseed for DUET.
  - `evaluator.num_cpus` is part of the PEP cache fingerprint, so changing it forces a recompute. The PEP is seeded with the trial seed, so the recompute gives the same draw.
  - That holds on GPU, as configured. On CPU with these NLL metrics, `evaluator.num_cpus` 1 versus 2 or more can change PEP counts; values of 2 or more agree ([Numerical determinism](../README.md#numerical-determinism)).
- **Memory and batch sizes affect speed only.** The symmetrize batch (8 GB budget in the log) and the GPU sample batch (set from free device memory) accumulate exact integer counts.
- **`num_samples` changes numbers.** One value sets both the ground-truth K_eval and DUET's PEP K. It must stay ≤ 32,767 (uint16 PEP counts).
- **Seeds change numbers.** The trial seed is `seed + trial index` (42..46). It seeds the pool, the ground-truth evaluator, DUET's PEP and initial selection, and the greedy shuffles.
- **λ:**
  - The per-λ optimizer seed is still seed + int(λ·1000), so re-runs are new draws, not relabelings. Here `seed` is the trial seed.
  - DUET did not move in this experiment anyway. Its initial selection is seeded by the trial seed. At temperature 0, the per-λ stream only breaks exact ties among maximal swap deltas (`src/duet/pareto_optimization.py` ~1232-1236), and no such tie changed an outcome.
  - The greedy baselines have no per-λ seed (one shuffle per call, seeded by the trial seed). They reproduce at the mirrored λ, up to rounding in 1−λ.

## Smoke test

`VARIANT=smoke bash experiments/synthetic_hvr/run.sh` reads `config.smoke.yaml`, which is tracked beside `config.yaml`. It uses trials 1, error_rates [0.2], num_samples 500 and λ [0.0, 1.0] for all three methods, with outdir `results/smoke/synthetic_hvr`. The device (`gpu:all`) and code path are unchanged. It runs in under 30 s on the reference machine.
