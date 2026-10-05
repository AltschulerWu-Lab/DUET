# synthetic_objective_correlation: Fig 2a,b

For each trial, a random pool of 15 length-8 binary codewords is drawn and all 3,003 size-5 codebooks are enumerated. Each codebook gets a Monte Carlo decode accuracy (random tie-break, 10,000 reads per codeword) and five objective scores: DUET, mean/min pairwise NLL (diagonal-centered) and mean/min pairwise Hamming. The panels show Spearman ρ between accuracy and each objective, for 20 trials × 4 noise channels at 20% error. This experiment has no λ.

## Run

<!-- docs-test: skip (runs the experiment) -->
```bash
conda activate duet          # any env whose `duet` imports from this clone (run.sh checks)
bash experiments/synthetic_objective_correlation/run.sh
```

- **Environment.** The DUET environment from `environment.yml` (this clone installed editable with the `benchmark` extra) is all it needs. See [Reproducing the paper](../../docs/reproducing_the_paper.md#setup).
- Works from any cwd. run.sh cd's into this folder and stops unless the config exists and the active env imports `duet` from this repo's `src/`. It then runs `run_rtb.py`, then runs `scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives` and `visualize_baseline_summary` as `python -m` with `PYTHONPATH` set to the repo root.
- To redraw the figures only, run from the repo root: `PYTHONPATH=. python -m scripts.benchmark.synthetic.visualize_duet_vs_baseline_objectives --indir results/experiments/synthetic_objective_correlation` (Fig 2a) and the same with `visualize_baseline_summary` (Fig 2b). The first writes only `trial_01.svg` by default; add `--debug-plots` for trials 2-20, which run.sh never writes.
- No other env, no GPU, no data files. `run_rtb.py` reads only the config and builds the noise channels in memory. The visualizers read only the parquet and the shared stylesheet `src/duet/plotting/duet_publication.mplstyle`.
- run.sh reads `outdir` from the config and stops if it is missing or null. It writes logs to `<outdir>/logs/` and exports `PYTHONUNBUFFERED=1`.
- **Cost.** About 10 min on one CPU core of the reference machine (40 logical CPUs): the runner takes 9 to 11 min (21-50 s per trial) and the figures about 1 min.

## Settings

- `seed: 42` and `trials: 20`.
- `pool` (1 group, 15 candidates, length 8, alphabet 2, quota 5).
- `evaluator.num_samples: 10000`, and `evaluator.num_cpus: 20`, which is never read.
- All four `noise_channels` and `error_rate: 0.2`.
- There is no device key and no memory budget.
- `outdir` is `../../results/experiments/synthetic_objective_correlation`, resolved from this folder: `BaselineObjectivesConfig` takes `outdir` verbatim, relative to the cwd.
- No λ: no DUET λ appears in `run_rtb.py`, the config or either visualizer.
- No cache or scratch: `run_rtb.py` writes no cache or temp files, and `evaluator.scratch_dir` is unset (it would not be read anyway). run.sh sets no `TMPDIR`.

## Outputs

Under `results/experiments/synthetic_objective_correlation/`:

| file | use |
|---|---|
| `figures/baseline_comparison_per_trial/trial_01.svg` | **Fig 2a** (5 objectives × 4 channels, trial 1, `ρ = {:.3f}` per cell) |
| `figures/summary_spearman.svg` | **Fig 2b** (one panel per channel, one horizontal box per objective over 20 trials, whiskers 5-95%) |
| `figures/summary_regret.svg` | not in the paper (a diagnostic, written on every run) |
| `figures/baseline_comparison_per_trial/trial_02..20.svg` | not in the paper; debug plots, written only with `--debug-plots`. The visualizer names `trial_01` (Fig 2a) in its `PAPER_PANELS`, so that one is always written |
| `objective_correlation.parquet` | 240,240 rows (20 × 4 × 3,003), 12 columns; input to both panels |
| `config.yaml` | copy of the config used |
| `logs/` | `run.raw.log`, `run.log`, `visualize_scatter.log`, `visualize_summary.log` |

ρ is computed per (trial, channel, objective) between `decode_accuracy` and the objective. The objective, but not the accuracy, is first rounded to 9 decimals (`round_for_ranking`). ρ is NaN if either column has fewer than 3 distinct values.

## Compare after a re-run

The paper's panels are `figures/baseline_comparison_per_trial/trial_01.svg` (Fig 2a) and `figures/summary_spearman.svg` (Fig 2b) of a run with this config.

Median ρ over 20 trials, recomputed from the paper run's parquet with the visualizer's own function:

| objective | symmetric | position_varying | asymmetric | pos_var_asym | mean of 4 | range |
|---|---|---|---|---|---|---|
| DUET | 0.9776 | 0.9797 | 0.9791 | 0.9782 | 0.9786 | 0.0021 |
| mean NLL | 0.8087 | 0.7714 | 0.7907 | 0.7602 | 0.7828 | 0.0486 |
| min NLL | 0.6616 | 0.7696 | 0.6648 | 0.7518 | 0.7120 | 0.1080 |
| mean Hamming | 0.8087 | 0.7032 | 0.7907 | 0.7031 | 0.7514 | 0.1056 |
| min Hamming | 0.6616 | 0.6493 | 0.6648 | 0.6375 | 0.6533 | 0.0273 |

Fig 2a (trial 1) ρ, sym / pv / asym / pva: DUET .973 / .980 / .976 / .983; mean NLL .828 / .753 / .824 / .757; min NLL .593 / .828 / .667 / .838; mean Hamming .828 / .706 / .824 / .718; min Hamming .593 / .721 / .667 / .741. Per-trial DUET ρ spans 0.949-0.991 over all channels (per-channel minimum 0.9493 position_varying to 0.9648 symmetric). The lowest single-trial baseline ρ is 0.2242, trial 5 under symmetric noise, where min Hamming and min NLL tie. Median argmax regret for DUET is 0 in every channel.

**Expect bit-identical numbers.** There is no λ, and the seeds and code are unchanged. A full re-run (2026-09-23) confirmed it:
- `objective_correlation.parquet` had 240,240 rows × 12 columns, and every column was bit-identical to the paper run's (`DataFrame.equals` is True; float columns were compared as uint64 bit patterns).
- The file sha256 differed only because of the parquet writer's `created_by` metadata (another pyarrow version). Compare the parquet by content (`DataFrame.equals`, or floats as bit patterns), not by file hash.
- Each of the 22 SVGs matched the paper run's except for the embedded `dc:date` and matplotlib's generated element ids. The figures now render at a full-page width of 180 mm, so a re-run's SVG sizes differ from the paper run's.

## Determinism

- CPU only, single process, float64. Ties use a fixed tolerance (`_TIE_TOL = 1e-9`). There is no device setting and no GPU path.
- The per-trial seed is `seed + (trial - 1)`, i.e. 42-61, whatever `trials` is. It seeds the pool and, for each noise channel, a fresh `default_rng(trial seed)` stream that is consumed serially over the 15 pool codewords.
- Knobs that change the numbers: `seed`, `evaluator.num_samples` (reads per codeword; it also shifts the random stream), the `pool` keys, `noise_channels`, `error_rate`, and the `duet` objective code.
- Knobs with no effect: `evaluator.num_cpus` and `evaluator.scratch_dir` (neither is read), `TMPDIR`, and the cwd beyond resolving `outdir`. No setting trades speed for memory. The cost cache (15 × 10,000 × 15 float64, rebuilt per trial and channel) is 18 MB.
- There is no λ, so there is no per-λ seed.
- **Code.** `run_rtb.py` imports the objectives (including `compute_pairwise_nll`) from `src/duet/benchmark/`. A change there changes the output.

## Smoke test

`VARIANT=smoke bash experiments/synthetic_objective_correlation/run.sh` reads `config.smoke.yaml`, which is tracked beside `config.yaml`. It sets `trials: 1`, `num_samples: 200` and outdir `results/smoke/synthetic_objective_correlation`, and runs in under 20 s.
