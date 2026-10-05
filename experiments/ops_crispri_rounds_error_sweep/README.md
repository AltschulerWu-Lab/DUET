# ops_crispri_rounds_error_sweep: Fig 3c

This experiment reproduces the sequencing-rounds × error-rate robustness sweep on the Weissman CRISPRi library. It has 18 points: sequencing rounds 8–13 × symmetric per-position error ε ∈ {0.03, 0.05, 0.10}. Each point runs the Fig 3b design over 5 trials: 1,000 genes × 2 guides plus 200 non-targeting controls (2,200 guides), Hamming metric and unique-minimum decoding. It compares DUET's 31-point λ sweep with Feldman et al. (ED 1, 2), Sivanandan et al. (ED 1, 2, 3) and maximum activity. Each heatmap cell is the delta between one DUET arm and a baseline. The arm is, per point and trial, the highest-mean-decode-accuracy λ whose mean activity is at least 97.5% of maximum activity.

Feldman et al. (2019) measured per-round errors of 0.016–0.046 (cell level) and 0.020–0.059 (read level), so ε = 0.10 is a stress test.

## Run

<!-- docs-test: skip (runs the experiment) -->
```bash
conda activate duet        # any env whose `duet` imports from this clone (run.sh checks)
bash experiments/ops_crispri_rounds_error_sweep/run.sh
```

- **Environment.** Use the DUET environment from `environment.yml` (this clone installed editable with the `benchmark` extra), with the `gpu` extra added for the GPUs: `pip install -e ".[benchmark,gpu]"`. See [Reproducing the paper](../../docs/reproducing_the_paper.md#setup).
- run.sh runs `scripts/benchmark/run_benchmark.py` once per point, in grid order (`sr08_eps0.03` … `sr13_eps0.10`), then this folder's `visualize_sweep.py`.
  - It works from any cwd.
  - It stops before any work unless the 18 configs exist, their `outdir`s share one parent and one `scratch_dir`, and the active env imports `duet` from this repo's `src/`.
  - The points run one after another. Each uses all four GPUs for its PEP build and ground truth, and 31 CPU workers for the λ sweep.
- **A failed point does not stop the sweep.** The 18 points are independent, so run.sh runs every selected point, records the failed ones, writes each point's cleaned log either way, and then runs the visualizer on whatever results exist. It prints a summary (points ok, skipped, failed, and the visualizer's exit code). It exits 1 if any point failed, otherwise with the visualizer's exit code, so a Tier A reference-check failure still exits 1 after the figures are written. When a point failed, `logs/visualize.log` starts with a warning naming it: any cells for that point come from results already on disk before the run. Re-run failed points with `SWEEP_POINTS="<stems>"` or `SKIP_COMPLETE=1`.
  - A signal (Ctrl-C, SIGTERM, SIGHUP) stops the whole sweep, not just the running point: run.sh exits 128 + the signal number once the running point has stopped.
  - The cost of carrying on: a failure that hits every point (a broken `ops` env, say) runs the whole sweep (8h50m in the 2026-09-26 re-run) before the next experiment starts. `run_all.sh`'s preflight checks the `ops` env and the configs first, which catches the likeliest such causes.
- **Order.** Run it after `experiments/ops_crispri_symmetric`. The visualizer checks this sweep's rounds-10 points against Fig 3b's outputs; see Reference check. If those outputs are absent it prints `SKIPPED` and does not fail.
- **A subset:** `SWEEP_POINTS="sr10_eps0.10 sr08_eps0.03" bash .../run.sh`. It runs those points in grid order; an unknown name stops the driver. The visualizer always renders every point that has results.
- **Resuming:** `SKIP_COMPLETE=1 bash .../run.sh` skips a point only if its run is complete:
  - `<outdir>/config.yaml` is a byte copy of the point's config (the runner copies it after writing `results.csv`);
  - `results.csv` has every configured method (31 DUET arms, maximum activity, Sivanandan ED 1–3, Feldman ED 1–2) in every trial.

  The default reruns every selected point. `run_benchmark.py` writes `results.csv` only after all 5 trials, so an interrupted point restarts from trial 1.
- **Figures only,** from existing results: `cd experiments/ops_crispri_rounds_error_sweep && python visualize_sweep.py` (about 1 min). It resolves every path from this folder, so the `cd` is optional. It draws the 8 default heatmaps (see Outputs); add `--debug-plots` for all 60. run.sh never passes the flag.
- **Per-point panels.** run.sh does not run `visualize_benchmark.py` per point (it would cost about 9m40s and 127 MB per point). To render one point: `cd experiments/ops_crispri_rounds_error_sweep && python ../../scripts/benchmark/visualize_benchmark.py --config sr10_eps0.10.yaml`. That writes the aggregated panels; add `--debug-plots` for the per-trial ones.
- **Feldman.** The baseline shells out to `conda run -n ops` (`feldman.conda_env: ops`). Create that environment from `environments/ops.yml` ([environments/README.md](../../environments/README.md)).
- **Configs are generated.** Edit `make_configs.py`, not the YAMLs, then run `python make_configs.py`. `python make_configs.py --check` renders into a temp dir and fails unless all 18 committed YAMLs are byte-identical to the render; it also rejects stray `sr*.yaml` files.
- **Logs** go to `results/experiments/ops_crispri_rounds_error_sweep/logs/`: `<stem>.raw.log` and `<stem>.log` per point, and `visualize.log`. run.sh prints `nvidia-smi` first and sets `TMPDIR=<scratch_dir>/tmp`.

## Inputs

- **The pool.** `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv` (hCRISPRi-v2.1, Horlbeck et al. 2016, eLife 5:e19760, CC BY 4.0; in the repository; sha256 starts `2aa08896`), read as the `WeissmanCRISPRi` package default.
- **The `ops` environment** for the Feldman baseline (above).
- **For the reference check:** the outputs of `experiments/ops_crispri_symmetric` (optional; see Reference check).

## Settings

- **`duet.optimizer.lambda`**, ascending exact literals. λ weights decoding accuracy, so λ = 1 is decode-only ([ADR 0001](../../docs/adr/0001-lambda-weights-decoding-accuracy.md)):

  `[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]`

  - It is a strict superset of Fig 3b's 18 values; the 13 extras are 0.3–0.95 without 0.5.
  - Labels use `:.2f`. 0.125 reads `0.12` and 0.175 reads `0.17`, and no two labels collide.
  - Every literal gives an exact seed offset `int(λ·1000)`. Computing `1 − 0.9` would give offset 99 instead of 100, and `1 − 0.825` would label `0.18`. `make_configs.py` checks the offsets and the labels on every render.
- **Seeds.** `evaluator.seed: 42` and `duet.pep.seed: 43`, as in `ops_crispri_symmetric`, which explains why the two differ. The PEP seed enters the PEP fingerprint.
- **Paths.** `outdir` is `../../results/experiments/ops_crispri_rounds_error_sweep/<stem>`, `cache_dir` is `../../results/cache/ops_crispri_rounds_error_sweep` and `scratch_dir` is `../../results/scratch/ops_crispri_rounds_error_sweep`. All are cwd-relative (`OpsBenchmarkConfig` takes them verbatim), and run.sh cd's here first.
- **Other keys:**
  - top level: `trials: 5`, `seed: 42`, `device: "gpu:all"`, `mem_budget_gb: 64.0`;
  - `candidate_pool`: WeissmanCRISPRi, `seq_rounds` per point, quota 2, 200 controls, 1,000 groups, `min_rank` 10;
  - `evaluator`: symmetric ε per point, hamming, unique_minimum, `num_samples` 2000, `num_cpus` 30;
  - `duet`: mmap, no forced rebuild, random init, temperature 0, `max_iter` 100000, `max_patience` 3000, duplicate penalty 100, **optimizer `num_cpus: 31`**; `pep` with the same channel, `num_samples` 5000, `num_cpus` 30;
  - Sivanandan ED [1, 2, 3] with 8 CPUs; Feldman ED [1, 2] via `ops`.
- **`num_cpus: 31` does not change results:** the λ jobs reseed themselves, and an 18-worker and a 31-worker run chose identical codebooks from the same PEP. It is more than the reference machine's 20 physical cores (see Cost).
- **`sr10_eps0.10.yaml`** differs from `ops_crispri_symmetric/config.yaml` only in `outdir`, `cache_dir`, `scratch_dir`, `duet.optimizer.lambda` and `duet.optimizer.num_cpus`.
- **The arm pick** is `select_duet_lambda`, the first `idxmax` over label-sorted rows. There is no other hard-coded λ.

## Outputs

Under `results/experiments/ops_crispri_rounds_error_sweep/`:

| Path | Content |
|---|---|
| `sweep_figures/heatmap_maximum_activity_mean_decode_accuracy_10th_percentile.svg` | **Fig 3c top**: "Absolute gain in mean decode accuracy (≤ 10th percentile)" vs maximum activity, in pp; 57.33 × 32.29 mm |
| `sweep_figures/heatmap_maximum_activity_standard_deviation_decode_accuracy.svg` | **Fig 3c bottom**: "Relative reduction in standard deviation of decode accuracy" vs maximum activity, in %; 57.33 × 32.29 mm |
| `sweep_figures/heatmap_<baseline>_mean_decode_accuracy.svg` | diagnostics, written by default: mean decode accuracy against each of the 6 baselines. Blank cells mark points where that baseline has no valid codebook |
| `sweep_figures/heatmap_<baseline>_<metric>.svg`, the other 52 | debug plots, written only with `--debug-plots`. With the 8 above, 6 baselines × 10 metrics, one per (baseline, metric) with at least one valid cell (60 in the first run) |
| `sweep_figures/sweep_summary.csv` | one row per (point, baseline, metric): mean, sd, se, n and the chosen DUET arm(s), for the caption |
| `<stem>/results.csv`, `<stem>/trial_NN/guides.csv`, `<stem>/config.yaml` | per-point results (per-guide rows with a `Valid` column), pools and config snapshot |
| `logs/<stem>.raw.log`, `logs/<stem>.log`, `logs/visualize.log` | runner logs per point and the visualizer log with the reference-check report |

- **Figure tiers.** By default the visualizer draws only the two Fig 3c panels (`FIG3C_PANELS`) and the six mean-decode-accuracy heatmaps (`DIAGNOSTIC_METRICS`). The other 52 heatmaps are debug plots, drawn only with `--debug-plots`. `sweep_summary.csv` holds every cell either way. Each heatmap's colour scale comes from its own cells, so skipping some changes none of the others.
- **Panel layout.** Every heatmap is a third-page slot panel with fixed millimetre margins (`heatmap_slot_figure`), saved at exactly its size with `save_panel`.
  - The two Fig 3c panels are 57.33 × 32.29 mm. Stacked with the 4 mm gap, they span Fig 3b right (68.58 mm), and their cells and colour bars line up.
  - Titles wrap to two lines and follow the Fig 3c caption.
  - Cell labels are 7 pt, with at most one decimal and no percent sign (`+7.4`, `-85`, `+0.4`); a cell is about 5.8 mm wide. The unit, "pp" or "%", sits at the top of the colour bar.
  - A debug heatmap whose title takes three lines is one line taller; its cells keep the same size.
- **Stale points.** A point whose snapshotted `config.yaml` differs from its current config prints a `WARN`.
- **Another copy.** `--results-base`, `--figures-dir`, `--reference-dir` and `--no-reference-check` allow re-rendering another copy of the results. `--variant` (or `VARIANT`) reads `<stem>.<variant>.yaml`.

## Reference check

`visualize_sweep.py` checks this sweep against `experiments/ops_crispri_symmetric` (Fig 3b). The reference directory is that experiment's `outdir`: `config.yaml`'s for the real run, `config.smoke.yaml`'s with `VARIANT=smoke`. Fig 3b is this sweep's rounds-10 design, with the same pool, optimizer, seeds 42/43 and, at ε = 0.1, the same channel, PEP and evaluator. Its 18 λ values are a subset of the sweep's 31.

The check runs after the figures are written.
- If any Tier A comparison fails, it prints `REFERENCE CHECK FAILED` and exits 1, and run.sh exits non-zero with it.
- If the reference outputs are missing, it prints `SKIPPED` and passes. The same applies when no loaded point has 10 rounds.
- It compares the reference's snapshotted `config.yaml` with each point's. If the pool or DUET settings that each tier relies on differ, that is a Tier A failure: the premise of the check is broken.

**Tier A, exact (fails the run).** At each rounds-10 point (ε 0.03, 0.05 and 0.10), per trial, it checks the following against the reference, each identical:
- `trial_NN/guides.csv`, byte for byte. The pool depends only on `seed` and `candidate_pool`, not on ε.
- Every baseline's rows from the raw `results.csv`: `Index`, `Group`, `Sequence`, `Activity score` and `Valid`, in order, invalid rows included. This covers maximum activity, Sivanandan ED 1–3 and Feldman ED 1–2, which are deterministic given the pool.
- DUET λ = 0.00. Its decode weight is exactly 0 (`combined += weight · deltas`), so the PEP, the only ε-dependent input, cannot influence it. In the first run, the activity-only arm was identical across ε at rounds 8, 10 and 13. With the current configs it is also identical to the Fig 3b λ = 0 arm at rounds 10.
- **At ε = 0.10 only:** DUET at all 18 shared λ values. The ingredients are the same:
  - pool, random init and per-λ seed (`trial_seed + int(λ·1000)`);
  - a seeded, integer-valued Hamming PEP. Its fingerprint hashes the library, channel, metric, rule, `num_samples`, `num_cpus` and seed, not the path.

  The λ jobs are independent. The 13 extra λ values and the worker count therefore change nothing.
- **The PEP is rebuilt, not shared.** This sweep's `cache_dir` differs from Fig 3b's, so the sweep builds its own copy of the same-fingerprint PEP. A seeded Hamming rebuild gives the same counts on either device, so sharing would not matter. The DUET comparison then also tests that the rebuild reproduces those counts.

**Why decode accuracy cannot match exactly.** The ground truth draws one random stream per codeword. The stream is indexed by the codeword's position in the sorted union of every codebook evaluated in that trial (`benchmark/metrics.py`, `SeedSequence(seed).spawn(N)[pos]`). In the first run, the 13 extra DUET codebooks added 390–434 pool indices per trial. The smallest new index was 8–42, so 99.6–99.9% of codewords shift position and get fresh draws. Every decode statistic therefore moves, the baselines' included, and an exact comparison of the decode cells (tolerance 1e-9) fails in every cell.

**Tier B, statistical (report only; never fails).** It runs only at ε = 0.10, the point with an equivalent reference.
- **Arms.** It prints both runs' 97.5% arm per trial and flags a trial whose arms differ. The expected arm is λ = 0.12 in all 5 trials. In a Fig 3b run with the current config, the gap to the runner-up was 0.0036–0.0046 decode accuracy, far above the noise.
- **Cells.** For each cell (baseline × metric) it compares the sweep's trial-mean delta with the reference's, which this script recomputes from Fig 3b's `results.csv`.
- **z-scores.** Each difference is scored against the Monte Carlo SD of the difference between two independent evaluations of the same codebooks. Given the codebooks, each guide's accuracy is Binomial(`num_samples`, p) / `num_samples`, independently across guides, so the script redraws the reference's per-guide accuracies (200 replicates, fixed seed). The SD comes from the per-trial variances: SD = √(2·Σ var) / n.
- **Flag.** Cells with |z| > 4 are flagged.
- **Conservative.** The model ignores the shared reads of a guide present in both codebooks, which makes the SD conservative. Its metric definitions reproduce the aggregation to 3e-16.
- **Validation.** Earlier runs of this design and of Fig 3b that shared cached PEPs but had different, unseeded evaluations evaluated the same codebooks independently. Compared at `sr10_eps0.10`, two such pairs gave a max |z| of **1.55** and **2.06** over the 40 cells, with none above 4. The largest raw differences were 0.0071 and 0.0086.

**ε = 0.05 and ε = 0.03.** No experiment is equivalent to either, so the delta comparison is **dropped** for these two points. Only the ε-independent Tier A checks (pools, baselines, DUET λ = 0) run there, against the ε = 0.10 reference.

## Compare after a re-run

Two full runs of this sweep give reference values. The first (2026-09-02) used these settings without `evaluator.seed` and `duet.pep.seed`, and with the opposite λ convention. The second (2026-09-26, below) ran these configs on the reference machine; a re-run with the same code, configs and hardware should repeat it exactly.

The first run's values, from its `sweep_figures/sweep_summary.csv`. They are DUET 97.5% arm minus maximum activity, mean ± s.e.m. over 5 trials, in %.

Fig 3c top: absolute gain in mean bottom-decile decode accuracy (percentage points).

| ε \ rounds | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|
| 0.10 | +45.40 ± 1.05 | +29.19 ± 1.22 | +18.57 ± 1.29 | +13.05 ± 0.54 | +9.08 ± 0.54 | +7.48 ± 0.64 |
| 0.05 | +65.59 ± 1.58 | +34.62 ± 1.71 | +19.11 ± 1.72 | +10.64 ± 0.66 | +6.73 ± 0.67 | +4.70 ± 0.65 |
| 0.03 | +74.89 ± 2.09 | +36.45 ± 1.92 | +17.60 ± 1.82 | +8.63 ± 0.64 | +4.85 ± 0.74 | +3.28 ± 0.65 |

Fig 3c bottom: relative change in the inter-guide SD of decode accuracy.

| ε \ rounds | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|
| 0.10 | −63.44 ± 0.45 | −62.09 ± 0.83 | −57.64 ± 1.75 | −57.70 ± 1.40 | −53.93 ± 2.88 | −60.44 ± 4.33 |
| 0.05 | −78.31 ± 0.33 | −74.29 ± 0.73 | −75.47 ± 1.63 | −76.90 ± 1.47 | −75.64 ± 4.14 | −79.40 ± 5.62 |
| 0.03 | −85.53 ± 0.16 | −83.18 ± 0.69 | −84.61 ± 1.07 | −87.40 ± 0.85 | −85.44 ± 4.07 | −85.74 ± 5.51 |

- **Rounds 10, ε 0.10** is also Fig 3b's operating point. A run of `ops_crispri_symmetric` with its current config gives **+18.53 ± 1.24** and **−57.64 ± 1.65**, with the arm λ = 0.12 in all 5 trials (recomputed with this script).
- **97.5% arms of the first run,** in the current λ convention. These are expectations, not guarantees.

  | ε \ rounds | 8 | 9 | 10 | 11 | 12 | 13 |
  |---|---|---|---|---|---|---|
  | 0.10 | 0.05 | 0.08–0.09 | 0.12 | 0.20 | 0.25–0.30 | 0.40 |
  | 0.05 | 0.12–0.15 | 0.20 | 0.35 | 0.50 | 0.65–0.70 | 0.80 |
  | 0.03 | 0.20 | 0.35–0.40 | 0.50–0.55 | 0.70 | 0.85 | 0.90 |
- **Signs and trend.** All 18 gains are positive (+3.3 to +74.9 pp), with the largest at 8 rounds. All 18 SD changes are reductions (−53.9% to −87.4%). Every panel is single-hue: no cell runs against its metric's direction.
- **Against the first run, a new draw.** Its PEPs and evaluations were unseeded, and it used the opposite λ convention, so every per-λ seed differs. The DUET codebooks and every decode statistic are therefore new draws. Expect cells within roughly its s.e.m. (0.5–2.1 pp for the gain; 0.2–5.6 pp for the SD change), and the signs and the rounds trend to hold. Baseline activities are not Monte Carlo and should match exactly, except Feldman ED=2, which now runs with a fixed string-hash seed.
- **Baseline validity in the first run.** Invalid baselines are dropped from their cells and named on stdout (`SKIP ... has no valid rows`):
  - Feldman ED=2: invalid at rounds 8–10 (1,635–2,197 guides of 2,200). Valid at rounds 11–13, except rounds 11 ε 0.05 trial 4 and ε 0.10 trial 2 (2,199 guides). With `PYTHONHASHSEED=0` its rounds-11 validity may change.
  - Sivanandan ED=3: valid only at rounds 12–13.
  - Sivanandan ED=2: invalid at rounds 8 and in trial 2 of rounds 9.
  - In all: 787 cells with n = 5, 50 with n = 4, and 3 with n = 0. The n = 0 cells are the 95th/5th ratio vs maximum activity at rounds 8, where maximum activity's 5th percentile is 0 (a divide-by-zero `RuntimeWarning`). The cell accounting line read 840/1080.
- **Overshoot warnings in the first run:** 10 of 90 picks (all 5 trials of sr13_eps0.03, plus sr08_eps0.03 t4, sr09_eps0.05 t2 and t4, sr12_eps0.10 t3, sr13_eps0.05 t2). Overshoot means the grid is still coarse above the floor at those points; the new draw may differ.

### Result of the 2026-09-26 re-run

A full re-run with these configs on the reference machine, 2026-09-26. **Everything reproduced.** A re-run with the same code, configs and hardware should repeat these values exactly (see Determinism).

- **Reference check** (`logs/visualize.log`): Tier A passed, with 15 `guides.csv`, 90 baseline and 100 DUET row sets identical to the Fig 3b run's. Tier B: the 97.5% arm was λ = 0.12 in all 5 trials of both runs; 40 cells, max |z| = 1.65, none above 4, max |diff| 0.0122.
- **Fig 3c**, from the re-run's `sweep_figures/sweep_summary.csv`, DUET 97.5% arm minus maximum activity, mean ± s.e.m. over 5 trials, in %:

  Top, absolute gain in mean bottom-decile decode accuracy (percentage points):

  | ε \ rounds | 8 | 9 | 10 | 11 | 12 | 13 |
  |---|---|---|---|---|---|---|
  | 0.10 | +45.25 ± 1.14 | +29.01 ± 1.19 | +18.59 ± 1.20 | +12.92 ± 0.39 | +9.03 ± 0.50 | +7.44 ± 0.60 |
  | 0.05 | +65.43 ± 1.60 | +34.67 ± 1.72 | +19.03 ± 1.82 | +10.67 ± 0.65 | +6.69 ± 0.69 | +4.70 ± 0.65 |
  | 0.03 | +74.86 ± 1.98 | +36.21 ± 1.90 | +17.79 ± 1.90 | +8.68 ± 0.67 | +4.88 ± 0.71 | +3.26 ± 0.65 |

  Bottom, relative change in the inter-guide SD of decode accuracy:

  | ε \ rounds | 8 | 9 | 10 | 11 | 12 | 13 |
  |---|---|---|---|---|---|---|
  | 0.10 | −63.43 ± 0.53 | −61.89 ± 0.88 | −57.59 ± 1.63 | −57.60 ± 1.13 | −53.53 ± 2.82 | −60.41 ± 4.18 |
  | 0.05 | −78.01 ± 0.31 | −74.45 ± 0.64 | −75.19 ± 1.70 | −77.15 ± 1.49 | −75.08 ± 4.47 | −79.59 ± 5.60 |
  | 0.03 | −85.47 ± 0.24 | −82.87 ± 0.61 | −84.71 ± 1.19 | −87.49 ± 0.96 | −85.67 ± 4.03 | −85.62 ± 5.49 |

- **Against the first run's values:** all 36 cells are within their s.e.m. The gains moved by −0.24 to +0.19 pp and the SD changes by −0.25 to +0.56 pp. All 18 gains are still positive (+3.26 to +74.86 pp, largest at 8 rounds) and all 18 SD changes are reductions (−53.53% to −87.49%).
- **97.5% arms** (λ, across trials) match the expected table above, except ε 0.10 at 11 rounds (0.17–0.20, against 0.20):

  | ε \ rounds | 8 | 9 | 10 | 11 | 12 | 13 |
  |---|---|---|---|---|---|---|
  | 0.10 | 0.05 | 0.08–0.09 | 0.12 | 0.17–0.20 | 0.25–0.30 | 0.40 |
  | 0.05 | 0.12 | 0.20 | 0.35 | 0.50 | 0.65–0.70 | 0.80 |
  | 0.03 | 0.20 | 0.35 | 0.50–0.55 | 0.70 | 0.85 | 0.90 |
- **Baselines and warnings:** the same cell accounting as the first run (840/1080) and the same skipped baselines. 9 overshoot warnings (10 in the first run): all 5 trials of sr13_eps0.03, plus sr09_eps0.05 t2, sr12_eps0.10 t3 and t5, and sr13_eps0.05 t2.

## Cost

Measured in the 2026-09-26 re-run on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 2 × 10-core Xeon E5-2640 v4 with 40 logical CPUs, 220 GB RAM):

- **8h50m03s** in all, every point and the visualizer. Points took 27.6–31.7 min each and ran one after another.
- Stage sums: DUET λ sweep 379 min (CPU, 31 workers), PEP 41 min (all 90 cold), evaluation 43 min, Sivanandan 32 min, Feldman 23 min. The sweep figures take about 1 min.
- **Workers.** `duet.optimizer.num_cpus: 31` oversubscribes the 20 physical cores; it does not change results (see Settings).
- **Disk.** The PEP cache (`results/cache/ops_crispri_rounds_error_sweep/`) holds about **46.5 GB**: 90 npy + sym.dat pairs, at rounds 8 0.38 GB per PEP, rounds 9 0.50 GB, rounds 10–13 0.54–0.57 GB. Outputs are about 0.7 GB (`results.csv` 33–39 MB per point). Scratch is transient, under about 1 GB per trial. Put `results/cache` and `results/scratch` on a volume with that much space.

## Determinism

- **Seeded Monte Carlo.** `evaluator.seed: 42` and `duet.pep.seed: 43`, distinct on purpose, as in `ops_crispri_symmetric`. `seed: 42` gives the trial seeds [191664963, 1662057957, 1405681631, 942484272, 929893137]. These drive the pool, the random initialization and the DUET tie-break streams. The same code, envs, input and config should repeat `results.csv` and `guides.csv` byte for byte, on either device. Hamming costs are integers, so CPU and GPU agree.
- **Re-runs are new draws, not relabelings.** The per-λ optimizer seed is still `trial_seed + int(λ·1000)` (`src/duet/runner/core.py`), and the PEP is a new seeded draw. At temperature 0 the per-λ seed only breaks exact ties among maximal swap deltas.
- **Union-indexed ground truth.** Draws are indexed by position in the per-trial union of all evaluated codebooks. Changing any method's selection, or the λ list, therefore shifts the draws of every method. The Tier B note above explains why this sweep and Fig 3b do not share decode numbers.
- **PEP cache.** It is shared by the 18 points; each fingerprint covers its library and channel. The seed is in the fingerprint, so an unseeded PEP is never reused. `duet.pep.num_cpus` is in the fingerprint but does not change the counts: the seed is split per 100-codeword batch.
- **Speed and memory only, not results:** `duet.optimizer.num_cpus`, `evaluator.num_cpus`, `sivanandan.num_cpus`, `mem_budget_gb`, `device`. With pools of at most 11,895 sequences, the evaluation's batched matmul fits in one batch at `mem_budget_gb: 64.0`, a few GB (unlike the genome-wide runs; see `ops_crispick_genome_wide/README.md`, Disk and memory).
  - With an NLL metric on CPU, `num_cpus` 1 versus 2 or more can change PEP counts ([Numerical determinism](../README.md#numerical-determinism)). That does not apply here: Hamming costs are small integers, which floating-point sums give exactly in any order.
- **Changes numbers:** `num_samples`, the λ list, `trials`, `seed`, `evaluator.seed`, `duet.pep.seed`.
- **Feldman ED=2** depends on Python's string hash (`set.pop()` in the `ops` package). `feldman_runner.py` pins `PYTHONHASHSEED=0` for the child.

## Smoke test

`VARIANT=smoke bash experiments/ops_crispri_rounds_error_sweep/run.sh` runs three tracked smoke configs: `sr08_eps0.03.smoke.yaml`, `sr10_eps0.05.smoke.yaml` and `sr10_eps0.10.smoke.yaml`. `python make_configs.py --smoke` writes them from the real configs. Run the `ops_crispri_symmetric` smoke first, so the reference check has its reference.

- **Shrink.** The same shrink as `ops_crispri_symmetric/config.smoke.yaml`: trials 1, 50 groups, 20 controls, evaluator and PEP `num_samples` 200, seeds 42/43.
- **λ.** `[0.0, 0.9, 1.0]`, a superset of that smoke's `[0.0, 1.0]`. Outputs, cache and scratch go under `results/smoke/`, with the same device (`gpu:all`) and code path.
- **Why this shape.** `sr10_eps0.10` is the Fig 3b smoke config plus λ = 0.9, so the smoke exercises every part of the reference check against `results/smoke/ops_crispri_symmetric`:
  - Tier A exactly: pool, 6 baselines, and DUET λ = 0 and 1;
  - Tier B, because the extra arm changes the evaluation union, as the 13 extra real values do. λ = 0.9 sits below the 97.5% activity floor, so the arm matches the reference's and the comparison is like for like;
  - the ε-independent checks at `sr10_eps0.05`;
  - a second rounds column at `sr08_eps0.03`.
- **Result.** The three points and the visualizer take about 1 min on the reference machine (16–20 s per point). Tier A passes (2 `guides.csv`, 12 baseline row sets and 3 DUET row sets identical). Tier B is like for like (97.5% arm λ = 0.00 in both), with no cell above |z| = 4. The sweep builds its own `sr10_eps0.10` PEP, which equals the Fig 3b smoke PEP byte for byte. `SKIP_COMPLETE=1` then skips all three points and re-runs only the visualizer.
- **Warnings.** With three λ values every smoke pick overshoots the floor (`WARN` lines). A few Sivanandan ED=3 smoke cells run against their metric's direction and render diverging. Both are expected at this size.
