# merfish_zhang2023_v2: Fig 4b (Zhang et al. 2023 1,147-gene MERFISH panel)

DUET warm-starts from the published Zhang v2 codebook (32-bit, MHD4, HW4) and selects from a mixed HW4/HW5 pool. It trades decode accuracy against optical crowding over a 9-value λ sweep, using whole-brain WMB-10X expression and the measured Zhang 2023 channel. The baselines are Bostrom HW4, Bostrom HW5 and the Zhang v2 codebook ("Zhang et al. codebook #2").

## Run

```bash
conda activate duet        # the DUET environment (environment.yml) with the benchmark and gpu extras (run.sh checks it imports this clone's duet)
bash experiments/merfish_zhang2023_v2/run.sh
```

- **Inputs not in git.** Build the Zhang v2 codebook and the whole-brain expression table first (see Inputs). The runner and the visualizer both need them. `run.sh` checks for both before any GPU work, and `experiments/run_all.sh` checks for both before starting.
- **Figures only,** from this folder: `python ../../scripts/benchmark/visualize_merfish.py --config config.yaml`. run.sh never passes `--debug-plots`; add it for the debug plots (see Outputs).
- **Hardware.** GPUs are used for the PEP build and the union evaluation (`gpu:all`). The DUET swap search runs on 9 CPU workers. No other env. `make_panels.sh` is optional; the committed `panels/` are its output. Works from any cwd.
- **Driver.** `run.sh` reads `config.yaml`, or `config.<VARIANT>.yaml` when `VARIANT` is set. It stops before running anything if that file is missing, if the active env does not import `duet` from this repo's `src/`, if the codebook or the expression table is missing, or if `outdir` or either `scratch_dir` is missing or null. It creates both scratch dirs, sets `TMPDIR` to `<scratch_dir>/tmp` and prints `nvidia-smi`. Logs go to `<outdir>/logs/{run.raw,run,visualize}.log`.

## Inputs

Two inputs are not in git. Build them once, from the repository root. `experiments/INPUTS.md` lists every input with its sha256.

- **Zhang v2 codebook** `data/processed/BIL_MERFISH/codebooks/zhang_2023_v2_processed.csv` (1,147 genes, 45,167 B, sha256 `3a6159f8ce07…`). It is the gene list, the warm start and the third baseline. The build script converts `codebook_32bit_v2.csv` of the Brain Image Library dataset of Zhang et al. (2023) (doi:10.35077/act-bag; 92,715 B, sha256 `7e45c01cab5b…`). With `--download` it first fetches that file to `data/raw/BIL_MERFISH/additional_files/` if it is absent:

  ```bash
  python scripts/data_processing/build_zhang2023_codebook.py --download
  ```

- **Whole-brain expression table** `examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv` (30,599 rows, one per gene symbol, sha256 `a8314155…`). It holds mean CPM values of the Allen Brain Cell Atlas WMB-10X data (Yao et al. 2023, CC BY-NC 4.0), so it is not shipped. Build it in the scanpy environment (`environments/scanpy.yml`). The script reads `wmb_precomputed_stats.h5` (1.38 GB) and `wmb_gene.csv` from `data/raw/WMB-10X/` and downloads them there if they are absent:

  ```bash
  conda run -n scanpy_env python scripts/wmb10x/build_whole_brain_cpm.py
  ```

  `merfish_2000_genes` and `merfish_expression_prior` read the same table.

Shipped inputs:
- the fitted Zhang 2023 channel `scripts/benchmark/noise_model_matrices/channels/merfish_zhang2023_channel.npy` (a 2x2 binary channel, P(1->0) = 0.0561, P(0->1) = 0.0145);
- the two Bostrom baseline panels `panels/zhang_v2_bostrom_hw{4,5}_panel.csv`, with the columns `Gene` and `Sequence` only. `make_panels.sh` rebuilds them from the two inputs above and the Bostrom codebooks in `examples/data/bostrom/`: it sorts the Zhang genes by whole-brain expression and pairs them with the Bostrom codewords in that order. `make_panels.sh` prints the sha256 of each panel; its header lists the expected values. The runner reads only `Gene` and `Sequence` from a panel; crowding takes CPM from the expression table. `panels/README.md` gives the panels' sources and licences.

The channel and the gene lists of the panels are adapted from the Brain Image Library dataset and are under CC BY-SA 4.0. This work used data from the Brain Image Library (RRID:SCR_017272), which is supported by the National Institutes of Mental Health of the National Institutes of Health under award number R24-MH-114793.

## Cost

From the log of the paper's run (2026-06-29). The log shows 4 GPUs but not their model; the run was most likely on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs, 220 GB RAM).

| stage | time |
|---|---|
| setup | 17 s |
| PEP GPU pass (99,999 codewords at 13.17 it/s) | 2h06m37s |
| host transpose + symmetrize | 19m21s |
| DUET 9-λ sweep (9 CPU workers) | 28m43s |
| union evaluation; crowding + metrics (100 trials in-run) | 50 s; about 34 s |
| **total**, plus visualizer (not logged) | **2h56m27s** + about 12 s |

- **Disk and memory.** The cold PEP cache was 38 GB (du). Budget about 40 GB under `results/cache/merfish_zhang2023_v2/`, peaking at about 60 GB during symmetrization (a 20 GB `.T.tmp` transpose file, deleted afterwards); file-backed RSS peaked around 39 GB.
- **The first run is a cold build.** Later runs with the same config reuse `results/cache/merfish_zhang2023_v2/`.
- **Crowding trials.** The paper's run simulated 100 crowding trials in-run, and its results were re-evaluated at 1,000 trials afterwards with `scripts/benchmark/recompute_crowding.py`. The config runs all 1,000 in-run: a full run of the current config took 3h19m on the reference hardware, against 2h56m for the paper's run. Trial t is seeded `seed + t`, so the two routes are equivalent.

## Outputs

Under `results/experiments/merfish_zhang2023_v2/`. Each figure is written as both `.svg` and `.png`.

| file | use |
|---|---|
| `figures/crowding_pareto_front.svg` | **Fig 4b left** |
| `figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg` | **Fig 4b right (inset)**: DUET λ = 0.80 vs Zhang #2 |
| `figures/decode_vs_identified_jointplot.svg` | **Not the panel.** Its matched rule picked λ = 0.50 on the paper's run. Written only with `--debug-plots` |
| `metrics.csv`, `results.csv` | text numbers; per-codeword decode and identified fraction |
| `selected_codewords_lambda{λ:.2f}.csv`, `optimization_history_lambda{λ:.2f}.csv` | one pair per λ |
| `summary.yaml`, `config.yaml`, `candidates.csv` | summary (key `true_accuracy_last_lambda`); verbatim config copy (its relative paths do not work from `outdir`); pool |
| `logs/` | `run.raw.log`, `run.log`, `visualize.log` |

The visualizer writes the inset by default because `config.yaml` lists it under `visualization.paper_panels`. Only `visualize_merfish.py` reads that block; the runner ignores it, and no cache fingerprint includes it.

Other figures under `figures/`:
- **By default**, the diagnostics: `crowding_pareto_front_lambda_labeled`, the crowding bar chart, both decode-accuracy histograms, the Hamming-weight distribution, and the mean and 5th-percentile decode-accuracy bar plots. The metric bar plots and the crowding bar chart have horizontal bars, one row per method. Of the jointplot sweep, only the λ = 0.80 inset is written.
- **With `--debug-plots`**, also the other six metric bar plots, the canonical jointplot, `identified_fraction_vs_count` and the three per-λ sweeps (`jointplot_sweep/`, `decode_hist_sweep/`, `decode_hist_by_hw_sweep/`).

## Compare after a re-run

Values of the paper's run (its `metrics.csv`, 1,000 crowding trials). IF is the identified fraction over 1000 simulated cells. Error cut and ΔIF are relative to Zhang #2.

| method | decode | error | error cut | IF | ΔIF (pp) |
|---|---|---|---|---|---|
| Zhang et al. codebook #2 | 0.92072 | 7.928% | – | 0.90588 | – |
| Bostrom HW4 / HW5 | 0.92014 / 0.94195 | 7.986 / 5.805% | – | 0.89031 / 0.87320 | −1.56 / −3.27 |
| DUET λ = 1.00 | 0.95433 | 4.567% | 42.4% | 0.87656 | −2.93 |
| DUET λ = 0.95 / 0.90 | 0.95411 / 0.95423 | 4.589 / 4.577% | 42.1 / 42.3% | 0.90164 / 0.90229 | −0.42 / −0.36 |
| **DUET λ = 0.80 (inset)** | **0.95402** | **4.598%** | **42.0%** | **0.90401** | **−0.19** |
| DUET λ = 0.70 | 0.95340 | 4.660% | 41.2% | 0.90465 | −0.12 |
| DUET λ = 0.50 | 0.95252 | 4.748% | 40.1% | 0.90594 | +0.01 |
| DUET λ = 0.30 / 0.10 / 0.00 | 0.95020 / 0.94381 / 0.90875 | 4.98 / 5.62 / 9.12% | 37.2 / 29.1 / −15.1% | 0.90700 / 0.90916 / 0.90928 | +0.11 / +0.33 / +0.34 |

Standard errors in the same file: decode 0.00016-0.00177, IF 0.00079-0.00094.

Expectations (not bit-identical overall):
- The pool is unchanged. The PEP is rebuilt cold with `duet.pep.seed: 43` (the paper's run used 42), so its counts are a new draw, and DUET's codebooks can differ from the paper's.
- **Zhang #2 should reproduce.** As the warm start it occupies the first 1,147 union rows, and in the paper's run `summary.yaml` `initial_accuracy` equals its `metrics.csv` decode to 15 digits. Its IF should also match closely. This is an expectation, not verified.
- **DUET rows are new draws.** The per-λ seeds differ from the paper's run, which wrote λ in the opposite convention (`docs/adr/0001-lambda-weights-decoding-accuracy.md`). Its crowding weight 1 − λ is not always bit-equal to the current one (λ = 0.90 gives 0.09999999999999998, not 0.1), and its union listed the DUET codebooks in the reverse order. So DUET decode accuracies are new Monte Carlo draws even if a codebook comes out the same; only codewords shared with the warm start keep their union rows. Bostrom rows come after the DUET rows and can shift too.
- **What to check:** the front still dominates; DUET at λ = 0.80 and 0.70 stays near 4.6-4.7% error (vs 7.9%), with IF within about 0.2 pp of Zhang.
- **The canonical jointplot's pick is fragile:** λ = 0.50 cleared Zhang's IF by only 0.00006.

## Determinism

- **Change the numbers:** top-level `seed` (per-λ seeds, the pool subsample fallback, crowding with per-trial seed = seed + trial); `evaluator.seed` (42) and `duet.pep.seed` (43; the paper's run used 42) (explicit, not inherited, and the run is unseeded if they are unset); `num_samples` (5000 eval, 30000 PEP); `per_codeword_sample_size`, `sample_seed` and `hamming_weight_limits`; the warm start and baselines, which are pool anchors; the expression table; and `device` for the PEP and the evaluation. CPU and GPU draw the same samples, but asymmetric NLL ties on the Zhang channel resolve differently (float32 CPU BLAS vs cuBLAS for the PEP; float64 vs float32 for the evaluation). Two 250-codeword, K = 300 probes (2026-09-23) found 4,375 and 4,495 of 62,500 PEP cells and 219 and 228 of 250 codeword accuracies different, by up to 0.12 (see `experiments/README.md`). Keep `gpu:all`.
- **Two seeds.** `duet.pep.seed` and `evaluator.seed` each spawn one child stream per index (PEP batch *j*, ground-truth union row *j*). With one seed, the two Monte Carlo estimates would share streams; distinct seeds make them independent, as in the OPS experiments.
- **`crowding.n_trials`** changes only the reported crowding values, not DUET. The first k trials are the same at any `n_trials`.
- **Union evaluation.** Eval streams are `SeedSequence(seed).spawn(N_union)`, indexed by union row. The union is in first-appearance order over [warm start, DUET λ ascending, baselines]. A changed λ list or codebook therefore shifts the stream of every codeword whose row moves.
- **Speed only:** `evaluator.num_cpus`, `duet.num_cpus`, `sym_mem_budget_gb` and `evaluator.mem_budget_gb`.
  - This holds for the GPU runs configured here. On CPU with NLL decoding metrics, `num_cpus` 1 versus 2 or more can change PEP counts; values of 2 or more agree ([Numerical determinism](../README.md#numerical-determinism)).
  - `duet.pep.num_cpus` does not change the counts. But it is in the PEP fingerprint, as is the `channel_matrix_path` string, so changing either forces a cold rebuild.
- **Per-λ seed.** The per-λ optimizer seed is `seed + int(λ·1000)`, so re-runs are new draws, not relabelings.
  - The seeds are 42, 142, …, 1042; λ = 0.80 uses 842. The paper's run used the opposite λ convention, so it used 242 at that operating point.
  - At temperature 0 the seed only breaks exact ties among maximal swap deltas. No such tie changed an outcome in `synthetic_hvr`; that has not been checked here.

## Smoke test

`VARIANT=smoke bash experiments/merfish_zhang2023_v2/run.sh` reads the tracked `config.smoke.yaml`. Outputs go under `results/smoke/`. It needs the same two inputs and GPUs. It shrinks the run to:
- all 1,147 genes, `per_codeword_sample_size` 3 (2,384 candidates), eval/PEP `num_samples` 50;
- λ [0.8, 1.0], `max_iter` 200, `max_patience` 50;
- crowding `n_trials` 100 (at 2 trials, no codeword reaches the jointplot's 25-transcript floor);
- `gpu:all`.

Expected: the runner takes about 70 s, about 100 s with figures, and writes `figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg`. Two warnings are expected: 11/1147 genes are absent from the table and set to 0.0, and "Mean of empty slice".

`config.smoke.yaml` must carry the `visualization` block of `config.yaml`. Without it, a smoke run writes no λ = 0.80 inset by default.
