# merfish_2000_genes: Fig 4c, 4d, 4e (MERFISH, 2,000 random brain genes)

MERFISH codebook design for 2,000 random whole-brain genes (32 bits; HW4 and HW5 candidates). DUET warm-starts from the Bostrom HW5 panel and trades decoding accuracy against optical crowding over a 9-value λ sweep. There are two baselines: Bostrom HW5 (expression-ordered assignment) and MERFISH MHD4 HW5 (the same codewords, randomly assigned with seed 42). `make_figures.py` then draws Fig 4d,e from DUET at λ = 0.90.

## Run

```bash
conda activate duet        # the DUET environment (environment.yml) with the benchmark and gpu extras (run.sh checks it imports this clone's duet)
bash experiments/merfish_2000_genes/run.sh
```

- The steps are `scripts/benchmark/run_merfish.py`, then `scripts/benchmark/visualize_merfish.py`, then `make_figures.py`. All three run in the active env; no other env is needed. `make_panels.sh` (same env) rebuilds `panels/`. It is optional, because the files in `panels/` are already its output.
- It needs GPUs: `device: gpu:all` for the PEP build and the union evaluation. The DUET swap search runs on the CPU (9 workers).
- **Build the expression table first** (see Inputs). It is not in git; `run.sh` stops before any GPU work if it is missing.
- **Run this before `merfish_expression_prior`**, which reads this run's outputs (see Outputs).
- **Figures only,** from this folder: `python ../../scripts/benchmark/visualize_merfish.py --config config.yaml`, then `python make_figures.py --config config.yaml`. run.sh never passes `--debug-plots`; add it to either command for the debug plots (see Outputs).
- **Driver.** `run.sh` `cd`s into this folder (all paths resolve from here), reads the outdir and both scratch dirs from the config, and chains the runner, the visualizer and `make_figures.py`. It stops before running anything if the config (`config.yaml`, or `config.<VARIANT>.yaml`) is missing, if the active env does not import `duet` from this repo's `src/`, if the expression table is missing, or if one of those keys is missing or null. It creates both scratch dirs and sets `TMPDIR=<scratch>/tmp`. Logs go to `<outdir>/logs/` (`run.raw.log`, `run.log`, `visualize.log`, `make_figures.log`).

## Inputs

- **Whole-brain expression table** `examples/data/processed/WMB-10X/whole_brain_per_gene_cpm.csv` (30,599 rows, one per gene symbol, sha256 `a8314155…`). It is not in git: it holds mean CPM values of the Allen Brain Cell Atlas WMB-10X data (Yao et al. 2023, CC BY-NC 4.0). Build it in the scanpy environment (`environments/scanpy.yml`). The script reads `wmb_precomputed_stats.h5` (1.38 GB) and `wmb_gene.csv` from `data/raw/WMB-10X/` and downloads them there if they are absent:

  ```bash
  conda run -n scanpy_env python scripts/wmb10x/build_whole_brain_cpm.py
  ```

  `merfish_zhang2023_v2` and `merfish_expression_prior` read the same table.
- **Shipped:** the fitted Zhang 2023 channel `scripts/benchmark/noise_model_matrices/channels/merfish_zhang2023_channel.npy`, and `panels/`:
  - `k2000_genes.csv`, the 2,000 gene symbols. The list is copied, not redrawn. It is a seed-42 draw over the non-zero genes of an earlier version of the table, so drawing it again would give a different list.
  - `k2000_bostrom_hw5_panel.csv` and `k2000_merfish_mhd4_hw5_panel.csv`, with the columns `Gene` and `Sequence` only. `make_panels.sh` rebuilds them from the gene list, the expression table and the Bostrom HW5 codebook in `examples/data/bostrom/`. The Bostrom rows are sorted by expression, and the MHD4 panel keeps that row order with a random codeword assignment (seed 42). `make_panels.sh` prints the sha256 of each file; its header lists the expected values. `panels/README.md` gives the panels' sources and licences.
- `experiments/INPUTS.md` lists every input with its sha256.

## Cost

From the log of the paper's run: 2026-06-28 21:18:53 to 06-29 01:19:33, **4h00m40s**, on 4 GPUs. The log records only the GPU count; the run was most likely on the reference machine (4 NVIDIA TITAN X (Pascal) 12 GB GPUs, 40 logical CPUs, 220 GB RAM).

| Stage | Time |
|---|---|
| Setup (pool of 100,000 candidates) | 23 s |
| PEP, GPU pass (100,000 codewords × 30,000 samples, 12.71 it/s) | 2h11m10s |
| PEP, host transpose + symmetrize | 4m46s + 17m23s |
| DUET, 9 λ in parallel (workers finished between 1h06m and 1h24m) | 1h24m41s |
| Union evaluation / crowding at 100 trials + metrics | 1m26s / ~37 s |
| Visualizer / `make_figures.py` (CPU) | ~12 s / ~6 s |

- Disk: the cold PEP cache takes about 38 GB (the cache of the paper's run, measured with `du`). Symmetrization writes a transient copy on top of that, for an estimated peak of about 60 GB. Logged host RSS peaked at about 39 GB after the transpose, almost all file-backed (`sym_mem_budget_gb: 64`).
- The re-run simulates 1,000 crowding trials in-run instead of 100, so that stage takes longer than 37 s: a full run of the current config took 6h04m on the reference hardware, against 4h01m for the paper's run. The paper's run re-evaluated its results at 1,000 trials afterwards with `scripts/benchmark/recompute_crowding.py`; trial t is seeded `seed + t`, so the two routes are equivalent.
- The first run is a cold build. Later runs with the same config reuse `results/cache/merfish_2000_genes/`, and so does every panel of `merfish_expression_prior`.

## Outputs

All outputs are under `results/experiments/merfish_2000_genes/` (untracked).

| Panel | File |
|---|---|
| 4c left | `figures/crowding_pareto_front.svg` |
| 4c right (inset) | `figures/jointplot_sweep/decode_vs_identified_jointplot_lambda0.80.svg` |
| 4d | `figures/fig4de/round_expression_uniformity_y0_k2000.svg` |
| 4e | `figures/fig4de/expr_by_hw_violin_k2000.svg` |

- By default the visualizer draws only the λ = 0.80 file of the jointplot sweep, because `config.yaml` lists it under `visualization.paper_panels`; `--debug-plots` draws every λ. Use the sweep file for the inset. The canonical `figures/decode_vs_identified_jointplot.svg` (written only with `--debug-plots`) uses a matched rule; on the paper's run it happened to pick the inset λ, but a re-run need not. Only the visualizer reads the `visualization` block; the runner ignores it, and no cache fingerprint includes it.
- Other figures under `figures/`, each as SVG + PNG. By default, the diagnostics: `crowding_pareto_front_lambda_labeled`, the crowding bar chart, both decode-accuracy histograms, the Hamming-weight distribution, and the mean and 5th-percentile decode-accuracy bar plots. The metric bar plots and the crowding bar chart have horizontal bars, one row per method. `--debug-plots` adds the other six metric bar plots, the canonical jointplot, `identified_fraction_vs_count` and the three per-λ sweeps (`jointplot_sweep/`, `decode_hist_sweep/`, `decode_hist_by_hw_sweep/`).
- Every λ gets `selected_codewords_lambda{λ:.2f}.csv` and `optimization_history_lambda{λ:.2f}.csv`.
- Also written: `results.csv`, `metrics.csv` (labels such as `DUET (lambda=0.80)`), `summary.yaml` (key `true_accuracy_last_lambda`), `candidates.csv`, a verbatim `config.yaml` copy (its relative paths are not valid from the outdir) and `figures/fig4de/`.
- **`make_figures.py`** draws the Fig 4d,e panels (SVG + PNG) from DUET at λ = 0.90 (`DEFAULT_LAMBDA`; `--lambda` takes one float). It reads `results.csv`, `metrics.csv` and the expression table through this experiment's config. The default output is `<outdir>/figures/fig4de/`. By default it writes only the Fig 4d and 4e panels; `--debug-plots` adds the other three (the autoscaled and sorted load panels and the box-plot version of 4e). The Fig 4d legend reads `DUET (λ=0.90)`.
- **Downstream:** `merfish_expression_prior` (Supp S4b) uses this `config.yaml` as the template for one DUET run per gene panel and reuses this run's PEP cache. Its build step reads `panels/k2000_bostrom_hw5_panel.csv` for a capture sanity check.

## Compare after a re-run

Values of the paper's run (its `metrics.csv`, 1,000 crowding trials) unless noted.

| Quantity | Paper's run | Expect |
|---|---|---|
| Bostrom HW5 identified fraction | 0.78044 | changes, because the assignment was rebuilt |
| MHD4 HW5 identified fraction | 0.77555 | changes, same reason |
| Baseline decode accuracy (both baselines) | 0.92496 | expected to repeat (see below) |
| DUET λ = 0.90, decode / identified | 0.93371 / 0.83336 | new draw |
| DUET λ = 0.80, decode / identified | 0.93400 / 0.83389 | new draw |
| Identified fraction, Bostrom HW5 → DUET λ = 0.90, as quoted in the paper: 78.0% → 83.3%, 24% fewer transcripts lost, "slightly improving decoding" | λ = 0.90: 24.1% | check the claims, not the digits |
| Fig 4d CV, DUET / Bostrom / MHD4 | 0.018 / 0.036 / 0.183 | see below |
| Fig 4e medians, DUET λ = 0.90, computed from `results.csv` | HW4 median about 760x the HW5 median (373 vs 1,627 genes) | new codebook; the paper says "nearly three orders of magnitude" |

- **Why the numbers move: the expression table.** The paper's run used an earlier version of the whole-brain table with 30,622 rows and 22 duplicated symbols (sha256 `dd029b27…`). In that table, a duplicated symbol took the value of its last row. The build script writes the de-duplicated table (30,599 rows, one per symbol).
  - Three of the 2,000 genes are de-duplicated symbols, and their values change: Aldoa goes from near zero to the top 2% of the table (rank 456 of 30,599), and Gm41392 and Zc3h11a also change. Ten more genes change only in the last digits.
  - The new values change the crowding objective, so they change the optimization, the crowding simulation and the Fig 4d,e expression maps.
  - They also re-rank both baseline panels (mainly Aldoa): in each panel 1,831 of the 2,000 rows carry a different gene than in the paper's run. The `Sequence` column (the codeword set and its order) is unchanged; the gene→codeword map is not.
- **Fig 4d CVs will differ.** 0.018 / 0.036 / 0.183 reproduce only with the earlier table (0.0185 / 0.0359 / 0.1835). On the de-duplicated table, the paper's codebooks give 0.0282 / 0.0397 / 0.1795.
  - The baselines are not optimized, so their re-run CVs depend only on the rebuilt panels and the table. Computed for this README with `compute_round_expression_load`: Bostrom 0.0356, MHD4 0.1821 (legend 0.036 / 0.182). DUET's CV comes from a new optimization.
- **The paper's operating points.** Its 78.0% → 83.3% (24%) matches λ = 0.90, the Fig 4d,e operating point. The Fig 4c inset shows λ = 0.80, which gives 83.4% (24.3%).
- **Expect new DUET and crowding numbers.** The expression table and the baseline gene assignments changed, the per-λ seeds changed, and codewords that only DUET picks get new union rows.
- **What should repeat.** The candidate pool: at the paper's PEP seed and the current path string, its PEP fingerprint is the same `10b6f878d0d24910` whether built from the earlier or the rebuilt panels (rechecked for this README). With `duet.pep.seed: 43` the fingerprint and the PEP counts are new. The baseline decode accuracy (0.9249566): the baselines use the warm-start codewords, which fill the first union rows in the same order as in the paper's run, so they keep their Monte Carlo streams. A small test with this config (80 codewords, 300 samples) confirmed that a codebook evaluated first keeps its accuracy exactly when other codebooks are added. The full 0.9249566 was not recomputed; a CPU evaluation was too slow.

## Determinism

- **These change the numbers:**
  - The expression table (above), `num_samples` (eval 5,000, PEP 30,000) and the `candidates` block (`per_codeword_sample_size`, `sample_seed`, the HW5 cap).
  - The seeds: top-level `seed` and `evaluator.seed` (42), and `duet.pep.seed` (43; the paper's run used 42). The last two do not inherit the top-level seed; unset means unseeded. Both spawn one child stream per index (PEP batch *j*, ground-truth union row *j*), so with one seed the two Monte Carlo estimates would share streams. Distinct seeds make them independent, as in the OPS experiments.
  - The warm start and baselines, which anchor the pool and set the crowding normalizer.
  - The λ list and the baselines together set the union of all codebooks. Each codeword's evaluation Monte Carlo stream is indexed by its row in that union. Rows go in order of first appearance: warm start, then DUET by ascending λ, then baselines. Changing either moves the DUET-only codewords to new streams; the warm-start codewords keep rows 0-1999.
  - Code defaults for unset keys (for example `crowding.diffraction_model: abbe`, `neighborhood: box`).
  - Device. Keep `gpu:all`. CPU and GPU draw the same samples, but the asymmetric NLL arithmetic differs (float32 CPU BLAS vs cuBLAS for the PEP; float64 vs float32 for the evaluation), and the binary Zhang channel gives many mathematically tied costs, which float rounding decides. Two 250-codeword, K = 300 probes (2026-09-23) found 4,375 and 4,495 of 62,500 PEP cells and 219 and 228 of 250 codeword accuracies different, by up to 0.12 (see `experiments/README.md`). `tests/test_gpu_pep.py` checks CPU-GPU equality only on its synthetic channels. A CPU-only PEP build would also take roughly 25 h at 30 processes (estimate).
- **`crowding.n_trials`** changes only the reported crowding values. DUET optimizes an exact crowding objective. Trial t is seeded `seed + t`, so the first k trials do not depend on `n_trials`.
- **Speed only:**
  - `evaluator.num_cpus`, `duet.num_cpus`, `evaluator.mem_budget_gb` and `sym_mem_budget_gb`.
  - This holds for the GPU runs configured here. On CPU with NLL decoding metrics, `num_cpus` 1 versus 2 or more can change PEP counts; values of 2 or more agree ([Numerical determinism](../README.md#numerical-determinism)).
  - `duet.pep.num_cpus` changes no numbers, but it is part of the PEP cache fingerprint (as is the `channel_matrix_path` string). Changing either forces a cold rebuild.
- **The per-λ optimizer seed is seed + int(λ·1000), so re-runs are new draws, not relabelings.**
  - The λ list gets seeds 42, 142, 342, 542, 742, 842, 942, 992, 1042. The paper's run used the opposite λ convention (`docs/adr/0001-lambda-weights-decoding-accuracy.md`), so its seed at each operating point was different.
  - At `temperature: 0` the seed only breaks exact ties among the best swaps.
  - The mirrored weights are not bit-equal either: at λ = 0.90 the crowding weight is 0.09999999999999998, not 0.1.

## Smoke test

`VARIANT=smoke bash experiments/merfish_2000_genes/run.sh` reads the tracked `config.smoke.yaml` and writes under `results/smoke/`. It needs the expression table and GPUs.
- **What was shrunk:** all 2,000 genes kept; `per_codeword_sample_size: 3` (a pool of 3,931 candidates instead of 100,000); eval and PEP `num_samples: 50`; λ [0.8, 0.9] (the two λ values read downstream); `max_iter: 200`; `max_patience: 50`; crowding `n_trials: 2`; still `gpu:all`.
- **Expected:** it takes about 80 s. It writes `figures/jointplot_sweep/..._lambda0.80.svg` and `figures/fig4de/*_k2000.svg`, and it feeds the `merfish_expression_prior` smoke, whose anchor check then matches exactly (to 1e-9). Under `VARIANT=smoke bash experiments/run_all.sh` it took about 5 min (3m54s of it in DUET).
- **The λ list must contain 0.90.** Without a λ = 0.90 row, `make_figures.py` fails.
- **Figure tiers.** `config.smoke.yaml` must carry the `visualization` block of `config.yaml`. Without it, a smoke run writes no λ = 0.80 inset by default.
