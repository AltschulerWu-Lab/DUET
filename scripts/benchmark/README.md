# Benchmark scripts

The scripts in this folder run the benchmarks behind the paper's figures and
draw their panels. Each experiment in [`experiments/`](../../experiments/) calls
them from its `run.sh`, with the configs kept in that experiment's folder.
[Reproducing the paper](../../docs/reproducing_the_paper.md) says which
experiment makes which result. To reproduce a result, run that experiment's
`run.sh`. This page describes the scripts themselves, for running them on your
own configs.

## Setup

Install the package with the `benchmark` extra from a clone of the repository:

```bash
pip install -e ".[benchmark]"
```

Configs with `device: "gpu:all"` need CuPy
([Installation](../../docs/installation.md)). The Feldman et al. baseline of
the OPS benchmark runs in a separate conda environment, named by
`feldman.conda_env` in the config. The experiments use one named `ops`, created
with `conda env create -f environments/ops.yml`
([environments/README.md](../../environments/README.md)).

## The scripts

| Script | What it does | Used by |
|---|---|---|
| `run_benchmark.py` | OPS benchmark. For each trial it samples a candidate pool, runs DUET over a λ sweep and the Feldman et al. and Sivanandan et al. baselines, and scores every selected library with one ground-truth evaluator. Writes `results.csv`, `trial_XX/guides.csv` (the pool of each trial) and a copy of the config. | `ops_crispri_symmetric`, `ops_crispri_cross_eval`, `ops_crispick_genome_wide`, `ops_crispri_rounds_error_sweep`; the `ops_uniform` regression fixture |
| `visualize_benchmark.py` | OPS figures from `results.csv`: Pareto fronts, hypervolume, guide-level comparisons. | `ops_crispri_symmetric`, `ops_crispri_cross_eval`, `ops_crispick_genome_wide`; imported by `ops_crispri_rounds_error_sweep/visualize_sweep.py` |
| `compare_benchmarks.py` | Cross-evaluation. Re-scores the selections of several `run_benchmark.py` runs with one evaluator, on the pools of `reference_dir`. Writes `reevaluated_results.csv` and `aggregated_metrics.csv`. | `ops_crispri_cross_eval`, `ops_crispick_genome_wide` |
| `visualize_comparison.py` | Hypervolume and Pareto plots of a `compare_benchmarks.py` run, from its `aggregated_metrics.csv`. | `ops_crispri_cross_eval`, `ops_crispick_genome_wide` |
| `comparison.py` | Library of the two scripts above: loading, merging and re-evaluating results. | `compare_benchmarks.py`, `visualize_comparison.py` |
| `run_merfish.py` | MERFISH benchmark. Builds the candidate codewords, runs DUET over a λ sweep (decoding accuracy against optical crowding), evaluates DUET and the baseline codebooks, and simulates optical crowding. Writes `results.csv`, `metrics.csv`, `summary.yaml`, `candidates.csv`, `selected_codewords_lambda*.csv` and a copy of the config. | `merfish_2000_genes`, `merfish_zhang2023_v2`, `merfish_expression_prior`; the `merfish_asymmetric` regression fixture |
| `visualize_merfish.py` | MERFISH figures from `results.csv` and `metrics.csv`. | `merfish_2000_genes`, `merfish_zhang2023_v2`; the `merfish_asymmetric` fixture |
| `make_bostrom_panel.py` | Turns a Boström et al. codebook into a `{Gene, Sequence}` panel CSV, used as a baseline and as a warm start. | `make_panels.sh` of `merfish_2000_genes` and `merfish_zhang2023_v2`; `make_panels.py` of `merfish_expression_prior` |
| `recompute_crowding.py` | Re-runs only the optical-crowding simulation of an existing MERFISH result, for example at more trials. | `merfish_2000_genes` and `merfish_zhang2023_v2` (see their READMEs) |

The synthetic benchmarks are in `synthetic/`:

| Script | What it does | Used by |
|---|---|---|
| `run_2d_synthetic_benchmark.py` | Synthetic pools with uniform random scores. Runs DUET and two greedy baselines (NLL and Hamming distance) over a λ sweep, enumerates the exact Pareto fronts of small pools, and computes normalized hypervolume. | `synthetic_hvr` |
| `visualize_2d_synthetic_benchmark.py` | Its figures: recovered fronts and hypervolume box plots. | `synthetic_hvr` |
| `visualize_duet_vs_baseline_objectives.py` | Per-trial scatter of each objective against decoding accuracy. | `synthetic_objective_correlation` |
| `visualize_baseline_summary.py` | Spearman correlation and regret of each objective across trials. | `synthetic_objective_correlation` |
| `run_duet_vs_baseline_objectives.py` | Enumerates every codebook of small synthetic pools and scores each objective (reject decoding). The experiment runs `run_rtb.py`, its random-tie-break sibling in `experiments/synthetic_objective_correlation/`. | `tests/regression/test_baseline_objectives_refactor_parity.py` |
| `run_duet_vs_baseline_objectives_sampled.py` | The same with sampled codebooks, for pools too large to enumerate. | `tests/test_baseline_objectives_module.py` |
| `run_objective_gap.py`, `visualize_objective_gap.py` | Decoding accuracy against DUET's union-bound objective for every codebook of a small synthetic pool. | `tests/test_objective_gap.py` |

Two more folders:

- `regression/`: the GPU regression harness ([README](regression/README.md)).
- `noise_model_matrices/channels/`: the fitted noise-channel matrices that the
  experiments read ([README.txt](noise_model_matrices/channels/README.txt),
  [Noise channels](../../docs/noise_channels.md)).

## Running

From the repository root:

```bash
python scripts/benchmark/run_benchmark.py --config ops.yaml -v
python scripts/benchmark/visualize_benchmark.py --config ops.yaml
python scripts/benchmark/compare_benchmarks.py --config cross_eval.yaml
python scripts/benchmark/visualize_comparison.py --config cross_eval.yaml
python scripts/benchmark/run_merfish.py --config merfish.yaml -v
python scripts/benchmark/visualize_merfish.py --config merfish.yaml
PYTHONPATH=. python -m scripts.benchmark.synthetic.run_2d_synthetic_benchmark --config synthetic.yaml -v
PYTHONPATH=. python -m scripts.benchmark.synthetic.visualize_2d_synthetic_benchmark --indir <outdir>
```

- Run the scripts of this folder as `python <path>`. `compare_benchmarks.py`
  and `visualize_comparison.py` import `comparison` as a plain module, so
  `python -m` fails for them.
- The `synthetic/` scripts are modules of the package
  `scripts.benchmark.synthetic`. Run them with `python -m` and the repository
  root on `PYTHONPATH`.
- `-v` logs at INFO level, `-vv` at DEBUG level.
- Paths in a config resolve against the working directory. In a MERFISH config,
  `outdir`, `cache_dir` and the input tables (`expression`, `genes`,
  `initialization`, `baselines`) resolve against the config's folder instead.
  The experiment configs are written for their own folder, which their
  `run.sh` changes into first.

## Figures and `--debug-plots`

The visualizers sort their figures into three tiers
(`src/duet/plotting/tiers.py`):

1. **Paper panels**: the files an experiment README lists as outputs. Always
   written.
2. **Diagnostics**: the trial-aggregated figures needed to choose λ. Written by
   default.
3. **Debug plots**: per-trial copies, per-λ sweeps and the rest. Written only
   with `--debug-plots`.

| Script | Written by default | Added by `--debug-plots` |
|---|---|---|
| `visualize_benchmark.py` | aggregated Pareto fronts, hypervolume bar plot | every per-trial figure, `hypervolume_visualization/` |
| `visualize_comparison.py` | hypervolume bar chart, aggregated Pareto fronts | per-trial hypervolume and Pareto plots |
| `visualize_merfish.py` | crowding fronts and bar chart, decode-accuracy histograms, Hamming-weight distribution, mean and 5th-percentile bar plots | the other metric bar plots, the matched-λ jointplot, `identified_fraction_vs_count`, the three per-λ sweeps (`figures/*_sweep/`) |
| `synthetic/visualize_2d_synthetic_benchmark.py` | aggregate HV, HVR and IGD, both recovered-front figures | `raw_scatter_per_trial/`, `pareto_per_trial/` |
| `synthetic/visualize_duet_vs_baseline_objectives.py` | `trial_01.svg` (the paper panel) | the other trials |
| `synthetic/visualize_baseline_summary.py` | both figures | nothing (the flag is accepted for a uniform command line) |

- `visualize_benchmark.py`, `visualize_comparison.py` and `visualize_merfish.py`
  read a `visualization.paper_panels` list from the `--config` YAML. It names
  debug-tier files that are paper panels (one trial or one λ), as paths
  relative to the output folder without the extension, and the visualizer
  writes them by default too. A listed file that was not written gives a
  `WARNING` at the end of the run. The runners ignore the block.
- Each run prints how many debug plots it skipped. A default run creates no
  empty folders.
- The visualizers write in place and delete nothing, so re-rendering into a
  folder that holds older debug plots leaves them there.

## Config files

The worked examples are the YAML files of the experiments, for example
`experiments/ops_crispri_symmetric/config.yaml` (OPS),
`experiments/ops_crispri_cross_eval/*.yaml` (four OPS runs and their
cross-evaluation) and `experiments/merfish_2000_genes/config.yaml` (MERFISH).
The loaders reject most key names of older configs with an error at load time.

### Evaluator blocks

`evaluator:` (the ground truth that scores every library) and `duet.pep:` (the
error model that DUET optimizes against) take the same keys:

```yaml
noise_channel:
  type: symmetric
  epsilon: 0.1
decoding_metric:
  type: hamming
decoding_rule:
  type: unique_minimum
num_samples: 2000   # simulated reads per codeword
num_cpus: 8
seed: 42            # set it here: the top-level seed does not reach this block
scratch_dir: ../../results/scratch/my_run   # optional
```

| Key | Types (parameters) |
|---|---|
| `noise_channel.type` | `symmetric` (`epsilon`), `position_varying` (`epsilon` or `epsilon_path`: one rate per round), `asymmetric` (`channel_matrix` or `channel_matrix_path`, or a per-symbol `epsilon`), `position_varying_asymmetric` (`channel_matrices` or `channel_matrices_path`: one matrix per round) |
| `decoding_metric.type` | `hamming`, `weighted_hamming` (`weights`), `symmetric_nll` (`epsilon`), `position_varying_nll`, `asymmetric_nll`, `position_varying_asymmetric_nll` (each takes the parameters of the matching channel) |
| `decoding_rule.type` | `unique_minimum`, `margin` (`margin`), `pairwise_posterior_threshold` (`threshold`), `approximate_posterior_threshold` (`threshold`, `n_eff`) |

Channel matrices have the transmitted symbol in rows and the observed symbol in
columns. DNA matrices are in A, T, C, G order and binary (MERFISH) matrices in
0, 1 order ([Noise channels](../../docs/noise_channels.md)). `evaluator:` also
takes `device` and `mem_budget_gb`; `duet.pep:` does not (use `duet.device` and
`duet.sym_mem_budget_gb`).

### OPS (`run_benchmark.py`)

```yaml
outdir: ../../results/experiments/my_run
cache_dir: ../../results/cache/my_run       # PEP matrix cache (optional)
scratch_dir: ../../results/scratch/my_run   # default for both scratch_dir keys (optional)
trials: 5
seed: 42
device: "gpu:all"                           # default "cpu"

candidate_pool:
  source: WeissmanCRISPRi
  seq_rounds: 10
  quota: 2               # guides to select per gene
  num_controls: 200
  num_groups: 1000       # genes sampled per trial
  min_rank: 10

evaluator:               # an evaluator block (above)
  ...

duet:
  use_mmap: true
  force_rebuild: false
  initialization:
    type: random         # or best_score; or sivanandan / feldman with edit_distance
  optimizer:
    lambda: [0.0, 0.1, 0.5, 1.0]
    temperature: 0.0
    max_iter: 100000
    max_patience: 3000
    avoid_duplicates: true
    duplicate_penalty: 100.0
    num_cpus: 8
  pep:                   # required: an evaluator block (above)
    ...

sivanandan:
  edit_distances: [1, 2, 3]
  num_cpus: 8
feldman:
  edit_distances: [1, 2]
  conda_env: ops
```

- `lambda` weights decoding accuracy:
  J = λ · decoding accuracy + (1 - λ) · secondary objective, so λ = 1 optimizes
  decoding alone and λ = 0 the secondary objective (activity score for OPS,
  optical crowding for MERFISH) alone
  ([ADR 0001](../../docs/adr/0001-lambda-weights-decoding-accuracy.md)).
- `candidate_pool.source` is `WeissmanCRISPRi` (hCRISPRi-v2.1, read from
  `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv`), `CRISPick` (build its table
  first with `scripts/data_processing/build_crispick_candidates.py`; it also
  takes `csv_path`, `control_quotas` and `score_method`), `synthetic` (with
  `candidates_per_group` and `alphabet_size`), or the path of your own CSV.
  `create_pool_from_source` in `src/duet/candidate_pool_factory.py` lists every
  source.
- Leave out `sivanandan:`, `feldman:` or `duet:` to skip that method.

### MERFISH (`run_merfish.py`)

```yaml
seed: 42
outdir: ../../results/experiments/my_merfish
cache_dir: ../../results/cache/my_merfish

candidates:
  seq_rounds: 16           # bits per codeword
  codebook_size: 140       # genes in the panel
  hamming_weights: [3, 4, 5]

evaluator:                 # an evaluator block (above), with a binary channel
  ...

expression:                # mean expression per gene, for optical crowding
  path: expression.csv
  gene_col: gene_name
  expression_col: mean_raw_counts

genes:                     # the panel's genes
  type: random             # or from_csv, with path and gene_col
  pool_filter: top_percentile
  pool_filter_value: 1

crowding:
  n_trials: 5
  total_reads: 100000
  cell_size_um: 100.0
  wavelength_nm: 500.0
  numerical_aperture: 1.4
  expansion_factor: 1.0

initialization:
  type: warm_start         # or random
  path: panel.csv
  sequence_col: Sequence

duet:                      # the optimizer keys sit directly under duet:
  lambda: [0.0, 0.5, 1.0]
  max_iter: 100000
  max_patience: 500
  num_cpus: 9
  pep:                     # required: an evaluator block (above)
    ...

baselines:
  - name: my baseline
    path: baseline_panel.csv
    gene_col: Gene
    sequence_col: Sequence
```

Without `expression`, `genes` and `crowding`, DUET optimizes decoding accuracy
only.

### Cross-evaluation (`compare_benchmarks.py`, `visualize_comparison.py`)

```yaml
outdir: ../../results/experiments/my_cross_eval
cache_dir: ../../results/cache/my_cross_eval
seed: 42                  # the seed of the runs being compared
device: "gpu:all"
eval_mem_budget_gb: 64.0
reference_dir: ../../results/experiments/run_a   # its trial_XX/guides.csv give the pools

evaluator:                # the ground truth for every method: an evaluator block (above)
  ...

methods:                  # one Pareto front per label
  - path: ../../results/experiments/run_a/results.csv
    regex: "DUET.*"
    label: "DUET (model A)"
  - path: ../../results/experiments/run_b/results.csv
    regex: "DUET.*"
    label: "DUET (model B)"
  - path: ../../results/experiments/run_a/results.csv
    regex:
      - "Feldman et al.*"
      - "Maximum activity"
    label: "Feldman et al."
```

- The runs being compared must share the top-level `seed`, so that their trials
  sample the same pools.
- All methods that the regexes of one `label` match form that label's Pareto
  front. List `Maximum activity` with a baseline to draw its full trade-off
  curve. `"DUET.*"` matches every `DUET (lambda=...)` row.

### Device, memory and scratch

| Key | Value used |
|---|---|
| `duet.device` | `duet.device`, else the top-level `device`, else `"cpu"` |
| `duet.sym_mem_budget_gb` | `duet.sym_mem_budget_gb`, else the top-level `mem_budget_gb`, else 32 |
| `evaluator.device` | `evaluator.device`, else the top-level `device`, else `"cpu"` |
| `evaluator.mem_budget_gb` | `evaluator.mem_budget_gb`, else the top-level `mem_budget_gb`, else 20 |
| `duet.pep.scratch_dir` | `duet.pep.scratch_dir`, else the top-level `scratch_dir` |
| `evaluator.scratch_dir` | `evaluator.scratch_dir`, else the top-level `scratch_dir` |

`duet.sym_mem_budget_gb` bounds only the symmetrization of the PEP matrix, not
its Monte Carlo build. `duet.pep` is required whenever a `duet:` block is
present.

### Synthetic configs

`experiments/synthetic_hvr/config.yaml` and
`experiments/synthetic_objective_correlation/config.yaml` document their keys
in their comments.
