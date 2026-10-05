# Benchmark regression harness

Re-runs two medium-scale optimization experiments and checks that their numeric
outputs have not drifted from a frozen reference committed in this folder.
Figures are regenerated on every run for a person to look at, but they are not
part of pass or fail.

Fixtures:

- `ops_uniform`: OPS benchmark (`run_benchmark.py`) on the hCRISPRi-v2.1 library
  (`WeissmanCRISPRi`), symmetric noise, `unique_minimum` decoding, 3 trials,
  with the Feldman et al. and Sivanandan et al. baselines.
- `merfish_asymmetric`: MERFISH benchmark (`run_merfish.py`), 16 bits, an
  asymmetric channel written inline in the config (the Chen et al. 2015 error
  rates), warm start from `codebook_1`, and the optical-crowding objective. It
  reads `examples/data/processed/MERFISH/codebook_1.csv` and the expression
  table in `examples/data/processed/Kastriki_et_al_2022/`.

Both fixtures run on a GPU (`device: gpu:all`). `ops_uniform` should pass on any
GPU. `merfish_asymmetric` may drift on other GPU models.

Data: the frozen references in `*/reference/` are outputs of these runs, so they
contain material derived from the inputs: sgRNA sequences and activity scores of
hCRISPRi-v2.1 (Horlbeck et al. 2016, CC BY 4.0), the gene names and barcodes of the
Chen et al. (2015) codebook, and expression values of the Kastriti et al. (2022)
table (CC BY 4.0). Sources and licenses are in
[examples/data/README.md](../../../examples/data/README.md) and
[data/processed/README.md](../../../data/processed/README.md).

## Setup

From a clone of the repository:

```bash
pip install -e ".[benchmark,test]"
```

You also need CuPy for the GPU ([Choosing a CuPy package](../../../docs/installation.md#choosing-a-cupy-package)).
The Feldman et al. baseline of `ops_uniform` runs in a separate conda
environment named `ops`; create it with

```bash
conda env create -f environments/ops.yml
```

(see [environments/README.md](../../../environments/README.md)). The harness
calls bare `python`, so activate the environment that has `duet` installed
before you run it.

## Running

Run the driver from the repository root:

```bash
# Compare a fresh run with the committed reference (exit 0 = pass, 1 = drift, 2 = a run failed):
python scripts/benchmark/regression/run_regression.py

# Only one fixture:
python scripts/benchmark/regression/run_regression.py --only merfish_asymmetric

# Compare existing results again without re-running (fast iteration):
python scripts/benchmark/regression/run_regression.py --compare-only

# (Re)record the reference after an intended, reviewed change of behavior:
python scripts/benchmark/regression/run_regression.py --record

# Run both fixtures even if the first one fails to run:
python scripts/benchmark/regression/run_regression.py --keep-going
```

The driver runs each fixture's runner and visualizer with the fixture's folder
as the working directory, so the relative paths in the two YAML files resolve
against that folder. Everything a run writes goes under the repository's
`results/` folder:

| What | Where |
|---|---|
| Outputs, figures, `run.log`, `visualize.log`, `comparison_report.json` | `results/benchmark/regression/<fixture>/` |
| PEP matrix cache | `results/cache/regression/` |
| Scratch files (temporary memory-mapped arrays) | `results/scratch/regression/` |

A check of both fixtures needs about 2 GB under `results/`: 1.7 GB of PEP cache, which later
runs overwrite, and about 0.5 GB of scratch while a fixture runs.

The driver does not clear the output folder before a run. Move old outputs
aside before a `--check` (the default mode), for example
`mv results/benchmark/regression results/benchmark/regression.old`: the
comparison reads every gated file it finds there, so a file left over from an
older run (such as a `selected_codewords_lambda*.csv` for a λ the fixture no
longer uses) is reported as missing from the reference, a false FAIL.

## Reading the report

The driver prints PASS or FAIL per fixture. For each gated file it reports the
number of added and removed keys and the worst deviation per column, and it
writes a machine-readable `comparison_report.json` into the fixture's output
folder. Tolerances default to `rtol=1e-6` and `atol=1e-9` (`--rtol` and `--atol`
override them).

## Updating the reference

`--record` is intentional and reviewed: it re-runs the fixtures and freezes the
gated outputs as gzipped files under `<fixture>/reference/`, plus a
`manifest.json` (git SHA, time, tolerances). Run `--check` first to see what
changed, inspect the diff, then commit the updated `reference/`.

The `git_sha` values in the committed `manifest.json` files refer to the
private development history the references were recorded in. That history is
not part of the public repository.

## Determinism

`--check` compares a fresh run with the recorded one, so it can pass only if
the fixture's runs reproduce (within `rtol` and `atol`):

- Every fixture sets `seed` under `evaluator:` and under `duet.pep:`. The
  top-level `seed` reaches neither. Unseeded, the final evaluation and the PEP
  (rebuilt on every run because of `force_rebuild: true`) draw fresh OS entropy.
- Feldman et al. runs with `PYTHONHASHSEED=0`
  (`src/duet/benchmark/feldman_runner.py`): the OPS package picks ED=2 guides
  with `set.pop()` over gene-ID strings.
- `--compare-only` right after `--record` compares the recorded outputs with
  themselves. To confirm that a new or re-recorded fixture reproduces, run
  `--check` twice in a row.

## Comparator unit tests (no GPU)

The comparison code and the driver's helpers have CPU tests, which CI runs:

```bash
python -m pytest scripts/benchmark/regression/tests -q
```
