# Auxiliary environments for reproducing the paper

The `duet-codebook` package itself needs only the environment in
`../environment.yml` (or `pip install -e ".[benchmark]"` from a clone); the NIS-seq
error analysis also needs the `notebooks` extra (`pip install -e ".[benchmark,notebooks]"`).
Two parts of the paper's figure reproduction run in separate conda environments:

| File | Env name | Used for |
|---|---|---|
| `ops.yml` | `ops` | The Feldman et al. baseline in the OPS benchmarks. `src/duet/benchmark/feldman_runner.py` runs `conda run -n ops python src/duet/benchmark/external_ops/ops_wrapper.py`. |
| `scanpy.yml` | `scanpy_env` | The WMB-10X atlas scripts: `scripts/wmb10x/` (including `build_whole_brain_cpm.py`, which builds the whole-brain expression table the MERFISH experiments read), `experiments/wmb10x_landscape/` and the build step of `experiments/merfish_expression_prior/`. |

Create them with `conda env create -f environments/ops.yml` and
`conda env create -f environments/scanpy.yml`. Both were exported on
2026-09-24 from the lab's environments with `conda env export --from-history`
plus their pip packages.

## OpticalPooledScreens in `ops`

- Source: <https://github.com/feldman4/OpticalPooledScreens>, commit
  `5417bf437290e263c72a7d6875afe26e8921858c` (2024-01-28, branch `master`).
- In the lab's `ops` env it is a git checkout of that commit installed with
  `python setup.py develop` (distribution `ops-lasagna 0.1`, package `ops`).
  `ops.yml` installs the same commit with pip instead.
- The checkout has local edits to `ops/pool_design.py` and `ops/constants.py`,
  the two modules the wrapper imports. They are whitespace-only
  (`git diff -w` is empty), so the pinned upstream commit gives the same code.
- The Feldman ED=2 selection depends on Python's string hashing;
  `feldman_runner.py` runs the wrapper with `PYTHONHASHSEED=0`.
