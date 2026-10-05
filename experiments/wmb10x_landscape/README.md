# wmb10x_landscape: WMB-10X cell-type landscape (Supp S4a)

PCA and cosine UMAP of the 5,322 Allen whole-mouse-brain 10x (WMB-10X) cluster centroids, colored by the 34 classes. This experiment has no DUET run and no λ. `scripts/wmb10x/README.md` documents the pipeline in detail.

## Run

```bash
bash experiments/wmb10x_landscape/run.sh                      # steps 1-4
WMB10X_STEPS="3 4" bash experiments/wmb10x_landscape/run.sh   # any subset of steps
```

- The driver works from any cwd, and no env needs to be active. Every step runs in `scanpy_env` (`environments/scanpy.yml`). The driver finds that env's interpreter once with `conda run` and then calls it by path, because `scripts/wmb10x/README.md` warns against running the steps through `conda run`. Unless `SCANPY_PYTHON` is set, `conda` must be on `PATH` or `CONDA_EXE` must be set. Under `experiments/run_all.sh`, this experiment does not use the active DUET env.
- `duet` is not installed in `scanpy_env`. The driver sets `PYTHONPATH=src` so that step 4 imports `duet.plotting` from this checkout.
- `scanpy_env` versions: Python 3.11, numpy 2.3.5, pandas 2.3.3, h5py 3.15.1, scikit-learn 1.8.0, scipy 1.16.3, umap-learn 0.5.11, numba 0.63.1, matplotlib 3.10.8. The paper's SVGs report Matplotlib v3.10.8.
- The driver runs the four scripts unmodified and passes neither `--force` (step 1) nor `--no-cache` (step 3).

| Override | Default | Effect |
|---|---|---|
| `SCANPY_ENV` | `scanpy_env` | Conda env to run in. It needs umap-learn. |
| `SCANPY_PYTHON` | that env's `python` | Interpreter path (a relative one is resolved against your cwd). Setting it skips the lookup. |
| `WMB10X_STEPS` | `1 2 3 4` (`1 2` if `VARIANT=smoke`) | Which steps run. |
| `VARIANT` | unset | Moves the driver's logs, scratch and numba cache to `results/$VARIANT/...`. Pipeline outputs stay at their fixed paths. |

| Step | Script (`scripts/wmb10x/`) | Writes (fixed, gitignored paths) |
|---|---|---|
| 1 | `download_wmb_metadata.py` | `data/raw/WMB-10X/`: 5 metadata files, ~16 MB. Files already present are skipped. |
| 2 | `build_centroid_matrix.py` | `data/processed/WMB-10X/{centroids.npy (687 MB), genes.csv, marker_genes.txt, cluster_metadata.csv}` |
| 3 | `embed_centroids.py` | `results/wmb10x/*.csv`, `embedding_summary.json`, and the SVD cache in `results/wmb10x/cache/` |
| 4 | `plot_embeddings.py` | `results/wmb10x/figures/*.svg` (6 files) and `results/wmb10x/captions.md` |

- **Output paths.** The scripts fix the output locations themselves (`ROOT = Path(__file__).resolve().parents[2]`): `data/raw/WMB-10X/`, `data/processed/WMB-10X/` and `results/wmb10x/`. These are gitignored and not under `results/experiments/`. A re-run overwrites `results/wmb10x/` in place.
- **Logs.** Each step runs under `script(1)`: `results/experiments/wmb10x_landscape/logs/<i>-<step>.raw.log` (verbatim) and `<i>-<step>.log` (cleaned).
- **Environment variables.** `PYTHONPATH=src`. `MPLCONFIGDIR` and `TMPDIR` point to `results/scratch/wmb10x_landscape/{mpl,tmp}` and `NUMBA_CACHE_DIR` to `results/cache/wmb10x_landscape/numba`. `PYTHONDONTWRITEBYTECODE=1` stops `scripts/wmb10x/__pycache__/` from being written; `PYTHONUNBUFFERED=1` keeps the logs in order. Step 4's PNG review copies therefore land in `results/scratch/wmb10x_landscape/tmp/wmb_figure_review/`.

## Inputs

Not in git:
- **Large inputs.** Step 2 reads `wmb_precomputed_stats.h5` (1,376,366,584 B, sha256 `b21ca985652f…`) and `wmb_gene.csv` (2,297,493 B, `ec5a211c23fa…`), from the Allen Brain Cell Atlas (Yao et al. 2023, CC BY-NC 4.0). Both go in `data/raw/WMB-10X/`. Download them with the commands in `docs/reproducing_the_paper.md` (Atlas files). `merfish_expression_prior` and `scripts/wmb10x/build_whole_brain_cpm.py` read the same two files.
- **The driver's guard.** When step 2 is requested, `run.sh` checks `data/raw/WMB-10X/` for both files. If either is missing (a dangling symlink counts as missing), it exits before running any step. `resolve_large_input()` in `download_wmb_metadata.py` would otherwise download a missing file from the ABC S3 bucket into `data/raw/WMB-10X/` (1.38 GB for the h5); the guard keeps the driver from triggering that download.
- **Metadata.** Step 1 downloads the five metadata files (~16 MB) into `data/raw/WMB-10X/`, so it needs network access the first time. Files already present are skipped. Step 1 downloads `term_with_counts.csv`, but no step reads it.
- **Link files, never directories.** You can link the inputs from elsewhere, but the scripts write through links: a directory link for `data/raw/WMB-10X`, `data/processed/WMB-10X` or `results/wmb10x` would send writes into the linked folder. `experiments/INPUTS.md` lists all inputs.

## Cost

- CPU only, in `scanpy_env`, on one machine.
- Expect about 6 min in total. On the reference machine (40 logical CPUs), step 2 took 5.1 s (max RSS 2.1 GB) and step 4 took 9.1 s. In the paper's run, step 3 took at most 5 min 45 s cold; its 7 UMAP fits dominate.

## Outputs

| File | Becomes |
|---|---|
| `results/wmb10x/figures/wmb_umap_panel_nokey.svg` | S4a scatter |
| `results/wmb10x/figures/wmb_umap_key.svg` | S4a 34-class key |
| `results/wmb10x/captions.md` | Caption source for S4a. The SVGs carry no caveat on their face, so this is the only place the "cell types, not cells" caveat is written. |

- **Which file is the panel.** S4a uses the split pair above: the nokey scatter and the separate key. The pipeline asserts that the nokey scatter is identical to the scatter of the keyed `wmb_umap_panel.svg`, so both routes give the same picture.
- **Other outputs.**
  - `wmb_umap_panel.svg` (keyed).
  - Diagnostics: `wmb_umap_variants.svg`, `wmb_pca_classes.svg`, `wmb_pca_scree.svg`.
  - Tables: `pca_coords.csv`, `pca_loadings_top.csv`, `pca_variance.csv`, `umap_coords.csv`, `umap_variants.csv`, `umap_knn_purity.csv`, `embedding_summary.json`.
  - SVD cache: `results/wmb10x/cache/{all_genes,marker_genes}_<32hex>.npz`.
  - Intermediates in `data/processed/WMB-10X/`.
- The panels carry no panel letter. Add the letter at assembly.
- **Figure widths.** `plot_embeddings.py` sizes every canvas from `duet.plotting.FIGURE_WIDTHS`. Its half_page and full_page widths are now 88 and 180 mm; the paper's S4a SVGs were drawn at 89 and 183 mm, so re-rendered SVGs are narrower.

## Compare after a re-run

The tables and captions should be byte-identical to the paper's run. Their sha256 values start with: `captions.md` (`d2c6c739ec94…`), `embedding_summary.json` (`f33f11ef57a7…`), `pca_coords.csv` (`407f16eff945…`), `pca_loadings_top.csv` (`0b4d0bd88d1c…`), `pca_variance.csv` (`e94a1d4d23dc…`), `umap_coords.csv` (`76774acb4c74…`), `umap_variants.csv` (`939dd5d89e69…`) and `umap_knn_purity.csv` (`9755e5aa187c…`).

```bash
cd results/wmb10x
sha256sum captions.md embedding_summary.json pca_coords.csv pca_loadings_top.csv \
    pca_variance.csv umap_coords.csv umap_variants.csv umap_knn_purity.csv
```

- **Checked.** Step 2's output was byte-identical to the paper's run. Step 4, run on the paper's tables at the paper's figure widths, gave a byte-identical `captions.md`, and its six SVGs were identical to the paper's after stripping ids and the date.
- **SVGs.** Raw bytes always differ between runs. matplotlib writes random `id`/`href`/`clip-path` ids and a `<dc:date>`, and the stylesheet sets no `svg.hashsalt`. Compare SVGs after stripping those.
- **Cache.** Cache file names change on every rebuild, so do not compare `cache/` by name.
- **Values** from the paper run's `embedding_summary.json`:
  - 5,322 centroids, 32,285 genes, 6,558 marker genes.
  - PC1 / PC2 variance ratio 0.1914 / 0.0837.
  - Cosine-UMAP class purity (k = 25): 0.8574 unweighted, 0.8883 cluster-size-weighted.
  - `umap_reproducible_bitwise: true`, `umap_repeat_max_coord_deviation: 0.0`.
- **Marker set and metric.** The marker set is the 6,558-gene union of Allen's MapMyCells markers, not the 8,460-gene list of Yao et al. The cosine metric is our choice; Yao et al. state none. `scripts/wmb10x/README.md` explains both.

## Determinism

- This experiment has no λ, no DUET, no GPU and no CPU-count, batch or memory knob.
- The scripts hard-code the seeds and parameters: UMAP with n_neighbors 25, min_dist 0.4, metric cosine, on 100 PCs of the 6,558 marker genes; scatters rasterized at 600 dpi.
  - `UMAP_SEED = 42` is passed as `random_state` to all 7 UMAP fits. This makes umap-learn single-threaded and deterministic; the refit check reports 0.0 deviation.
  - `PCA_SEED = 0` has no effect with `svd_solver="full"`, but it is part of the cache key.
  - `RNG_SEED = 0` sets the draw order in `plot_embeddings.py`.
- The numbers depend only on the inputs and the library versions. Byte-identity is expected in the same `scanpy_env`. It has not been checked across library versions.
- The SVD cache is the only speed knob the scripts expose. Its key includes the mtime of `centroids.npy`, and step 2 rewrites that file. So a default four-step run always does both SVDs cold. Running `WMB10X_STEPS="3 4"` after a completed run reuses the cache.

## Smoke test

- **Command.** `VARIANT=smoke bash experiments/wmb10x_landscape/run.sh`. It runs steps 1-2 only. There is no smoke config, because the scripts take no parameters. The logs go to `results/smoke/wmb10x_landscape/logs/`. Step 2 still writes the real `data/processed/WMB-10X/` files, because those paths are fixed.
- **Expected.**
  - Step 1 skips the metadata files already present (or downloads them, about 16 MB).
  - Step 2 writes the four `data/processed/WMB-10X/` files, byte-identical to those of the paper's run.
  - Steps 3-4 are not part of the smoke test, because there is no smaller setting. A smoke run therefore does not check them.
