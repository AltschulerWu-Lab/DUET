# nisseq_error_analysis: Supp Fig S3a (NIS-seq error rates) and the NIS-seq noise matrices

This experiment estimates the NIS-seq sequencing-noise channel from spot-level base calls of two HeLa IL-1β screens, E-9 and E-9C (8 files, 32,889,778 reads). It:

1. matches each read (14 nt) to the Brunello library by minimum Hamming distance, against the first 14 nt of each guide's reverse complement. Tied guides share the read equally;
2. applies a per-screen intensity threshold at the 50th percentile, after sweeping the 0th-90th percentiles;
3. counts transmitted vs observed bases per position and row-normalizes them into the four channel classes: uniform (symmetric), positional, channel (asymmetric) and positional channel;
4. saves the figures and the four `.npy` matrices.

The **Supp S3a** panel is the per-true-base error-rate plot with 95% Wilson intervals. This experiment also regenerates the matrices behind **Supp S3b** and **Fig 3d-f / Supp S3c**, and checks them byte for byte against the copies those experiments read.

The NIS-seq channel matrices are fitted from spot-level base calls that the authors of Fandrey et al. (2025) shared on request, and are included only with their permission. If one is missing from your copy, rebuild it with this experiment from the spot calls, which are available from those authors on request (see [Inputs](#inputs)).

## Run

```bash
conda activate duet        # the DUET environment (environment.yml) with the benchmark and notebooks extras (run.sh checks it imports this clone's duet)
bash experiments/nisseq_error_analysis/run.sh                       # N = 5000, 50000, 100000
SUBSAMPLE_NS="5000" bash experiments/nisseq_error_analysis/run.sh   # the S3a panel only
VARIANT=smoke bash experiments/nisseq_error_analysis/run.sh         # the smoke config
```

The notebook tools (`nbconvert`, `nbformat` and `ipykernel` for the `python3` kernel) come from the `notebooks` extra: `pip install -e ".[benchmark,notebooks]"`.

For each subsample size N (reads per file), `run.sh`:

1. runs the notebook copy headless: `python -m nbconvert --to notebook --execute` under `script(1)`, with a per-cell timeout of 7,200 s (`NISSEQ_CELL_TIMEOUT`; the slowest cell took 610 s at N = 100,000). A kernel that is alive but stuck therefore fails the run within 2 h instead of blocking an unattended launch; a kernel that dies fails it at once. A notebook failure stops `run.sh`.
   - The kernel starts in this folder with the active env's `python`.
   - The kernel reads its parameters from `NISSEQ_*` environment variables, which `run.sh` sets from `config.yaml`.
2. runs `check_provenance.py`. It compares the four regenerated matrices with the reference ones listed for that N under `provenance:` in `config.yaml`, by sha256 and by `np.array_equal`. It exits 1 on any difference or on a missing regenerated file. `run.sh` then carries on with the other sizes, so every size's matrices and check are produced, and exits 1 at the end, listing each N whose check failed. A size with no reference matrices in the config is reported as skipped, and so is a size whose four reference matrices are all missing from your copy. Some but not all four missing fails the check.

Papermill is not installed, so the notebook is run through `nbconvert`. Papermill would also not fit this notebook, which has no `parameters` cell.

Before any work, `run.sh` stops if:
- the config is missing;
- the active env does not import `duet` from this repo's `src/`;
- a subsample size is malformed;
- any of the 9 inputs is missing or differs from its sha256 in `config.yaml`. `CHECK_INPUT_HASHES=0` skips the hash check but keeps the existence check. The hash check takes about 8 s.

Everything runs on the CPU in one kernel process. The distance matmul uses the env's multithreaded BLAS. There is no GPU path.

**Environment overrides:**

| Variable | Default | Meaning |
|---|---|---|
| `SUBSAMPLE_NS` | config `subsample_ns`: `5000 50000 100000` | sizes to run, in order. `none` means all reads: untested here, and the original notebook estimated 5-7 h. |
| `NISSEQ_DATA_DIR` | config `data_dir`: `../../data/raw/NISseq_HeLa_IL1b` | folder holding `NIS-seq_HeLa_IL1b/` and `Brunello_sgRNAs/`. A relative value is resolved against your cwd. |
| `NISSEQ_LIBRARY_PATH` | config `library_path` | the Brunello file. A relative value is resolved against your cwd. |
| `CHECK_INPUT_HASHES` | `1` | `0` skips the input sha256 check. |
| `NISSEQ_CELL_TIMEOUT` | `7200` | per-cell timeout in seconds for `nbconvert` (`--ExecutePreprocessor.timeout`); `-1` disables it. |

The notebook can also be opened and run by hand from this folder. Its cell-1 defaults are the same relative paths, with N = 50,000 (the original notebook's value).

**Cost** (measured 2026-09-25 on the reference machine: Xeon E5-2640 v4, 40 logical CPUs, shared with other jobs):

| N per file | Notebook wall time | Matching cell (5) | Threshold sweep (cell 9) | Peak kernel RSS |
|---|---|---|---|---|
| 2,000 (smoke) | 1m00s | 39 s | 2 s | 1.8 GB |
| 5,000 | 1m11s | 47 s | 5 s | 3.4 GB |
| 50,000 | 7m34s | 6m19s | 44 s | 9.2 GB |
| 100,000 | 12m09s | 10m10s | 1m22s | 9.3 GB |

- The default run (all three sizes) took **21m06s** of wall time, including the 8 s input hash check and the three provenance checks (about 1 s each).
  - `/usr/bin/time -v` gave a maximum RSS of 9.8 GB over the whole run.
  - The per-size peaks above were sampled from the kernel every 2 s.
- The smoke row's RSS is the `/usr/bin/time` figure for the smoke run.
- Every file is read in full (2.6-6.9 M rows) before `head(N)` is taken.
- Matching runs in chunks of 10,000 reads against 77,441 guides. Each chunk's distance block is 10,000 × 77,441 float32 = 3.1 GB, plus a `np.partition` copy.
- So about 10 GB of free RAM is needed for any N ≥ 10,000.

## The notebook

- **A parameterized copy.** `02_error_analysis.ipynb` is the analysis notebook behind the paper's NIS-seq fits, with outputs and execution counts cleared, and **only cell 1 changed**. All other cells are verbatim, including the stale YAML snippets in the last markdown cell, which use the old `noise_model:` / `type: channel` names and old file names; [Noise channels](../../docs/noise_channels.md) gives the current config keys, and the Inputs section below the paths the experiments read.
  - **Paths and switches:** each comes from an environment variable with a relative default: `NISSEQ_DATA_DIR`, `NISSEQ_LIBRARY_PATH`, `NISSEQ_OUTDIR`, `NISSEQ_SUBSAMPLE_N` (`none` means all reads), `NISSEQ_SAVE_NOISE_MODEL` and `NISSEQ_SAVE_FIGURES`.
  - **Outputs:** matrices go to `<NISSEQ_OUTDIR>/noise_matrices/`, and figures to `<NISSEQ_OUTDIR>/figures/`.
  - **Guard:** cell 1 raises an error if the output folder lies inside any `noise_model_matrices/` folder, so a run cannot overwrite the reference matrices.
  - **`SAVE_NOISE_MODEL` defaults to on.**
  - **Provenance printout:** cell 1 also prints `duet`'s location, the kernel's `python`, and the resolved input and output paths.
- **`error_utils.py`** sits beside the notebook in this folder. It holds the loading, matching, counting and plotting code. Cell 1 puts this folder on `sys.path` and imports it. `run.sh` sets `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` is written here.
- **Logs.** Each notebook run runs under `script(1)`, leaving `<tag>.raw.log` verbatim and `<tag>.log` cleaned. `run.sh` also extracts what each cell printed, with its run time, to `<tag>.outputs.log`, because `nbconvert` does not echo cell output. The check output goes to `<tag>.provenance.log` and the input hashes to `inputs.log`.
- **Settings:**
  - every compute cell: whole-file reads, then `df.head(N)`; `chunk_size=10000`; the exact-match sanity check; the 7-percentile per-screen sweep; `SELECTED_PERCENTILE = 50`;
  - `NUM_ROUNDS = 14`, the file list, and the export names;
  - `apply_style()`, all figure calls, sizes and file names. `channel_matrices` keeps its hard-coded `figsize=(21, 6)`; it is not a paper panel;
  - the `python3` kernel.
- **Figure width.** `FIGURE_WIDTHS["half_page"]` is now 88 mm; the paper's S3a panel was drawn at 89 mm. So the re-rendered panel is 246.75 pt wide instead of 249.59 pt; its height is unchanged at 177.16 pt. The data are identical: every marker's y coordinate matches (see below). Everything else matches the paper's SVG: fill and stroke colors (same counts), font families and sizes, stroke widths and the 28 text elements.
- **Not regenerated:** the 80th-percentile matrices and figures. They need `SELECTED_PERCENTILE = 80` in cell 12, and no figure uses them.

## Inputs

The inputs are not in git. Put them in `data/raw/NISseq_HeLa_IL1b/` (gitignored; a folder or a link to one), or point `NISSEQ_DATA_DIR` and `NISSEQ_LIBRARY_PATH` elsewhere:

- `NIS-seq_HeLa_IL1b/` holds the 8 spot files. They are available from the authors of Fandrey et al. 2025 on request (see [Data availability](#data-availability)).
- `Brunello_sgRNAs/Brunello_sgRNAs.txt` is the Brunello library (see Brunello below).

| File (under the data folder) | Size (B) | Rows | sha256 |
|---|---|---|---|
| `Brunello_sgRNAs/Brunello_sgRNAs.txt` | 2,222,847 | 77,441 | `5daee4c28dcb777ee19b29823dab1216d8f41a453492d6e4b75c0a82c427bb79` |
| `NIS-seq_HeLa_IL1b/E-9_B2_seq_combined.txt` | 119,313,782 | 2,569,670 | `8e8449828fe17b0098376aba0612dd719a5f5d726f22db3058e8c9e9c39e4ee9` |
| `NIS-seq_HeLa_IL1b/E-9_B3_seq_combined.txt` | 133,578,381 | 2,874,102 | `07cc21ee18f7cbe1741fa9cf256c4037538c69981f1c02d22ea67b00b55750e7` |
| `NIS-seq_HeLa_IL1b/E-9_B4_seq_combined.txt` | 126,376,488 | 2,720,463 | `3825a3ed5d6ada47d3412e5d982ebf4eddd8f46b50e5e3dc5f6f53390ccdd70b` |
| `NIS-seq_HeLa_IL1b/E-9_B5_seq_combined.txt` | 141,855,496 | 3,053,594 | `29a7cf8fa59e5048402324d64e78c49c02641da18069bab03d54054f8b31cdd5` |
| `NIS-seq_HeLa_IL1b/E-9C_Sequences_B2.txt` | 241,735,654 | 5,189,057 | `d618161f1c5848a7c6b6d871a2aa9ff05d6789fe806b592c621ac20dfad0cfef` |
| `NIS-seq_HeLa_IL1b/E-9C_Sequences_B3.txt` | 320,017,282 | 6,868,569 | `66e0e29b688c3a10b144d2b420b01872da545fd48b3b0c170a34d63a96baf450` |
| `NIS-seq_HeLa_IL1b/E-9C_Sequences_B4.txt` | 240,773,689 | 5,166,013 | `b0c8dce56815bcefc852a7e4a17a7e8deb6e115639f65e21f5e47b0d049c508c` |
| `NIS-seq_HeLa_IL1b/E-9C_Sequences_B5.txt` | 207,131,231 | 4,448,310 | `ff0ae0aa93623fc641cf32ad7d221c91730b3860d9b3d21db27e7a19e0ab3bef` |

The spot files are the total 1,530,782,003 B. Their format is tab-separated with no header: `tile, x, y, sequence, intensity`.

- All reads are 14 nt of ACGT, so `filter_valid_reads` removes nothing.
- Rows are sorted by tile. So `head(N)` takes the first tiles of each well: 1-2 tiles at N = 5,000, 4-15 at 50,000, and 7-27 at 100,000 (checked 2026-09-25).
- The run also reads the reference matrices it compares against (see the table under Subsample sizes), when your copy includes them.
- It reads the tracked `src/duet/plotting/duet_publication.mplstyle` through `apply_style()`.

**Brunello.** `Brunello_sgRNAs.txt` has CRLF line endings, no header, and two columns (gene, spacer).

- Its 77,441 spacers are identical, **in the same row order**, to the public Broad GPP / Addgene table `broadgpp-brunello-library-contents.txt` (column `sgRNA Target Sequence`; sha256 `77001ee1…905c`), from <https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/>. Verified 2026-09-25.
- Only 56 gene labels differ: the Excel-mangled `SEPT1`-`SEPT14` and `SEP15`, which read as "Sep 01" and so on (14 genes × 4 guides).
- Gene labels do not enter the matching or the counts. The matrices depend only on the spacer order.
- So a public build script can rebuild this file from the Addgene table and give identical matrices. That is inferred from the code, not re-run. Its sha256 would differ, so run such a build with `CHECK_INPUT_HASHES=0` or with the new hash in the config.

## Data availability

The spot-level base calls are not in the public NIS-seq Zenodo record (doi:10.5281/zenodo.13375079); they are available from the authors of Fandrey et al. 2025 (doi:10.1038/s41587-024-02516-5) on request.

## Subsample sizes and the figures that use them

**Three different read subsets feed the paper's NIS-seq panels.** Each is the first N rows of each file, then a per-screen 50th-percentile intensity threshold, so the subsets are nested.

| N / file | Matched | Kept at pctl50 | Thresholds E-9 / E-9C | avg ε | Consumer | Compared against |
|---|---|---|---|---|---|---|
| 5,000 | 40,000 | 20,000 | 1.3706 / 0.5989 | 0.158689 | **Supp S3a** panel | `channels/nisseq/nisseq_hela_{kind}_pctl50_subsample5000.npy` (no other reader) |
| 50,000 | 400,000 | 200,000 | 1.4130 / 0.6469 | 0.144656 | **Supp S3b**: `experiments/ops_crispri_cross_eval`. Its `noise_matrices/` holds a copy of the channel matrix and the first 10 rounds of the positional and positional-channel matrices, and its `epsilon: 0.14465571` literal comes from the uniform matrix. | `channels/nisseq/nisseq_hela_{kind}_pctl50_subsample50000.npy` |
| 100,000 | 800,000 | 400,000 | 1.3978 / 0.6456 | 0.142798 | **Fig 3d-f, Supp S3c**: the genome-wide CRISPick runs (`channels/archive/…`, read by `experiments/ops_crispick_genome_wide`) | `channels/archive/nisseq_hela_{kind}_subsample.npy` |

Here {kind} is uniform, positional, channel and positional_channel, and avg ε is the mean per-position error rate, which is 1 − the diagonal of the uniform matrix. The `channels/` paths are under `scripts/benchmark/noise_model_matrices/`.

- **S3a renders at N = 5,000 reads per file.** The 50,000 and 100,000 runs stay in the default run as provenance checks for S3b and for Fig 3d-f / S3c. At 50,000 the panel's y-axis stops at 0.35 instead of 0.40, and its intervals are narrower.
- **Using rebuilt matrices.** If your copy lacks the reference matrices, create their folders from the repository root (`mkdir -p scripts/benchmark/noise_model_matrices/channels/nisseq scripts/benchmark/noise_model_matrices/channels/archive experiments/ops_crispri_cross_eval/noise_matrices`), then copy the regenerated ones (`subsample<N>/noise_matrices/`, see Outputs) to the paths in the last column; the N = 100,000 files take the names `nisseq_hela_{kind}_subsample.npy`. For S3b, `experiments/ops_crispri_cross_eval/noise_matrices/` takes the N = 50,000 channel matrix as `nisseq_channel.npy`, and the first 10 rounds (`[:10]`) of the positional and positional-channel matrices as `nisseq_positional_10rounds.npy` and `nisseq_positional_channel_10rounds.npy`.

## Outputs

All outputs are under `results/experiments/nisseq_error_analysis/`. Here `<N>` is the subsample size and `{kind}` is uniform, positional, channel or positional_channel.

| File | Use |
|---|---|
| `subsample5000/figures/error_lines_4_pctl50_subsample5000.svg` | **Supp S3a** |
| `subsample<N>/figures/error_lines_4_pctl50_subsample<N>.svg` (N = 50,000, 100,000) | the same panel at the other sizes; not in the paper |
| `subsample<N>/figures/{channel_matrices,positional_channel_grid,error_heatmap,error_lines_12}_pctl50_subsample<N>.svg` | diagnostics; not in the paper |
| `subsample<N>/noise_matrices/nisseq_hela_{kind}_pctl50_subsample<N>.npy` | the four channels. Byte-identical to the reference files in the table above. |
| `subsample<N>/02_error_analysis.executed.ipynb` | the executed notebook. Its inline plots include the threshold sweep and the per-replicate and per-base views, which are not saved as files. |
| `logs/inputs.log` | input sha256 check |
| `logs/subsample<N>.{raw.log,log}` | `nbconvert` under `script(1)` |
| `logs/subsample<N>.outputs.log` | every code cell's printed output and run time: thresholds, sweep, avg ε, matrices |
| `logs/subsample<N>.provenance.log` | `check_provenance.py`: both sha256s and `np.array_equal` per matrix |

## Compare after a re-run

**All 12 regenerated matrices should be byte-identical to the reference ones.** All are float64. Their sha256 values, which you can check with `sha256sum` even if your copy lacks the reference files:

| N / file | Reference file | uniform | positional | channel | positional_channel |
|---|---|---|---|---|---|
| 5,000 | `channels/nisseq/nisseq_hela_{kind}_pctl50_subsample5000.npy` | `dd0958d0a2447ac7732a969e6abe617366e9a769dd6a16e78c9aa16a4fc0ef63` | `57a870fcb390d69513131d9c0842d9b776c87b0b5e5e7bce0375b7d855e91c3b` | `af2120c8dd5b62fd1e34d164708ee6ebaf0db1f98ec78cd21cbe667dbf2fd6a0` | `baa3ebc0add405c6329aa59e838c0d5c489e5a053819e29064193e2f4f8023e8` |
| 50,000 (S3b) | `channels/nisseq/nisseq_hela_{kind}_pctl50_subsample50000.npy` | `d028e4ca5faa7f848a2921149d06c62d1dccb3181b17b9b26d5d274a774a1a33` | `0c82c5698b2396c63b8cb1701fb80d5140bbf495f96a28378472c7a54b7b93ee` | `a85aee4091523badbd5c001603c18cc23bf999e50ce052d3bf494270a8abe648` | `fa1a90566ba76e09628469c87bb795e072b983520f14f5f46078ab3ab7938679` |
| 100,000 (Fig 3d-f, S3c) | `channels/archive/nisseq_hela_{kind}_subsample.npy` | `b899d43997dddb7e077ff79c6ffbcd9d918e0e2a8292c13d6707f46a98c87068` | `a66a64fe5aa4f9cb3a2c1fdb166a2f2be34ba8ebda3b486f6c2e5c0304f584cf` | `27fcda8b163ec02bff141313fd7b606a61589adb4e47c761799fe8b530a6ddf4` | `0f2a4d93d671066f39d686a2235ac57b0e58e9674319076581630893474def50` |

The full hashes and paths of each run are in `logs/subsample<N>.provenance.log`. A re-run of 2026-09-25 matched all 12, which confirms two things:
- **S3b:** its matrices are the N = 50,000 fit.
- **Fig 3d-f and S3c:** the archive `_subsample` matrices are this pipeline at N = 100,000, with the per-screen p50 threshold and 400,000 reads kept. Their avg ε is 0.142798.

**S3a panel.** SVG bytes never match (see Determinism), so compare panels by the coordinates of their 201 `<use>` markers. In that re-run, all 201 y coordinates of the regenerated N = 5,000 panel equalled those of the paper's panel, as identical strings in the same order; the x coordinates differ because of the 89 → 88 mm width. The regenerated 50,000 panel also matched its counterpart from the paper's run: 200/200 y coordinates in order.

## Determinism

- **There is no randomness.** Subsampling is `df.head(N)`, and the files are sorted by tile.
  - The Hamming distances come from a float32 one-hot matmul whose results are small exact integers (at most 14). So BLAS, thread count and chunking cannot change them.
  - Ties are resolved in index order.
  - So the `.npy` outputs repeat byte for byte. A re-run with numpy 2.4.2 and pandas 3.0.0 reproduced every matrix of the paper's runs exactly.
- **The SVGs are not byte-reproducible.** matplotlib writes a `dc:date`, and `svg.hashsalt` is unset, so marker and clip ids are random. Compare panels by coordinates.
- **Knobs that change the numbers:** N, `SELECTED_PERCENTILE`, `THRESHOLD_PERCENTILES` (only the selected one is exported), `NUM_ROUNDS`, the file list and order, the library order, and the inputs.
- **Knobs that do not:** `chunk_size` and BLAS threads affect only speed and memory.

## Smoke test

`VARIANT=smoke bash experiments/nisseq_error_analysis/run.sh` reads the tracked `config.smoke.yaml`. It needs the same inputs. It is `config.yaml` with these changes:
- `subsample_ns: [2000]`;
- `outdir: ../../results/smoke/nisseq_error_analysis`;
- `provenance: {}`: no reference matrices exist for 2,000, so the check reports "skipped".

It takes the same code path as the real run: input hashes, nbconvert, all cells, both figure and matrix export, and the check script.

Expected, from the smoke run of 2026-09-25: about 1m10s wall time.
- All 9 input hashes were OK.
- The notebook ran all cells in 1m00s: 16,000 reads, 8,000 kept at pctl50, avg ε 0.160339.
- It wrote 5 SVGs and 4 matrices under `results/smoke/nisseq_error_analysis/subsample2000/`.
- The provenance step printed `PROVENANCE CHECK SKIPPED`.
- Peak RSS was 1.8 GB. With N < 10,000, the distance chunk is only N rows.

The check's failure path was tested by hand. The N = 2,000 matrices, renamed as N = 5,000, were compared with the reference 5,000 files. The check exited 1 with `PROVENANCE CHECK FAILED` and each kind's max |diff| (0.0017-0.063).
