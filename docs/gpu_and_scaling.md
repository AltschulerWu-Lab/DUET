# GPUs and scaling

Most of DUET's run time goes into two Monte Carlo computations: the pairwise error probability
(PEP) matrix and the evaluation of the returned codebooks. Both run on NVIDIA GPUs through CuPy,
or on the CPU. This page covers devices, measured times, memory and disk, and PEP caching.

## Devices

`design`, `design_ops`, `design_merfish` and `evaluate` take `device=`:

| `device` | Meaning |
|---|---|
| `"auto"` (default) | Every visible GPU when CuPy finds one, else the CPU |
| `"cpu"` | CPU only, with `workers` processes |
| `"gpu"` | GPU 0 only |
| `"gpu:all"` | Every visible GPU |
| `"gpu:0,2"` | The listed GPUs; list each once |

- **What runs where.** The PEP and the evaluation run on the chosen device. The λ sweep always runs
  on the CPU, one process per λ (`min(workers, len(lambdas))` processes), so on a GPU machine it can
  be the longest stage. `duet.design` runs its one search in the main process.
- **Choosing GPUs.** `CUDA_VISIBLE_DEVICES=1,3 python my_design.py` makes physical GPUs 1 and 3
  visible as `gpu:0` and `gpu:1`; `CUDA_VISIBLE_DEVICES=""` hides all of them. `"auto"` does not
  check whether a GPU is busy.
- **CPU warning.** With `"auto"` and no GPU, a problem with U >= 2,000 unique codewords runs on CPU
  with a warning; pass `device="cpu"` to silence it.
- **Provenance.** `res.provenance["devices"]` records the device of each stage. CPU and GPU give
  identical results with Hamming decoding but can differ with likelihood metrics
  ([Reproducibility](design.md#reproducibility)).

## Measured scaling

[`examples/scaling_ops.py`](../examples/scaling_ops.py) times one `duet.design_ops` call on an OPS
problem the size of the paper's medium-scale CRISPRi benchmark: 1,000 genes of hCRISPRi-v2.1 with
their top 10 sgRNAs plus 1,895 controls (U = 11,628), 2 guides per gene and 200 controls, 10 rounds,
symmetric channel (ε = 0.1), default settings. Hardware: 2 x Intel Xeon E5-2640 v4 (40 threads),
220 GB RAM, 4 x NVIDIA TITAN X (Pascal, 12 GB), CUDA 12.7 driver, CuPy 14.0.1.

| Device | PEP | λ sweep (CPU) | Evaluation | Total |
|---|---|---|---|---|
| CPU only (40 threads) | 4,027 s (67 min) | 86 s | 1,945 s (32 min) | 6,060 s (101 min) |
| 1 GPU (`gpu:0`) | 118 s | 94 s | 51 s | 265 s |
| 4 GPUs (`gpu:all`) | 54 s | 83 s | 36 s | 176 s |

To re-run it (it reads `data/processed/Horlbeck_2016/CRISPRi_v2_1.csv` and prints the stage times
as one JSON line):

<!-- docs-test: skip (needs the full Horlbeck table, GPUs, and minutes to hours) -->
```bash
python examples/scaling_ops.py --device cpu
python examples/scaling_ops.py --device gpu:all --json scaling.jsonl
```

## Memory and disk

`pool.describe()` reports U, the number of unique candidate codewords, before anything is
computed. The PEP takes time proportional to U² x `num_samples`: on 40 CPU threads, roughly
4,027 s x (U / 11,628)² x (`num_samples` / 10,000), and about ten times longer on 4 cores.

- **Memory.** The PEP is a U x U matrix of 16-bit counts: 2U² bytes. DUET holds it in memory when
  2U² <= 64 MiB (U up to about 5,792), the codebook has at most 256 codewords and the RAM is free;
  otherwise it memory-maps it from disk. Storage never changes the results.
- **Disk.** A memory-mapped PEP needs 6U² bytes free while it is built and takes 4U² once built (a
  raw and a symmetric copy), in `cache_dir` if given, else in a temporary directory deleted after
  the run.
- **Temporary directory.** Temporary PEP files and the evaluation's files go to `$TMPDIR` (else
  `/tmp`); the evaluation's files stay until the Python process exits, up to about 0.5 TB for a
  genome-wide library. For large runs set `TMPDIR` to a large disk. Killed runs can leave
  `duet_pep_*` and `duet_cache_*` folders there; delete them by hand.

| U | Scale | PEP, 2U² | Memory-mapped on disk, 4U² | Free while built, 6U² | CPU PEP, 40 threads (4 cores) |
|---|---|---|---|---|---|
| 2,000 | the CPU warning | 8 MB | 16 MB | 24 MB | about 2 min (about 20 min) |
| 11,628 | the medium-scale CRISPRi benchmark | 270 MB | 541 MB | 811 MB | 67 min, measured (about 11 h) |
| 100,000 | the paper's 32-bit MERFISH designs | 20 GB | 40 GB | 60 GB | use GPUs |
| 289,113 | the paper's genome-wide CRISPick library | 167 GB | 334 GB | 501 GB | use GPUs |

The PEP of the paper's MERFISH designs (`num_samples=30_000`) took about 2.5 hours on four GPUs.
For the genome-wide library (`num_samples=5_000`) the PEP took 5.7 to 7.2 hours on four GPUs, and
each design, with its baselines, took 16 to 19 hours and 0.6 to 0.85 TB of disk. Each
experiment's time, disk and memory are in [Reproducing the paper](reproducing_the_paper.md#hardware-and-time).

## Caching the PEP

Pass `cache_dir=` to keep the PEP on disk. A later run that changes only λ, the scores, the
quotas, the optimizer or the evaluation settings reads it from the cache:

```python
import pandas as pd
import duet

table = pd.read_csv("examples/data/quickstart_ops.csv")
pool = duet.CandidatePool.from_table(table, group="gene", sequence="sequence",
                                     score="activity", quota=2, seq_length=10)
kw = dict(seed=0, num_samples=2_000, eval_samples=0, workers=3, verbose=False,
          cache_dir="my_pep_cache")
first = duet.design_ops(pool, duet.channels.symmetric(0.1), lambdas=[0, 1], **kw)
again = duet.design_ops(pool, duet.channels.symmetric(0.1), lambdas=[0.25, 0.5], **kw)
print(first.provenance["pep"]["source"], again.provenance["pep"]["source"])
```

```text
computed cache
```

The cache key covers the unique codewords in table order, the channel, the decoding rule,
`num_samples`, the seed and `workers`, but not the device: a GPU-computed PEP is reused on CPU
(`provenance["pep"]["computed_on"]` says which device computed it). `workers` defaults to the
machine's CPU count, so pass it explicitly to reuse a cache on another machine. You can delete the
folder whenever no run is using it.

## Troubleshooting GPUs

| Problem | Fix |
|---|---|
| DUET runs on CPU, or `ValueError: device='gpu:all' requested but no GPUs are available` | CuPy cannot open a GPU. `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"` shows the CUDA error; see [Installation](installation.md#troubleshooting). Common causes: `CUDA_VISIBLE_DEVICES=""`, or a container or batch job without GPU access. |
| `ImportError: CuPy is required for GPU devices` | Install the `gpu` extra from your clone, `pip install ".[gpu]"` (DUET is not on PyPI), or use `device="auto"`. |
| `ValueError: GPU ID(s) [3] out of range` | GPU ids count the visible GPUs from 0, after `CUDA_VISIBLE_DEVICES`. |
| `cupy.cuda.memory.OutOfMemoryError` | DUET sizes its batches from each GPU's free memory when a stage starts, so another process probably took memory. Check `nvidia-smi` and use idle GPUs. |
| `OSError: The PEP for U = ... needs ... GB of free disk` | Pass `cache_dir=` on a disk with 6U² bytes free, or set `TMPDIR`. `No space left on device` during the evaluation means the temporary directory is too small. |
| `MemoryError` with `pep_storage="memory"` | Use the default `pep_storage="auto"`, which memory-maps a PEP that does not fit, or fewer `workers`. |
