# Examples

## Quickstart demos

Two toy-scale demos of the public API that run on CPU in under a minute, so
a reviewer without a GPU can run them. Real libraries should run on GPU
(install the `gpu` extra; `device="auto"` then uses every visible GPU).

| Script | What it designs | Data | Runtime (4 CPU cores) |
|---|---|---|---|
| `quickstart_ops.py OUTDIR` | an OPS sgRNA library: 2 of 10 sgRNAs for each of 50 genes, 10 of 50 controls, 10 sequencing rounds, symmetric 10% error, 6 lambdas | `data/quickstart_ops.csv`, from the hCRISPRi-v2.1 library (Horlbeck et al. 2016, CC BY 4.0; see `data/README.md`) | 15 s |
| `quickstart_merfish.py OUTDIR` | a 40-gene, 16-bit MERFISH codebook (Hamming weight 4, 40 candidate barcodes per gene), asymmetric binary error, 9 lambdas, starting from a Boström et al. 2025 codebook | genes and barcodes of `data/bostrom/bostrom_hw4_16bit_panel.csv`; **synthetic** log-normal expression | 44 s |

Each prints the result table (one row per codebook: evaluated accuracy,
`surrogate_accuracy`, the secondary objective, duplicate codewords, Pareto
front) and writes `table.csv`, `codebooks.csv`, `settings.json` and
`pareto_front.svg` under `OUTDIR`. The expected output is recorded in each
script's docstring, and `tests/test_quickstarts.py` checks it
(`python -m pytest tests/test_quickstarts.py -m slow`). Both use the default
sample counts (`num_samples=10_000`, `eval_samples=5_000`); the MERFISH demo
samples 40 candidate barcodes per gene instead of the default 1,000.

Runtimes were measured on an Intel Xeon E5-2640 v4 at 2.4 GHz, restricted to
4 cores (`taskset -c 0-3`), CPU only.

## Scaling: CPU vs 1 GPU vs 4 GPUs

`scaling_ops.py` times one `duet.design_ops` call on an OPS problem the size of
the paper's medium-scale CRISPRi benchmark, stage by stage, with the engine's own stopwatches.

- **Problem.** 1,000 genes of hCRISPRi-v2.1 with their top 10 sgRNAs, plus
  200 of 1,895 non-targeting controls, 10 sequencing rounds, quota 2 per gene
  (`create_pool_from_source("WeissmanCRISPRi", seq_rounds=10, quota=2,
  num_controls=200, min_rank=10, num_groups=1000, seed=0)`): 11,895
  candidates, 1,001 groups, 2,200 codewords, **U = 11,628** unique codewords.
- **Settings.** The facade defaults: `num_samples=10_000`,
  `eval_samples=5_000`, six lambdas, `max_patience=3000`, unique-minimum
  decoding, symmetric 10% error, `seed=0`. The PEP (2U² = 270 MB) is
  memory-mapped.
- **Hardware.** 2 x Intel Xeon E5-2640 v4 (20 cores, 40 threads), 220 GB RAM,
  4 x NVIDIA TITAN X (Pascal, 12 GB), CUDA 12.7 driver, CuPy 14.0.1. CPU
  stages used all 40 threads; the lambda sweep used 6 processes (one per
  lambda). Measured on 2026-09-24 on an otherwise idle machine.

| Device | PEP | Lambda sweep (CPU) | Evaluation | Total |
|---|---|---|---|---|
| CPU only (40 threads) | 4,027 s (67 min) | 86 s | 1,945 s (32 min) | 6,060 s (101 min) |
| 1 GPU (`gpu:0`) | 118 s | 94 s | 51 s | 265 s |
| 4 GPUs (`gpu:all`) | 54 s | 83 s | 36 s | 176 s |

GPUs speed up the PEP (75x with 4 GPUs) and the evaluation (54x); the lambda
sweep runs on CPU in every case. The PEP costs about U² x `num_samples`
reads-times-codewords, so on CPU it takes roughly 4,027 s x (U / 11,628)² x
(`num_samples` / 10,000) on 40 threads: about 2 minutes at U = 2,000, and
about ten times longer on 4 cores. `duet.design`, `duet.design_ops` and
`duet.design_merfish` therefore warn, when no GPU is found, for U >= 2,000
(`duet.api.CPU_WARN_UNIQUE_CODEWORDS`).

Re-run with, on an otherwise idle machine:

```bash
python examples/scaling_ops.py --device cpu
python examples/scaling_ops.py --device gpu:0
python examples/scaling_ops.py --device gpu:all
```
