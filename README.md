# DUET

DUET (Decoding Unified with Experimental Trade-offs) designs codebooks, the sets of barcodes
that identify genes, perturbations or other targets in barcode-based assays. It selects barcodes
that decode accurately under a model of the readout errors and, optionally, trades decoding
accuracy against an experiment-specific objective, returning the Pareto front of codebooks
across that trade-off. The two worked applications are sgRNA libraries for optical pooled
screens (OPS) and MERFISH codebooks; the method is general.

The method and its benchmarks are described in the manuscript "DUET: Multi-objective
optimization of codebooks for barcode-based assays" (Rao, Wu & Altschuler, submitted). This
repository holds the software, its documentation and the scripts that reproduce the paper.

## Overview

1. **Candidates.** You give candidate sets (for example the candidate sgRNAs of each gene) and
   how many to select from each. A candidate's codeword is the part of its barcode that is read.
2. **Decoding model.** You give a noise channel (how the readout corrupts each symbol) and a
   decoding rule. DUET simulates reads to estimate the pairwise error probabilities (PEP)
   between all candidate codewords.
3. **Search.** From an initial codebook (random, or one you already use), DUET swaps candidates
   within their sets to maximize λ · decoding accuracy + (1 - λ) · objective; a sweep over λ
   traces the trade-off. The objective is a per-candidate score such as predicted sgRNA activity
   (`duet.design_ops`) or, for MERFISH, optical crowding (`duet.design_merfish`). DUET can also
   run without a secondary objective: `duet.design` optimizes decoding accuracy alone (λ = 1;
   see [Improving a codebook for decoding](#improving-a-codebook-for-decoding)).
4. **Evaluation.** Every codebook is evaluated on fresh simulated reads. The result table gives
   its decoding accuracy and, in a trade-off design, its objective and whether it lies on the
   Pareto front.

## System requirements

- **Operating system:** Linux. Tested on Ubuntu 20.04.6 LTS.
- **Python:** 3.11, 3.12 or 3.13. Tested on 3.11.14; continuous integration covers 3.11 to 3.13.
- **Dependencies:** NumPy, pandas, SciPy, matplotlib, psutil, threadpoolctl and tqdm, installed
  by pip. Tested with numpy 2.4.2, pandas 3.0.0, scipy 1.17.0, matplotlib 3.10.8, psutil 7.2.2,
  threadpoolctl 3.6.0 and tqdm 4.67.3, the minimum versions in `pyproject.toml`.
- **GPU (recommended, especially for large libraries):** one or more NVIDIA GPUs whose driver
  supports CUDA 12, and CuPy. Tested on NVIDIA TITAN X (Pascal, 12 GB) with driver 565.57.01,
  CuPy 14.0.1 (conda-forge) and cupy-cuda12x 14.2.0 (PyPI).
- **CPU only:** fine for the demos and small problems. The demos run on an ordinary desktop
  computer in under a minute and about 1 GB of RAM.

By default DUET uses every visible GPU, splitting the PEP and the evaluation across them; their
cost grows with the square of the number of unique candidate codewords. Designing a medium-scale
CRISPRi library (2 sgRNAs for each of 1,000 genes, from 11,628 unique candidate codewords) with
default settings took 3 min on 4 GPUs, 4.4 min on one GPU and 101 min on 40 CPU threads.
Genome-wide libraries need substantially more time and disk; see
[GPUs and scaling](docs/gpu_and_scaling.md#memory-and-disk) for more details. All timings come
from one machine: 2 x Intel Xeon E5-2640 v4 (2.4 GHz, 40 threads), 220 GB RAM, 4 x NVIDIA
TITAN X.

## Installation

DUET is not on PyPI. Install it from a clone into a new environment:

<!-- docs-test: skip (installs packages) -->
```bash
git clone https://github.com/AltschulerWu-Lab/DUET.git
cd DUET
conda create -n duet python=3.11
conda activate duet
pip install ".[gpu]"    # CPU only: pip install .
```

The `gpu` extra installs CuPy for CUDA 12. Installation takes about a minute: creating the
environment took 42 s, `pip install .` 27 s and the `gpu` extra 10 s more. Check it (this prints
`0.2.0`):

```bash
python -c "import duet; print(duet.__version__)"
```

[Installation](docs/installation.md) covers other CuPy packages, checking that DUET sees your
GPUs, the development install and troubleshooting.

## Demo

From the repository root:

```bash
python examples/quickstart_ops.py demo_ops
```

This designs a small OPS library: 2 of 10 candidate sgRNAs for each of 50 genes of the
hCRISPRi-v2.1 library (Horlbeck et al. 2016) and 10 of 50 non-targeting controls, read in 10
sequencing rounds with a 10% error per base, for six values of λ. It runs in about 20 s on 4
CPU cores and ends with:

```text
   codebook  accuracy_mean  accuracy_p10  surrogate_accuracy  mean_score  duplicate_codewords  on_pareto_front
    initial         0.9501        0.9187              0.9072      0.7169                    0            False
   lambda=0         0.9449        0.8959              0.8973      0.8447                    0             True
lambda=0.05         0.9495        0.9032              0.9056      0.8447                    0             True
 lambda=0.1         0.9511        0.9040              0.9085      0.8444                    0             True
lambda=0.25         0.9543        0.9252              0.9140      0.8433                    0             True
 lambda=0.5         0.9614        0.9453              0.9247      0.8355                    0             True
   lambda=1         0.9726        0.9670              0.9476      0.6907                    0             True
  max_score         0.9456        0.8989              0.8987      0.8447                    0            False
...
Wrote demo_ops/table.csv, codebooks.csv, settings.json, pareto_front.svg
Runtime: 20 s
```

Each row is a codebook: the random initial codebook, one per λ, and the maximum-activity
selection. `accuracy_mean` is the decoding accuracy on fresh simulated reads, `accuracy_p10` its
10th percentile across codewords and `mean_score` the mean predicted activity
(`surrogate_accuracy` is the search's objective, not an accuracy). From λ = 0 to λ = 1 the
codebooks trade activity for decoding accuracy. The table is the same on any machine.

```bash
python examples/quickstart_merfish.py demo_merfish
```

This redesigns a 40-gene, 16-bit MERFISH codebook of Boström et al. (2025) against optical
crowding, with synthetic expression values, in about 45 s; its expected output is recorded at
the top of the script. Both demos also write their results and a plot of the front to the folder
you name.

## Using DUET on your own data

Describe the candidates in a table with one row per candidate: its candidate set, its sequence
and, optionally, a score. This example uses the OPS demo's table; replace it with yours.

```python
import pandas as pd
import duet

table = pd.read_csv("examples/data/quickstart_ops.csv")  # one row per candidate sgRNA
pool = duet.CandidatePool.from_table(
    table,
    group="gene",         # candidate set: the sgRNAs of one gene, or the controls
    sequence="sequence",  # spacer, 5' to 3'
    score="activity",     # optional: predicted activity
    quota=2,              # how many to select from each candidate set
    seq_length=10,        # 10 rounds of in situ sequencing read the first 10 bases
)
channel = duet.channels.symmetric(0.1)  # each base is misread with probability 0.1
```

### Improving a codebook for decoding

`duet.design` optimizes decoding accuracy alone and ignores the score (a pool built without
`score` works the same). Here it starts from a library you already use:

```python
current = table.groupby("gene").head(2)  # stand-in for a library you already use
res = duet.design(pool, channel, init=current, seed=0, verbose=False)
print(res.table[["codebook", "accuracy_mean", "accuracy_p10"]].to_string(index=False))
improved = table.iloc[res.codebook()["candidate"]]  # the improved library, full rows
```

```text
codebook  accuracy_mean  accuracy_p10
 initial       0.947929       0.90246
designed       0.974353       0.96800
```

`initial` is your library and `designed` the improved one, evaluated together on fresh
simulated reads. Leave out `init` to start from a random codebook. [Designing a
codebook](docs/design.md#improving-a-codebook-for-decoding) shows how to keep candidates fixed
and how to score any codebook with `duet.evaluate`.

### Trading decoding accuracy against another objective

With a score, a sweep over λ returns the Pareto front of decoding accuracy against the mean
score:

```python
res = duet.design_ops(pool, channel, lambdas=[0, 0.1, 0.25, 0.5, 1], seed=0, verbose=False)
print(res.table[["codebook", "accuracy_mean", "mean_score", "on_pareto_front"]])
op = res.operating_point()  # the most accurate codebook that keeps 97.5% of the maximum score
library = table.iloc[res.codebook(op["lambda"])["candidate"]]  # the chosen sgRNAs, full rows
res.save("my_library")      # table.csv, codebooks.csv, settings.json
```

The table reads like the demo's. Scores of order 1 (here 0.25 to 1) suit these λ values; rescale
others ([Adding an objective](docs/design.md#adding-an-experiment-specific-objective)). Each
design takes about 15 s on 4 CPU cores.

The guides take it from here: [Designing a codebook](docs/design.md) for any assay, and the two
worked applications, [OPS libraries](docs/design_ops.md) and
[MERFISH codebooks](docs/design_merfish.md).

## Reproducing the paper

[`experiments/`](experiments/) has one folder per analysis of the paper, each with a `run.sh`
driver and a README. [Reproducing the paper](docs/reproducing_the_paper.md) gives the hardware,
inputs and run times. Most experiments need NVIDIA GPUs and take hours. The largest one designs
the genome-wide OPS library under each of four noise channels: each design, with its baselines,
took 16 to 19 hours on 4 GPUs and 0.6 to 0.85 TB of disk (a 334 GB PEP cache plus 0.24 to
0.51 TB of temporary evaluation files). A few inputs are not in the repository and must be
downloaded, built or requested.

## Running the tests

<!-- docs-test: skip (installs packages and runs the whole test suite) -->
```bash
pip install ".[test]"
python -m pytest
```

The default run takes about 2 minutes on CPU and skips slow tests, GPU tests without a GPU and
benchmark tests without the `benchmark` extra; nothing should fail (tests marked xfail document
known floating-point behavior). `python -m pytest -m slow` runs the slow tests, including both
demos checked against their recorded output. `tests/test_docs.py` checks the documentation: its
links and style by default, and with `-m slow` it runs every command and code snippet of this
README and `docs/` on CPU, except those that install packages, need a GPU or take hours.

## Documentation

| Page | Contents |
|---|---|
| [Installation](docs/installation.md) | CuPy packages, checking the installation, extras and development install, troubleshooting |
| [Designing a codebook](docs/design.md) | The workflow for any assay: inputs, decoding-only designs (`duet.design`), objectives and λ, reading results, evaluating codebooks, reproducibility |
| [OPS libraries](docs/design_ops.md) | Designing an sgRNA library for an optical pooled screen, from a candidate table to an exported library |
| [MERFISH codebooks](docs/design_merfish.md) | Designing a MERFISH codebook from a gene panel, and the crowding objective |
| [Noise channels](docs/noise_channels.md) | Channel classes, transition matrices, estimating a channel from reads, decoding rules |
| [GPUs and scaling](docs/gpu_and_scaling.md) | Devices, measured scaling, memory and disk, caching the PEP |
| [API reference](docs/api.md) | Public functions and classes, result tables, saved files |
| [Reproducing the paper](docs/reproducing_the_paper.md) | Hardware, inputs and commands for the analyses in `experiments/` |

Report problems and ask questions in the [GitHub issues](https://github.com/AltschulerWu-Lab/DUET/issues),
with the `settings.json` of the run.

## Citation

If you use DUET, please cite:

Rao, L., Wu, L. F. & Altschuler, S. J. DUET: Multi-objective optimization of codebooks for
barcode-based assays. Submitted.

The software citation is in [CITATION.cff](CITATION.cff); each release is archived on Zenodo.
<!-- TODO(release): add the Zenodo DOI of the release -->

## License

The code is released under the MIT license (see [LICENSE](LICENSE)). The MIT license covers the
code only. Bundled data keep their own licenses:

- the hCRISPRi-v2.1 tables (Horlbeck et al. 2016): CC BY 4.0;
- the Boström et al. (2025) codebooks, from Dryad
  ([doi:10.5061/dryad.zkh1893m5](https://doi.org/10.5061/dryad.zkh1893m5)): CC0 1.0;
- the sensory-neuron expression table (Kastriti et al. 2022): CC BY 4.0;
- the 140-gene MERFISH codebook of Chen et al. (2015): a factual gene-to-barcode table, included
  with its citation;
- the MERFISH noise channel and the gene lists of the `merfish_zhang2023_v2` panels, derived from
  the Zhang et al. (2023) data in the Brain Image Library: CC BY-SA 4.0. This work used data from
  the Brain Image Library (RRID:SCR_017272), which is supported by the National Institutes of
  Mental Health of the National Institutes of Health under award number R24-MH-114793.

The NIS-seq channel matrices are included only with the permission of the authors of Fandrey et
al. (2025) ([Noise channels](docs/noise_channels.md#channels-in-the-repository)).

Not included, because their terms restrict commercial use: the per-gene values of the
whole-mouse-brain atlas of Yao et al. (2023) (CC BY-NC 4.0), the CRISPick candidate table (Broad
Institute GPP terms, research use) and the MERFISH data of Xia et al. (2019) (CC BY-NC-ND 4.0).
Scripts in the repository build the first two from the original downloads
([Reproducing the paper](docs/reproducing_the_paper.md#inputs-not-in-the-repository)).

Sources, attributions and changes are recorded in [examples/data/README.md](examples/data/README.md),
[data/processed/README.md](data/processed/README.md), the
[channels README](scripts/benchmark/noise_model_matrices/channels/README.txt) and the panel
READMEs of [merfish_zhang2023_v2](experiments/merfish_zhang2023_v2/panels/README.md) and
[merfish_2000_genes](experiments/merfish_2000_genes/panels/README.md).
