# Designing a MERFISH codebook

This tutorial designs a MERFISH codebook for a small gene panel bundled with the repository. In
MERFISH each gene gets a binary barcode, one bit per readout image. Readout errors flip bits, and
every 1 adds the gene's transcripts to that image, so abundant genes crowd the images.
`duet.design_merfish` chooses a barcode for every gene, selecting barcodes and assigning them to
genes in one optimization, and trades decoding accuracy against optical crowding.
[Designing a codebook](design.md) explains the result table.

The Python blocks run in order in one session, in about 65 to 100 s on 3 CPU cores, and write
to `my_merfish_design/`.

## Gene table and candidate barcodes

| You need | In this tutorial | Argument |
|---|---|---|
| A gene panel with one expression value per gene, on a linear scale (for example mean counts per million) | the first 40 genes of a 140-gene panel, with **synthetic** expression | `genes`, `gene=`, `expression=` |
| The barcode length, in readout bits | 16 | `n_bits=` |
| The allowed Hamming weights (numbers of 1s), each with an optional cap | weight 4 (all 1,820 barcodes) and weight 5 (1,000 of 4,368) | `hamming_weights=` |
| Candidate barcodes per gene | 40 (the default is 1,000) | `candidates_per_gene=` |
| A binary noise channel | the error rates measured from the data of Zhang et al. 2023 | `duet.channels.asymmetric(..., alphabet="01")` |
| Optional: a codebook to start from, and codebooks to compare with | the 16-bit codebooks of Boström et al. 2025, of Hamming weight 4 and 5 | `init=`, `include=` |

```python
import numpy as np
import pandas as pd
import duet

panel = pd.read_csv("examples/data/bostrom/bostrom_hw4_16bit_panel.csv",
                    dtype={"Sequence": str}).head(40)          # barcodes as text
rng = np.random.default_rng(0)
genes = pd.DataFrame({
    "gene": panel["Gene"],
    "expression": rng.lognormal(mean=3.0, sigma=1.5, size=len(panel)),  # SYNTHETIC
})
print(genes.head(3).round(1).to_string(index=False))
```

```text
 gene  expression
Rn45s        24.3
 Actb        16.5
Tubb5        52.5
```

- **Genes.** Row *i* of the gene table gets codeword position *i*. Each gene appears once.
- **Expression.** Finite and non-negative, proportional to transcript abundance (not
  log-transformed). The unit does not matter.
- **Candidate barcodes.** DUET enumerates every `n_bits`-bit barcode of each allowed weight, keeps
  a random `cap` of them where a cap is set, and draws `candidates_per_gene` candidates for each
  gene; genes can share candidates. Barcodes given through `init` and `include` always stay in
  their gene's candidate set (and count toward the caps). The cost grows with U², the number of
  distinct candidate barcodes squared.
- **Barcodes are text.** Read barcode columns with `dtype=str`, or pandas drops their leading
  zeros.

## Noise channel

MERFISH errors are asymmetric: a 1 read as 0 (a missed spot) is more common than a 0 read as 1.

```python
channel = duet.channels.asymmetric([[0.985, 0.015],    # sent 0: read 0, read 1
                                    [0.056, 0.944]],   # sent 1: read 0, read 1
                                   alphabet="01")
```

The matrix is `T[sent, read]`, in the order of `alphabet`, which must be `"01"`. The rates are
the readout error rates estimated from the Zhang et al. 2023 data and used for the paper's
designs. The unrounded estimate ships with the repository;
[Channels in the repository](noise_channels.md#channels-in-the-repository) says how it was
fitted.

## Warm start and reference codebooks

- `init=` a table with the gene column and a barcode column (named by `barcode=`), one barcode per
  panel gene, starts the search from that codebook. It also sets the zero of the crowding
  objective.
- `include=` a dict `{name: table}` adds codebooks that are evaluated alongside the designs, one row
  each. Only a reference that covers exactly the panel genes compares directly with the designs.

The tutorial starts from the Boström et al. weight-4 codebook and compares with their weight-5
codebook for the same genes:

```python
initial = pd.DataFrame({"gene": panel["Gene"], "barcode": panel["Sequence"]})
hw5 = pd.read_csv("examples/data/bostrom/bostrom_hw5_16bit_panel.csv",
                  dtype={"Sequence": str}).head(40)
references = {"bostrom_hw5": pd.DataFrame({"gene": hw5["Gene"], "barcode": hw5["Sequence"]})}
```

## Running the λ sweep

One call builds the candidate pool, runs the λ sweep (by default the paper's nine values, 0, 0.1,
0.3, 0.5, 0.7, 0.8, 0.9, 0.95 and 1) and evaluates every codebook on fresh simulated reads:

```python
res = duet.design_merfish(
    genes, channel,
    n_bits=16,
    hamming_weights={4: None, 5: 1000},  # all weight-4 barcodes, 1,000 of weight 5
    candidates_per_gene=40,              # default 1,000; lowered to fit a CPU budget
    init=initial,
    include=references,
    seed=0,
    verbose=False,
)
print("unique candidate barcodes:", res.provenance["pool"]["unique_codewords"])
```

```text
unique candidate barcodes: 1221
```

This call takes nearly all of the page's run time, on GPUs when there are any. **On a GPU, or on
another CPU model, the numbers below can differ**, and some codebooks too: the binary channel's
likelihood costs tie often, and the devices break ties differently (see
[Reproducibility](design.md#reproducibility)).

## Reading the results

```python
cols = ["codebook", "accuracy_mean", "accuracy_p10", "crowding", "duplicate_codewords",
        "on_pareto_front"]
print(res.table[cols].round(4).to_string(index=False))
```

```text
   codebook  accuracy_mean  accuracy_p10  crowding  duplicate_codewords  on_pareto_front
    initial         0.9632        0.9553    0.0000                    0            False
   lambda=0         0.9374        0.8755    0.2488                    0             True
 lambda=0.1         0.9632        0.9496    0.2482                    0             True
 lambda=0.3         0.9656        0.9609    0.2460                    0             True
 lambda=0.5         0.9727        0.9641    0.2404                    0             True
 lambda=0.7         0.9796        0.9666    0.2232                    0             True
 lambda=0.8         0.9824        0.9716    0.2093                    0             True
 lambda=0.9         0.9865        0.9769    0.1913                    0             True
lambda=0.95         0.9870        0.9818    0.1560                    0             True
   lambda=1         0.9894        0.9837   -0.1410                    0             True
bostrom_hw5         0.9737        0.9662   -0.2439                    0            False
```

- **`crowding`** is 0 for the starting codebook, higher when a codebook is less crowded and
  negative when it is more crowded. The weight-5 reference lights up five images per gene instead
  of four, so it carries more load.
- **`duplicate_codewords`** counts genes that share a barcode and cannot be decoded. At λ = 0
  decoding has no weight, so check this column there.
- **Choosing a codebook.** `operating_point()` is for OPS designs only. Choose from
  `res.pareto_front()` with a rule that fits your experiment, for example the most accurate
  codebook that keeps the crowding objective at 0.2 or more:

```python
front = res.pareto_front()
choice = front[front["crowding"] >= 0.2].sort_values("accuracy_mean").iloc[-1]
codebook = res.codebook(choice["lambda"])   # gene, barcode, expression, accuracy
res.save("my_merfish_design")
codebook.to_csv("my_merfish_design/codebook.csv", index=False)
print(choice["codebook"])
print(codebook.head(3).round(3).to_string(index=False))
```

```text
lambda=0.8
 gene          barcode  expression  accuracy
Rn45s 1000010001001000      24.254     0.979
 Actb 0000001011000010      16.475     0.983
Tubb5 0000001100110000      52.491     0.977
```

## The crowding objective

For a codebook $S$, with $e_i$ the expression of gene $i$ and $b_{i,r} = 1$ when gene $i$'s
barcode is 1 in bit $r$:

$$T_r(S) = \sum_i e_i \, b_{i,r}, \qquad C(S) = \sum_r T_r(S)^2, \qquad \text{crowding} = 1 - \frac{C(S)}{C(S_0)}.$$

$T_r$ is the expression-weighted load of bit $r$. $C$ penalizes both a high total load and an
uneven one, so with mixed Hamming weights the search tends to give low-weight barcodes to
abundant genes. The objective is anchored at the starting codebook $S_0$; to compare codebooks
from runs with different starting codebooks, use
$C(S)$ = `res.provenance["merfish"]["crowding_anchor_C_S0"] * (1 - crowding)`.

$C$ is a proxy that the search can update quickly. The paper also evaluates optical crowding
with the spot simulation of Boström et al. (2025), whose resolved fraction `design_merfish` does
not compute; the MERFISH experiments in `experiments/` do
([Reproducing the paper](reproducing_the_paper.md#experiments)). The crowding objective uses one
expression value per gene; `experiments/merfish_expression_prior/` evaluates designs under
cell-type-specific profiles.

## Settings used in the paper

The paper designed 32-bit codebooks for a 1,147-gene panel (Zhang et al. 2023, starting from their
published codebook) and a 2,000-gene panel (starting from a Boström et al. weight-5 codebook),
with expression from the whole-mouse-brain atlas of Yao et al. (2023):

| Setting | Paper | Default |
|---|---|---|
| `n_bits` | 32 | (required) |
| `hamming_weights` | `{4: None, 5: 64040}` (100,000 barcodes) | (required) |
| `candidates_per_gene` | 1,000 | 1,000 |
| `num_samples` | 30,000 | 10,000 |
| `eval_samples` | 5,000 | 5,000 |
| `max_patience` | 500 | 3,000 |
| `avoid_duplicates` | `False` | `True` |
| `lambdas` | 0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1 | the same |
| channel | `merfish_zhang2023_channel.npy`, unrounded | (required) |

With these settings on your own panel `genes` (and, optionally, a published codebook `published`
for it):

<!-- docs-test: skip (needs GPUs and hours; about 20 GB PEP) -->
```python
channel = duet.channels.asymmetric(
    "scripts/benchmark/noise_model_matrices/channels/merfish_zhang2023_channel.npy",
    alphabet="01")
res = duet.design_merfish(
    genes, channel,
    n_bits=32, hamming_weights={4: None, 5: 64040}, candidates_per_gene=1000,
    init=published,                       # or "random"
    num_samples=30_000, eval_samples=5_000, max_patience=500, avoid_duplicates=False,
    cache_dir="merfish_pep_cache",        # on a disk with at least 60 GB free
    workers=9, seed=42,
)
```

With a thousand or more genes almost every barcode of the pool becomes a candidate, so U is about
100,000: the PEP is 20 GB, and building it needs 60 GB of free disk (40 GB once built). Each of
the paper's MERFISH experiments took about 3 to 6 hours on 4 GPUs, including the baselines and
the crowding simulation. With `cache_dir` and a fixed `workers`, later runs that change only λ,
the optimizer settings or the expression values reuse the PEP. `design_merfish` reproduces the
paper's designs statistically; the exact runs are in `experiments/`.
