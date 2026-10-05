# Designing an OPS library

This tutorial designs an sgRNA library for an optical pooled screen (OPS), from a table of
candidate sgRNAs to a CSV of the chosen ones, on a small table bundled with the repository. In
situ sequencing reads the first n bases of each spacer, so those bases are the sgRNA's codeword.
DUET chooses the sgRNAs of each gene so that their codewords decode accurately while their
predicted activity stays high. [Designing a codebook](design.md) explains the general workflow
and the result table.

The Python blocks run in order in one session, in about 30 s on 4 CPU cores, and write to
`my_library/`.

## Candidate table

| You need | In this tutorial |
|---|---|
| One row per candidate sgRNA: its gene, its spacer and a predicted activity | `examples/data/quickstart_ops.csv`: 10 sgRNAs for each of 50 genes of the hCRISPRi-v2.1 library (Horlbeck et al. 2016), and 50 non-targeting controls |
| How many sgRNAs to keep per gene, and how many controls | 2 per gene, 10 controls |
| The number of sequencing rounds | 10 |
| A noise channel | a 10% error per base, spread evenly over the other bases |

```python
import pandas as pd
import duet

table = pd.read_csv("examples/data/quickstart_ops.csv")
print(table.head(3).to_string())

quota = {gene: 10 if gene == "negative_control" else 2 for gene in table["gene"].unique()}
pool = duet.CandidatePool.from_table(
    table,
    group="gene",         # candidate set
    sequence="sequence",  # spacer, 5' to 3'
    score="activity",     # predicted activity
    quota=quota,          # 2 per gene, 10 controls
    seq_length=10,        # sequencing rounds: keep the first 10 bases
)
print(pool.describe())
```

```text
                       sgID   gene gene_name             sequence  activity
0  ABCB5_+_20687051.23-P1P2  ABCB5     ABCB5  GCTCACAGAGAGATATTAG  0.781216
1  ABCB5_+_20687242.23-P1P2  ABCB5     ABCB5  ATAACCTGGAAAATATGAA  0.510780
2  ABCB5_+_20687218.23-P1P2  ABCB5     ABCB5  GCTCCTCGGGCTATTGCGA  0.463202
Candidate pool
  groups                  51
  candidates              550
  codebook size           110
  unique codewords (U)    550
  duplicate candidates    0 (share a codeword with an earlier candidate)
  sequence length         10
  swaps per iteration     1,390
  PEP reads               5,500,000 (10,000 per unique codeword)
  PEP in memory           605.0 kB (U x U uint16)
  PEP on disk if cached   1.2 MB (raw and symmetric copies)
```

- **Spacers** are uppercase A, C, G and T, written in the order they are read; DUET keeps the
  first `seq_length` bases.
- **Activity scores** should be of order 1 (here 0.25 to 1); see
  [Adding an experiment-specific objective](design.md#adding-an-experiment-specific-objective).
  Every row needs a finite score. Give non-targeting controls 1.0, as the paper does, so that
  decoding alone chooses among them.
- **Shared codewords.** With fewer rounds, different sgRNAs can share their first n bases.
  `describe()` counts them as duplicate candidates; if there are many, read more rounds.
- **Row order** seeds the simulated reads, so keep it stable between runs.

## Running the λ sweep

One call runs the λ sweep (by default 0, 0.05, 0.1, 0.25, 0.5 and 1) and evaluates every library
on fresh simulated reads. `cache_dir` keeps the PEP for later runs.

```python
channel = duet.channels.symmetric(0.1)
res = duet.design_ops(pool, channel, seed=0, cache_dir="my_library/pep_cache", verbose=False)
cols = ["codebook", "accuracy_mean", "accuracy_p10", "mean_score", "on_pareto_front"]
print(res.table[cols].round(4).to_string(index=False))
```

```text
   codebook  accuracy_mean  accuracy_p10  mean_score  on_pareto_front
    initial         0.9501        0.9187      0.7169            False
   lambda=0         0.9449        0.8959      0.8447             True
lambda=0.05         0.9495        0.9032      0.8447             True
 lambda=0.1         0.9511        0.9040      0.8444             True
lambda=0.25         0.9543        0.9252      0.8433             True
 lambda=0.5         0.9614        0.9453      0.8355             True
   lambda=1         0.9726        0.9670      0.6907             True
  max_score         0.9456        0.8989      0.8447            False
```

`initial` is the random library the search started from and `max_score` the most active sgRNAs
of each gene. On a GPU machine, `device="auto"` (the default) uses every GPU; here it fell back
to the CPU. For your own sequencing chemistry, estimate a channel from reads of known sgRNAs
([Noise channels](noise_channels.md#estimating-a-channel-from-your-reads)); error rates usually
rise over the rounds, which `duet.channels.position_varying_asymmetric` captures.

You can only choose among the λ values you ran. Here the mean score falls between λ = 0.5 and 1,
so add values there. The PEP now comes from the cache:

```python
fine = duet.design_ops(pool, channel, lambdas=[0, 0.05, 0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                       seed=0, cache_dir="my_library/pep_cache", verbose=False)
print("PEP from:", fine.provenance["pep"]["source"])
print(fine.table[fine.table["lambda"] >= 0.5][cols].round(4).to_string(index=False))
```

```text
PEP from: cache
  codebook  accuracy_mean  accuracy_p10  mean_score  on_pareto_front
lambda=0.5         0.9614        0.9453      0.8355             True
lambda=0.6         0.9645        0.9488      0.8288             True
lambda=0.7         0.9671        0.9565      0.8194             True
lambda=0.8         0.9695        0.9610      0.8023             True
lambda=0.9         0.9720        0.9642      0.7621             True
  lambda=1         0.9727        0.9662      0.6907             True
```

## Choosing and exporting a library

The paper's rule for OPS takes the most accurate library that keeps at least 97.5% of the
maximum mean activity. `operating_point` applies it; lower `score_fraction` to trade more
activity for accuracy. `codebook(...)["candidate"]` gives the row positions of the chosen sgRNAs
in your table:

```python
op = fine.operating_point(score_fraction=0.975)
chosen = fine.codebook(op["lambda"])
library = table.iloc[chosen["candidate"]].reset_index(drop=True)
library["predicted_accuracy"] = chosen["accuracy"]
library.to_csv("my_library/library.csv", index=False)
fine.save("my_library/design")
print(op["codebook"], "|", len(library), "sgRNAs")
print(library[["sgID", "sequence", "activity", "predicted_accuracy"]].head(3).to_string())
```

```text
lambda=0.6 | 110 sgRNAs
                       sgID             sequence  activity  predicted_accuracy
0  ABCB5_+_20687242.23-P1P2  ATAACCTGGAAAATATGAA  0.510780              0.9732
1  ABCB5_+_20687051.23-P1P2  GCTCACAGAGAGATATTAG  0.781216              0.9460
2  ABHD8_-_17414169.23-P1P2  AGGCCGGGTGCGTCTACGC  0.795083              0.9564
```

`predicted_accuracy` is the simulated decoding accuracy of each sgRNA under your channel, not a
measurement. `duet.load("my_library/design")` reads the saved design back, and
`init=` a saved library starts a later design from it.

## Genome-wide libraries

The cost of a design is set by U, the number of unique candidate codewords: the PEP takes time
proportional to U² x `num_samples` and 2U² bytes. Build the pool and call `describe()` before you
run anything. This pool has the size of the paper's medium-scale CRISPRi benchmark: 1,000 genes of
hCRISPRi-v2.1 with their 10 top-ranked sgRNAs and all 1,895 non-targeting controls, selecting 2
per gene and 200 controls.

```python
import numpy as np

src = pd.read_csv("data/processed/Horlbeck_2016/CRISPRi_v2_1.csv", low_memory=False)
src = src[(src["Gene"] == "negative_control") | (src["Rank"] <= 10)].reset_index(drop=True)
is_control = src["Gene"] == "negative_control"
genes = np.random.default_rng(0).choice(src.loc[~is_control, "Gene"].unique(), size=1000, replace=False)
big = src[src["Gene"].isin(genes) | is_control].copy()
big.loc[big["Gene"] == "negative_control", "Activity score"] = 1.0
big_pool = duet.CandidatePool.from_table(
    big, group="Gene", sequence="Sequence", score="Activity score",
    quota={g: 200 if g == "negative_control" else 2 for g in big["Gene"].unique()},
    seq_length=10,
)
d = big_pool.describe(num_samples=5_000)
print(f"U = {d.unique_codewords:,}; PEP {d.pep_memory_bytes / 1e6:.0f} MB")
```

```text
U = 11,628; PEP 270 MB
```

| Problem | U | PEP (2U²) | Time with default settings |
|---|---|---|---|
| This tutorial | 550 | 0.6 MB | 15 to 20 s on 4 CPU cores |
| Medium-scale CRISPRi benchmark | 11,628 | 270 MB | 101 min on 40 CPU threads, 3 min on 4 GPUs |
| The paper's genome-wide CRISPick library | 289,113 | 167 GB | see [Reproducing the paper](reproducing_the_paper.md#hardware-and-time) |

For large libraries, use GPUs, put `cache_dir` on a disk with 6U² bytes free, pass `workers`
explicitly (it is part of the PEP cache key), and try a few hundred genes or fewer λ values
first. [GPUs and scaling](gpu_and_scaling.md) gives memory, disk and runtime estimates.

## Settings used in the paper

The paper's medium-scale CRISPRi benchmark used the pool above, the symmetric channel with
ε = 0.1 and unique-minimum decoding. It differs from the `design_ops` defaults in its sample
counts and λ values:

| Setting | Paper | Default |
|---|---|---|
| `num_samples` (PEP reads per unique codeword) | 5,000 | 10,000 |
| `eval_samples` (evaluation reads per codeword) | 2,000 | 5,000 |
| `lambdas` | 18 values, dense below 0.25 | 6 values |
| `init`, `max_patience`, `avoid_duplicates` | random, 3,000, True | the same |
| Operating point | `score_fraction=0.975` | the same |
| Repeats | 5 trials, each with its own 1,000 genes | |

<!-- docs-test: skip (U = 11,628: needs a GPU, or over an hour on CPU) -->
```python
paper_lambdas = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 1]
res_paper = duet.design_ops(big_pool, duet.channels.symmetric(0.1), lambdas=paper_lambdas,
                            num_samples=5_000, eval_samples=2_000, seed=0,
                            cache_dir="my_library/paper_pep_cache")
print(res_paper.operating_point(score_fraction=0.975))
```

This reproduces the paper's design statistically. The paper's numbers come from
`experiments/ops_crispri_symmetric/`, which draws its own genes and seeds for each trial and
evaluates DUET together with the baseline methods ([Reproducing the paper](reproducing_the_paper.md)).
The genome-wide library (3 sgRNAs for each of 20,114 genes and 1,000 controls, from 305,202
CRISPick candidates, read over 14 rounds) is in `experiments/ops_crispick_genome_wide/`. Its
CRISPick candidate table is not in the repository; you build it from CRISPick downloads
([CRISPick table](reproducing_the_paper.md#crispick-table)).
