# Designing a codebook

This guide covers the DUET workflow for any barcode assay: describing the candidates, improving
a codebook for decoding alone, adding an experiment-specific objective, reading the results and
scoring codebooks you already have. The two worked applications have their own pages,
[OPS libraries](design_ops.md) and [MERFISH codebooks](design_merfish.md). The method is
described in the paper.

The Python blocks on this page run top to bottom in one session, in about a minute on CPU.

## Inputs

| Input | Meaning | In the API |
|---|---|---|
| Candidate set | The candidates for one target: the sgRNAs of a gene, the candidate tags of an antibody, a set of blank barcodes or controls | the `group` column of the candidate table |
| Quota | How many candidates to select from each set (exactly) | `quota=`: one number, a dict per set, or a column |
| Codeword | The symbols that are read out, all of one length | the `sequence` column, cut to its first `seq_length` symbols |
| Alphabet | DNA (`"ACGT"`, the bases in any order) or binary (`"01"`), uppercase | `alphabet=` of `from_table` and of the channel |
| Noise channel | How the readout corrupts each symbol | [`duet.channels`](noise_channels.md) |
| Decoding rule | Unique minimum (the default) or a margin decoder | `rule=` |
| Objective (optional) | None for decoding alone (`duet.design`); a score per candidate, whose mean the design keeps high (`duet.design_ops`); or MERFISH optical crowding | the `score` column; [`duet.design_merfish`](design_merfish.md) |

`duet.design` and `duet.design_ops` take any pool built with `CandidatePool.from_table`,
whatever the assay; the name `design_ops` refers to its first application. The examples on this
page use a synthetic panel of 24 DNA-barcoded antibodies read by 6 cycles of in situ sequencing.
Each antibody has 6 candidate oligo tags with a quality score, and 4 blank barcodes are picked
from 12 candidates.

```python
import numpy as np
import pandas as pd
import duet

rng = np.random.default_rng(7)
rows = [{"target": f"Ab{a:02d}", "tag": "".join(rng.choice(list("ACGT"), size=12)),
         "quality": round(float(rng.uniform(0.3, 1.0)), 3)}
        for a in range(24) for _ in range(6)]
rows += [{"target": "blank", "tag": "".join(rng.choice(list("ACGT"), size=12)), "quality": 1.0}
         for _ in range(12)]
tags = pd.DataFrame(rows)

quota = {t: 1 for t in tags["target"].unique()}
quota["blank"] = 4
pool = duet.CandidatePool.from_table(tags, group="target", sequence="tag",
                                     score="quality", quota=quota, seq_length=6)
print(pool.describe(num_samples=1_000))
```

```text
Candidate pool
  groups                  25
  candidates              156
  codebook size           28
  unique codewords (U)    150
  duplicate candidates    6 (share a codeword with an earlier candidate)
  sequence length         6
  swaps per iteration     164
  PEP reads               150,000 (1,000 per unique codeword)
  PEP in memory           45.0 kB (U x U uint16)
  PEP on disk if cached   90.0 kB (raw and symmetric copies)
```

`describe()` sizes the problem before anything runs. The cost of a design is set by U, the number
of unique candidate codewords: DUET simulates reads for each of them and builds a U x U matrix of
pairwise error probabilities (PEP); see [GPUs and scaling](gpu_and_scaling.md#memory-and-disk).

- **Shared codewords.** Candidates with the same codeword (here 6, after cutting the tags to 6
  bases) cannot be told apart, and for λ > 0 the search keeps them out of one codebook. A codeword may
  also be listed in several sets, once per set; DUET then decides which set gets it.
- **Controls and blank barcodes** form an ordinary set with its own quota. Give them one constant
  score (the paper uses 1.0).
- **Row order** seeds the simulated reads. Keep it stable between runs.

## Improving a codebook for decoding

`duet.design` optimizes decoding accuracy alone. It ignores scores, so a pool built without a
`score` column works the same. This picks 12 barcodes out of 200 random 6-mers for a sequencing
error of 10% per base:

```python
barcodes = pd.DataFrame({"set": "all",
                         "barcode": ["".join(rng.choice(list("ACGT"), size=6)) for _ in range(200)]})
pool_dec = duet.CandidatePool.from_table(barcodes, group="set", sequence="barcode", quota=12)
res_dec = duet.design(pool_dec, duet.channels.symmetric(0.1),
                      seed=0, num_samples=1_000, eval_samples=1_000, verbose=False)
print(res_dec.table[["codebook", "accuracy_mean", "accuracy_p10"]].to_string(index=False))
```

```text
codebook  accuracy_mean  accuracy_p10
 initial        0.95700        0.9453
designed        0.97325        0.9711
```

`initial` is the random codebook the search started from and `designed` the result;
`res_dec.codebook()` returns it, one row per entry. `duet.design` is the λ = 1 end of the
trade-off described below: with the same seed and settings, `duet.design_ops(..., lambdas=[1])`
selects the same codebook. A decoding-only result has no objective column and no Pareto front,
so `pareto_front()`, `operating_point()` and `plot()` do not apply.

**Start from your own codebook.** `init=` takes a codebook table with the pool's group and
sequence columns (or a `candidate` column of row positions) and as many rows per set as its
quota. It becomes the `initial` row, so the result compares your codebook with DUET's. The
search keeps the best codebook it finds, so the result is never worse than the starting codebook
on DUET's decoding objective (`surrogate_accuracy`).

**Keep candidates fixed.** DUET only swaps a candidate for another of the same set. To keep
candidates, put them in a set of their own whose quota equals its size. Here 8 barcodes are added
to 4 already in use:

```python
extend = barcodes.assign(set=["in_use"] * 4 + ["new"] * 196)
pool_ext = duet.CandidatePool.from_table(extend, group="set", sequence="barcode",
                                         quota={"in_use": 4, "new": 8})
res_ext = duet.design(pool_ext, duet.channels.symmetric(0.1),
                      seed=0, num_samples=1_000, eval_samples=1_000, verbose=False)
book = res_ext.codebook()
print(book[book["group"] == "in_use"][["sequence", "accuracy", "candidate"]].to_string(index=False))
```

```text
sequence  accuracy  candidate
  CGCATC     0.975          1
  CAAGTA     0.976          2
  GCGCTG     0.968          3
  TTAAAT     0.967          0
```

The four barcodes in use (candidates 0 to 3) stay; DUET chose the other 8 to decode well with them.

## Adding an experiment-specific objective

With a score, each λ codebook maximizes λ times DUET's decoding objective plus (1 - λ) times the
mean score of the selected candidates. λ = 1 optimizes decoding alone (the codebook
`duet.design` returns) and λ = 0 the score alone. The two terms are not rescaled, so:

- **Scores of order 1** (about 0 to 1) suit the default λ values, 0, 0.05, 0.1, 0.25, 0.5 and 1.
  Rescale other scores to about [0, 1], by min-max scaling or percentile ranks: in z-score units
  the score decides every swap until λ is close to 1. Several criteria can be combined into one
  score column.
- **A constant added to every score** changes no codebook, but it moves the threshold of
  `operating_point()`, which assumes positive scores.
- **Refine the sweep.** Run the default λ values, then add values where the front bends. With
  `cache_dir=`, later runs reuse the PEP ([Caching the PEP](gpu_and_scaling.md#caching-the-pep)).

The antibody panel, under a substitution matrix measured for its chemistry and with a margin
decoder that assigns a read only when its codeword is more than $e^2$ times as likely as any
other ([Decoding rules](noise_channels.md#decoding-rules)):

```python
T = pd.DataFrame([[0.93, 0.01, 0.05, 0.01],     # rows: sent base A, C, G, T
                  [0.01, 0.95, 0.01, 0.03],     # columns: read base A, C, G, T
                  [0.07, 0.01, 0.91, 0.01],
                  [0.01, 0.04, 0.01, 0.94]],
                 index=list("ACGT"), columns=list("ACGT"))
channel = duet.channels.asymmetric(T)
res = duet.design_ops(pool, channel, lambdas=[0, 0.25, 0.5, 1], rule=duet.MarginDecoding(2.0),
                      seed=0, num_samples=1_000, eval_samples=1_000, verbose=False)
cols = ["codebook", "accuracy_mean", "accuracy_p10", "mean_score", "duplicate_codewords",
        "on_pareto_front"]
print(res.table[cols].to_string(index=False))
```

```text
   codebook  accuracy_mean  accuracy_p10  mean_score  duplicate_codewords  on_pareto_front
    initial       0.924679        0.8469    0.754107                    0            False
   lambda=0       0.936786        0.8741    0.911464                    0             True
lambda=0.25       0.957393        0.9417    0.908643                    0             True
 lambda=0.5       0.965964        0.9497    0.905071                    0             True
   lambda=1       0.975036        0.9670    0.711393                    0             True
  max_score       0.942250        0.8994    0.911464                    0            False
```

## Reading the results

`res.table` has one row per codebook: `initial`, one `lambda=<value>` row per λ, then reference
codebooks: `max_score` (the highest-scoring candidates of each set) for `design_ops`, the
`include=` codebooks for `design_merfish`. A `duet.design` result has two rows, `initial` and
`designed`, and no `lambda`, objective or `on_pareto_front` column.

| Column | Meaning |
|---|---|
| `accuracy_mean` | Decoding accuracy: the fraction of simulated reads assigned to the right entry, averaged over the codebook's entries, from fresh reads (`eval_samples` per codeword). The paper reports this accuracy. |
| `accuracy_p10` | 10th percentile of the per-entry accuracy |
| `accuracy_p95_p5_ratio` | 95th over 5th percentile of the per-entry accuracy (1 is perfectly even) |
| `surrogate_accuracy` | The decoding objective the search maximizes, from the design reads. Not an accuracy: it is lower, and can be negative. |
| `mean_score` or `crowding` | The experiment-specific objective (not in `duet.design` results) |
| `duplicate_codewords` | Entries whose codeword occurs more than once (they cannot be decoded). Check it at λ = 0, where decoding has no weight. |
| `on_pareto_front` | No other λ codebook is at least as good on both `accuracy_mean` and the objective and better on one. Reference rows are not part of the comparison. Not in `duet.design` results. |

```python
op = res.operating_point()           # design_ops only: the paper's OPS rule
print(op["codebook"])
print(res.codebook(op["lambda"]).head(3).to_string(index=False))
res.save("my_design")                # table.csv, codebooks.csv, settings.json
print(duet.load("my_design").table.equals(res.table))
```

```text
lambda=0.5
group sequence  score  accuracy  candidate
 Ab00   TATGAA  0.942     0.971          3
 Ab01   AGACCT  0.961     0.984         10
 Ab02   TATACT  0.911     0.959         14
True
```

- `res.codebook(...)` takes a λ value or a row label and returns that codebook, one row per entry,
  with its per-entry `accuracy`. With no argument it returns the designed codebook of a
  `duet.design` result. Its `candidate` column is the row position in the table you gave
  `from_table`: `tags.iloc[book["candidate"]]` returns the full rows.
- `res.operating_point(score_fraction=0.975)` returns the most accurate λ codebook that keeps
  97.5% of the `max_score` mean score, the rule the paper used for OPS. Choosing from the front is
  otherwise up to you: `res.pareto_front()` returns its rows and `res.plot()` draws it.
- `res.save` and `duet.load` keep every codebook and the settings; [Saved
  results](api.md#saved-results) describes the files.

## Evaluating existing codebooks

`duet.evaluate` scores codebooks you already have, such as a published library, the panel you
use now or the codebooks of a saved result, under any channel. Its `num_samples` is the number of
reads per codeword, which the design functions call `eval_samples`.

```python
books = {label: res.codebook(label) for label in ["max_score", "lambda=0.5"]}
ev = duet.evaluate(books, duet.channels.symmetric(0.1), num_samples=2_000, seed=0)
print(ev.summary[["codebook", "accuracy_mean", "accuracy_p10"]].to_string(index=False))
```

```text
  codebook  accuracy_mean  accuracy_p10
 max_score       0.892643       0.83025
lambda=0.5       0.919964       0.89380
```

A codebook is a table with a sequence column, as here, or a list of codewords. To judge every
codebook of a design under a second channel, pass `eval_channel=` to the design function instead
([Evaluating under another channel](noise_channels.md#evaluating-under-another-channel)).

## Reproducibility

- **Seeds.** One `seed` gives each stage (pool, initial codebook, PEP, evaluation, search) its
  own seed. `res.provenance`, saved as `settings.json`, records them with every setting and the
  DUET, Python and NumPy versions. With `seed=None` DUET draws a seed and records it.
- **Identical results** come from the same inputs, settings and seed on the same device and
  machine. Re-sorting the candidate table changes the simulated reads, and the accuracy of a
  codebook depends slightly, within Monte Carlo error, on the other codebooks evaluated with it.
- **CPU, GPU and `workers`.** With the symmetric channel (Hamming decoding) CPU and GPU agree
  exactly. The other channels decode by negative log-likelihood, which the devices compute with
  different floating-point arithmetic, so reads whose costs tie or nearly tie can be decoded
  differently. This is rare for DNA channels and frequent for binary MERFISH channels, where
  accuracies, and sometimes codebooks, can differ by more than the Monte Carlo error
  ([measurements](../experiments/README.md#numerical-determinism)). For the same reason, on CPU,
  `workers=1` can differ from `workers` of 2 or more, and one CPU model from another. Compare
  such numbers only between runs on the same device and machine.
- **The paper.** The public API reproduces the paper's designs statistically. The paper's numbers
  come from the scripts in `experiments/` ([Reproducing the paper](reproducing_the_paper.md)).

## Terms

| Paper | Code |
|---|---|
| candidate set $C_i$, number selected $k_i$ | `group`, `quota` |
| codeword, codeword length $n$ | `sequence` (cut to `seq_length`), `barcode`; `seq_length`, `n_bits` |
| noise channel, decoding rule | `channel` and `eval_channel`, `rule` |
| pairwise error probability (PEP) | computed internally; kept on disk with `cache_dir` |
| reads per codeword for the PEP and for the evaluation | `num_samples`, `eval_samples` (`num_samples` in `duet.evaluate`) |
| union-bound decoding objective | `surrogate_accuracy` |
| decoding accuracy | `accuracy_mean`; per entry, the `accuracy` column of `res.codebook(...)` |
| bottom-decile accuracy | not a column, and not `accuracy_p10`: `acc[acc <= acc.quantile(0.1)].mean()` with `acc = res.codebook(...)["accuracy"]` |
| experiment-specific (secondary) objective | `mean_score`, `crowding` |
| trade-off weight λ | `lambdas`; the rows `lambda=<value>` |
| decoding-only design | `duet.design`; the `designed` row |
| initial codebook | `init`; the `initial` row |
| maximum-activity selection | the `max_score` row |
| operating point | `res.operating_point()` |

## Limits

- **Alphabets and channels.** DNA or binary codewords of one length. Channels are square (the
  read alphabet equals the sent one) and model substitutions, independent across positions: no
  insertions, deletions or phasing.
- **Objectives.** The public API offers decoding alone (`duet.design`), decoding plus the mean
  per-candidate score (`duet.design_ops`), and decoding plus MERFISH crowding
  (`duet.design_merfish`). Other objectives need the provisional
  [advanced layer](api.md#advanced-layer) and internal code; please
  [open an issue](https://github.com/AltschulerWu-Lab/DUET/issues) to describe yours.
- **Quotas are exact**, and `num_samples` and `eval_samples` are at most 32,767.
- **Local search.** Each λ gives the best codebook the search finds, not a guaranteed optimum,
  and the sweep visits only the λ values you give.
- **Cost** grows with U², the square of the number of unique candidate codewords.
- **The gain depends on the errors.** It is largest when decoding errors are common (short reads,
  noisy chemistry), and every estimate is only as good as the channel: estimate it from your own
  reads where you can ([Estimating a channel](noise_channels.md#estimating-a-channel-from-your-reads)).
