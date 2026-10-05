# API reference

The public names of the `duet` package: signatures, parameters, return values and saved files.
For workflows, see [Designing a codebook](design.md) and the guides for [OPS](design_ops.md)
and [MERFISH](design_merfish.md). The runnable blocks form one Python session on toy data.

## Stability

| Tier | Names | Promise |
|---|---|---|
| Public | The names on this page, except the advanced layer and the four `DesignResult` attributes marked provisional | Changes follow semantic versioning |
| Advanced, provisional | The [Advanced layer](#advanced-layer) | May change in a minor version before 1.0 |
| Internal | Everything else, including names that start with an underscore | None |

Tiers: [ADR 0003](adr/0003-public-api.md). Arguments after the first one or two are keyword-only.
λ = 1 is decoding only, λ = 0 the other objective only ([ADR 0001](adr/0001-lambda-weights-decoding-accuracy.md)).

## CandidatePool

<!-- docs-test: skip (signature) -->
```python
duet.CandidatePool.from_table(df, *, group, sequence, score=None, quota=1,
                              seq_length=None, alphabet="ACGT") -> CandidatePool
pool.describe(num_samples=10_000) -> duet.PoolDescription
```

| Parameter | Default | Meaning |
|---|---|---|
| `df` | | DataFrame, one row per candidate. Row order seeds the Monte Carlo reads: keep it fixed between runs. |
| `group` | required | Column naming each candidate's candidate set (a gene, a target, the controls). |
| `sequence` | required | Column with each candidate's sequence: uppercase symbols of `alphabet`. |
| `score` | None | Column with a finite per-candidate score, controls included. None scores every candidate 1.0, so only decoding matters. |
| `quota` | 1 | Candidates to select per set: an int, a dict naming every set, or a column constant within each set. 0 leaves a set out. |
| `seq_length` | None | Keep the first `seq_length` symbols as the codeword (for example the number of sequencing rounds). Without it, all sequences must have one length. |
| `alphabet` | `"ACGT"` | The four DNA bases in any order, or `"01"` for binary barcodes. |

Candidates that share a codeword stay separate and share one row of the pairwise error
probability (PEP) matrix. To keep a candidate in every codebook, put it in a set whose quota
equals its size ([Inputs](design.md#inputs)).

`describe` computes nothing; it reports the size of the problem. `duet.PoolDescription` has the
fields `groups`, `candidates`, `codebook_size`, `unique_codewords` (U), `duplicate_candidates`,
`seq_length`, `swaps_per_iteration`, `num_samples`, `pep_reads` (U x `num_samples`),
`pep_memory_bytes` (2U²) and `pep_disk_bytes` (4U², a memory-mapped PEP), plus `to_dict()`.
[Memory and disk](gpu_and_scaling.md#memory-and-disk) turns them into time and space.

```python
import pandas as pd
import duet

table = pd.read_csv("examples/data/quickstart_ops.csv")
genes = list(table["gene"].unique()[:8]) + ["negative_control"]
library = table[table["gene"].isin(genes)].reset_index(drop=True)
quota = {g: (4 if g == "negative_control" else 2) for g in genes}
pool = duet.CandidatePool.from_table(library, group="gene", sequence="sequence",
                                     score="activity", quota=quota, seq_length=10)
print(pool.describe(num_samples=500))
```

```text
Candidate pool
  groups                  9
  candidates              130
  codebook size           20
  unique codewords (U)    130
  duplicate candidates    0 (share a codeword with an earlier candidate)
  sequence length         10
  swaps per iteration     340
  PEP reads               65,000 (500 per unique codeword)
  PEP in memory           33.8 kB (U x U uint16)
  PEP on disk if cached   67.6 kB (raw and symmetric copies)
```

## Channels

<!-- docs-test: skip (signature) -->
```python
duet.channels.symmetric(epsilon, alphabet="ACGT") -> Channel
duet.channels.position_varying(epsilon, alphabet="ACGT") -> Channel
duet.channels.asymmetric(matrix, alphabet="ACGT") -> Channel
duet.channels.position_varying_asymmetric(matrices, alphabet="ACGT") -> Channel
```

| Constructor | First argument | Decoding metric |
|---|---|---|
| `symmetric` | error probability in [0, 1) | Hamming distance |
| `position_varying` | one error probability per position, shape `(n,)` | negative log-likelihood |
| `asymmetric` | one matrix `T[sent, read]`, shape `(q, q)` | negative log-likelihood |
| `position_varying_asymmetric` | one matrix per position, shape `(n, q, q)` | negative log-likelihood |

Matrices are row-stochastic (`T[sent, read]`) in the order of `alphabet`; labeled DataFrames are
reordered by their labels. Arrays may also be given as the path of a `.npy` file. A
position-varying channel needs one entry per codeword position. Details:
[Noise channels and decoding rules](noise_channels.md#channel-classes).

A `Channel` has `kind`, `alphabet`, `alphabet_size`, `seq_length`, `params` (DNA in the internal
order A, T, C, G), `to_dict()`, `Channel.from_dict(d)` and `check_compatible(alphabet, seq_length)`.

## design

Decoding accuracy alone: one search at λ = 1, then a fresh-read evaluation of the starting and
the designed codebook.

<!-- docs-test: skip (signature) -->
```python
duet.design(pool, channel, *, seed=None, num_samples=10_000, eval_samples=5_000,
            device="auto", init="random", rule="unique_minimum", eval_channel=None,
            workers=None, cache_dir=None, max_patience=3000, max_iter=100_000,
            avoid_duplicates=True, duplicate_penalty=100.0, temperature=0.0,
            pep_storage="auto", verbose=True) -> DesignResult
```

The parameters are those of [`design_ops`](#design_ops) without `lambdas`, with two differences:

| Parameter | Default | Meaning |
|---|---|---|
| `init` | `"random"` | `"random"` or a codebook table, as in `design_ops`; `"best_score"` is not offered. |
| `workers` | None | CPU processes of the PEP and the evaluation (default: the CPUs available); the search runs in one process. Part of the PEP cache key. |

The pool's scores are not used. With the same seed and settings, `design` selects the same
codebook as `design_ops(pool, channel, lambdas=[1])`; its accuracies can differ slightly, within
Monte Carlo error, because they depend on the codebooks evaluated together.

**Returns** a [`DesignResult`](#designresult) of kind `"decoding"` with the rows `initial` and
`designed`, and no `lambda`, objective or `on_pareto_front` column. `res.codebook()` returns the
designed codebook (`group`, `sequence`, `accuracy`, `candidate`); `pareto_front()`,
`operating_point()` and `plot()` raise `ValueError`.

```python
channel = duet.channels.symmetric(0.1)
res_dec = duet.design(pool, channel, seed=0, num_samples=500, eval_samples=200, workers=3,
                      verbose=False)
print(res_dec.table.to_string())
```

```text
   codebook      kind  accuracy_mean  accuracy_p10  accuracy_p95_p5_ratio  surrogate_accuracy  n_codewords  duplicate_codewords
0   initial   initial        0.99075        0.9845               1.020669              0.9864           20                    0
1  designed  designed        0.99425        0.9895               1.015228              0.9976           20                    0
```

## design_ops

The λ sweep with the mean candidate score as the second objective, then a fresh-read evaluation.

<!-- docs-test: skip (signature) -->
```python
duet.design_ops(pool, channel, *, lambdas=(0, 0.05, 0.1, 0.25, 0.5, 1), seed=None,
                num_samples=10_000, eval_samples=5_000, device="auto", init="random",
                rule="unique_minimum", eval_channel=None, workers=None, cache_dir=None,
                max_patience=3000, max_iter=100_000, avoid_duplicates=True,
                duplicate_penalty=100.0, temperature=0.0, pep_storage="auto",
                verbose=True) -> DesignResult
```

| Parameter | Default | Meaning |
|---|---|---|
| `pool` | | From `CandidatePool.from_table`. |
| `channel` | | Channel of the design stage (the PEP); its alphabet must match the pool's. |
| `lambdas` | `(0, 0.05, 0.1, 0.25, 0.5, 1)` | Values in [0, 1], sorted ascending; no two may share `int(1000 * λ)`. For decoding alone, use [`design`](#design). |
| `seed` | None | One root seed; five stage seeds are derived from it. None draws one and records it in `provenance["seed"]`. |
| `num_samples` | 10,000 | Simulated reads per unique codeword for the PEP, at most 32,767. |
| `eval_samples` | 5,000 | Fresh reads per codeword for the evaluation, at most 32,767. 0 skips it; the front then uses `surrogate_accuracy`. |
| `device` | `"auto"` | `"auto"` (every visible GPU, else CPU), `"cpu"`, `"gpu"` (GPU 0 only), `"gpu:all"` or `"gpu:0,2"`. See [Devices](gpu_and_scaling.md#devices). |
| `init` | `"random"` | The initial codebook, shared by every λ: `"random"`, `"best_score"` or a table with the pool's group column and a sequence (or `candidate`) column, `quota` rows per set. `res.codebook(...)` of a design on the same pool works as is. |
| `rule` | `"unique_minimum"` | `"unique_minimum"`, `duet.UniqueMinimum()` or `duet.MarginDecoding(k)`. |
| `eval_channel` | None | Channel of the evaluation stage; None uses `channel`. |
| `workers` | None | CPU processes (default: the CPUs available). The sweep uses `min(workers, len(lambdas))`. Part of the PEP cache key. |
| `cache_dir` | None | Folder that keeps the PEP for later runs ([Caching the PEP](gpu_and_scaling.md#caching-the-pep)). |
| `max_patience` | 3000 | Stop a λ search after this many swaps without a new best objective. |
| `max_iter` | 100,000 | Maximum swaps per λ. |
| `avoid_duplicates` | True | Penalize two entries that share a codeword (no effect at λ = 0). |
| `duplicate_penalty` | 100.0 | Size of that penalty. |
| `temperature` | 0.0 | 0 takes the best swap; above 0 samples swaps from a softmax of their gains. |
| `pep_storage` | `"auto"` | `"auto"`, `"memory"` (codebooks of at most 256 codewords) or `"mmap"`. Never changes the selections. |
| `verbose` | True | Show progress output. |

**Returns** a [`DesignResult`](#designresult) of kind `"ops"`, with rows `initial`, one
`lambda=<value>` per λ and `max_score` (the highest-scoring candidates of each set). For
decoding alone, [`design`](#design) returns only the starting and the designed codebook.
`design_ops` warns when the scores do not vary within any candidate set (for example a pool
built without `score`): every codebook then has the same mean score.

```python
channel = duet.channels.symmetric(0.1)
res = duet.design_ops(pool, channel, lambdas=[1, 0, 0.1], seed=0,
                      num_samples=500, eval_samples=200, workers=3, verbose=False)
print(res.table.to_string())
```

```text
     codebook       kind  lambda  accuracy_mean  accuracy_p10  accuracy_p95_p5_ratio  surrogate_accuracy  mean_score  n_codewords  duplicate_codewords  on_pareto_front
0     initial    initial     NaN        0.99075        0.9845               1.020669              0.9864    0.726680           20                    0            False
1    lambda=0     lambda     0.0        0.98925        0.9800               1.021450              0.9818    0.867163           20                    0            False
2  lambda=0.1     lambda     0.1        0.98950        0.9800               1.021450              0.9851    0.867163           20                    0             True
3    lambda=1     lambda     1.0        0.99625        0.9900               1.010356              0.9976    0.713579           20                    0             True
4   max_score  max_score     NaN        0.98625        0.9800               1.016858              0.9810    0.867163           20                    0            False
```

## design_merfish

Builds candidate barcodes for a gene panel and runs the λ sweep against optical crowding. It
selects one barcode per gene, so it chooses barcodes and assigns them to genes at once.

<!-- docs-test: skip (signature) -->
```python
duet.design_merfish(genes, channel, *, gene="gene", expression="expression", n_bits,
                    hamming_weights, candidates_per_gene=1000, init="random", include=None,
                    barcode="barcode", lambdas=(0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1),
                    ...) -> DesignResult   # the other keywords and defaults of design_ops
```

| Parameter | Default | Meaning |
|---|---|---|
| `genes` | | DataFrame, one row per gene in panel order; codeword position i belongs to row i. |
| `channel` | | A binary channel (`alphabet="01"`). |
| `gene` | `"gene"` | Gene column, in `genes` and in the `init` and `include` tables. |
| `expression` | `"expression"` | Expression column: finite, non-negative, linear scale (any unit). |
| `n_bits` | required | Barcode length (readout bits). |
| `hamming_weights` | required | `{weight: cap or None}`: the allowed weights, each capped at a random subset of `cap` barcodes (anchored ones included) or kept whole. |
| `candidates_per_gene` | 1000 | Barcodes sampled per gene, its anchored barcodes included. None gives every gene the whole pool. |
| `init` | `"random"` | Starting codebook S0: `"random"` or a table with the gene and barcode columns, one barcode per panel gene. It also sets the zero of `crowding`. |
| `include` | None | Reference codebooks evaluated alongside: a table, a list or a `{name: table}` dict. |
| `barcode` | `"barcode"` | Barcode column of the `init` and `include` tables. |
| `lambdas` | the paper's nine values | As in `design_ops`. |
| others | | As in [`design_ops`](#design_ops). |

**Returns** a [`DesignResult`](#designresult) of kind `"merfish"`, with rows `initial`, one
`lambda=<value>` per λ and the `include` names. Its objective column is `crowding`,
1 - C(S)/C(S0): 0 for the starting codebook, higher when less crowded
([The crowding objective](design_merfish.md#the-crowding-objective)). Read barcode columns with
`dtype=str`, or leading zeros are lost. The defaults are not the paper's settings
([Settings used in the paper](design_merfish.md#settings-used-in-the-paper)).

## evaluate

The decoding accuracy of codebooks you already have.

<!-- docs-test: skip (signature) -->
```python
duet.evaluate(codebooks, channel, *, num_samples=5_000, seed=None, device="auto",
              rule="unique_minimum", workers=None, sequence=None) -> EvaluationResult
```

| Parameter | Default | Meaning |
|---|---|---|
| `codebooks` | | `{name: codebook}`; a codebook is a list of sequences or a table with a sequence column (and optionally `group` or `gene`). |
| `channel` | | Channel of the readout. All sequences must have one length (no truncation). |
| `num_samples` | 5,000 | Reads per codeword (a design's `eval_samples`), at most 32,767. |
| `seed` | None | Root seed, as in the design functions. |
| `device`, `rule`, `workers` | | As in [`design_ops`](#design_ops). |
| `sequence` | None | Sequence column of table inputs; None tries `sequence`, `barcode`, `Sequence`, `Barcode`. |

Codebooks passed together share reads for shared codewords, so adding or removing one changes
the others' accuracies slightly. Passing every codebook of a design in
`res.provenance["evaluation"]["codebook_order"]`, with its seed, evaluation channel, rule,
device and `num_samples=eval_samples`, reproduces its accuracies exactly:

```python
order = res.provenance["evaluation"]["codebook_order"]
ev = duet.evaluate({label: res.codebook(label) for label in order}, channel,
                   num_samples=200, seed=0, device=res.provenance["evaluation"]["device"],
                   workers=3)
print(ev.summary[["codebook", "accuracy_mean", "accuracy_p10"]].to_string())
print((ev.summary["accuracy_mean"] == res.table["accuracy_mean"]).all())
```

```text
     codebook  accuracy_mean  accuracy_p10
0     initial        0.99075        0.9845
1    lambda=0        0.98925        0.9800
2  lambda=0.1        0.98950        0.9800
3    lambda=1        0.99625        0.9900
4   max_score        0.98625        0.9800
True
```

## DesignResult

Returned by `design`, `design_ops` and `design_merfish`, and by `duet.load`.

| Member | Meaning |
|---|---|
| `table` | One row per codebook (columns below) |
| `pareto_front()` | The rows of `table` with `on_pareto_front` True (λ sweeps only: `ValueError` for a `design` result) |
| `codebook(which=None)` | One codebook, by λ value (`0.5`) or row label (`"initial"`, `"lambda=1"`, `"max_score"`, an `include` name, `"designed"`). Without `which`: the designed codebook of a `design` result (a λ sweep needs `which`). `design` columns: `group`, `sequence`, `accuracy`, `candidate`. OPS columns: `group`, `sequence` (the codeword), `score`, `accuracy`, `candidate` (row position in the table given to `from_table`; use `.iloc`). MERFISH: `gene`, `barcode`, `expression`, `accuracy` |
| `operating_point(score_fraction=0.975)` | OPS only: the most accurate λ codebook that keeps `score_fraction` of the `max_score` mean score (the paper's OPS rule), as a row of `table` |
| `plot(ax=None, annotate=True)` | The λ codebooks, the front and the references; returns a matplotlib `Axes` (λ sweeps only) |
| `save(outdir)` | Writes the [saved files](#saved-results); returns the folder |
| `provenance` | Versions, the root and stage seeds, devices, workers, PEP storage and source, channel, rule, optimizer and pool settings, and `table_sha256`, a hash of the ordered candidate table |
| `kind` (`"decoding"`, `"ops"` or `"merfish"`), `codebooks`, `secondary_objective`, `front_axis` | Provisional (not covered by ADR 0003); `secondary_objective` and `front_axis` are None for a `design` result |

Columns of `table`:

| Column | Meaning |
|---|---|
| `codebook`, `kind`, `lambda` | Row label; `initial`, `lambda`, `max_score`, `include` or `designed`; λ (NaN for other rows; no `lambda` column in `design` results) |
| `accuracy_mean` | Decoding accuracy: fraction of evaluation reads decoded to the right codeword, averaged over codewords. The accuracy the paper reports. NaN, like the other accuracy columns, when `eval_samples=0` |
| `accuracy_p10` | 10th percentile of per-codeword accuracy. Not the paper's bottom-decile accuracy (the mean of the codewords at or below this percentile) |
| `accuracy_p95_p5_ratio` | 95th over 5th percentile of per-codeword accuracy; 1 is perfectly even |
| `surrogate_accuracy` | The union-bound objective the search maximizes, on the design PEP. Not an accuracy: it can be negative |
| `mean_score` (OPS) / `crowding` (MERFISH) | The experiment-specific objective; absent from `design` results |
| `n_codewords`, `duplicate_codewords` | Entries; entries whose codeword occurs more than once (they cannot be decoded) |
| `on_pareto_front` | True for λ codebooks that no other λ codebook dominates; always False for `initial` and references; absent from `design` results |

```python
print(res.pareto_front()["codebook"].tolist(), res.operating_point()["codebook"])
cb = res.codebook(0.1)
print(library.iloc[cb["candidate"]][["sgID", "sequence"]].head(2))
res.save("my_design")
back = duet.load("my_design")
pd.testing.assert_frame_equal(back.table, res.table)
```

```text
['lambda=0.1', 'lambda=1'] lambda=0.1
                       sgID             sequence
1  ABCB5_+_20687242.23-P1P2  ATAACCTGGAAAATATGAA
0  ABCB5_+_20687051.23-P1P2  GCTCACAGAGAGATATTAG
```

## EvaluationResult

| Field | Content |
|---|---|
| `summary` | One row per codebook: `codebook`, `n_codewords`, `accuracy_mean`, `accuracy_p10`, `accuracy_p95_p5_ratio`, `duplicate_codewords` |
| `per_codeword` | One row per codeword: `codebook`, `position`, `sequence`, `group` (when the input had one), `accuracy` |
| `provenance` | Version, seeds, `num_samples`, device, channel, rule, `codebook_order` |

## load

<!-- docs-test: skip (signature) -->
```python
duet.load(outdir) -> DesignResult
```

Reads a folder written by `DesignResult.save`, with the column types restored. A loaded codebook
can serve as `init` for a new design on the same pool.

## Decoding rules

<!-- docs-test: skip (signature) -->
```python
duet.UniqueMinimum()
duet.MarginDecoding(k)
```

| Rule | A read of codeword s decodes correctly when | Paper name |
|---|---|---|
| `duet.UniqueMinimum()` (default) | s alone has the lowest cost; ties are errors | unique-minimum decoding |
| `duet.MarginDecoding(k)`, k >= 0 | every other codeword costs more than the cost of s plus k | margin-γ decoder, γ = k |

Pass an instance, with parentheses. The rule applies to the PEP and to the evaluation. See
[Decoding rules](noise_channels.md#decoding-rules).

## Saved results

`save(outdir)` writes three files, overwriting files of the same names:

| File | Content |
|---|---|
| `table.csv` | `DesignResult.table` |
| `codebooks.csv` | Every codebook, one row per codeword: `codebook`, `position`, then the columns of `codebook()` |
| `settings.json` | `format` (`"duet-design-result"`), `format_version`, `kind` (`"ops"`, `"merfish"` or, from 0.2.0, `"decoding"`), the column types and the provenance |

A result of kind `decoding` needs DUET 0.2.0 or later.

Read them with `duet.load`; with pandas alone, read label and barcode columns as text
(`dtype=str`) and pass `keep_default_na=False, na_values=[""]` (NaN is an empty field).

## Common errors

| Message (shortened) | Fix |
|---|---|
| `sequences use symbols ['a', 'c', 'g', 't'] outside the alphabet 'ACGT'` | Uppercase the sequences; write RNA `U` as `T`; binary barcodes need `alphabet="01"` |
| `matrix: each row T[sent, :] must sum to 1` | The matrix is probably `T[read, sent]`: pass its transpose |
| `barcode '1100010000001' is not a string of 16 bits (0/1)` | Leading zeros were lost: read the CSV with `dtype={"barcode": str}` |
| `CuPy is required for GPU devices ... pip install "duet-codebook[gpu]"` | DUET is not on PyPI: run `pip install ".[gpu]"` in your clone, or pass `device="cpu"` |
| `no lambda codebook keeps 97.5% of the maximum-score codebook's mean score` | Add smaller λ values, lower `score_fraction`, or rescale scores that are not positive |
| `pareto_front() needs a lambda sweep with a second objective` | A `duet.design` result has one designed codebook: use `res.codebook()` and compare it with `res.codebook("initial")` |

## Advanced layer

These names are provisional: documented and importable, but free to change in a minor version
before 1.0. They return raw selections and PEP counts, without evaluation, seed derivation or
provenance.

| Name | What it is |
|---|---|
| `duet.runner.run_duet_ops`, `run_duet_merfish` | The λ sweep on engine objects |
| `duet.runner.DuetOptimizerConfig` | Optimizer settings |
| `duet.runner.ObjectiveSpec` | Intended extension point for custom objectives; no runner accepts it yet |
| `duet.evaluator_config.EvaluatorConfig` | The decoding model by name, including metrics, rules and non-square channels the public API does not expose |
| Other `CandidatePool` members | The dataclass constructor, `from_dataframe` and its attributes |
| `duet.api.derive_stage_seeds(seed)` | The five stage seeds a design derives from `seed` |
| `duet.results.select_operating_point(...)` | The `operating_point()` rule on plain arrays |
