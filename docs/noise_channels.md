# Noise channels and decoding rules

DUET simulates reads through a **noise channel** and decodes them with a **decoding rule**; the
decoding metric follows from the channel. The snippets run in one Python session, in about a minute
on CPU. Under non-symmetric channels a GPU or another CPU can decode near-tie reads differently
([Reproducibility](design.md#reproducibility)).

## Channel classes

All classes model substitutions, independently per position. Each constructor in `duet.channels`
takes an `alphabet`: the DNA bases in any order (default `"ACGT"`) or `"01"`. q is the number of
symbols and n the codeword length (after `seq_length` truncation, or `n_bits`).

| Class | Constructor | Parameters | Decoding metric | Use it when |
|---|---|---|---|---|
| symmetric | `symmetric(epsilon)` | one error rate | Hamming distance | you know only an average error rate |
| position-varying | `position_varying(epsilon)` | shape `(n,)` | negative log-likelihood (NLL) | the error rate changes over rounds |
| asymmetric | `asymmetric(matrix)` | shape `(q, q)` | NLL | some substitutions dominate, similarly at every position |
| position-varying asymmetric | `position_varying_asymmetric(matrices)` | shape `(n, q, q)` | NLL | you have enough reads for a matrix per position |

```python
import numpy as np
import pandas as pd
import duet

T = np.array([[0.92, 0.02, 0.04, 0.02],    # T[sent, read]; rows and columns in A, C, G, T order
              [0.01, 0.94, 0.04, 0.01],
              [0.01, 0.01, 0.97, 0.01],
              [0.01, 0.01, 0.04, 0.94]])
off = T - np.diag(np.diag(T))
Ts = np.stack([off * s + np.diag(1 - (off * s).sum(axis=1)) for s in np.linspace(0.5, 2, 10)])
sym = duet.channels.symmetric(0.1)                                # 10% per base
pv = duet.channels.position_varying(np.linspace(0.05, 0.15, 10))  # 10 rounds, rising
asym = duet.channels.asymmetric(T)
pva = duet.channels.position_varying_asymmetric(Ts)                # (10, 4, 4)
```

## Transition matrices

- **`T[sent, read]`**: rows are the transmitted symbol, columns the symbol read. Entries are
  probabilities (not percentages) and every row sums to 1. A matrix written as `T[read, sent]`
  usually fails this check (`matrix: each row T[sent, :] must sum to 1 ...`); one whose columns
  also sum to 1 cannot be detected, so check the orientation.
- **Order**: rows and columns follow `alphabet`; a labeled DataFrame (index = sent, columns =
  read) is reordered by its labels. Matrices are square, and a `.npy` path works too.
- **Stored order**: `Channel.params`, the provenance and `settings.json` hold DNA matrices in A, T,
  C, G order (`internal_symbol_order`).

## Estimating a channel from your reads

1. **Collect (true, read) pairs**, best from reads of known identity such as control barcodes. If
   you match reads to your library, keep every matched read (filters on the match bias the error
   rate low) and split a read tied between m members with weight 1/m each. Filter on quality only.
2. **Count substitutions** per position, `counts[l, a, b]`, and add a pseudocount: a zero entry
   makes that substitution impossible for the decoder.
3. **Normalize** each row per position (position-varying asymmetric), pool positions first
   (asymmetric), or take the mismatch fraction per position (position-varying) or overall
   (symmetric). Here 20,000 reads simulated through `Ts` stand in for your data:

```python
rng = np.random.default_rng(1)
library = ["".join(rng.choice(list("ACGT"), 10)) for _ in range(300)]
sent = np.array([["ACGT".index(c) for c in s] for s in rng.choice(library, 20_000)])
cdf = np.cumsum(Ts, axis=2)[np.arange(10), sent]
read = (rng.random(sent.shape)[..., None] < cdf).argmax(axis=2)   # your reads, as indices
weights = np.ones(len(sent))           # 1 per read, or 1/m for a read tied between m members
counts = np.zeros((10, 4, 4))
for pos in range(10):
    np.add.at(counts[pos], (sent[:, pos], read[:, pos]), weights)
pseudo = counts + 1.0
T_pos = pseudo / pseudo.sum(axis=2, keepdims=True)                  # (n, 4, 4)
T_pooled = pseudo.sum(axis=0) / pseudo.sum(axis=(0, 2))[:, None]    # (4, 4)
eps_pos = 1 - np.einsum("lii->l", counts) / counts.sum(axis=(1, 2))  # (n,)
eps_all = 1 - np.einsum("lii->", counts) / counts.sum()
fitted = [duet.channels.position_varying_asymmetric(T_pos), duet.channels.asymmetric(T_pooled),
          duet.channels.position_varying(eps_pos), duet.channels.symmetric(eps_all)]
print("estimated:", np.round(eps_pos, 3))
print("true:     ", np.round(1 - np.einsum("lii->l", Ts) / 4, 3))
```

```text
estimated: [0.029 0.038 0.049 0.055 0.067 0.075 0.087 0.098 0.106 0.117]
true:      [0.029 0.038 0.048 0.057 0.067 0.077 0.086 0.096 0.105 0.115]
```

A matrix per position needs many reads per position and symbol; one error rate needs few. Design
under your best estimate and evaluate under the other fits ([below](#evaluating-under-another-channel)).

## Channels in the repository

`channels/` is `scripts/benchmark/noise_model_matrices/channels/`; its
[README](../scripts/benchmark/noise_model_matrices/channels/README.txt) gives the source of
each file and, where one applies, its license.

| File | Shape | Build with | Used by |
|---|---|---|---|
| `channels/merfish_zhang2023_channel.npy` | (2, 2) | `asymmetric(path, alphabet="01")` | MERFISH experiments |

The MERFISH readout error rates were estimated with MERlin v0.1.6 from four samples of the Zhang
et al. (2023) data, pooled, and fitted as a binary channel: a 1 is read as 0 with probability
0.0561, a 0 as 1 with probability 0.0145. The fitted matrix is included, licensed CC BY-SA 4.0
because it is derived from Brain Image Library data; the estimation scripts are not included.

**NIS-seq channels.** The OPS cross-channel experiment and the genome-wide design use NIS-seq
channels, one file per class: `uniform` (build it with `symmetric(1 - T[0, 0])`, where
`T = np.load(path)`), `positional` (`position_varying(path)`), `channel`
(`asymmetric(path, alphabet="ATCG")`) and `positional_channel`
(`position_varying_asymmetric(path, alphabet="ATCG")`). The matrices are in **A, T, C, G**
order, hence `alphabet="ATCG"`.

- The genome-wide design reads `channels/archive/nisseq_hela_*_subsample.npy` (14 rounds).
- The cross-channel experiment reads `nisseq_positional_10rounds.npy` (10,),
  `nisseq_positional_channel_10rounds.npy` (10, 4, 4) and `nisseq_channel.npy` (4, 4) in
  `experiments/ops_crispri_cross_eval/noise_matrices/`: the first 10 rounds of the `positional`
  and `positional_channel` files of `channels/nisseq/nisseq_hela_*_pctl50_subsample50000.npy`,
  and its `channel` file. Its symmetric arm is `symmetric(0.14465571)`.
- `experiments/nisseq_error_analysis` checks the matrices it fits against these files and
  `channels/nisseq/nisseq_hela_*_pctl50_subsample5000.npy`.

The NIS-seq channel matrices are fitted from spot-level base calls that the authors of Fandrey et
al. (2025) shared on request, and are included only with their permission. If one is missing from
your copy, rebuild it with `experiments/nisseq_error_analysis` from the spot calls, which are
available from those authors on request
([Inputs](reproducing_the_paper.md#inputs-not-in-the-repository)).

## Decoding rules

Pass `rule=` to `design`, `design_ops`, `design_merfish` or `evaluate`; a design uses it
throughout.

- **Unique minimum** (default; `"unique_minimum"` or `duet.UniqueMinimum()`): a read is assigned
  only if one codeword has a strictly lower cost than all others. Ties count as errors.
- **Margin decoding** (`duet.MarginDecoding(k)`, k >= 0; the paper's margin-γ decoder): a read is
  assigned only if every other codeword costs more by more than k; otherwise it is rejected.

k is in the units of the metric: mismatches under Hamming, a natural-log likelihood ratio under
NLL (k = 2.94 means a pairwise posterior above 0.95), so the same k is much stricter under Hamming.
Choose k for the decoder you will run, and design with the same rule. Other metrics, rectangular
channels and indels are outside the public API ([Limits](design.md#limits)).

## Evaluating under another channel

`channel` drives the PEP, the λ sweep and `surrogate_accuracy`; `eval_channel` (default: `channel`)
drives the `accuracy_*` columns; `duet.evaluate` scores any codebooks under any channel. Here a
design under the one error rate estimated above (`eps_all`) is evaluated under `pva`, the
position-varying asymmetric channel that made the reads, then re-scored:

```python
one_rate = duet.channels.symmetric(eps_all)
table = pd.read_csv("examples/data/quickstart_ops.csv")
quota = {g: (10 if g == "negative_control" else 2) for g in table["gene"].unique()}
pool = duet.CandidatePool.from_table(table, group="gene", sequence="sequence",
                                     score="activity", quota=quota, seq_length=10)
res = duet.design_ops(pool, one_rate, eval_channel=pva, lambdas=[0, 0.25, 1],
                      seed=0, num_samples=2_000, eval_samples=2_000, verbose=False)
books = {label: res.codebook(label) for label in res.table["codebook"]}
acc = pd.DataFrame({"codebook": list(books)})
for name, channel in (("one_rate", one_rate), ("pva", pva)):
    acc[name] = duet.evaluate(books, channel, num_samples=2_000, seed=0).summary["accuracy_mean"]
print(acc.round(4).to_string(index=False))
print("equals the design's evaluation:", acc["pva"].equals(res.table["accuracy_mean"]))
```

```text
   codebook  one_rate    pva
    initial    0.9751 0.9908
   lambda=0    0.9710 0.9888
lambda=0.25    0.9758 0.9909
   lambda=1    0.9880 0.9956
  max_score    0.9716 0.9893
equals the design's evaluation: True
```

The two channels rank the codebooks alike but score them differently. With all of a design's
codebooks in table order, the same `seed`, channel and rule, and `num_samples` equal to the design's
`eval_samples`, `duet.evaluate` reproduces its accuracies exactly. The paper's cross-channel
experiment does this at full scale ([Experiments](reproducing_the_paper.md#experiments)).
