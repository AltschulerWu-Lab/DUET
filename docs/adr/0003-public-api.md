# ADR 0003: Public API

- **Status:** accepted
- **Date:** 2026-09-24
- **Reviewed:** an internal API surface review, 2026-09-24.
- **Amended:** 2026-10-01 (0.2.0): `duet.design`, a decoding-only design (see
  the amendment at the end).

## Context

Before 0.1.0, going from "my candidate table" to "my library" meant
assembling a `CandidatePool`, an `EvaluatorConfig`, a `DuetOptimizerConfig`,
an initial-codebook array and an alphabet size, calling `run_duet_ops`, and
then evaluating the result the way the benchmark runners do. The runners are
the paper's record (YAML schemas, as-run bundles, regression goldens), so
they cannot be moved onto a new API before submission.

## Decision

A thin facade in `duet.api`, `duet.channels` and `duet.results`, re-exported
from `duet/__init__.py`. It builds the same internal objects the runners
build and calls the same functions. No logic that affects a number exists
only in the facade.

### Names and tiers

| Tier | Names | Promise |
|---|---|---|
| Public | `duet.design`, `duet.design_ops`, `duet.design_merfish`, `duet.evaluate`, `duet.load`, `duet.channels` (`symmetric`, `position_varying`, `asymmetric`, `position_varying_asymmetric`, `Channel`), `duet.DesignResult` (`table`, `pareto_front()`, `codebook()`, `operating_point()`, `plot()`, `save()`, `provenance`; `codebook()` without argument for `duet.design` results), `duet.EvaluationResult`, `duet.CandidatePool.from_table`, `CandidatePool.describe()` and the `duet.PoolDescription` it returns, `duet.UniqueMinimum`, `duet.MarginDecoding`, `duet.__version__` | The public API. Changes follow semantic versioning. |
| Advanced, provisional | `duet.runner.run_duet_ops`, `run_duet_merfish`, `DuetOptimizerConfig`, `ObjectiveSpec` (the extension point for custom objectives); `duet.evaluator_config.EvaluatorConfig`; the other members of `CandidatePool`; `duet.api.derive_stage_seeds` and `duet.results.select_operating_point` | Documented and importable, but free to change in minor versions until 1.0. |
| Internal | `duet.runner.run_duet_from_primitives`, the PEP accessors (`duet.pep_accessor`), private helpers (leading underscore) and every other module | No promise. |

The PEP accessors are no longer exported from `duet/__init__.py`; nothing
live imported them from there.

Deviations from the proposed shape, and why:

- `rule=` accepts `"unique_minimum"`, `duet.UniqueMinimum()` or
  `duet.MarginDecoding(k)` with k >= 0, the engine's own classes, exposed
  lazily from `duet`, rather than a new rule type. Other engine rules stay in
  the advanced layer. A negative margin is refused: a codeword would no longer
  compete with itself, which the memory-mapped PEP assumes (its diagonal is
  taken as 2).
- `design_merfish` takes `barcode=` (the barcode column of `init` and
  `include` tables) and `include=` as a table, a list or a name -> table dict.
- `evaluate` takes `sequence=` for the sequence column of table inputs.
- Advanced keywords on both design functions: `max_patience`, `max_iter`,
  `avoid_duplicates`, `duplicate_penalty`, `temperature`, `pep_storage`,
  `cache_dir`, `verbose`.
- `lambdas` has defaults: `(0, 0.05, 0.1, 0.25, 0.5, 1)` for OPS and the
  paper's nine values for MERFISH.
- `duet.design` (added in 0.2.0): decoding alone. It takes the keywords of
  `design_ops` except `lambdas`; `init` is `"random"` or a table (no
  `"best_score"`).
- Row labels of `res.table`: `initial`, `lambda=<value>`, `max_score` (OPS),
  the `include` names (MERFISH) and `designed` (`duet.design`). The
  secondary-objective column is `mean_score` (OPS) or `crowding` (MERFISH,
  1 - C(S)/C(S0)); a `duet.design` table has no `lambda`, objective or
  `on_pareto_front` column.

### Two Monte Carlo stages

1. **Design.** The PEP (`num_samples` reads per unique codeword) and the
   lambda sweep (for `duet.design`, lambda = 1 alone), through
   `run_duet_ops` / `run_duet_merfish`. The optimizer sees only the
   union-bound surrogate, reported as `surrogate_accuracy`
   (`duet.benchmark.metrics.compute_duet_objective`, without the duplicate
   penalty). It is not an accuracy, can be negative, and the paper never
   reports it.
2. **Evaluation.** One call of `evaluate_codebooks_by_sequence` on, in this
   order: the initial codebook; the lambda codebooks in ascending lambda; the
   references (OPS: the maximum-score codebook; MERFISH: the `include`
   codebooks). Then `compute_metrics` per codebook. Fresh reads
   (`eval_samples`) under an independent seed. `eval_samples=0` skips it.
   For `duet.design`: the initial and the designed codebook.

The Pareto front is the set of lambda codebooks not dominated on (mean
evaluated accuracy, secondary objective), computed in numpy. Only the
accuracy axis carries Monte Carlo error. Without the evaluation stage the
front uses `surrogate_accuracy`. A `duet.design` result has no front:
`pareto_front()`, `operating_point()` and `plot()` raise `ValueError`.

### Parity contract

Two facts limit what "identical" can mean:

- **Row order matters.** PEP reads are seeded per block of 100 unique
  sequences in table order (`codebook_evaluator.py`, `compute_pep_matrix`).
  Re-sorting the candidate table redraws them and changes the codebooks.
- **Companion codebooks matter.** Evaluation reads are seeded by position in
  the union of the codebooks evaluated together (`initialize_cache`). A
  codebook's accuracy depends on the other codebooks in the same evaluation.

Hence:

- **Selections are bit-identical.** Given the same pool (rows in the same
  order), initial codebook, stage seeds and device, the facade returns the
  same selections as calling `run_duet_ops` / `run_duet_merfish` directly.
- **Accuracies match the evaluation function.** They equal
  `evaluate_codebooks_by_sequence` on the same ordered list of codebooks with
  the same evaluation seed.
- **`duet.design` selects what `design_ops(lambdas=[1])` selects.** It makes
  the same `run_duet_ops` call (`lambda_=[1.0]`, the same stage seeds, PEP
  configuration, initial codebook and storage), so its selections and
  `surrogate_accuracy` equal that call's. Its accuracies equal
  `evaluate_codebooks_by_sequence` on [initial, designed]; they equal the
  `design_ops` rows only when the `max_score` codebook adds no extra copy of a
  codeword to the front of the evaluation union, and otherwise agree within
  Monte Carlo error.
- **Paper accuracies match only statistically,** within Monte Carlo error:
  the benchmarks evaluate DUET together with the baselines.
- **Storage does not change selections.** The engine has two decode-cache
  paths: in memory (`cache_dir=None`) it sums the codebook's PEP rows in one
  call, memory-mapped (`build_shared_decode_state`) in batches of 256 rows.
  Above 256 codewords the two sums differ in the last bits and the optimizer
  breaks exact ties differently (found by the API review; the paper's runs
  all used the memory-mapped path). The facade therefore holds the PEP in
  memory only for codebooks of at most 256 codewords, where the two paths are
  bitwise identical, and refuses `pep_storage="memory"` above that. So the
  facade's selections equal `run_duet_*` called with the same storage: with
  `cache_dir=None` up to 256 codewords and a small PEP, with
  `cache_dir=..., use_mmap=True` otherwise.

Tests: `tests/test_api.py` (equivalence against direct calls for OPS and
MERFISH, evaluation equivalence, storage, RNG, channels, validation,
save/load, operating point, the decoding-only design (`TestDecodingDesign`)),
`tests/test_api_parity.py` (MERFISH pool parity against the
`merfish_asymmetric` reference and a per-gene runner run; the `ops_uniform`
paper-parity test, GPU, `slow`).

### Seeds

The public API takes one `seed`. `np.random.SeedSequence(seed).spawn(5)`
gives one child per stage, and each child becomes a 32-bit seed with
`generate_state(1, dtype=uint32)[0]`. The order is fixed and must never
change:

| Child | Stage | Used by |
|---|---|---|
| 0 | `pool` | MERFISH pool sampling (`subsample_seed` and `sample_seed` of `MERFISHFactory`) |
| 1 | `init` | the initial codebook (`RandomInit`) |
| 2 | `pep` | `EvaluatorConfig.seed` of the PEP |
| 3 | `evaluation` | `EvaluatorConfig.seed` of the evaluation stage (and of `duet.evaluate`) |
| 4 | `optimizer` | the `seed` of `run_duet_*`; each lambda then uses `(seed + int(1000 * lambda)) % (2**31 - 1)` |

When `seed` is None, one is drawn from OS entropy
(`SeedSequence().entropy`) and recorded in `provenance`. The PEP and the
evaluation never share a seed (shared seeds line up their child streams by
index); the facade rejects it. The private keyword `_stage_seeds=` sets stage
seeds directly; only the paper-parity test uses it.

`DUET.optimize` reseeds the global `random` and `np.random` state; the
facade saves and restores both around the lambda sweep, so a serial run
leaves the caller's next draws unchanged.

### Devices, workers and PEP storage

- `device="auto"` resolves to `"gpu:all"` when CuPy sees a GPU, else `"cpu"`
  with a warning when U, the number of unique codewords, is at least
  `CPU_WARN_UNIQUE_CODEWORDS` = 2,000. The threshold comes from the scaling
  measurement in `examples/README.md`: on 40 CPU threads the PEP took 67 min
  at U = 11,628 (54 s on 4 GPUs), so about 2 min at U = 2,000. GPUs compute the PEP and the evaluation; the
  lambda sweep runs on CPU, one process per lambda.
- `workers`: the sweep uses min(workers, len(lambdas)) processes (default
  min(len(lambdas), CPU count)); CPU PEP and evaluation use `workers`
  (default: the CPU count). `duet.design` runs its one search in the main
  process.
- The PEP is a dense U x U uint16 count matrix: 2U² bytes in memory, 4U² on
  disk when cached (a raw and a symmetric copy; 6U² while the symmetric copy
  is built). It is held in memory when 2U² <= 64 MiB, the codebook has at
  most 256 codewords and it fits in RAM (with `cache_dir`, the raw counts
  are cached and loaded); otherwise it is memory-mapped from `cache_dir` or
  from a temporary directory deleted after the run. The facade checks free
  memory and disk first and says how much is needed. The result never holds
  the PEP.
- The PEP cache key covers the unique codewords in order, the channel, rule,
  `num_samples`, the PEP seed and the number of CPU processes (`workers`,
  default: the CPU count), but not the device.
- Provenance records the device per stage and, for the PEP, whether it was
  computed or read from the cache and which device computed it (cache
  metadata `device`, added in 0.1.0; the cache key still ignores the device).

### Defaults, and the paper's settings as facade arguments

| Setting | Facade default |
|---|---|
| `num_samples` (PEP) | 10,000 |
| `eval_samples` | 5,000 |
| `max_patience` | 3,000 |
| `max_iter` | 100,000 |
| `avoid_duplicates` | True |
| decoding rule | unique minimum |
| `init` | `"random"` |

These serve users; the paper's runs used other settings per benchmark:

| Paper run (config) | `num_samples` | `eval_samples` | `max_patience` | `avoid_duplicates` | `lambdas` | Other |
|---|---|---|---|---|---|---|
| OPS CRISPRi medium-scale benchmark (`experiments/ops_crispri_symmetric/config.yaml`) | 5,000 | 2,000 | 3,000 | True | 0, 0.01, ..., 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 1 (18) | `channels.symmetric(0.1)`, `seq_length=10`, `quota=2`, 1,000 genes + 200 of 1,895 controls, `init="random"`; seeds: PEP 43, evaluation 42 via `_stage_seeds` |
| OPS cross-channel (`experiments/ops_crispri_cross_eval/*.yaml`) | 5,000 | 2,000 | 3,000 | True | as above | the NIS-seq channels in `noise_matrices/` (`position_varying`, `asymmetric`, `position_varying_asymmetric`; matrices in A, T, C, G order, so pass `alphabet="ATCG"`), which the repository includes only with the permission of the NIS-seq authors (see `docs/noise_channels.md`); `eval_channel=` for the cross-evaluation |
| OPS cross-channel, symmetric arm (`experiments/ops_crispri_cross_eval/symmetric.yaml`) | 5,000 | 2,000 | 3,000 | True | as above | `channels.symmetric(0.14465571)` (1 - the NIS-seq diagonal) |
| MERFISH Zhang 2023 (`experiments/merfish_zhang2023_v2/config.yaml`) | 30,000 | 5,000 | 500 | False | 0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1 | `n_bits=32`, `hamming_weights={4: None, 5: 64040}`, `candidates_per_gene=1000`, `init=` the published Zhang v2 codebook, `include=` the three baselines the runner anchors (Boström HW4 and HW5 panels, Zhang codebook #2), `channels.asymmetric(T, alphabet="01")`; seeds: pool 42, PEP 43, evaluation 42 |
| MERFISH 2,000 genes (`experiments/merfish_2000_genes/config.yaml`) | 30,000 | 5,000 | 500 | False | as above | as above, 2,000 genes, `init=` the Boström HW5 panel, `include=` the Boström HW5 and MERFISH MHD4 panels |

All paper runs use unique-minimum decoding. The paper's seeds come from the
runners' own scheme (per-trial seeds for OPS; one top-level seed for
MERFISH), so reproducing a run exactly needs the private `_stage_seeds=`, as
`tests/test_api_parity.py` does. The synthetic studies
(`experiments/synthetic_objective_correlation`; `experiments/synthetic_hvr`,
101 lambdas, `num_samples` 20,000) run on integer-encoded synthetic pools
against exhaustive enumeration and greedy baselines; they have no facade
equivalent and stay in `experiments/`. The paper's accuracies come from
evaluating DUET together with the baselines, so the facade reproduces them
statistically, not bit for bit.

## Consequences

- The runners (`duet.ops_benchmark`, `duet.merfish_benchmark`) are unchanged
  and keep reproducing the paper. The facade is pinned to them by the parity
  tests. `select_operating_point` is a package copy of the paper's OPS
  operating-point rule (`scripts/benchmark/visualize_benchmark.py`,
  `select_duet_lambda` and `_reference_arms`), pinned by a test on the
  recorded `ops_uniform` reference; the script switches to it after
  publication.
- `import duet` stays light: of the heavy packages it loads numpy and
  pandas only, not scipy, cupy, pymoo or matplotlib (the engine loads when a
  design function is called).

### Post-publication work

Each item changes numbers or touches the frozen paths, so none is done before
publication:

- **Seed by sequence identity.** PEP and evaluation reads are seeded by
  position (table row order; position in the evaluation union). Seeding by
  sequence would make results independent of row order and of the other
  codebooks evaluated alongside.
- **The per-lambda seed rule** `seed + int(1000 * lambda)` collides for
  lambdas closer than 0.001 (the facade rejects them).
- **The global RNG in `DUET.optimize`** (the facade restores the caller's
  state).
- **The device in the PEP cache key.** CPU and GPU give different counts for
  NLL metrics, yet a GPU-computed PEP is reused by a CPU run (and the key
  includes `num_cpus`, which changes nothing but blocks reuse across
  machines).
- **One decode-cache summation order.** The in-memory and memory-mapped
  engine paths sum PEP rows in different orders above 256 codewords and can
  select different codebooks; the facade routes around it (see Parity).
- **float32 ties.** The BLAS decoding-metric path computes costs in float32,
  which breaks some exact cost ties that unique-minimum decoding counts as
  errors (xfails in `tests/test_blas_decoding_metric.py`). Which ties break
  depends on the CPU model, through numpy's OpenBLAS kernel: the tests fail on
  the paper machine and pass on some GitHub runners, so CPU-path decode counts
  at exact ties for NLL metrics reproduce only on a similar CPU.
- **Move the runners onto the facade**: the runners become YAML adapters
  over `duet.api`.

## Amendment 2026-10-01 (0.2.0): `duet.design`

- **Context.** `design_ops(lambdas=[1])` was the decoding-only route: OPS-named, a default sweep
  of 6 λ, and a result with a meaningless `mean_score`, `max_score` row, front,
  `operating_point()` and `plot()`.
- **Decision.** `duet.design(pool, channel, ...)` runs the λ = 1 search alone through the shared
  `_design` path and evaluates [initial, designed]. Result kind `"decoding"`, rows `initial` and
  `designed`, no `lambda`, objective or `on_pareto_front` column; the provenance has the same
  keys as OPS with `front_axis` null; `codebook()` without argument returns the designed codebook.
- **Rejected.** A decode-only `ObjectiveSpec` (new engine path; parity only by 0·Δ = 0); a new
  result class (a second save format); evaluating a hidden `max_score` (not reproducible by
  `duet.evaluate`).
- **Compatibility.** Additive for existing callers; `format_version` stays 1. DUET 0.1.0 loads a
  `decoding` result but misbehaves (`secondary_objective` returns "crowding", `pareto_front()`
  raises KeyError). `codebook()` without argument on a sweep raises a TypeError with a new
  message; `secondary_objective` returns None for unknown kinds.
- **Future.** An objective-taking `duet.design` can add `objective=None, lambdas=None`; with
  `objective=None` it must return exactly this result.
