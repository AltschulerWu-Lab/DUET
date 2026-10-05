# ADR 0001: λ weights decoding accuracy

- **Status:** accepted
- **Date:** 2026-09-11

## Context

DUET scalarizes two objectives with a single sweep parameter λ. Until this
decision the code and the manuscript disagreed about which objective λ
weights:

- **Code (2026-04 to 2026-09-10):** `J = (1-λ)·decode + λ·secondary`.
  λ = 0 was decode-only, λ = 1 was score-only (OPS) or crowding-only (MERFISH).
  The greedy multi-objective baselines used the same blend.
- **Manuscript:** `J = λ·(1 - P̃(error)) + (1-λ)·ā`, "λ = 1 gives the
  decoding-only objective". The method schematic, Results, Methods and the
  supplementary greedy-baseline description all use this form, and every λ
  quoted in prose (0.12, 0.07, 0.80, 0.90, ...) is in this convention.

Panels assembled from result files therefore carried labels in the code
convention (`DUET (λ=0.88)`) that read backwards against the captions, and
several hand edits were needed to reconcile them. Two internal notes had
recommended keeping the code and relabelling figures instead.

## Decision

Flip the **code** to the manuscript convention rather than the manuscript to
the code:

- `_ops_weights(λ)` returns `{"decode": λ, "score": 1-λ}` and
  `_merfish_weights(λ)` returns `{"decode": λ, "crowding": 1-λ}`.
- `GreedyHammingMOOptimizer` and `GreedyNLLMOOptimizer` blend
  `λ·distance + (1-λ)·score`.
- λ = 1 is decode-only everywhere; λ = 0 is secondary-only everywhere.
- The YAML key stays `lambda`; no compatibility marker or key rename. The
  public repository starts from a fresh history, so its users only see the
  new convention.
- The per-λ optimizer seed stays `seed + int(λ·1000)`. A run at a given
  physical operating point therefore draws a different tie-break RNG stream
  than the archived run at the same point; regenerated results are new trials,
  not bit-for-bit relabelings.
- Every live config's λ list was rewritten as `1 - x` so each config still
  sweeps the same physical operating points (dense near the activity-heavy
  end for OPS, matching the Methods MERFISH grid verbatim).
- `MerfishBenchmarkConfig` now rejects any λ < 1 when no `crowding` block is
  present, because the runner substitutes a placeholder uniform expression
  vector that is only sound at zero crowding weight.
- Code that located the decode-only endpoint by "smallest λ" / `λ == 0.0`
  was mirrored in place (largest λ / `λ == 1.0`); derived names follow the
  literal flip (`*_at_lambda_one`, `true_accuracy_last_lambda`).

## Consequences

- **Archived results are untouched and stay in the old convention.**
  Archived results, scripts, knowledge notes and notebooks dated before
  2026-09-11 (not part of this repository) name runs by `λ_old = 1 - λ`: their
  result directories, `selected_codewords_lambda*.csv` files, `lambda` columns
  of `metrics.csv` and SVG legends. Convert before quoting into prose.
- Re-running an archived YAML with current code produces the mirrored sweep.
  Rewrite its λ list as `1 - x` first.
- Regression goldens (`tests/regression/data/*_snapshot.json` and
  `scripts/benchmark/regression/*/reference/`) were regenerated under the new
  convention and seeds.
- Regenerating any manuscript panel with current code yields labels in the
  manuscript convention directly (for example `DUET (λ=0.12)` for the
  97.5%-activity operating point of the medium-scale CRISPRi benchmark).
