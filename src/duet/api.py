"""Public entry points of DUET: :func:`design`, :func:`design_ops`, :func:`design_merfish` and :func:`evaluate`.

The functions here are a thin facade over the engine the paper's benchmark
runners use. They build the same internal objects (``CandidatePool``,
``EvaluatorConfig``, ``DuetOptimizerConfig``) and call the same functions
(``run_duet_ops``, ``run_duet_merfish``, ``evaluate_codebooks_by_sequence``,
``compute_metrics``). What the facade adds changes no number: seed
derivation, validation, the choice of PEP storage, saving and restoring the
caller's random state, and provenance. See ``docs/adr/0003-public-api.md``.

Two Monte Carlo stages run by default:

1. *Design.* The pairwise error probability (PEP) matrix is estimated from
   ``num_samples`` simulated reads per unique codeword, and a lambda sweep
   (lambda = 1 alone for :func:`design`) optimizes the union-bound surrogate
   built from it (``surrogate_accuracy``).
2. *Evaluation.* Every codebook of the sweep, together with the initial
   codebook and the reference codebooks (none for :func:`design`), is
   evaluated in one pass with fresh reads (``eval_samples``) under an
   independent seed. This is the decoding accuracy the paper reports;
   ``eval_samples=0`` skips it.

GPUs compute the PEP and the evaluation (all visible GPUs with
``device="gpu:all"``); the lambda sweep runs on CPU, one process per lambda.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from duet.candidate_pool import CandidatePool
from duet.channels import Channel
from duet.results import (
    DesignResult,
    EvaluationResult,
    accuracy_summary,
    duplicate_count,
    non_dominated,
)

__all__ = [
    "design",
    "design_ops",
    "design_merfish",
    "evaluate",
    "derive_stage_seeds",
    "STAGES",
]

# Order of the per-stage child seeds spawned from the user's seed. Fixed:
# reordering it would change every derived seed (docs/adr/0003-public-api.md).
STAGES: Tuple[str, ...] = ("pool", "init", "pep", "evaluation", "optimizer")

# PEP counts are uint16 and hold 2 * num_samples (CodebookEvaluator.MAX_SAMPLES_UINT16).
MAX_SAMPLES = 32_767

DEFAULT_OPS_LAMBDAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
DEFAULT_MERFISH_LAMBDAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)

LAMBDA_CONVENTION = (
    "J = lambda * decoding accuracy + (1 - lambda) * secondary objective; "
    "lambda = 1 is decode-only, lambda = 0 secondary-only (docs/adr/0001)"
)

# The U x U uint16 PEP (2 U^2 bytes) is held in memory up to this size and
# memory-mapped from disk above it. Each lambda worker also builds its own
# in-memory copy, so the limit is kept small.
MEMORY_PEP_LIMIT_BYTES = 64 * 2**20


def _shared_state_batch() -> int:
    """Rows per batch of the memory-mapped path's decode state (build_shared_decode_state).

    The in-memory path sums the codebook's PEP rows in one call, the
    memory-mapped path in batches of this many rows. Up to one batch the two
    sums are bitwise identical; above it they can differ in the last bits,
    which changes how the optimizer breaks exact ties. The facade therefore
    holds the PEP in memory only for codebooks of at most this size.
    """
    import inspect

    from duet.pareto_optimization import build_shared_decode_state

    return int(inspect.signature(build_shared_decode_state).parameters["batch_size"].default)

# With device="auto" and no GPU, warn when the pool has at least this many
# unique codewords U. The PEP costs U^2: measured on 40 CPU threads it took
# 67 min at U = 11,628 with num_samples=10,000 (54 s on 4 GPUs), so about
# 2 min at U = 2,000 and about ten times that on 4 cores
# (examples/README.md, "Scaling").
CPU_WARN_UNIQUE_CODEWORDS = 2_000


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def derive_stage_seeds(seed: int) -> Dict[str, int]:
    """Independent per-stage seeds derived from one user seed.

    ``np.random.SeedSequence(seed).spawn(5)`` gives one child per stage, in
    the fixed order of :data:`STAGES`: pool sampling, initial codebook, PEP,
    evaluation, optimizer. Each child is turned into a 32-bit integer seed.

    Examples
    --------
    >>> from duet.api import derive_stage_seeds
    >>> sorted(derive_stage_seeds(0))
    ['evaluation', 'init', 'optimizer', 'pep', 'pool']
    """
    children = np.random.SeedSequence(seed).spawn(len(STAGES))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(STAGES, children)
    }


def _check_seed(seed, name: str = "seed") -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"{name} must be a non-negative integer or None, got {seed!r}")
    return int(seed)


def _resolve_seeds(seed, stage_seeds) -> Tuple[int, Dict[str, int]]:
    if seed is None:
        # Fresh OS entropy; the caller's random state is not touched.
        seed = int(np.random.SeedSequence().entropy)
    seed = _check_seed(seed)
    seeds = derive_stage_seeds(seed)
    if stage_seeds:
        unknown = set(stage_seeds) - set(STAGES)
        if unknown:
            raise ValueError(f"_stage_seeds has unknown stages {sorted(unknown)}; stages: {list(STAGES)}")
        seeds.update({k: _check_seed(v, f"_stage_seeds[{k!r}]") for k, v in stage_seeds.items()})
    if seeds["pep"] == seeds["evaluation"]:
        raise ValueError(
            "the PEP and the evaluation must not share a seed: their Monte Carlo "
            "streams would line up by index"
        )
    return seed, seeds


@contextlib.contextmanager
def _preserve_global_rng():
    """Restore Python's and NumPy's global random state on exit.

    ``DUET.optimize`` reseeds both global generators; in a serial sweep that
    would change the caller's next draws.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_samples(value, name: str, allow_zero: bool = False) -> int:
    low = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= MAX_SAMPLES:
        extra = " (0 skips the evaluation)" if allow_zero else ""
        raise ValueError(
            f"{name} must be an integer from {low} to {MAX_SAMPLES:,}{extra}: the PEP "
            f"and evaluation counts are stored as uint16; got {value!r}"
        )
    return int(value)


def _check_lambdas(lambdas) -> List[float]:
    if isinstance(lambdas, (int, float, np.floating, np.integer)):
        lambdas = [lambdas]
    lams = [float(x) for x in lambdas]
    if not lams:
        raise ValueError("lambdas is empty; give at least one value in [0, 1]")
    bad = [x for x in lams if not np.isfinite(x) or not 0.0 <= x <= 1.0]
    if bad:
        raise ValueError(
            f"lambda values must lie in [0, 1] (lambda = 1 is decode-only), got {bad}"
        )
    if len(set(lams)) != len(lams):
        raise ValueError(f"lambdas repeats a value: {lams}")
    # The optimizer seeds lambda with seed + int(1000 * lambda)
    # (duet.runner.core._run_duet_core), so two lambdas with the same
    # int(1000 * lambda) would share an optimizer seed.
    keys: Dict[int, float] = {}
    for x in lams:
        k = int(x * 1000)
        if k in keys:
            raise ValueError(
                f"lambdas {keys[k]} and {x} would share an optimizer seed "
                "(seed + int(1000 * lambda)); use values that differ by at least 0.001"
            )
        keys[k] = x
    return sorted(lams)


def _check_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _pool_alphabet(pool: CandidatePool) -> str:
    alphabet = pool.metadata.get("alphabet")
    if alphabet:
        return alphabet
    if isinstance(pool.sequences, np.ndarray):
        raise ValueError(
            "the public API needs sequences as strings; build the pool with "
            "duet.CandidatePool.from_table"
        )
    symbols = set().union(*(set(s) for s in pool.unique_sequences))
    if symbols <= set("01"):
        return "01"
    if symbols <= set("ACGT"):
        return "ACGT"
    raise ValueError(
        f"candidate sequences use the symbols {sorted(symbols)}; DUET's public API "
        "supports DNA (A, C, G, T) and binary (0, 1) codewords"
    )


def _rule_config(rule) -> Dict[str, Any]:
    from duet.codebook_evaluator import MarginDecoding, UniqueMinimum

    if rule is None or (isinstance(rule, str) and rule == "unique_minimum") or isinstance(rule, UniqueMinimum):
        return {"type": "unique_minimum"}
    if isinstance(rule, MarginDecoding):
        k = float(rule.k)
        if not np.isfinite(k) or k < 0:
            raise ValueError(
                f"MarginDecoding margin k must be finite and >= 0, got {rule.k!r}: with "
                "k < 0 a codeword no longer competes with itself, which the "
                "memory-mapped PEP assumes"
            )
        return {"type": "margin", "margin": k}
    raise ValueError(
        "rule must be 'unique_minimum' (the default), duet.UniqueMinimum() or "
        "duet.MarginDecoding(k); other decoding rules are available through "
        "duet.evaluator_config.EvaluatorConfig (advanced layer)"
    )


def _evaluator_config(channel: Channel, rule_cfg, num_samples: int, seed: int, n_jobs: int):
    from duet.evaluator_config import ComponentConfig, EvaluatorConfig

    noise, metric = channel.component_configs()
    return EvaluatorConfig(
        noise_channel=ComponentConfig.from_dict(noise),
        decoding_metric=ComponentConfig.from_dict(metric),
        decoding_rule=ComponentConfig.from_dict(rule_cfg),
        num_samples=num_samples,
        num_cpus=n_jobs,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Devices and workers
# ---------------------------------------------------------------------------


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def _resolve_device(device, n_unique: int) -> str:
    """Resolve ``device`` to an engine device string, validating GPU requests."""
    if device == "auto":
        from duet.gpu_utils import has_gpu

        if has_gpu():
            return "gpu:all"
        if n_unique >= CPU_WARN_UNIQUE_CODEWORDS:
            warnings.warn(
                f"No GPU found, so DUET runs on CPU. This problem has U = {n_unique:,} "
                "unique codewords; the PEP (U x U) and the evaluation will be slow on "
                'CPU. GPUs are recommended: install the GPU extra (pip install '
                '"duet-codebook[gpu]"). Pass device="cpu" to silence this warning.',
                stacklevel=3,
            )
        return "cpu"
    if device == "cpu":
        return "cpu"
    if isinstance(device, str) and device.startswith("gpu"):
        from duet.gpu_utils import parse_device, require_cupy

        require_cupy(device)
        parse_device(device)  # validates the GPU ids now, before any compute
        return device
    raise ValueError(
        f'device must be "auto", "cpu", "gpu", "gpu:all" or "gpu:<ids>", got {device!r}'
    )


def _workers(workers, n_lambdas: int) -> Tuple[int, int]:
    """(lambda-sweep processes, CPU processes for the PEP and evaluation)."""
    ncpu = _available_cpus()
    if workers is None:
        return min(n_lambdas, ncpu), ncpu
    workers = _check_positive_int(workers, "workers")
    return min(workers, n_lambdas), workers


@contextlib.contextmanager
def _output(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# ---------------------------------------------------------------------------
# PEP storage
# ---------------------------------------------------------------------------


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def _plan_storage(pool, pep_storage: str, cache_dir, sweep_workers: int) -> str:
    """Decide "memory" or "mmap" for the PEP and check that it fits, before any compute.

    Raises MemoryError / OSError with the space needed when it does not fit.
    Disk space in ``cache_dir`` is checked later, once cache hits are known.
    """
    import psutil

    U = len(pool.unique_sequences)
    mem_bytes = 2 * U * U
    # Each lambda worker builds its own in-memory copy of X = M + M^T with
    # uint32 temporaries: about 16 U^2 bytes per worker at peak.
    mem_needed = mem_bytes * (1 + 8 * sweep_workers)
    available = psutil.virtual_memory().available
    if pep_storage not in ("auto", "memory", "mmap"):
        raise ValueError(f'pep_storage must be "auto", "memory" or "mmap", got {pep_storage!r}')
    batch = _shared_state_batch()
    storage = pep_storage
    if storage == "auto":
        small = (mem_bytes <= MEMORY_PEP_LIMIT_BYTES and mem_needed <= available
                 and pool.total_selections <= batch)
        storage = "memory" if small else "mmap"
    if storage == "memory" and pool.total_selections > batch:
        raise ValueError(
            f'pep_storage="memory" is available for codebooks of at most {batch} '
            f"codewords; this one has {pool.total_selections:,}. Above that size the "
            "in-memory and memory-mapped engine paths sum the PEP in a different "
            "order and can break exact ties differently, so DUET uses the "
            'memory-mapped path, as the paper\'s runs did. Use pep_storage="auto" or "mmap".'
        )
    if storage == "memory" and mem_needed > available:
        raise MemoryError(
            f"The PEP for U = {U:,} unique codewords needs about {_gb(mem_needed)} of "
            f"memory with {sweep_workers} lambda workers, but {_gb(available)} is "
            'available. Use pep_storage="mmap" (with cache_dir= on a large disk) or '
            "fewer workers."
        )
    if storage == "mmap" and cache_dir is None:
        free = shutil.disk_usage(tempfile.gettempdir()).free
        if 6 * U * U > free:
            raise OSError(
                f"The PEP for U = {U:,} unique codewords needs {_gb(6 * U * U)} of free "
                f"disk while it is built ({_gb(4 * U * U)} once finished: a raw and a "
                f"symmetric copy), but only {_gb(free)} is free in the temporary "
                f"directory {tempfile.gettempdir()}. Pass cache_dir= on a larger disk."
            )
    return storage


@contextlib.contextmanager
def _pep_storage(pool, pep_config, alphabet_size: int, pep_storage: str, cache_dir,
                 sweep_workers: int, device: str):
    """Choose where the PEP lives and yield the run_duet_* storage arguments.

    Yields a dict with ``cache_dir``, ``use_mmap`` and ``provenance``.

    The PEP is held in memory only when it is small and the codebook has at
    most :func:`_shared_state_batch` codewords, where the in-memory and
    memory-mapped engine paths are bitwise identical; otherwise it is
    memory-mapped, the path of the paper's runs. So the storage choice never
    changes the selections (tests/test_api.py pins it).
    """
    from duet.providers import PEPMatrixProvider

    U = len(pool.unique_sequences)
    mem_bytes = 2 * U * U
    storage = _plan_storage(pool, pep_storage, cache_dir, sweep_workers)

    tmpdir = None
    directory = None
    if cache_dir is not None:
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
    elif storage == "mmap":
        tmpdir = tempfile.mkdtemp(prefix="duet_pep_")
        directory = Path(tmpdir)

    prov: Dict[str, Any] = {
        "storage": storage,
        "unique_codewords": U,
        "memory_bytes": mem_bytes,
        "cache_dir": None if directory is None else ("temporary" if tmpdir else str(directory)),
    }
    try:
        if directory is None:
            prov.update(source="computed", computed_on=device)
        else:
            provider = PEPMatrixProvider(directory)
            status = provider.cache_status(pool.unique_sequences, pep_config, alphabet_size)
            hit = status["sym_hit"] or status["raw_hit"] if storage == "mmap" else status["raw_hit"]
            if storage == "mmap":
                need = 0 if status["sym_hit"] else (4 if status["raw_hit"] else 6) * U * U
            else:
                need = 0 if status["raw_hit"] else 2 * U * U
            free = shutil.disk_usage(directory).free
            if need > free:
                raise OSError(
                    f"The PEP for U = {U:,} unique codewords needs {_gb(need)} of free disk "
                    f"in {directory} while it is built ({_gb(4 * U * U)} once finished: a raw "
                    f"and a symmetric copy), but only {_gb(free)} is free. Pass cache_dir= "
                    "on a larger disk."
                )
            prov.update(
                disk_bytes=4 * U * U if storage == "mmap" else 2 * U * U,
                fingerprint=status["fingerprint"],
                source="cache" if hit else "computed",
                computed_on=(status["device"] or "unknown (cached before device metadata)")
                if hit else device,
            )
        yield {
            "cache_dir": directory,
            "use_mmap": storage == "mmap",
            "provenance": prov,
        }
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _surrogate_accuracies(pep_source, n_samples, unique_index_lists) -> List[float]:
    """The optimizer's union-bound surrogate for each codebook (NaN if a codeword has no PEP row).

    Computed with :func:`duet.benchmark.metrics.compute_duet_objective` on
    the raw PEP ``M`` (or on ``X / 2 = (M + M^T) / 2`` for a memory-mapped
    PEP, whose off-diagonal sums are the same), without the duplicate penalty.
    """
    from duet.benchmark.metrics import compute_duet_objective
    from duet.pep_accessor import PEPAccessor

    out = []
    for u in unique_index_lists:
        if u is None:
            out.append(float("nan"))
            continue
        u = np.asarray(u, dtype=np.int64)
        if isinstance(pep_source, PEPAccessor):
            saved = pep_source._duplicate_offset
            pep_source._duplicate_offset = 0.0
            try:
                sub = np.empty((len(u), len(u)), dtype=np.float64)
                for start in range(0, len(u), 256):  # bounded memory: 256 rows of U at a time
                    sub[start:start + 256] = 0.5 * pep_source.get_rows(u[start:start + 256])[:, u]
            finally:
                pep_source._duplicate_offset = saved
        else:
            sub = pep_source[np.ix_(u, u)].astype(np.float64) / n_samples
        out.append(float(compute_duet_objective(sub, np.arange(len(u)))))
    return out


# ---------------------------------------------------------------------------
# Shared design machinery
# ---------------------------------------------------------------------------


def _table_hash(pool: CandidatePool) -> str:
    """SHA-256 of the ordered candidate table (sequence, groups, score) plus quotas."""
    groups: Dict[int, List[str]] = {}
    for g, members in pool.group_to_candidates.items():
        for c in members:
            groups.setdefault(int(c), []).append(str(g))
    h = hashlib.sha256()
    for i, s in enumerate(pool.sequences):
        h.update(f"{s}\t{'|'.join(groups.get(i, []))}\t{float(pool.scores[i])!r}\n".encode())
    h.update(json.dumps([[str(g), int(q)] for g, q in pool.quotas.items()]).encode())
    return h.hexdigest()


def _unique_index_lists(pool: CandidatePool, sequence_lists) -> List[List[int] | None]:
    seq_to_u = {s: u for u, s in enumerate(pool.unique_sequences)}
    out = []
    for seqs in sequence_lists:
        if all(s in seq_to_u for s in seqs):
            out.append([seq_to_u[s] for s in seqs])
        else:
            out.append(None)
    return out


def _design(
    *,
    kind: str,
    pool: CandidatePool,
    channel: Channel,
    eval_channel: Channel,
    lambdas: List[float],
    root_seed: int,
    seeds: Dict[str, int],
    num_samples: int,
    eval_samples: int,
    device: str,
    rule,
    workers,
    cache_dir,
    pep_storage: str,
    optimizer: Dict[str, Any],
    verbose: bool,
    init_indices: np.ndarray,
    init_note: Dict[str, Any],
    secondary_fn,
    run_fn,
    references: List[Tuple[str, str, List[str], np.ndarray | None]],
    codebook_table_fn,
    extra_provenance: Dict[str, Any],
) -> DesignResult:
    """Run the sweep, evaluate every codebook together and assemble the result.

    ``secondary_fn=None`` is a decoding-only design (:func:`design`): lambda = 1
    alone and no references. The lambda codebook becomes the ``designed`` row,
    and the table has no ``lambda``, objective or ``on_pareto_front`` column;
    ``provenance["front_axis"]`` is None.
    """
    import duet
    from duet.benchmark.metrics import compute_metrics, evaluate_codebooks_by_sequence
    from duet.runner.core import DuetOptimizerConfig

    decoding_only = secondary_fn is None
    if decoding_only and (list(lambdas) != [1.0] or references):
        raise ValueError("internal: a decoding-only design runs lambda = 1 alone, without references")
    alphabet_size = channel.alphabet_size
    rule_cfg = _rule_config(rule)
    sweep_workers, cpu_jobs = _workers(workers, len(lambdas))
    pep_config = _evaluator_config(channel, rule_cfg, num_samples, seeds["pep"], cpu_jobs)
    optimizer_config = DuetOptimizerConfig(
        lambda_=list(lambdas),
        temperature=optimizer["temperature"],
        max_iter=optimizer["max_iter"],
        max_patience=optimizer["max_patience"],
        avoid_duplicates=optimizer["avoid_duplicates"],
        duplicate_penalty=optimizer["duplicate_penalty"],
        num_cpus=sweep_workers,
    )

    # Stage 1: PEP and lambda sweep.
    with _pep_storage(pool, pep_config, alphabet_size, pep_storage, cache_dir,
                      sweep_workers, device) as storage:
        with _output(verbose), _preserve_global_rng():
            run = run_fn(
                pep_config=pep_config,
                optimizer_config=optimizer_config,
                init=init_indices,
                alphabet_size=alphabet_size,
                seed=seeds["optimizer"],
                cache_dir=storage["cache_dir"],
                use_mmap=storage["use_mmap"],
                device=device,
            )
        lambda_indices = [np.asarray(ix, dtype=np.int64) for ix in run["best_indices"]]
        seq = pool.sequences
        rows = [("initial", "initial", np.nan, [seq[i] for i in init_indices], init_indices)]
        if decoding_only:
            (ix,) = lambda_indices
            rows.append(("designed", "designed", np.nan, [seq[i] for i in ix], ix))
        else:
            rows += [
                (f"lambda={lam:g}", "lambda", lam, [seq[i] for i in ix], ix)
                for lam, ix in zip(run["lambdas"], lambda_indices)
            ]
        rows += [(label, rkind, np.nan, seqs, ix) for label, rkind, seqs, ix in references]
        surrogate = _surrogate_accuracies(
            run["pep_count_matrix"], run["n_samples"],
            _unique_index_lists(pool, [r[3] for r in rows]),
        )
        del run
    pep_prov = storage["provenance"]

    # Stage 2: evaluation of every codebook together, in this order.
    per_codeword: List[np.ndarray | None] = [None] * len(rows)
    eval_prov: Dict[str, Any] = {"eval_samples": eval_samples}
    if eval_samples > 0:
        eval_config = _evaluator_config(eval_channel, rule_cfg, eval_samples,
                                        seeds["evaluation"], cpu_jobs)
        with _output(verbose):
            results = evaluate_codebooks_by_sequence(
                [r[3] for r in rows],
                eval_config=eval_config,
                alphabet_size=eval_channel.alphabet_size,
                n_jobs=cpu_jobs,
                device=device,
            )
        per_codeword = [r.codeword_accuracy for r in results]
        eval_prov.update(device=device, seed=seeds["evaluation"],
                         codebook_order=[r[0] for r in rows],
                         channel=eval_channel.to_dict())

    table_rows = []
    for (label, rkind, lam, seqs, ix), surr, acc in zip(rows, surrogate, per_codeword):
        row = {"codebook": label, "kind": rkind}
        if not decoding_only:
            row["lambda"] = lam
        if acc is not None:
            row.update(accuracy_summary(compute_metrics(np.asarray(acc))))
        else:
            row.update({"accuracy_mean": np.nan, "accuracy_p10": np.nan,
                        "accuracy_p95_p5_ratio": np.nan})
        row["surrogate_accuracy"] = surr
        if not decoding_only:
            row[secondary_fn.column] = secondary_fn(ix) if ix is not None else np.nan
        row["n_codewords"] = len(seqs)
        row["duplicate_codewords"] = duplicate_count(seqs)
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    acc_axis = None
    if not decoding_only:
        is_lambda = (table["kind"] == "lambda").to_numpy()
        acc_axis = "accuracy_mean" if eval_samples > 0 else "surrogate_accuracy"
        front = np.zeros(len(table), dtype=bool)
        front[is_lambda] = non_dominated(table.loc[is_lambda, acc_axis],
                                         table.loc[is_lambda, secondary_fn.column])
        table["on_pareto_front"] = front

    codebooks = {
        label: codebook_table_fn(label, rkind, seqs, ix, acc)
        for (label, rkind, lam, seqs, ix), acc in zip(rows, per_codeword)
    }

    provenance = {
        "duet_version": duet.__version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": kind,
        "lambda_convention": LAMBDA_CONVENTION,
        "lambdas": list(lambdas),
        "seed": root_seed,
        "stage_seed_order": list(STAGES),
        "stage_seeds": dict(seeds),
        "per_lambda_optimizer_seed": "(stage_seeds['optimizer'] + int(1000 * lambda)) % (2**31 - 1)",
        "devices": {"pep": device, "lambda_sweep": "cpu",
                    "evaluation": device if eval_samples > 0 else None},
        "workers": {"lambda_sweep": sweep_workers, "cpu_processes": cpu_jobs},
        "pep": {**pep_prov, "num_samples": num_samples, "seed": seeds["pep"]},
        "evaluation": eval_prov,
        "channel": channel.to_dict(),
        "decoding_rule": rule_cfg,
        "optimizer": dict(optimizer),
        "init": init_note,
        "pool": {
            "candidates": pool.pool_size,
            "groups": pool.num_groups,
            "codebook_size": pool.total_selections,
            "unique_codewords": len(pool.unique_sequences),
            "seq_length": pool.seq_length,
            "alphabet": _pool_alphabet(pool),
            "table_sha256": _table_hash(pool),
        },
        "front_axis": acc_axis,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        **extra_provenance,
    }
    return DesignResult(kind=kind, table=table, codebooks=codebooks, provenance=provenance)


class _Secondary:
    """Callable secondary objective with its table column name."""

    def __init__(self, column: str, fn):
        self.column = column
        self._fn = fn

    def __call__(self, indices) -> float:
        return float(self._fn(np.asarray(indices, dtype=np.int64)))


# ---------------------------------------------------------------------------
# OPS
# ---------------------------------------------------------------------------


def _ops_init(pool: CandidatePool, init, init_seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    from duet.initialization import BestScoreInit, RandomInit

    if isinstance(init, str):
        if init == "random":
            idx, meta = RandomInit().get_initial_indices(pool, init_seed)
        elif init == "best_score":
            idx, meta = BestScoreInit().get_initial_indices(pool, init_seed)
        else:
            raise ValueError(f'init must be "random", "best_score" or a codebook table, got {init!r}')
        return np.asarray(idx, dtype=np.int64), {"type": meta.description, "seed": init_seed}
    if isinstance(init, pd.DataFrame):
        return _codebook_table_to_indices(pool, init), {"type": "table", "rows": len(init)}
    raise ValueError(f'init must be "random", "best_score" or a codebook table, got {type(init).__name__}')


def _codebook_table_to_indices(pool: CandidatePool, df: pd.DataFrame) -> np.ndarray:
    """Map an initial codebook table onto the codeword positions of ``pool``.

    Each row names a group and a candidate of that group, by its ``candidate``
    index (the row in the candidate table, as in :meth:`DesignResult.codebook`)
    or by its sequence. A sequence as long as the table's input sequence is
    matched in full; a sequence of ``seq_length`` symbols is matched as a
    codeword and must then identify one candidate. When both a candidate
    index and a sequence are given they must agree. Rows of a group fill that
    group's positions in order.
    """
    cols = pool.metadata.get("columns", {})
    gcol = next((c for c in (cols.get("group"), "group") if c and c in df.columns), None)
    scol = next((c for c in (cols.get("sequence"), "sequence") if c and c in df.columns), None)
    has_candidate = "candidate" in df.columns
    if gcol is None or (scol is None and not has_candidate):
        raise ValueError(
            "the initial codebook table needs a group column "
            f"({cols.get('group') or 'group'!r}) and a sequence column "
            f"({cols.get('sequence') or 'sequence'!r}) or a 'candidate' column"
        )
    L = pool.seq_length
    full = pool.metadata.get("input_sequences")  # untruncated, from from_table
    need = dict(pool.quotas)
    positions: Dict[Any, List[int]] = {}
    for pos, g in enumerate(pool.codeword_to_group):
        positions.setdefault(g, []).append(pos)
    init = np.full(pool.total_selections, -1, dtype=np.int64)
    taken: Dict[Any, List[int]] = {g: [] for g in need}

    for row, r in df.iterrows():
        g = r[gcol]
        if g not in need and str(g) in need:
            g = str(g)
        if g not in need:
            raise ValueError(f"initial codebook, row {row}: group {r[gcol]!r} is not in the candidate pool")
        if len(taken[g]) >= need[g]:
            raise ValueError(f"initial codebook: group {g!r} has more rows than its quota ({need[g]})")
        members = pool.group_to_candidates[g]
        free = [c for c in members if c not in taken[g]]
        seq = None if scol is None else str(r[scol])

        def seq_matches(c):
            return seq == pool.sequences[c] or (full is not None and seq == full[c])

        if has_candidate:
            c = int(r["candidate"])
            if c not in members:
                raise ValueError(f"initial codebook, row {row}: candidate {c} is not in group {g!r}")
            if seq is not None and not seq_matches(c):
                raise ValueError(
                    f"initial codebook, row {row}: candidate {c} has sequence "
                    f"{pool.sequences[c]!r}, not {seq!r}; the table may come from a "
                    "different candidate pool"
                )
        else:
            if full is not None and len(seq) > L:
                match = [c for c in free if full[c] == seq]
            else:
                match = [c for c in free if pool.sequences[c] == seq[:L]]
                if full is not None and len({full[c] for c in match}) > 1:
                    raise ValueError(
                        f"initial codebook, row {row}: codeword {seq[:L]!r} matches several "
                        f"candidates of group {g!r} with different full sequences; give the "
                        "full sequence or a 'candidate' column"
                    )
            if not match:
                if any(seq_matches(c) for c in members):
                    raise ValueError(
                        f"initial codebook: sequence {seq!r} appears more often in group "
                        f"{g!r} than the group has candidates with that sequence"
                    )
                raise ValueError(
                    f"initial codebook, row {row}: sequence {seq!r} is not a candidate of group {g!r}"
                )
            c = match[0]
        init[positions[g][len(taken[g])]] = c
        taken[g].append(c)

    short = {g: need[g] - len(taken[g]) for g in need if len(taken[g]) < need[g]}
    if short:
        some = dict(list(short.items())[:5])
        raise ValueError(
            f"initial codebook: {len(short)} groups have fewer rows than their quota, "
            f"for example {some} (group: missing rows)"
        )
    return init


def _score_is_constant_within_groups(pool: CandidatePool) -> bool:
    """True when no within-group swap can change the mean score (the OPS objective)."""
    scores = np.asarray(pool.scores, dtype=float)
    return all(np.ptp(scores[list(ix)]) == 0 for ix in pool.group_to_candidates.values() if len(ix))


def design_ops(
    pool: CandidatePool,
    channel: Channel,
    *,
    lambdas: Sequence[float] = DEFAULT_OPS_LAMBDAS,
    seed: int | None = None,
    num_samples: int = 10_000,
    eval_samples: int = 5_000,
    device: str = "auto",
    init="random",
    rule="unique_minimum",
    eval_channel: Channel | None = None,
    workers: int | None = None,
    cache_dir=None,
    max_patience: int = 3000,
    max_iter: int = 100_000,
    avoid_duplicates: bool = True,
    duplicate_penalty: float = 100.0,
    temperature: float = 0.0,
    pep_storage: str = "auto",
    verbose: bool = True,
    _stage_seeds: Mapping[str, int] | None = None,
) -> DesignResult:
    """Design OPS codebooks: decoding accuracy against mean candidate score.

    Runs one lambda sweep, J = lambda * decoding + (1 - lambda) * mean score,
    then evaluates every resulting codebook, the initial codebook and the
    maximum-score codebook together with fresh reads. Returns the Pareto front
    to choose from. GPUs are recommended (install the ``gpu`` extra): they
    compute the PEP and the evaluation, while the lambda sweep runs on CPU,
    one process per lambda.

    Lambda weighs the mean score against decoding accuracy on their raw
    scales, so what a given lambda means depends on the units of your scores.

    Parameters
    ----------
    pool : CandidatePool
        Candidates, from :meth:`CandidatePool.from_table`. Row order seeds the
        Monte Carlo draws: re-sorting the table redraws them.
    channel : Channel
        Noise channel from :mod:`duet.channels`; its alphabet must match the
        pool's.
    lambdas : sequence of float, default (0, 0.05, 0.1, 0.25, 0.5, 1)
        Values in [0, 1]; 1 is decode-only (:func:`design` returns that
        codebook alone), 0 score-only. Must be distinct, also after
        ``int(1000 * lambda)`` (the per-lambda seed rule).
    seed : int, optional
        One seed for the whole design; independent per-stage seeds are
        derived from it (see :func:`derive_stage_seeds`). None draws one and
        records it in ``provenance``.
    num_samples : int, default 10,000
        Simulated reads per unique codeword for the PEP (at most 32,767).
    eval_samples : int, default 5,000
        Reads per codeword for the evaluation stage; 0 skips it.
    device : str, default "auto"
        ``"auto"`` (all GPUs when CuPy sees one, else CPU with a warning for
        large problems), ``"cpu"``, ``"gpu"`` (GPU 0 only), ``"gpu:all"`` or
        ``"gpu:0,2"``.
    init : "random", "best_score" or DataFrame, default "random"
        The initial codebook, shared by every lambda. A table needs the
        pool's group and sequence columns (or a ``candidate`` column).
    rule : default "unique_minimum"
        Decoding rule: ``"unique_minimum"``, ``duet.UniqueMinimum()`` or the
        margin decoder ``duet.MarginDecoding(k)``.
    eval_channel : Channel, optional
        Channel for the evaluation stage, to judge designs under a channel
        other than the one they were optimized for. Defaults to ``channel``.
        The evaluation then decodes with the metric matched to
        ``eval_channel``; ``surrogate_accuracy`` stays under ``channel``.
    workers : int, optional
        CPU processes. The lambda sweep uses min(workers, len(lambdas)); the
        CPU PEP and evaluation use ``workers``. Default: min(len(lambdas),
        CPU count) for the sweep and the CPU count otherwise.
    cache_dir : path, optional
        Directory for the PEP cache. A cached PEP is reused by a later run
        with the same unique codewords (in the same order), channel, rule,
        ``num_samples``, PEP seed and number of CPU processes (``workers``,
        which defaults to the machine's CPU count, so pass it explicitly to
        reuse a cache on another machine). The cache key ignores the device,
        so a GPU-computed PEP is reused on CPU; ``provenance`` records which
        device computed it. Without ``cache_dir`` the PEP is held in memory
        when small, else memory-mapped from a temporary directory deleted
        after the run.
    max_patience : int, default 3000
        Stop a lambda's search after this many consecutive swaps without a
        new best objective.
    max_iter : int, default 100,000
        Maximum number of swaps per lambda.
    avoid_duplicates : bool, default True
        Penalize codebooks in which two entries share a codeword (such
        entries cannot be decoded). The penalty sits in the decoding term, so
        it has no effect at lambda = 0.
    duplicate_penalty : float, default 100.0
        Size of that penalty.
    temperature : float, default 0.0
        0 applies the best swap at every step (ties broken at random); a
        positive value samples swaps with probability proportional to
        exp(gain / temperature).
    pep_storage : {"auto", "memory", "mmap"}, default "auto"
        Where the PEP lives. ``"auto"`` holds it in memory when it is small
        (2U² <= 64 MiB) and the codebook has at most 256 codewords, and
        memory-maps it otherwise; ``"memory"`` is refused above 256 codewords.
        Storage never changes the selections.
    verbose : bool, default True
        Show the engine's progress output.

    Returns
    -------
    DesignResult

    Warns
    -----
    UserWarning
        When the scores do not vary within any candidate set (for example a
        pool built without ``score``): every codebook then has the same mean
        score, so use :func:`duet.design` to optimize decoding alone.

    Examples
    --------
    >>> import duet
    >>> pool = duet.CandidatePool.from_table(df, group="gene", sequence="barcode",
    ...                                      score="activity", quota=2, seq_length=10)  # doctest: +SKIP
    >>> res = duet.design_ops(pool, duet.channels.symmetric(0.1), seed=0)            # doctest: +SKIP
    >>> res.pareto_front()                                                           # doctest: +SKIP
    """
    from duet.initialization import BestScoreInit
    from duet.runner.ops import _build_score_cache, run_duet_ops

    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a duet.CandidatePool (see CandidatePool.from_table)")
    alphabet = _pool_alphabet(pool)
    channel.check_compatible(alphabet, pool.seq_length)
    eval_channel = channel if eval_channel is None else eval_channel
    eval_channel.check_compatible(alphabet, pool.seq_length)
    if not np.isfinite(np.asarray(pool.scores, dtype=float)).all():
        raise ValueError(
            "pool scores must be finite numbers; every candidate needs a score, "
            "controls included (the paper's pool gives non-targeting controls 1.0)"
        )
    if pool.total_selections < 1:
        raise ValueError("every group has quota 0: the codebook would be empty")
    if _score_is_constant_within_groups(pool):
        warnings.warn(
            "The scores do not vary within any candidate set, so every codebook has the same "
            "mean score: lambda = 0 has nothing to optimize and every lambda > 0 optimizes "
            "decoding accuracy alone. duet.design does that in one search, without the "
            "score columns.",
            stacklevel=2,
        )
    lambdas = _check_lambdas(lambdas)
    num_samples = _check_samples(num_samples, "num_samples")
    eval_samples = _check_samples(eval_samples, "eval_samples", allow_zero=True)
    root_seed, seeds = _resolve_seeds(seed, _stage_seeds)
    device = _resolve_device(device, len(pool.unique_sequences))
    init_indices, init_note = _ops_init(pool, init, seeds["init"])

    score_cache = _build_score_cache(pool, codeword_to_group=pool.codeword_to_group)
    secondary = _Secondary("mean_score", score_cache.compute_objective)
    max_idx, _ = BestScoreInit().get_initial_indices(pool, seeds["init"])
    max_idx = np.asarray(max_idx, dtype=np.int64)
    references = [("max_score", "max_score", [pool.sequences[i] for i in max_idx], max_idx)]

    def run_fn(**kw):
        return run_duet_ops(candidates=pool, **kw)

    def codebook_table(label, rkind, seqs, ix, acc):
        return pd.DataFrame({
            "group": pool.codeword_to_group,
            "sequence": seqs,
            "score": pool.scores[ix],
            "accuracy": np.nan if acc is None else np.asarray(acc, dtype=float),
            "candidate": ix,
        })

    return _design(
        kind="ops", pool=pool, channel=channel, eval_channel=eval_channel, lambdas=lambdas,
        root_seed=root_seed, seeds=seeds, num_samples=num_samples, eval_samples=eval_samples,
        device=device, rule=rule, workers=workers, cache_dir=cache_dir,
        pep_storage=pep_storage,
        optimizer={"max_patience": int(max_patience), "max_iter": int(max_iter),
                   "avoid_duplicates": bool(avoid_duplicates),
                   "duplicate_penalty": float(duplicate_penalty),
                   "temperature": float(temperature)},
        verbose=verbose, init_indices=init_indices, init_note=init_note,
        secondary_fn=secondary, run_fn=run_fn, references=references,
        codebook_table_fn=codebook_table, extra_provenance={},
    )


# ---------------------------------------------------------------------------
# Decoding-only design
# ---------------------------------------------------------------------------


def design(
    pool: CandidatePool,
    channel: Channel,
    *,
    seed: int | None = None,
    num_samples: int = 10_000,
    eval_samples: int = 5_000,
    device: str = "auto",
    init="random",
    rule="unique_minimum",
    eval_channel: Channel | None = None,
    workers: int | None = None,
    cache_dir=None,
    max_patience: int = 3000,
    max_iter: int = 100_000,
    avoid_duplicates: bool = True,
    duplicate_penalty: float = 100.0,
    temperature: float = 0.0,
    pep_storage: str = "auto",
    verbose: bool = True,
    _stage_seeds: Mapping[str, int] | None = None,
) -> DesignResult:
    """Design a codebook for decoding accuracy alone.

    Starts from an initial codebook and swaps candidates within their sets to
    maximize decoding accuracy (one search: the lambda = 1 run of
    :func:`design_ops`), then evaluates the initial and the designed codebook
    together with fresh reads. The pool's scores are not used. With the same
    seed and settings, ``design_ops(pool, channel, lambdas=[1])`` selects the
    same codebook. GPUs are recommended for large pools (install the ``gpu``
    extra): they compute the PEP and the evaluation, while the search runs on
    CPU in one process.

    Parameters
    ----------
    pool : CandidatePool
        Candidates, from :meth:`CandidatePool.from_table` (DNA or binary).
        Row order seeds the Monte Carlo draws: re-sorting the table redraws
        them. Scores, if any, are ignored.
    channel : Channel
        Noise channel from :mod:`duet.channels`; its alphabet must match the
        pool's.
    init : "random" or DataFrame, default "random"
        The initial codebook: random, or a table with the pool's group and
        sequence columns (or a ``candidate`` column), ``quota`` rows per
        group, such as a library you already use or ``res.codebook()`` of a
        design on the same pool.
    seed, num_samples, eval_samples, device, rule, eval_channel, workers,
    cache_dir, max_patience, max_iter, avoid_duplicates, duplicate_penalty,
    temperature, pep_storage, verbose
        As in :func:`design_ops`. The search runs in one process; ``workers``
        sets the CPU processes of the PEP and the evaluation.

    Returns
    -------
    DesignResult
        Of kind ``"decoding"``, with the rows ``initial`` and ``designed``;
        ``res.codebook()`` returns the designed codebook.

    Examples
    --------
    >>> import duet
    >>> pool = duet.CandidatePool.from_table(df, group="set", sequence="barcode", quota=12)  # doctest: +SKIP
    >>> res = duet.design(pool, duet.channels.symmetric(0.1), seed=0)                       # doctest: +SKIP
    >>> res.codebook()                                                                       # doctest: +SKIP
    """
    from duet.runner.ops import run_duet_ops

    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be a duet.CandidatePool (see CandidatePool.from_table)")
    if isinstance(init, str) and init == "best_score":
        raise ValueError(
            'init="best_score" is not offered: duet.design ignores scores. Pass the codebook '
            "to start from as a table, or use duet.design_ops to trade decoding against the score"
        )
    if not (isinstance(init, pd.DataFrame) or (isinstance(init, str) and init == "random")):
        got = repr(init) if isinstance(init, str) else type(init).__name__
        raise ValueError(
            'init must be "random" or a codebook table (the pool\'s group column and a '
            f"sequence or candidate column), got {got}"
        )
    alphabet = _pool_alphabet(pool)
    channel.check_compatible(alphabet, pool.seq_length)
    eval_channel = channel if eval_channel is None else eval_channel
    eval_channel.check_compatible(alphabet, pool.seq_length)
    if not np.isfinite(np.asarray(pool.scores, dtype=float)).all():
        # The lambda = 1 search still builds the score cache and weighs its
        # deltas by 0.0; a NaN or infinite score would turn that into NaN.
        raise ValueError(
            "pool scores must be finite numbers: duet.design gives them no weight, but the "
            "search engine reads them; build the pool with duet.CandidatePool.from_table"
        )
    if pool.total_selections < 1:
        raise ValueError("every group has quota 0: the codebook would be empty")
    num_samples = _check_samples(num_samples, "num_samples")
    eval_samples = _check_samples(eval_samples, "eval_samples", allow_zero=True)
    root_seed, seeds = _resolve_seeds(seed, _stage_seeds)
    device = _resolve_device(device, len(pool.unique_sequences))
    init_indices, init_note = _ops_init(pool, init, seeds["init"])

    def run_fn(**kw):
        return run_duet_ops(candidates=pool, **kw)

    def codebook_table(label, rkind, seqs, ix, acc):
        return pd.DataFrame({
            "group": pool.codeword_to_group,
            "sequence": seqs,
            "accuracy": np.nan if acc is None else np.asarray(acc, dtype=float),
            "candidate": ix,
        })

    return _design(
        kind="decoding", pool=pool, channel=channel, eval_channel=eval_channel, lambdas=[1.0],
        root_seed=root_seed, seeds=seeds, num_samples=num_samples, eval_samples=eval_samples,
        device=device, rule=rule, workers=workers, cache_dir=cache_dir,
        pep_storage=pep_storage,
        optimizer={"max_patience": int(max_patience), "max_iter": int(max_iter),
                   "avoid_duplicates": bool(avoid_duplicates),
                   "duplicate_penalty": float(duplicate_penalty),
                   "temperature": float(temperature)},
        verbose=verbose, init_indices=init_indices, init_note=init_note,
        secondary_fn=None, run_fn=run_fn, references=[],
        codebook_table_fn=codebook_table, extra_provenance={},
    )


# ---------------------------------------------------------------------------
# MERFISH
# ---------------------------------------------------------------------------


def _barcode_column(df: pd.DataFrame, barcode: str, what: str) -> str:
    for c in (barcode, "barcode", "sequence", "Sequence"):
        if c in df.columns:
            return c
    raise ValueError(f"{what} needs a barcode column ({barcode!r}); columns: {list(df.columns)}")


def _check_barcode(b, n_bits: int, what: str) -> str:
    b = str(b)
    if len(b) != n_bits or set(b) - {"0", "1"}:
        raise ValueError(f"{what}: barcode {b!r} is not a string of {n_bits} bits (0/1)")
    return b


def _build_merfish_pool(
    gene_names: List[str],
    n_bits: int,
    hamming_weights: Mapping[int, int | None],
    candidates_per_gene: int | None,
    init_barcodes: List[str] | None,
    includes: List[Tuple[str, List[str], List[str]]],
    pool_seed: int,
) -> CandidatePool:
    """The MERFISH candidate pool, built as the MERFISH benchmark runner builds it.

    Anchoring follows ``duet/merfish_benchmark/runner.py`` (Step 4): the
    initial codebook's barcode of gene ``i`` is required at position ``i``,
    and each ``include`` codebook anchors its barcode at the positions of the
    panel genes it covers. Both factory seeds are the pool-stage seed.
    Pinned by the MERFISH pool parity tests.
    """
    from duet.merfish_factory import MERFISHFactory

    required: List[set] = [set() for _ in gene_names]
    if init_barcodes is not None:
        for i, b in enumerate(init_barcodes):
            required[i].add(b)
    for _name, genes, barcodes in includes:
        gene_to_seq = dict(zip(genes, barcodes))
        for i, g in enumerate(gene_names):
            if g in gene_to_seq:
                required[i].add(gene_to_seq[g])
    limits = {int(w): int(c) for w, c in hamming_weights.items() if c is not None}
    factory = MERFISHFactory(
        seq_rounds=int(n_bits),
        codebook_size=len(gene_names),
        hamming_weights=sorted(int(w) for w in hamming_weights),
        hamming_weight_limits=limits or None,
        subsample_seed=int(pool_seed),
        required_codewords=required,
        per_codeword_sample_size=None if candidates_per_gene is None else int(candidates_per_gene),
        sample_seed=int(pool_seed),
    )
    return factory.create()


def design_merfish(
    genes: pd.DataFrame,
    channel: Channel,
    *,
    gene: str = "gene",
    expression: str = "expression",
    n_bits: int,
    hamming_weights: Mapping[int, int | None],
    candidates_per_gene: int | None = 1000,
    init="random",
    include=None,
    barcode: str = "barcode",
    lambdas: Sequence[float] = DEFAULT_MERFISH_LAMBDAS,
    seed: int | None = None,
    num_samples: int = 10_000,
    eval_samples: int = 5_000,
    device: str = "auto",
    rule="unique_minimum",
    eval_channel: Channel | None = None,
    workers: int | None = None,
    cache_dir=None,
    max_patience: int = 3000,
    max_iter: int = 100_000,
    avoid_duplicates: bool = True,
    duplicate_penalty: float = 100.0,
    temperature: float = 0.0,
    pep_storage: str = "auto",
    verbose: bool = True,
    _stage_seeds: Mapping[str, int] | None = None,
) -> DesignResult:
    """Design MERFISH codebooks: decoding accuracy against optical crowding.

    Builds the candidate pool (binary barcodes of the allowed Hamming weights,
    ``candidates_per_gene`` per gene), runs one lambda sweep, J = lambda *
    decoding + (1 - lambda) * crowding objective, and evaluates every codebook
    together with fresh reads. The crowding objective is 1 - C(S)/C(S0),
    anchored at the initial codebook S0. GPUs are recommended (install the
    ``gpu`` extra): they compute the PEP and the evaluation, while the lambda
    sweep runs on CPU, one process per lambda.

    Parameters
    ----------
    genes : DataFrame
        The gene panel, one row per gene, in panel order; codeword position
        ``i`` belongs to row ``i``.
    channel : Channel
        A binary channel, e.g. ``duet.channels.asymmetric(T, alphabet="01")``.
    gene, expression : str
        Column names in ``genes`` (expression: non-negative, for example
        counts per million). ``gene`` also names the gene column of the
        ``init`` and ``include`` tables.
    n_bits : int
        Barcode length.
    hamming_weights : dict {weight: cap or None}
        Allowed Hamming weights; a cap keeps that many barcodes of that
        weight in the candidate pool, the anchored ones (from ``init`` and
        ``include``) included and the rest sampled at random; None keeps all.
        More anchored barcodes than the cap is an error. Without caps,
        32 bits at weights 4 and 5 give 237,336 barcodes (about 113 GB per
        PEP copy).
    candidates_per_gene : int or None, default 1000
        Barcodes sampled per gene (the gene's anchored barcodes included).
        None puts every barcode in one group shared by all genes.
    init : "random" or DataFrame, default "random"
        Initial codebook S0: a table with the ``gene`` and ``barcode``
        columns, one barcode per panel gene.
    include : DataFrame, list or dict of DataFrames, optional
        Codebooks (``gene``, ``barcode`` columns) whose barcodes stay
        available to their genes, and which are evaluated alongside as
        references (for example a published codebook).
    barcode : str, default "barcode"
        Barcode column of ``init`` and ``include`` tables.
    lambdas, seed, num_samples, eval_samples, device, rule, eval_channel,
    workers, cache_dir, max_patience, max_iter, avoid_duplicates,
    duplicate_penalty, temperature, pep_storage, verbose
        As in :func:`design_ops`. Default lambdas: the paper's nine values.

    Returns
    -------
    DesignResult

    Examples
    --------
    >>> ch = duet.channels.asymmetric([[0.985, 0.015], [0.056, 0.944]], alphabet="01")  # doctest: +SKIP
    >>> res = duet.design_merfish(genes_df, ch, gene="gene", expression="cpm", n_bits=16,
    ...                           hamming_weights={4: None}, candidates_per_gene=200,
    ...                           init=published_df, seed=0)                           # doctest: +SKIP
    """
    from duet.initialization import RandomInit, _lookup_indices
    from duet.runner.merfish import _build_crowding_cache, run_duet_merfish

    # -- inputs ---------------------------------------------------------------
    if not isinstance(genes, pd.DataFrame):
        raise TypeError("genes must be a pandas DataFrame with gene and expression columns")
    for col in (gene, expression):
        if col not in genes.columns:
            raise ValueError(f"genes has no column {col!r}; columns: {list(genes.columns)}")
    gene_names = genes[gene].astype(str).tolist()
    dup = pd.Series(gene_names)[pd.Series(gene_names).duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"genes lists some genes more than once: {dup[:5]}")
    expr = pd.to_numeric(genes[expression], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(expr)) or np.any(expr < 0) or not np.any(expr > 0):
        raise ValueError(
            f"expression column {expression!r} must be finite and non-negative, with "
            "at least one positive value"
        )
    n_bits = _check_positive_int(n_bits, "n_bits")
    if not isinstance(hamming_weights, Mapping) or not hamming_weights:
        raise ValueError("hamming_weights must be a dict such as {4: None, 5: 64040}")
    for w, cap in hamming_weights.items():
        if isinstance(w, bool) or not isinstance(w, (int, np.integer)) or not 1 <= w <= n_bits:
            raise ValueError(f"hamming_weights: weight {w!r} must be an integer in [1, {n_bits}]")
        if cap is not None:
            _check_positive_int(cap, f"hamming_weights[{w}]")
    allowed = {int(w) for w in hamming_weights}
    if candidates_per_gene is not None:
        candidates_per_gene = _check_positive_int(candidates_per_gene, "candidates_per_gene")
    if frozenset(channel.alphabet) != frozenset("01"):
        raise ValueError(
            f"MERFISH barcodes are binary; build the channel with alphabet='01', "
            f"not {channel.alphabet!r}"
        )
    channel.check_compatible("01", n_bits)
    eval_channel = channel if eval_channel is None else eval_channel
    eval_channel.check_compatible("01", n_bits)
    lambdas = _check_lambdas(lambdas)
    num_samples = _check_samples(num_samples, "num_samples")
    eval_samples = _check_samples(eval_samples, "eval_samples", allow_zero=True)

    init_barcodes = None
    if isinstance(init, pd.DataFrame):
        bcol = _barcode_column(init, barcode, "init")
        if gene not in init.columns:
            raise ValueError(f"init needs a {gene!r} column naming each barcode's gene")
        by_gene = dict(zip(init[gene].astype(str), init[bcol]))
        missing = [g for g in gene_names if g not in by_gene]
        extra = sorted(set(by_gene) - set(gene_names))
        if missing or extra or len(init) != len(gene_names):
            raise ValueError(
                "init must give exactly one barcode per panel gene: "
                f"missing {missing[:5]}, not in the panel {extra[:5]}"
            )
        init_barcodes = [_check_barcode(by_gene[g], n_bits, f"init, gene {g}") for g in gene_names]
        bad_w = [(g, b.count("1")) for g, b in zip(gene_names, init_barcodes) if b.count("1") not in allowed]
        if bad_w:
            raise ValueError(
                f"init: barcodes with Hamming weights outside hamming_weights "
                f"{sorted(allowed)}, e.g. {bad_w[:3]}"
            )
    elif not (isinstance(init, str) and init == "random"):
        raise ValueError('init must be "random" or a table with gene and barcode columns')

    if include is None:
        include_items: List[Tuple[str, pd.DataFrame]] = []
    elif isinstance(include, pd.DataFrame):
        include_items = [("include_1", include)]
    elif isinstance(include, Mapping):
        include_items = [(str(k), v) for k, v in include.items()]
    else:
        include_items = [(f"include_{i + 1}", v) for i, v in enumerate(include)]
    includes = []
    for name, df in include_items:
        if name in ("initial",) or name.startswith("lambda="):
            raise ValueError(f"include name {name!r} is reserved")
        bcol = _barcode_column(df, barcode, f"include {name!r}")
        if gene not in df.columns:
            raise ValueError(f"include {name!r} needs a {gene!r} column")
        g_list = df[gene].astype(str).tolist()
        b_list = [_check_barcode(b, n_bits, f"include {name!r}") for b in df[bcol]]
        panel = set(gene_names)
        bad_w = [(g, b.count("1")) for g, b in zip(g_list, b_list)
                 if g in panel and b.count("1") not in allowed]
        if bad_w:
            raise ValueError(
                f"include {name!r}: barcodes of panel genes have Hamming weights outside "
                f"hamming_weights {sorted(allowed)}, e.g. {bad_w[:3]}"
            )
        includes.append((name, g_list, b_list))

    root_seed, seeds = _resolve_seeds(seed, _stage_seeds)

    # -- pool, device, initial codebook ----------------------------------------
    pool = _build_merfish_pool(gene_names, n_bits, hamming_weights, candidates_per_gene,
                               init_barcodes, includes, seeds["pool"])
    pool.metadata["alphabet"] = "01"
    device = _resolve_device(device, len(pool.unique_sequences))
    # Fail on a PEP that cannot fit before the crowding cache and the sweep.
    _plan_storage(pool, pep_storage, cache_dir, _workers(workers, len(lambdas))[0])
    if init_barcodes is not None:
        init_indices = _lookup_indices(init_barcodes, pool.sequences)
        init_note = {"type": "table", "rows": len(init_barcodes)}
    else:
        idx, _ = RandomInit().get_initial_indices(pool, seeds["init"])
        init_indices = np.asarray(idx, dtype=np.int64)
        init_note = {"type": "random", "seed": seeds["init"]}

    codewords = pool.get_sequences_as_array(alphabet_size=2).astype(np.float64)
    crowding_cache = _build_crowding_cache(codewords, expr, pool,
                                           codeword_to_group=pool.codeword_to_group)
    crowding_cache.build_cache(np.asarray(init_indices))  # anchors C(S0)
    secondary = _Secondary("crowding", crowding_cache.compute_objective)

    references = []
    for name, g_list, b_list in includes:
        gene_to_seq = dict(zip(g_list, b_list))
        ix = None
        if all(g in gene_to_seq for g in gene_names):
            ix = _lookup_indices([gene_to_seq[g] for g in gene_names], pool.sequences)
        references.append((name, "include", b_list, ix))
    include_genes = {name: g_list for name, g_list, _ in includes}
    expr_by_gene = dict(zip(gene_names, expr))

    def run_fn(**kw):
        return run_duet_merfish(candidates=pool, codewords=codewords, expression=expr, **kw)

    def codebook_table(label, rkind, seqs, ix, acc):
        g = include_genes[label] if rkind == "include" else gene_names
        return pd.DataFrame({
            "gene": g,
            "barcode": seqs,
            "expression": [expr_by_gene.get(x, np.nan) for x in g],
            "accuracy": np.nan if acc is None else np.asarray(acc, dtype=float),
        })

    return _design(
        kind="merfish", pool=pool, channel=channel, eval_channel=eval_channel, lambdas=lambdas,
        root_seed=root_seed, seeds=seeds, num_samples=num_samples, eval_samples=eval_samples,
        device=device, rule=rule, workers=workers, cache_dir=cache_dir,
        pep_storage=pep_storage,
        optimizer={"max_patience": int(max_patience), "max_iter": int(max_iter),
                   "avoid_duplicates": bool(avoid_duplicates),
                   "duplicate_penalty": float(duplicate_penalty),
                   "temperature": float(temperature)},
        verbose=verbose, init_indices=np.asarray(init_indices, dtype=np.int64),
        init_note=init_note, secondary_fn=secondary, run_fn=run_fn, references=references,
        codebook_table_fn=codebook_table,
        extra_provenance={
            "merfish": {
                "n_bits": n_bits,
                "hamming_weights": {str(w): c for w, c in hamming_weights.items()},
                "candidates_per_gene": candidates_per_gene,
                "genes": len(gene_names),
                "crowding_anchor_C_S0": float(crowding_cache._C_anchor),
                "include": [name for name, _, _ in includes],
            },
        },
    )


# ---------------------------------------------------------------------------
# Evaluation of existing codebooks
# ---------------------------------------------------------------------------


def _normalize_codebooks(codebooks, sequence) -> Tuple[List[str], List[List[str]], List[List[str] | None]]:
    if not isinstance(codebooks, Mapping) or not codebooks:
        raise ValueError(
            "codebooks must be a non-empty mapping of name -> sequences (a list of "
            "strings or a table with a sequence column)"
        )
    names, seqs, groups = [], [], []
    for name, cb in codebooks.items():
        names.append(str(name))
        if isinstance(cb, pd.DataFrame):
            col = sequence or next(
                (c for c in ("sequence", "barcode", "Sequence", "Barcode") if c in cb.columns), None
            )
            if col is None or col not in cb.columns:
                raise ValueError(
                    f"codebook {name!r}: no sequence column; pass sequence= (columns: {list(cb.columns)})"
                )
            seqs.append(cb[col].astype(str).tolist())
            gcol = next((c for c in ("group", "gene") if c in cb.columns), None)
            groups.append(cb[gcol].astype(str).tolist() if gcol else None)
        elif isinstance(cb, str):
            raise ValueError(
                f"codebook {name!r} is a single string; give a list of sequences or a table"
            )
        else:
            seqs.append([str(s) for s in cb])
            groups.append(None)
        if not seqs[-1]:
            raise ValueError(f"codebook {name!r} is empty")
    return names, seqs, groups


def evaluate(
    codebooks: Mapping[str, Any],
    channel: Channel,
    *,
    num_samples: int = 5_000,
    seed: int | None = None,
    device: str = "auto",
    rule="unique_minimum",
    workers: int | None = None,
    sequence: str | None = None,
) -> EvaluationResult:
    """Estimate the decoding accuracy of codebooks, evaluated together.

    Every codebook is decoded against its own codewords, but the reads are
    drawn once for the union of all codebooks (by sequence), so codebooks
    passed together share reads for shared codewords. Adding or removing a
    codebook changes that union and redraws the reads. Results equal
    :func:`duet.benchmark.metrics.evaluate_codebooks_by_sequence` on the same
    ordered list. GPUs are recommended for large codebooks (install the
    ``gpu`` extra).

    Parameters
    ----------
    codebooks : dict
        Name -> codebook: a list of sequences, or a table with a sequence
        column (``sequence``, ``barcode``, or the ``sequence`` argument) and
        optionally a ``group`` or ``gene`` column.
    channel : Channel
        Noise channel from :mod:`duet.channels`.
    num_samples : int, default 5,000
        Simulated reads per codeword (at most 32,767).
    seed : int, optional
        The evaluation uses the evaluation-stage seed derived from ``seed``
        (:func:`derive_stage_seeds`), as :func:`design_ops` does, so the
        same codebooks in the same order reproduce a design's accuracies:
        pass the design's ``seed`` (not ``provenance["evaluation"]["seed"]``),
        every codebook in the order of
        ``res.provenance["evaluation"]["codebook_order"]``, the design's
        evaluation channel and rule, ``num_samples=eval_samples`` and the
        same device. A subset of the codebooks redraws the reads.
    device, rule, workers
        As in :func:`design_ops`.
    sequence : str, optional
        Sequence column name for table inputs.

    Returns
    -------
    EvaluationResult

    Examples
    --------
    >>> import duet
    >>> acc = duet.evaluate({"mine": ["ACGTAC", "TTGACA", "GGGCCA"]},
    ...                     duet.channels.symmetric(0.1), num_samples=500, seed=0, device="cpu")
    >>> list(acc.summary.columns)[:2]
    ['codebook', 'n_codewords']
    """
    import duet
    from duet.benchmark.metrics import compute_metrics, evaluate_codebooks_by_sequence

    names, seqs, groups = _normalize_codebooks(codebooks, sequence)
    lengths = {len(s) for cb in seqs for s in cb}
    if len(lengths) != 1:
        raise ValueError(f"all sequences must have the same length; found lengths {sorted(lengths)}")
    L = lengths.pop()
    symbols = set().union(*(set(s) for cb in seqs for s in cb))
    if not symbols <= set(channel.alphabet):
        raise ValueError(
            f"sequences use symbols {sorted(symbols - set(channel.alphabet))} outside the "
            f"channel's alphabet {channel.alphabet!r}"
        )
    channel.check_compatible(channel.alphabet, L)
    num_samples = _check_samples(num_samples, "num_samples")
    root_seed, seeds = _resolve_seeds(seed, None)
    union = {s for cb in seqs for s in cb}
    device = _resolve_device(device, len(union))
    _, cpu_jobs = _workers(workers, 1)
    rule_cfg = _rule_config(rule)
    cfg = _evaluator_config(channel, rule_cfg, num_samples, seeds["evaluation"], cpu_jobs)
    results = evaluate_codebooks_by_sequence(
        seqs, eval_config=cfg, alphabet_size=channel.alphabet_size, n_jobs=cpu_jobs, device=device,
    )
    per_rows, summary_rows = [], []
    for name, cb, grp, res in zip(names, seqs, groups, results):
        acc = np.asarray(res.codeword_accuracy, dtype=float)
        frame = pd.DataFrame({"codebook": name, "position": np.arange(len(cb)),
                              "sequence": cb, "accuracy": acc})
        if grp is not None:
            frame.insert(3, "group", grp)
        per_rows.append(frame)
        summary_rows.append({"codebook": name, "n_codewords": len(cb),
                             **accuracy_summary(compute_metrics(acc)),
                             "duplicate_codewords": duplicate_count(cb)})
    provenance = {
        "duet_version": duet.__version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": root_seed,
        "evaluation_seed": seeds["evaluation"],
        "stage_seed_order": list(STAGES),
        "num_samples": num_samples,
        "device": device,
        "channel": channel.to_dict(),
        "decoding_rule": rule_cfg,
        "codebook_order": names,
    }
    return EvaluationResult(
        per_codeword=pd.concat(per_rows, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        provenance=provenance,
    )
