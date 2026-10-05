# src/duet/runner/core.py
"""DUET core: shared infrastructure for OPS and MERFISH DUET runners.

Contains:
- DuetOptimizerConfig (unified optimizer config; no crowding_weight)
- ObjectiveSpec (N-objective-ready spec for secondary caches + weight schedule)
- _run_duet_core (private: PEP, shared state, parallel lambda sweep)
- run_duet_from_primitives (thin wrapper kept for an internal benchmark script)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping

import numpy as np
from tqdm import tqdm

from duet.evaluator_config import EvaluatorConfig, create_evaluator
from duet.candidate_pool import CandidatePool
from duet.pareto_optimization import (
    DUET,
    DecodingSwapCache,
    ObjectiveSwapCache,
    SharedDecodeState,
    build_shared_decode_state,
)
from duet.pep_accessor import PEPAccessor
from duet.providers import PEPMatrixProvider
from duet.utils import Stopwatch, _worker_init_blas, log_memory


# Short alias for ObjectiveSpec annotations.
SwapCache = ObjectiveSwapCache


class _FrozenMap(Mapping):
    """Immutable, picklable dict wrapper.

    MappingProxyType from types is not picklable on Python <3.12, so we use
    this lightweight wrapper instead. Raises TypeError on any mutation attempt,
    satisfying the same contract as MappingProxyType while remaining picklable
    for multiprocessing.Pool use in the parallel lambda sweep.
    """

    def __init__(self, data: dict) -> None:
        # Store a copy so the caller's original dict cannot mutate us.
        self._data: dict = dict(data)

    # --- Mapping ABC ---
    def __getitem__(self, key: str):
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # --- Block mutations ---
    def __setitem__(self, key, value):
        raise TypeError(f"'{type(self).__name__}' object does not support item assignment")

    def __delitem__(self, key):
        raise TypeError(f"'{type(self).__name__}' object does not support item deletion")

    def __repr__(self) -> str:
        return f"_FrozenMap({self._data!r})"

    # Pickling: reconstruct via __init__ from the stored dict.
    def __reduce__(self):
        return (self.__class__, (self._data,))


@dataclass
class DuetOptimizerConfig:
    """Configuration for DUET optimizer parameters.

    Unified across OPS and MERFISH. `lambda_` is the universal sweep parameter:
    for each lambda_ the runner calls `ObjectiveSpec.weight_schedule(lambda_)` to
    obtain the weight dict passed to DUET.

    Convention (matches the manuscript's Methods): lambda_ is the weight on
    decoding accuracy, J = lambda * decode + (1 - lambda) * secondary, so
    lambda_ = 1.0 is decode-only and lambda_ = 0.0 is secondary-objective-only.
    Result directories produced before 2026-09-11 used the opposite convention;
    see docs/adr/0001-lambda-weights-decoding-accuracy.md.

    num_cpus bounds peak concurrent CPU usage during the lambda sweep. Each
    lambda worker is pinned to 1 BLAS thread via _worker_init_blas, so the
    peak number of busy cores is exactly min(num_cpus, len(lambda_)). No
    per-lambda BLAS parallelism; multi-threaded BLAS within a worker is
    explicitly out of scope (small matmuls in CrowdingSwapCache do not
    benefit from it).
    """

    lambda_: List[float]
    temperature: float = 0.0
    max_iter: int = 100_000
    max_patience: int = 3000
    avoid_duplicates: bool = True
    duplicate_penalty: float = 100.0
    store_all_solutions: bool = False
    num_cpus: int = 1
    annealing: bool = False
    initial_temperature: float | None = None
    final_temperature: float = 1e-6
    cooling_rate: float = 0.9995
    reheat_patience: int | None = None
    reheat_factor: float = 5.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DuetOptimizerConfig":
        """Construct from a YAML-derived dict.

        Raises ValueError on known-deprecated keys to prevent silent YAML
        migrations.
        """
        if "crowding_weight" in d:
            raise ValueError(
                "duet.optimizer.crowding_weight is removed. MERFISH now uses "
                "duet.optimizer.lambda, the weight on decoding accuracy: an "
                "old optical_crowding.crowding_weights value w becomes "
                "lambda = 1 - w (docs/adr/0001-lambda-weights-decoding-accuracy.md)."
            )

        lambda_ = d["lambda"]  # required; natural KeyError if a config omits it
        if isinstance(lambda_, (int, float)):
            lambda_ = [lambda_]

        return cls(
            lambda_=list(lambda_),
            temperature=d.get("temperature", 0.0),
            max_iter=d.get("max_iter", 100_000),
            max_patience=d.get("max_patience", 3000),
            avoid_duplicates=d.get("avoid_duplicates", True),
            duplicate_penalty=d.get("duplicate_penalty", 100.0),
            store_all_solutions=d.get("store_all_solutions", False),
            num_cpus=d.get("num_cpus", 1),
            annealing=d.get("annealing", False),
            initial_temperature=d.get("initial_temperature"),
            final_temperature=d.get("final_temperature", 1e-6),
            cooling_rate=d.get("cooling_rate", 0.9995),
            reheat_patience=d.get("reheat_patience"),
            reheat_factor=d.get("reheat_factor", 5.0),
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    """Describes the non-decode objectives for a DUET run.

    The decode objective is always present and built internally by the core
    (it owns PEP matrix and shared decode state). This spec declares the
    remaining objectives and how lambda_ maps to the full weight dict.

    Picklability requirement: cache_factories values and weight_schedule
    must be picklable for parallel lambda sweep via multiprocessing.Pool.
    Use module-level functions + functools.partial — never local lambdas
    or closures over local variables.

    Frozen so the key-set invariant validated in __post_init__ cannot
    silently drift after construction.
    """

    cache_factories: Dict[str, Callable[..., SwapCache]]
    # Factory signature:
    #   (candidates, *, codeword_to_group, candidate_to_sequence_idx=None) -> SwapCache.
    # codeword_to_group is the per-codeword group assignment (length |S|);
    # candidate_to_sequence_idx is the optional dedup map (length pool_size).
    # Caches that don't use the latter should accept and discard it.
    weight_schedule: Callable[[float], Dict[str, float]]

    def __post_init__(self):
        # Freeze the mapping before validating so the stored object matches
        # the validated shape. _FrozenMap copies internally, so the caller's
        # original dict can still be held; we hold our own immutable view.
        object.__setattr__(
            self, "cache_factories", _FrozenMap(self.cache_factories)
        )
        expected = {"decode", *self.cache_factories.keys()}
        for probe in (0.0, 0.5, 1.0):
            observed = set(self.weight_schedule(probe).keys())
            if observed != expected:
                raise ValueError(
                    f"ObjectiveSpec weight_schedule(lambda={probe}) keys "
                    f"{observed} do not match decode + cache_factories "
                    f"{expected}"
                )
        try:
            pickle.dumps(self)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            raise ValueError(
                "ObjectiveSpec is not picklable; parallel lambda sweep will fail. "
                "Ensure cache_factories values and weight_schedule are module-level "
                "callables (or functools.partial over module-level functions). "
                f"Original error: {e!r}"
            ) from e


def _compute_pep_matrix(
    library: List[str],
    config: EvaluatorConfig,
    alphabet_size: int,
    cache_dir: Path | None = None,
    use_mmap: bool = True,
    sample_batch_size: int | None = None,
    device: str = "cpu",
    force_rebuild: bool = False,
    sym_mem_budget_gb: float = 8.0,
) -> "tuple[np.ndarray, int] | PEPAccessor":
    """Compute PEP matrix for the given library, with optional caching."""
    if cache_dir is not None:
        provider = PEPMatrixProvider(cache_dir, force_rebuild=force_rebuild)
        if use_mmap:
            return provider.get_mmap(
                library, config, alphabet_size=alphabet_size,
                n_jobs=config.num_cpus, sample_batch_size=sample_batch_size,
                device=device, sym_mem_budget_gb=sym_mem_budget_gb,
            )
        return provider.get(
            library, config, alphabet_size=alphabet_size,
            n_jobs=config.num_cpus, sample_batch_size=sample_batch_size,
            device=device,
        )
    else:
        # No caching - compute directly
        evaluator = create_evaluator(library, config, alphabet_size=alphabet_size)
        return evaluator.compute_pep_matrix(
            n_jobs=config.num_cpus, sample_batch_size=sample_batch_size,
            device=device,
        )


@dataclass
class _LambdaJob:
    """Pickle-safe bundle of per-lambda worker arguments.

    All fields are picklable by construction (no lambdas, no closures).
    """
    init: np.ndarray
    pep_source: "np.ndarray | PEPAccessor"
    n_samples: int | None
    candidates: CandidatePool
    sequences: List[str]
    lambda_: float
    optimizer_config: DuetOptimizerConfig
    seed: int | None
    objective_spec: ObjectiveSpec
    shared_decode_state: SharedDecodeState | None


def _run_single_lambda(job: _LambdaJob) -> Dict:
    """Run a single DUET optimization for one lambda value. Worker-safe."""
    # Decode cache — always present, owned by the core (not by ObjectiveSpec).
    # Duplicate penalty is encoded as a scalar diagonal offset on the accessor;
    # there is no separate sparse penalty object anymore.
    duplicate_offset = (
        2.0 * job.optimizer_config.duplicate_penalty
        if job.optimizer_config.avoid_duplicates
        else 0.0
    )
    codeword_to_group = job.candidates.codeword_to_group
    c_to_u = job.candidates.candidate_to_sequence_idx

    if isinstance(job.pep_source, PEPAccessor):
        # Apply runtime overlay to the accessor (idempotent; same value across workers).
        job.pep_source._duplicate_offset = duplicate_offset
        job.pep_source._c_to_u = c_to_u

        if job.shared_decode_state is not None:
            decode_cache = DecodingSwapCache.from_shared_state(
                pep_accessor=job.pep_source,
                state=job.shared_decode_state,
                group_to_candidates=job.candidates.group_to_candidates,
                codeword_to_group=codeword_to_group,
                candidate_to_sequence_idx=c_to_u,
            )
        else:
            decode_cache = DecodingSwapCache(
                pep_accessor=job.pep_source,
                group_to_candidates=job.candidates.group_to_candidates,
                codeword_to_group=codeword_to_group,
                candidate_to_sequence_idx=c_to_u,
            )
    else:
        decode_cache = DecodingSwapCache.from_pep_matrix(
            pep_matrix=job.pep_source,
            group_to_candidates=job.candidates.group_to_candidates,
            codeword_to_group=codeword_to_group,
            candidate_to_sequence_idx=c_to_u,
            n_samples=job.n_samples,
            duplicate_offset=duplicate_offset,
        )

    # Secondary caches — built via ObjectiveSpec factories. Factory signature is
    # (candidates, *, codeword_to_group, candidate_to_sequence_idx=None) -> SwapCache.
    # codeword_to_group is required so the cache can enumerate within-group swaps;
    # candidate_to_sequence_idx is optional and used only by caches that operate
    # in sequence space (currently only DecodingSwapCache, built by the core).
    secondary_caches = [
        factory(
            job.candidates,
            codeword_to_group=job.candidates.codeword_to_group,
            candidate_to_sequence_idx=job.candidates.candidate_to_sequence_idx,
        )
        for factory in job.objective_spec.cache_factories.values()
    ]

    weights = job.objective_spec.weight_schedule(job.lambda_)

    optimizer = DUET(
        group_to_candidates=job.candidates.group_to_candidates,
        codeword_to_group=job.candidates.codeword_to_group,
        objective_caches=[decode_cache, *secondary_caches],
        weights=weights,
        temperature=job.optimizer_config.temperature,
        max_iter=job.optimizer_config.max_iter,
        max_patience=job.optimizer_config.max_patience,
        track_history=True,
        verbose=False,
        store_all_solutions=job.optimizer_config.store_all_solutions,
        annealing=job.optimizer_config.annealing,
        initial_temperature=job.optimizer_config.initial_temperature,
        final_temperature=job.optimizer_config.final_temperature,
        cooling_rate=job.optimizer_config.cooling_rate,
        reheat_patience=job.optimizer_config.reheat_patience,
        reheat_factor=job.optimizer_config.reheat_factor,
    )
    optimizer.optimize(list(job.init), seed=job.seed)
    log_memory(f"[worker lambda={job.lambda_}] after optimize")

    return {
        "lambda": job.lambda_,
        "best_indices": optimizer.best_S,
        "history": dict(optimizer.history),
    }


def _run_duet_core(
    candidates: CandidatePool,
    pep_config: EvaluatorConfig,
    optimizer_config: DuetOptimizerConfig,
    init: np.ndarray,
    alphabet_size: int,
    objective_spec: ObjectiveSpec,
    seed: int | None = None,
    cache_dir: Path | None = None,
    use_mmap: bool = True,
    device: str = "cpu",
    force_rebuild: bool = False,
    sym_mem_budget_gb: float = 8.0,
) -> Dict:
    """Run DUET with the given ObjectiveSpec, sweeping over optimizer_config.lambda_.

    Returns:
        {
            "lambdas": List[float],
            "best_indices": List[np.ndarray],
            "histories": List[Dict],
            "pep_count_matrix": np.ndarray | PEPAccessor,
            "n_samples": int | None,
        }
    """
    mode = "mmap" if use_mmap else "in-memory"
    print(f"Computing PEP matrix ({mode}, device={device})...")
    log_memory("before _compute_pep_matrix")
    with Stopwatch("PEP matrix computation"):
        result = _compute_pep_matrix(
            # Build PEP over the DEDUPLICATED unique sequences so accessor.N == U
            # and candidate_to_sequence_idx (compacted [0,U)) indexes the right
            # rows/cols. Pass unique_sequences VERBATIM — never np.unique/sorted,
            # which would reorder rows relative to candidate_to_sequence_idx and
            # silently corrupt the decode objective.
            library=candidates.unique_sequences,
            config=pep_config,
            alphabet_size=alphabet_size,
            cache_dir=cache_dir,
            use_mmap=use_mmap,
            device=device,
            force_rebuild=force_rebuild,
            sym_mem_budget_gb=sym_mem_budget_gb,
        )
    log_memory("after _compute_pep_matrix")

    if isinstance(result, PEPAccessor):
        pep_source = result
        n_samples = None
        print(f"PEP accessor loaded: N={pep_source.N}")
    else:
        pep_source, n_samples = result
        print(f"PEP matrix computed: shape {pep_source.shape}")

    # Shared decode state when PEP source is an accessor.
    can_share = isinstance(pep_source, PEPAccessor)
    if can_share:
        duplicate_offset = (
            2.0 * optimizer_config.duplicate_penalty
            if optimizer_config.avoid_duplicates
            else 0.0
        )
        # Apply the runtime overlay BEFORE building shared state so the streaming
        # row reads inside build_shared_decode_state see the offset-adjusted diagonal.
        pep_source._duplicate_offset = duplicate_offset
        pep_source._c_to_u = candidates.candidate_to_sequence_idx
        shared_state = build_shared_decode_state(
            S=init,
            accessor=pep_source,
            group_to_candidates=candidates.group_to_candidates,
            codeword_to_group=candidates.codeword_to_group,
            candidate_to_sequence_idx=candidates.candidate_to_sequence_idx,
        )
    else:
        shared_state = None

    # Build per-lambda jobs.
    jobs: List[_LambdaJob] = []
    for lambda_ in optimizer_config.lambda_:
        if seed is not None:
            run_seed = int((seed + int(lambda_ * 1000)) % (2**31 - 1))
        else:
            run_seed = None
        jobs.append(_LambdaJob(
            init=init,
            pep_source=pep_source,
            n_samples=n_samples,
            candidates=candidates,
            sequences=candidates.sequences,
            lambda_=lambda_,
            optimizer_config=optimizer_config,
            seed=run_seed,
            objective_spec=objective_spec,
            shared_decode_state=shared_state,
        ))

    num_lambdas = len(optimizer_config.lambda_)
    with Stopwatch("DUET optimization"):
        if optimizer_config.num_cpus > 1 and num_lambdas > 1:
            with Pool(
                processes=min(optimizer_config.num_cpus, num_lambdas),
                initializer=_worker_init_blas,
            ) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_run_single_lambda, jobs),
                    total=num_lambdas,
                    desc="DUET lambda sweep",
                ))
        else:
            results = [_run_single_lambda(job) for job in tqdm(jobs, desc="DUET lambda sweep")]
    log_memory("after DUET optimization")

    results.sort(key=lambda r: r["lambda"])

    return {
        "lambdas": [r["lambda"] for r in results],
        "best_indices": [r["best_indices"] for r in results],
        "histories": [r["history"] for r in results],
        "pep_count_matrix": pep_source,
        "n_samples": n_samples,
    }


def _build_primitives_pool(
    sequences: List[str],
    group_to_candidates: Dict[str, List[int]],
    scores: np.ndarray,
) -> "CandidatePool":
    """Construct a CandidatePool from primitive inputs."""
    # Verify coverage early so we surface a clear error rather than letting
    # CandidatePool._validate report it.
    seen: set = set()
    for idxs in group_to_candidates.values():
        seen.update(idxs)
    missing = [i for i in range(len(sequences)) if i not in seen]
    if missing:
        raise ValueError(
            f"group_to_candidates does not cover all sequence indices; "
            f"missing indices: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    quotas = {g: len(idxs) for g, idxs in group_to_candidates.items()}
    return CandidatePool(
        sequences=sequences,
        group_to_candidates=group_to_candidates,
        quotas=quotas,
        scores=scores,
    )


def run_duet_from_primitives(
    sequences: List[str],
    group_to_candidates: Dict[str, List[int]],
    scores: np.ndarray,
    pep_config: EvaluatorConfig,
    optimizer_config: DuetOptimizerConfig,
    init: np.ndarray,
    alphabet_size: int,
    seed: int | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> Dict:
    """Run DUET optimization from primitive data types (no CandidatePool needed).

    Kept importable for an internal benchmark script.
    Signature matches the pre-refactor version — no new knobs. Routes through
    run_duet_ops under the hood since that's exactly what the old behavior was
    (decode + score objectives, identical weight schedule).
    """
    # Local import to avoid a circular dependency at module load.
    from duet.runner.ops import run_duet_ops

    candidates = _build_primitives_pool(sequences, group_to_candidates, scores)
    return run_duet_ops(
        candidates=candidates,
        pep_config=pep_config,
        optimizer_config=optimizer_config,
        init=init,
        alphabet_size=alphabet_size,
        seed=seed,
        cache_dir=cache_dir,
        use_mmap=False,   # primitives caller doesn't use PEP cache
        device=device,
    )
