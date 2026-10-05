#!/usr/bin/env python3
"""
2-D synthetic benchmark runner.

Generates synthetic candidate pools with iid U(0,1) scores, runs DUET,
GreedyNLLMOOptimizer (noise-model-aware NLL distance), and
GreedyHammingMOOptimizer (Hamming distance — noise-agnostic) over a
lambda sweep, optionally enumerates exhaustive Pareto fronts in the
small regime, and computes normalized hypervolume per cell. Sweeps over
noise channels and error rates the same way the 1-D synthetic benchmark
does. The three-method setup isolates two methodological axes:
"noise-channel-aware distance" (NLL vs Hamming) and "PEP-aware
optimization" (DUET vs greedy).

An optional top-level `device` key ("cpu" default, or "gpu", "gpu:all",
"gpu:<ids>") places the two Monte Carlo steps — the ground-truth evaluator
cache and DUET's PEP matrix — on GPU. The swap search, greedy baselines,
and exhaustive scoring always run on CPU.

Usage (the paper's configuration is experiments/synthetic_hvr/config.yaml;
its `outdir` is relative to the working directory, so run it from that folder,
as experiments/synthetic_hvr/run.sh does):
    cd experiments/synthetic_hvr
    PYTHONPATH=../.. python -m scripts.benchmark.synthetic.run_2d_synthetic_benchmark \
        --config config.yaml -v
"""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from duet.benchmark.exhaustive import (
    compute_total_codebooks,
    enumerate_all_codebooks,
)
from duet.benchmark.metrics import compare_pareto_fronts, compute_duet_objective
# These were moved to duet.benchmark.baseline_objectives so process_trial
# (which lives in src/) could call them. Re-exported here for back-compat
# with the runner's _build_evaluator_config below and any external
# importer of these private names.
from duet.benchmark.baseline_objectives import (
    EvaluatorRunConfig,
    PoolConfig,
    _EPSILON_FLOOR,
    _channel_class_partition,
    _class_factor_per_symbol,
    _create_noise_and_decoding_metric,
    _positional_ramp,
    _uniform_row_channel_matrix,
)
from duet.benchmark.config_common import normalized_top_device
from duet.benchmark.synthetic_pool import create_2d_synthetic_pool
from duet.codebook_evaluator import (
    AsymmetricNLL,
    AsymmetricChannel,
    CodebookEvaluator,
    DecodingMetric,
    NoiseChannel,
    PositionVaryingAsymmetricNLL,
    PositionVaryingAsymmetricChannel,
    PositionVaryingEpsilon,
    PositionVaryingNLL,
    SymmetricEpsilon,
    SymmetricNLL,
    UniqueMinimum,
)
from duet.runner.core import DuetOptimizerConfig
from duet.runner.ops import run_duet_ops
from duet.evaluator_config import EvaluatorConfig, ComponentConfig
from duet.greedy_optimizers import GreedyHammingMOOptimizer, GreedyNLLMOOptimizer

logger = logging.getLogger(__name__)

# Codebooks at-or-below this count are exhaustively enumerated and the
# resulting (decode, score) points form the small-regime "true PF".
ENUM_THRESHOLD = 1_000_000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GreedyMOConfig:
    lambda_: List[float]


def _parse_greedy_block(raw: Dict[str, Any], key: str) -> GreedyMOConfig | None:
    """Parse one greedy-baseline YAML block with strict validation.

    Returns None when `key` is absent from `raw` — that's the supported
    "skip this baseline" affordance. Present-but-null and missing /
    empty `lambda` raise ValueError; both are easy-to-make typos rather
    than a supported skip form.
    """
    if key not in raw:
        return None
    block = raw[key]
    if block is None:
        raise ValueError(
            f"Config key {key!r} is present but null. "
            f"Omit the key entirely to skip this baseline, "
            f"or provide a non-empty 'lambda' list."
        )
    lambda_ = block.get("lambda")
    if lambda_ is None or len(lambda_) == 0:
        raise ValueError(
            f"Config key {key!r}.lambda is missing or empty. "
            f"Omit {key!r} entirely to skip this baseline, "
            f"or provide a non-empty 'lambda' list."
        )
    return GreedyMOConfig(lambda_=list(lambda_))


@dataclass
class TwoDSyntheticBenchmarkConfig:
    outdir: str
    seed: int
    trials: int
    pool: PoolConfig
    evaluator: EvaluatorRunConfig
    noise_channels: List[str]
    error_rates: List[float]
    duet: DuetOptimizerConfig
    greedy_hamming_mo: GreedyMOConfig | None = None
    greedy_nll_mo: GreedyMOConfig | None = None
    # Placement of the evaluator cache and DUET PEP steps. Top-level key,
    # same semantics as the OPS/MERFISH benchmark configs; null/absent -> cpu.
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "TwoDSyntheticBenchmarkConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        pool = PoolConfig(**raw["pool"])
        evaluator = EvaluatorRunConfig(**raw["evaluator"])
        duet = DuetOptimizerConfig.from_dict(raw["duet"]["optimizer"])
        greedy_hamming = _parse_greedy_block(raw, "greedy_hamming_mo")
        greedy_nll = _parse_greedy_block(raw, "greedy_nll_mo")

        return cls(
            outdir=raw["outdir"],
            seed=raw["seed"],
            trials=raw["trials"],
            pool=pool,
            evaluator=evaluator,
            noise_channels=list(raw["noise_channels"]),
            error_rates=list(raw["error_rates"]),
            duet=duet,
            greedy_hamming_mo=greedy_hamming,
            greedy_nll_mo=greedy_nll,
            device=normalized_top_device(raw),
        )


# ---------------------------------------------------------------------------
# Per-cell logic
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """All outputs for a single (regime, noise, error, trial) cell."""
    regime: str
    # Per-method recovered points: list of (lambda_, codebook_indices, decode_acc, mean_score)
    duet_points: List[Tuple[float, np.ndarray, float, float]]
    greedy_hamming_points: List[Tuple[float, np.ndarray, float, float]]
    greedy_nll_points: List[Tuple[float, np.ndarray, float, float]]
    # Exhaustive front points (small regime only): list of (codebook_indices, decode_acc, mean_score)
    exhaustive_points: List[Tuple[np.ndarray, float, float]]
    # DUET's union-bound objective value at its lambda=1 pick, computed from
    # the evaluator's cached PEP matrix. NaN if DUET has no lambda=1 entry.
    duet_objective_at_lambda_one: float = float("nan")
    # Maximum objective over the full enumerated codebook list, computed from
    # the same evaluator-cached PEP matrix. NaN in the large regime (no
    # enumeration), matching the existing treatment of max_decode_accuracy.
    max_duet_objective: float = float("nan")


def _build_evaluator_config(
    *,
    noise_type: str,
    error_rate: float,
    seq_length: int,
    alphabet_size: int,
    num_samples: int,
    num_cpus: int,
    seed: int,
) -> EvaluatorConfig:
    """Build an EvaluatorConfig matching the runner's noise factory.

    Must produce eps values bit-identical to `_create_noise_and_decoding_metric`
    so DUET (which reconstructs the noise channel from this config) sees the
    same channel as the runner's local evaluator. Both factories share the
    same eps-construction helpers above.

    Note: `duet.evaluator_config.create_noise_channel` accepts type
    "position_varying" with param key "epsilon" (singular) — quirk of the
    factory, preserved here.
    """
    L = seq_length
    q = alphabet_size
    x = error_rate

    if noise_type == "symmetric":
        nm = ComponentConfig(type="symmetric", params={"epsilon": x})
        df = ComponentConfig(type="symmetric_nll", params={"epsilon": x})
    elif noise_type == "position_varying":
        eps = [
            max(e, _EPSILON_FLOOR)
            for e in _positional_ramp(0.5 * x, 1.5 * x, L)
        ]
        nm = ComponentConfig(type="position_varying", params={"epsilon": eps})
        df = ComponentConfig(type="position_varying_nll", params={"epsilon": eps})
    elif noise_type == "asymmetric":
        class_factor = _class_factor_per_symbol(q)
        eps_per_symbol = np.maximum(class_factor * x, _EPSILON_FLOOR)
        ch = _uniform_row_channel_matrix(eps_per_symbol, q)
        nm = ComponentConfig(type="asymmetric", params={"channel_matrix": ch.tolist()})
        df = ComponentConfig(type="asymmetric_nll", params={"channel_matrix": ch.tolist()})
    elif noise_type == "position_varying_asymmetric":
        class_factor = _class_factor_per_symbol(q)
        position_factor = np.array(_positional_ramp(0.5, 1.5, L))
        eps_pq = np.maximum(
            position_factor[:, None] * class_factor[None, :] * x,
            _EPSILON_FLOOR,
        )
        chs = np.empty((L, q, q), dtype=np.float64)
        for p in range(L):
            chs[p] = _uniform_row_channel_matrix(eps_pq[p], q)
        nm = ComponentConfig(type="position_varying_asymmetric", params={"channel_matrices": chs.tolist()})
        df = ComponentConfig(type="position_varying_asymmetric_nll", params={"channel_matrices": chs.tolist()})
    else:
        raise ValueError(f"Unknown noise type: {noise_type!r}")

    return EvaluatorConfig(
        noise_channel=nm,
        decoding_metric=df,
        decoding_rule=ComponentConfig(type="unique_minimum"),
        num_samples=num_samples,
        num_cpus=num_cpus,
        seed=seed,
    )


def process_cell(
    *,
    pool,
    noise_type: str,
    error_rate: float,
    trial_seed: int,
    eval_cfg: EvaluatorRunConfig,
    duet_cfg: DuetOptimizerConfig,
    greedy_hamming_cfg: GreedyMOConfig | None,
    greedy_nll_cfg: GreedyMOConfig | None,
    cache_dir: Path,
    device: str = "cpu",
) -> CellResult:
    """Process one (noise, error_rate) cell for one trial.

    Runs all three methods (DUET, Greedy-NLL-MO, Greedy-Hamming-MO) plus
    exhaustive enumeration in the small regime. Closes the evaluator and
    DUET's pep_count_matrix before returning so mmap files are released
    between cells.

    `device` places the evaluator cache initialization and DUET's PEP
    computation ("cpu" or a GPU spec understood by duet.gpu_utils.parse_device).
    """
    seq_length = pool.sequences.shape[1]
    alphabet_size = int(pool.sequences.max()) + 1
    if alphabet_size < 2:
        alphabet_size = 2

    noise_channel, decoding_metric = _create_noise_and_decoding_metric(
        noise_type, error_rate, seq_length, alphabet_size,
    )
    evaluator = CodebookEvaluator(
        codebook=pool.sequences,
        noise_channel=noise_channel,
        decoding_metric=decoding_metric,
        decoding_rule=UniqueMinimum(),
        n_samples=eval_cfg.num_samples,
        seed=trial_seed,
    )
    evaluator.initialize_cache(n_jobs=eval_cfg.num_cpus, device=device)

    g2c = pool.group_to_candidates

    # Enumerability check.
    total = compute_total_codebooks(g2c, pool.quotas)
    if total <= ENUM_THRESHOLD:
        regime = "small"
        codebooks = enumerate_all_codebooks(g2c, pool.quotas)
        all_acc = evaluator.batch_get_accuracy(codebooks)
        all_score = np.array(
            [float(np.mean(pool.scores[cb])) for cb in codebooks],
            dtype=np.float64,
        )
        points = np.stack([all_acc, all_score], axis=1)
        exhaustive_points = _filter_nondominated_max_with_codebooks(
            points=points, codebooks=codebooks,
        )
    else:
        regime = "large"
        exhaustive_points = []

    # DUET lambda sweep.
    init = pool.sample_initial_selection(strategy="random", seed=trial_seed)
    pep_config = _build_evaluator_config(
        noise_type=noise_type,
        error_rate=error_rate,
        seq_length=seq_length,
        alphabet_size=alphabet_size,
        num_samples=eval_cfg.num_samples,
        num_cpus=eval_cfg.num_cpus,
        seed=trial_seed,
    )
    duet_out = run_duet_ops(
        candidates=pool,
        pep_config=pep_config,
        optimizer_config=duet_cfg,
        init=init,
        alphabet_size=alphabet_size,
        seed=trial_seed,
        cache_dir=cache_dir,
        use_mmap=True,
        device=device,
    )
    duet_points: List[Tuple[float, np.ndarray, float, float]] = []
    for lambda_, indices in zip(duet_out["lambdas"], duet_out["best_indices"]):
        idx = np.asarray(indices, dtype=int)
        decode_acc = float(evaluator.get_accuracy(idx))
        mean_score = float(np.mean(pool.scores[idx]))
        duet_points.append((float(lambda_), idx, decode_acc, mean_score))

    # Greedy-NLL-MO sweep — uses the same decoding metric as the runner's
    # evaluator, so all three methods see the same noise channel in their
    # distance/cost metric. Skipped when greedy_nll_cfg is None (block
    # absent from YAML). The skip is logged at INFO level so `-v` runs
    # make the no-op observable rather than silently producing parquets
    # that lack greedy_nll_mo rows.
    greedy_nll_points: List[Tuple[float, np.ndarray, float, float]] = []
    if greedy_nll_cfg is None:
        logger.info(
            "greedy_nll_mo absent from config; skipping NLL sweep "
            "for noise=%s error_rate=%.3f",
            noise_type, error_rate,
        )
    else:
        mo_nll = GreedyNLLMOOptimizer(seed=trial_seed)
        mo_nll_out = mo_nll.optimize(
            sequences=pool.sequences,
            group_to_candidates=g2c,
            quotas=pool.quotas,
            scores=pool.scores,
            lambdas=greedy_nll_cfg.lambda_,
            decoding_metric=decoding_metric,
        )
        for lambda_ in sorted(mo_nll_out.keys()):
            idx = np.asarray(mo_nll_out[lambda_], dtype=int)
            decode_acc = float(evaluator.get_accuracy(idx))
            mean_score = float(np.mean(pool.scores[idx]))
            greedy_nll_points.append((float(lambda_), idx, decode_acc, mean_score))

    # Greedy-Hamming-MO sweep. Skipped when greedy_hamming_cfg is None
    # (block absent from YAML). The skip is logged at INFO level so `-v`
    # runs make the no-op observable rather than silently producing
    # parquets that lack greedy_hamming_mo rows.
    greedy_hamming_points: List[Tuple[float, np.ndarray, float, float]] = []
    if greedy_hamming_cfg is None:
        logger.info(
            "greedy_hamming_mo absent from config; skipping Hamming sweep "
            "for noise=%s error_rate=%.3f",
            noise_type, error_rate,
        )
    else:
        mo_h = GreedyHammingMOOptimizer(seed=trial_seed)
        mo_h_out = mo_h.optimize(
            sequences=pool.sequences,
            group_to_candidates=g2c,
            quotas=pool.quotas,
            scores=pool.scores,
            lambdas=greedy_hamming_cfg.lambda_,
        )
        for lambda_ in sorted(mo_h_out.keys()):
            idx = np.asarray(mo_h_out[lambda_], dtype=int)
            decode_acc = float(evaluator.get_accuracy(idx))
            mean_score = float(np.mean(pool.scores[idx]))
            greedy_hamming_points.append((float(lambda_), idx, decode_acc, mean_score))

    # DUET objective values for the goal-criterion gate.
    #
    # Both sides of the gate (DUET's lambda=1 objective and the max objective
    # over the enumerated codebooks) MUST come from the evaluator's cached
    # PEP matrix, not from run_duet_ops's pep_count_matrix. The two are
    # distinct reductions of the same trial seed and can differ at ULP
    # precision; using one matrix on both sides keeps the gate deterministic.
    pep_eval = evaluator.get_pairwise_error_from_cache()
    duet_lambda_one_idx = _codebook_at_lambda_one(duet_points)
    if duet_lambda_one_idx is None:
        duet_objective_at_lambda_one = float("nan")
    else:
        duet_objective_at_lambda_one = compute_duet_objective(
            pep_eval, duet_lambda_one_idx,
        )

    if regime == "small":
        # Iterate the full enumeration (not the non-dominated subset) —
        # the objective-argmax is decoupled from the (accuracy, score)
        # Pareto front and can be off-front.
        max_duet_objective = max(
            compute_duet_objective(pep_eval, np.asarray(cb, dtype=int))
            for cb in codebooks
        )
    else:
        max_duet_objective = float("nan")

    # Resource release.
    pep = duet_out.get("pep_count_matrix")
    if hasattr(pep, "close"):
        pep.close()
    evaluator.close()

    return CellResult(
        regime=regime,
        duet_points=duet_points,
        greedy_hamming_points=greedy_hamming_points,
        greedy_nll_points=greedy_nll_points,
        exhaustive_points=exhaustive_points,
        duet_objective_at_lambda_one=duet_objective_at_lambda_one,
        max_duet_objective=max_duet_objective,
    )


def _filter_nondominated_max_with_codebooks(
    *,
    points: np.ndarray,
    codebooks,
) -> List[Tuple[np.ndarray, float, float]]:
    """Filter to non-dominated (maximize both axes), preserving codebook arrays.

    pymoo's `find_non_dominated` assumes minimization, so we negate.
    """
    from pymoo.util.nds.non_dominated_sorting import find_non_dominated
    keep_idx = find_non_dominated(-points)
    out = []
    for i in keep_idx:
        out.append(
            (np.asarray(codebooks[i], dtype=int), float(points[i, 0]), float(points[i, 1]))
        )
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _build_results_rows(
    *,
    regime: str,
    trial: int,
    seed: int,
    noise: str,
    error: float,
    cell: CellResult,
    codebook_tag: str,
) -> List[Dict[str, Any]]:
    """Long-form rows for results.parquet from one cell."""
    rows: List[Dict[str, Any]] = []
    for lambda_, _idx, dec, sc in cell.duet_points:
        rows.append({
            "regime": regime, "trial": trial, "seed": seed,
            "noise_channel": noise, "error_rate": error,
            "method": "duet", "lambda": lambda_,
            "decode_accuracy": dec, "mean_score": sc,
            "codebook_id": f"{codebook_tag}::duet_lambda_{lambda_:.4f}",
        })
    for lambda_, _idx, dec, sc in cell.greedy_hamming_points:
        rows.append({
            "regime": regime, "trial": trial, "seed": seed,
            "noise_channel": noise, "error_rate": error,
            "method": "greedy_hamming_mo", "lambda": lambda_,
            "decode_accuracy": dec, "mean_score": sc,
            "codebook_id": f"{codebook_tag}::greedy_hamming_lambda_{lambda_:.4f}",
        })
    for lambda_, _idx, dec, sc in cell.greedy_nll_points:
        rows.append({
            "regime": regime, "trial": trial, "seed": seed,
            "noise_channel": noise, "error_rate": error,
            "method": "greedy_nll_mo", "lambda": lambda_,
            "decode_accuracy": dec, "mean_score": sc,
            "codebook_id": f"{codebook_tag}::greedy_nll_lambda_{lambda_:.4f}",
        })
    for i, (_idx, dec, sc) in enumerate(cell.exhaustive_points):
        rows.append({
            "regime": regime, "trial": trial, "seed": seed,
            "noise_channel": noise, "error_rate": error,
            "method": "exhaustive", "lambda": float("nan"),
            "decode_accuracy": dec, "mean_score": sc,
            "codebook_id": f"{codebook_tag}::exhaustive_pf_{i:03d}",
        })
    return rows


def _save_codebook_artifacts(
    *,
    cell: CellResult,
    codebook_dir: Path,
    tag: str,
) -> None:
    """Write all codebook indices for a cell into one .npz file."""
    arrs: Dict[str, np.ndarray] = {}
    for lambda_, idx, _dec, _sc in cell.duet_points:
        arrs[f"duet_lambda_{lambda_:.4f}"] = idx
    for lambda_, idx, _dec, _sc in cell.greedy_hamming_points:
        arrs[f"greedy_hamming_lambda_{lambda_:.4f}"] = idx
    for lambda_, idx, _dec, _sc in cell.greedy_nll_points:
        arrs[f"greedy_nll_lambda_{lambda_:.4f}"] = idx
    for i, (idx, _dec, _sc) in enumerate(cell.exhaustive_points):
        arrs[f"exhaustive_pf_{i:03d}"] = idx
    if arrs:
        np.savez(codebook_dir / f"{tag}.npz", **arrs)


def _save_exhaustive_front(
    *,
    cell: CellResult,
    exhaustive_dir: Path,
    tag: str,
) -> None:
    """Write exhaustive Pareto front as a small parquet (small regime only)."""
    if not cell.exhaustive_points:
        return
    rows = [
        {"point_index": i, "decode_accuracy": dec, "mean_score": sc}
        for i, (_idx, dec, sc) in enumerate(cell.exhaustive_points)
    ]
    pd.DataFrame(rows).to_parquet(exhaustive_dir / f"{tag}.parquet")


def _hv_for_cell(cell: CellResult) -> Dict[str, Dict[str, Any]]:
    """Compute per-method HV using a shared per-cell reference point.

    The reference-point auto-computation in compare_pareto_fronts uses the
    union of all input fronts, including the exhaustive front when present.

    HVR (Hypervolume Ratio) is also populated: HV(method) / HV(exhaustive)
    when an exhaustive front exists for the cell, set to 1.0 explicitly
    for the exhaustive method itself. In the large regime (no exhaustive),
    every method's hvr is NaN.
    """
    method_fronts: Dict[str, np.ndarray] = {}

    if cell.duet_points:
        method_fronts["duet"] = np.array(
            [[d, s] for _a, _i, d, s in cell.duet_points], dtype=np.float64
        )
    if cell.greedy_hamming_points:
        method_fronts["greedy_hamming_mo"] = np.array(
            [[d, s] for _a, _i, d, s in cell.greedy_hamming_points], dtype=np.float64
        )
    if cell.greedy_nll_points:
        method_fronts["greedy_nll_mo"] = np.array(
            [[d, s] for _a, _i, d, s in cell.greedy_nll_points], dtype=np.float64
        )
    if cell.exhaustive_points:
        method_fronts["exhaustive"] = np.array(
            [[d, s] for _i, d, s in cell.exhaustive_points], dtype=np.float64
        )

    methods = list(method_fronts.keys())
    fronts = [method_fronts[m] for m in methods]
    cmp = compare_pareto_fronts(fronts=fronts, maximize=True)

    # Find exhaustive's HV (if present) to compute HVR.
    exhaustive_hv: float | None = None
    for method, fr in zip(methods, cmp.fronts):
        if method == "exhaustive":
            exhaustive_hv = float(fr.hypervolume)
            break

    if exhaustive_hv is not None and exhaustive_hv == 0:
        # Defensive: a degenerate exhaustive front with zero HV would
        # divide by zero. Should not happen on real data (the auto-ref
        # point sits outside the union), but guard explicitly.
        logger.warning(
            "Exhaustive HV is zero in this cell; setting HVR to NaN for all methods."
        )
        exhaustive_hv = None

    out: Dict[str, Dict[str, Any]] = {}
    for method, fr in zip(methods, cmp.fronts):
        if exhaustive_hv is None:
            hvr = float("nan")
        elif method == "exhaustive":
            hvr = 1.0
        else:
            hvr = float(fr.hypervolume) / exhaustive_hv
        out[method] = {
            "hv": float(fr.hypervolume),
            "normalized_hv": float(fr.normalized_hv),
            "hvr": hvr,
            "n_pareto_points": int(len(fr.pareto_front)),
        }
    return out


def _decode_acc_at_lambda_one(points) -> float:
    """Return decode accuracy of the lambda=1 codebook (or NaN if none)."""
    for lambda_, _idx, dec, _sc in points:
        if lambda_ == 1.0:
            return float(dec)
    return float("nan")


def _codebook_at_lambda_one(points) -> np.ndarray | None:
    """Return the codebook indices at lambda=1 (or None if no such entry)."""
    for lambda_, idx, _dec, _sc in points:
        if lambda_ == 1.0:
            return np.asarray(idx, dtype=int)
    return None


def _max_decode_in_exhaustive(cell: CellResult) -> float:
    if not cell.exhaustive_points:
        return float("nan")
    return float(max(dec for _idx, dec, _sc in cell.exhaustive_points))


def _build_hv_rows(
    *,
    regime: str,
    trial: int,
    seed: int,
    noise: str,
    error: float,
    cell: CellResult,
) -> List[Dict[str, Any]]:
    """One row per method for hv_summary.parquet."""
    rows: List[Dict[str, Any]] = []
    hv = _hv_for_cell(cell)
    for method, summary in hv.items():
        if method == "duet":
            decode_at_one = _decode_acc_at_lambda_one(cell.duet_points)
            max_decode = float("nan")
            objective_at_one = cell.duet_objective_at_lambda_one
            max_objective = float("nan")
        elif method == "greedy_hamming_mo":
            decode_at_one = _decode_acc_at_lambda_one(cell.greedy_hamming_points)
            max_decode = float("nan")
            objective_at_one = float("nan")
            max_objective = float("nan")
        elif method == "greedy_nll_mo":
            decode_at_one = _decode_acc_at_lambda_one(cell.greedy_nll_points)
            max_decode = float("nan")
            objective_at_one = float("nan")
            max_objective = float("nan")
        else:  # exhaustive
            decode_at_one = float("nan")
            max_decode = _max_decode_in_exhaustive(cell)
            objective_at_one = float("nan")
            max_objective = cell.max_duet_objective
        rows.append({
            "regime": regime, "trial": trial, "seed": seed,
            "noise_channel": noise, "error_rate": error,
            "method": method,
            "hv": summary["hv"],
            "normalized_hv": summary["normalized_hv"],
            "hvr": summary["hvr"],
            "n_pareto_points": summary["n_pareto_points"],
            "decode_acc_at_lambda_one": decode_at_one,
            "max_decode_accuracy": max_decode,
            "duet_objective_at_lambda_one": objective_at_one,
            "max_duet_objective": max_objective,
        })
    return rows


# ---------------------------------------------------------------------------
# Trial loop / orchestration
# ---------------------------------------------------------------------------

def run_benchmark(
    config: TwoDSyntheticBenchmarkConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full sweep and return (results_df, hv_summary_df)."""
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pool_dir = outdir / "artifacts" / "pools"
    codebook_dir = outdir / "artifacts" / "codebooks"
    exhaustive_dir = outdir / "artifacts" / "exhaustive_fronts"
    cache_root = outdir / "artifacts" / "pep_cache"
    for d in (pool_dir, codebook_dir, exhaustive_dir, cache_root):
        d.mkdir(parents=True, exist_ok=True)

    all_result_rows: List[Dict[str, Any]] = []
    all_hv_rows: List[Dict[str, Any]] = []

    for trial in range(config.trials):
        trial_seed = config.seed + trial
        logger.info("Trial %d/%d (seed=%d)", trial + 1, config.trials, trial_seed)

        pool = create_2d_synthetic_pool(
            num_groups=config.pool.num_groups,
            candidates_per_group=config.pool.candidates_per_group,
            seq_length=config.pool.seq_length,
            alphabet_size=config.pool.alphabet_size,
            quota=config.pool.quota,
            seed=trial_seed,
        )
        # Per-candidate group label (length pool_size). Synthetic pools have
        # disjoint groups, so each candidate has exactly one label.
        candidate_group = np.empty(pool.pool_size, dtype=object)
        for group, idxs in pool.group_to_candidates.items():
            candidate_group[idxs] = group
        np.savez(
            pool_dir / f"trial_{trial:02d}.npz",
            sequences=pool.sequences,
            scores=pool.scores,
            groups=candidate_group,
            quotas_keys=np.array(list(pool.quotas.keys()), dtype=object),
            quotas_values=np.array(list(pool.quotas.values()), dtype=int),
        )

        for noise_type in config.noise_channels:
            for error_rate in config.error_rates:
                logger.info("  noise=%s error_rate=%.3f", noise_type, error_rate)
                tag_base = f"{noise_type}_{error_rate}_trial_{trial:02d}"
                cell_cache_dir = cache_root / tag_base
                cell_cache_dir.mkdir(parents=True, exist_ok=True)

                cell = process_cell(
                    pool=pool,
                    noise_type=noise_type,
                    error_rate=error_rate,
                    trial_seed=trial_seed,
                    eval_cfg=config.evaluator,
                    duet_cfg=config.duet,
                    greedy_hamming_cfg=config.greedy_hamming_mo,
                    greedy_nll_cfg=config.greedy_nll_mo,
                    cache_dir=cell_cache_dir,
                    device=config.device,
                )

                tag_full = f"{cell.regime}_{tag_base}"
                _save_codebook_artifacts(
                    cell=cell, codebook_dir=codebook_dir, tag=tag_full,
                )
                _save_exhaustive_front(
                    cell=cell, exhaustive_dir=exhaustive_dir, tag=tag_base,
                )

                all_result_rows.extend(_build_results_rows(
                    regime=cell.regime,
                    trial=trial + 1,
                    seed=trial_seed,
                    noise=noise_type,
                    error=error_rate,
                    cell=cell,
                    codebook_tag=tag_full,
                ))
                all_hv_rows.extend(_build_hv_rows(
                    regime=cell.regime,
                    trial=trial + 1,
                    seed=trial_seed,
                    noise=noise_type,
                    error=error_rate,
                    cell=cell,
                ))

    results_df = pd.DataFrame(all_result_rows)
    hv_df = pd.DataFrame(all_hv_rows)
    results_df.to_parquet(outdir / "results.parquet", index=False)
    hv_df.to_parquet(outdir / "hv_summary.parquet", index=False)
    return results_df, hv_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2-D synthetic benchmark for DUET multi-objective evaluation.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = TwoDSyntheticBenchmarkConfig.from_yaml(args.config)
    logger.info("Config loaded: %s", args.config)

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(args.config).resolve()
    dst = (outdir / "config.yaml").resolve()
    if src != dst:
        shutil.copy2(src, dst)

    results_df, hv_df = run_benchmark(config)
    print(f"Done. {len(results_df)} result rows, {len(hv_df)} HV rows -> {outdir}")


if __name__ == "__main__":
    main()
