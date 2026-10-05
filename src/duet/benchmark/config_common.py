# src/duet/benchmark/config_common.py
"""Shared device / memory-budget / scratch resolution for the benchmark configs.

Single source of truth so OpsBenchmarkConfig and MerfishBenchmarkConfig resolve
identically. Owns: top-level-default + per-step-override resolution, the
`duet.pep` guard, the moved-key rejects, and the inherit-visibility log.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cpu"
DEFAULT_SYM_MEM_GB = 32.0
DEFAULT_EVAL_MEM_GB = 20.0


def normalized_top_device(raw: Dict[str, Any]) -> str:
    """Top-level default device; null/empty -> "cpu"."""
    return raw.get("device") or DEFAULT_DEVICE


def resolve_step_device(top_device: str, block_device: str | None, *, step_name: str) -> str:
    """A step's device: an explicit (truthy) block value wins; otherwise inherit
    the top-level default. Inheriting a non-"cpu" device is logged so the
    implicit GPU placement is visible. ``top_device`` must already be normalized.
    """
    if block_device:
        return block_device
    if top_device != DEFAULT_DEVICE:
        logger.info(
            "%s step inheriting device=%r from top-level `device`", step_name, top_device
        )
    return top_device


def resolve_step_mem(top_mem: float | None, block_mem: float | None, default: float) -> float:
    """Per-batch memory ceiling: block value > top-level default > built-in default."""
    if block_mem is not None:
        return block_mem
    if top_mem is not None:
        return top_mem
    return default


def resolve_scratch(top_scratch: str | None, block_scratch: str | None) -> str | None:
    """Scratch dir: block value wins, else inherit top-level (mirrors cache_dir)."""
    return block_scratch if block_scratch is not None else top_scratch


def require_duet_pep(duet: Dict[str, Any]) -> None:
    """DUET must specify its own PEP evaluator (the evaluator fallback was removed)."""
    if not duet.get("pep"):
        raise ValueError(
            "`duet.pep` is required: specify DUET's PEP evaluator explicitly "
            "(the `evaluator` fallback was removed in the 2026-06-21 schema)."
        )


def reject_pep_runtime_keys(duet: Dict[str, Any]) -> None:
    """`device`/`mem_budget_gb` are step-level (whole DUET step), not PEP-strategy keys."""
    pep = duet.get("pep") or {}
    bad = {"device", "mem_budget_gb"} & set(pep)
    if bad:
        raise ValueError(
            f"{sorted(bad)} not allowed under `duet.pep`; set `duet.device` / "
            f"`duet.sym_mem_budget_gb` (they govern the whole DUET step, "
            f"including the optimizer)."
        )


def reject_nested_eval_mem(duet: Dict[str, Any]) -> None:
    """The eval memory budget lives on `evaluator`, not nested under `duet`."""
    if "eval_mem_budget_gb" in duet:
        raise ValueError(
            "`duet.eval_mem_budget_gb` moved to `evaluator.mem_budget_gb` "
            "in the 2026-06-21 schema."
        )
