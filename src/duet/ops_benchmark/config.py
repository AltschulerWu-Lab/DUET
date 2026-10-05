# src/duet/ops_benchmark/config.py
"""Configuration dataclasses for the OPS benchmark pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from duet.benchmark.config_common import (
    DEFAULT_EVAL_MEM_GB, DEFAULT_SYM_MEM_GB,
    normalized_top_device, resolve_step_device, resolve_step_mem, resolve_scratch,
    require_duet_pep, reject_pep_runtime_keys, reject_nested_eval_mem,
)
from duet.runner.core import DuetOptimizerConfig
from duet.evaluator_config import EvaluatorConfig


logger = logging.getLogger(__name__)


def _parse_num_groups(raw) -> int | None:
    if isinstance(raw, str) and raw.lower() == "all":
        return None
    return raw


@dataclass
class CandidatePoolConfig:
    source: str
    seq_rounds: int
    quota: int
    num_controls: int
    num_groups: int | None = None
    min_rank: int = 10
    pairing_strategy: str | None = None
    max_control_pairs: int = 1000
    chemistry: str = "dna"
    csv_path: str | None = None
    control_quotas: Dict[str, int] | None = None
    score_method: str = "picking_round"
    candidates_per_group: int | None = None
    alphabet_size: int = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidatePoolConfig":
        # Legacy-alias keys are rejected loudly.
        legacy_keys = {
            "guide_candidates": "candidate_pool (section name)",
            "guides_per_gene": "quota",
            "num_ntc": "num_controls",
            "num_genes": "num_groups",
            "min_guide_rank": "min_rank",
            "max_ntc_pairs": "max_control_pairs",
        }
        for legacy, replacement in legacy_keys.items():
            if legacy in d:
                raise ValueError(
                    f"Legacy key '{legacy}' is no longer supported. "
                    f"Use '{replacement}' instead."
                )

        return cls(
            source=d["source"],
            seq_rounds=d["seq_rounds"],
            quota=d["quota"],
            num_controls=d.get("num_controls", 0),
            num_groups=_parse_num_groups(d.get("num_groups")),
            min_rank=d.get("min_rank", 10),
            pairing_strategy=d.get("pairing_strategy"),
            max_control_pairs=d.get("max_control_pairs", 1000),
            chemistry=d.get("chemistry", "dna"),
            csv_path=d.get("csv_path"),
            control_quotas=d.get("control_quotas"),
            score_method=d.get("score_method", "picking_round"),
            candidates_per_group=d.get("candidates_per_group"),
            alphabet_size=d.get("alphabet_size", 4),
        )


@dataclass
class SivanandanConfig:
    edit_distances: List[int]
    num_cpus: int = 8

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SivanandanConfig":
        eds = d.get("edit_distances", [3])
        if isinstance(eds, int):
            eds = [eds]
        return cls(edit_distances=list(eds), num_cpus=d.get("num_cpus", 8))


@dataclass
class FeldmanConfig:
    edit_distances: List[int]
    conda_env: str | None = None
    python_path: str | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeldmanConfig":
        eds = d.get("edit_distances", [1, 2])
        if isinstance(eds, int):
            eds = [eds]
        return cls(
            edit_distances=list(eds),
            conda_env=d.get("conda_env"),
            python_path=d.get("python_path"),
        )


@dataclass
class OpsBenchmarkConfig:
    """Top-level OPS benchmark config (renamed from BenchmarkConfig)."""

    outdir: Path
    cache_dir: Path | None
    trials: int
    seed: int | None
    device: str
    candidate_pool: CandidatePoolConfig
    evaluator: EvaluatorConfig
    # DUET fields (previously bundled in BenchmarkDuetConfig — now inlined).
    duet_optimizer: DuetOptimizerConfig | None = None
    duet_pep: EvaluatorConfig | None = None
    duet_cache_dir: Path | None = None
    duet_use_mmap: bool = True
    duet_device: str = "cpu"
    duet_force_rebuild: bool = False
    duet_sym_mem_budget_gb: float = 32.0
    duet_initialization_raw: Dict[str, Any] | None = None  # parsed by runner via ops_benchmark.init.parse_init
    # Eval step resolution.
    eval_device: str = "cpu"
    eval_mem_budget_gb: float = 20.0
    mem_budget_gb: float | None = None
    scratch_dir: str | None = None
    # Baselines (optional).
    sivanandan: SivanandanConfig | None = None
    feldman: FeldmanConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "OpsBenchmarkConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpsBenchmarkConfig":
        top_device = normalized_top_device(d)
        top_mem = d.get("mem_budget_gb")
        top_scratch = d.get("scratch_dir")

        evaluator = EvaluatorConfig.from_dict(d["evaluator"])
        evaluator.scratch_dir = resolve_scratch(top_scratch, evaluator.scratch_dir)
        eval_device = resolve_step_device(top_device, evaluator.device, step_name="eval")
        eval_mem_budget_gb = resolve_step_mem(top_mem, evaluator.mem_budget_gb, DEFAULT_EVAL_MEM_GB)

        duet_kwargs: Dict[str, Any] = dict(
            duet_optimizer=None, duet_pep=None, duet_cache_dir=None,
            duet_use_mmap=True, duet_device=top_device, duet_force_rebuild=False,
            duet_sym_mem_budget_gb=(top_mem if top_mem is not None else DEFAULT_SYM_MEM_GB),
            duet_initialization_raw=None,
        )
        duet = d.get("duet")
        if duet is not None:
            reject_nested_eval_mem(duet)
            require_duet_pep(duet)
            reject_pep_runtime_keys(duet)
            duet_pep = EvaluatorConfig.from_dict(duet["pep"])
            duet_pep.scratch_dir = resolve_scratch(top_scratch, duet_pep.scratch_dir)
            duet_kwargs = dict(
                duet_optimizer=DuetOptimizerConfig.from_dict(duet["optimizer"]),
                duet_pep=duet_pep,
                duet_cache_dir=Path(duet["cache_dir"]) if duet.get("cache_dir") else None,
                duet_use_mmap=duet.get("use_mmap", True),
                duet_device=resolve_step_device(top_device, duet.get("device"), step_name="DUET"),
                duet_force_rebuild=duet.get("force_rebuild", False),
                duet_sym_mem_budget_gb=resolve_step_mem(top_mem, duet.get("sym_mem_budget_gb"), DEFAULT_SYM_MEM_GB),
                duet_initialization_raw=duet.get("initialization"),
            )

        return cls(
            outdir=Path(d["outdir"]),
            cache_dir=Path(d["cache_dir"]) if d.get("cache_dir") else None,
            trials=d.get("trials", 5),
            seed=d.get("seed"),
            device=top_device,
            mem_budget_gb=top_mem,
            scratch_dir=top_scratch,
            eval_device=eval_device,
            eval_mem_budget_gb=eval_mem_budget_gb,
            candidate_pool=CandidatePoolConfig.from_dict(d["candidate_pool"]),
            evaluator=evaluator,
            sivanandan=SivanandanConfig.from_dict(d["sivanandan"]) if "sivanandan" in d else None,
            feldman=FeldmanConfig.from_dict(d["feldman"]) if "feldman" in d else None,
            **duet_kwargs,
        )

    def __post_init__(self):
        """Validate warm-start config: if DUET initialization requires a
        baseline, confirm that baseline section is present and its edit_distances
        list covers the warm-start edit_distance.
        """
        if self.duet_initialization_raw is None:
            return
        init_type = self.duet_initialization_raw.get("type", "random")
        if init_type == "sivanandan":
            if self.sivanandan is None:
                raise ValueError(
                    "DUET initialization.type='sivanandan' requires a 'sivanandan' section."
                )
            ed = self.duet_initialization_raw.get("edit_distance")
            if ed is not None and ed not in self.sivanandan.edit_distances:
                logger.info(f"Auto-adding edit_distance={ed} to sivanandan.edit_distances")
                self.sivanandan.edit_distances.append(ed)
                self.sivanandan.edit_distances.sort()
        elif init_type == "feldman":
            if self.feldman is None:
                raise ValueError(
                    "DUET initialization.type='feldman' requires a 'feldman' section."
                )
            ed = self.duet_initialization_raw.get("edit_distance")
            if ed is not None and ed not in self.feldman.edit_distances:
                logger.info(f"Auto-adding edit_distance={ed} to feldman.edit_distances")
                self.feldman.edit_distances.append(ed)
                self.feldman.edit_distances.sort()

    @property
    def run_duet(self) -> bool:
        return self.duet_optimizer is not None

    @property
    def run_sivanandan(self) -> bool:
        return self.sivanandan is not None

    @property
    def run_feldman(self) -> bool:
        return self.feldman is not None

    @property
    def methods(self) -> List[str]:
        methods = []
        if self.run_duet:
            methods.append("duet")
        if self.run_sivanandan:
            methods.append("sivanandan")
        if self.run_feldman:
            methods.append("feldman")
        return methods
