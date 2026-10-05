# src/duet/merfish_benchmark/config.py
"""Configuration dataclasses for the MERFISH benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from duet.benchmark.config_common import (
    DEFAULT_EVAL_MEM_GB, DEFAULT_SYM_MEM_GB,
    normalized_top_device, resolve_step_device, resolve_step_mem, resolve_scratch,
    require_duet_pep, reject_pep_runtime_keys, reject_nested_eval_mem,
)
from duet.runner.core import DuetOptimizerConfig
from duet.evaluator_config import EvaluatorConfig


def resolve_config_path(path_str, config_dir):
    """Resolve a yaml path string against a config file's directory.

    Absolute paths are returned unchanged. Relative paths are joined with
    ``config_dir`` and `.resolve()`-d to fold any ``..`` traversal into a
    clean absolute path. When ``config_dir`` is None (legacy callers), the
    path is returned as-is — preserves the pre-config-relative behavior.

    Args:
        path_str: The path string from the yaml. May be None (returns None).
        config_dir: The directory of the config file. Resolved paths land
            relative to this directory.

    Returns:
        Absolute Path (when path is relative and config_dir is given);
        otherwise the original Path; None if path_str is None.
    """
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute() or config_dir is None:
        return p
    return (Path(config_dir) / p).resolve()


@dataclass
class MerfishCandidatesConfig:
    seq_rounds: int
    codebook_size: int
    hamming_weights: List[int] | None = None
    hamming_weight_limits: Dict[int, int] | None = None
    subsample_seed: int | None = None
    per_codeword_sample_size: int | None = None
    sample_seed: int | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MerfishCandidatesConfig":
        return cls(
            seq_rounds=d["seq_rounds"],
            codebook_size=d["codebook_size"],
            hamming_weights=d.get("hamming_weights"),
            hamming_weight_limits=d.get("hamming_weight_limits"),
            subsample_seed=d.get("subsample_seed"),
            per_codeword_sample_size=d.get("per_codeword_sample_size"),
            sample_seed=d.get("sample_seed"),
        )


@dataclass
class ExpressionConfig:
    path: Path
    gene_col: str = "gene_name"
    expression_col: str = "mean_raw_counts"

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config_dir: Path | None = None) -> "ExpressionConfig":
        return cls(
            path=resolve_config_path(d["path"], config_dir),
            gene_col=d.get("gene_col", "gene_name"),
            expression_col=d.get("expression_col", "mean_raw_counts"),
        )


@dataclass
class RandomGenesConfig:
    pool_filter: str
    pool_filter_value: int
    exclude_controls: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RandomGenesConfig":
        return cls(
            pool_filter=d["pool_filter"],
            pool_filter_value=d["pool_filter_value"],
            exclude_controls=d.get("exclude_controls", True),
        )


@dataclass
class FromCsvGenesConfig:
    path: Path
    gene_col: str = "Gene"

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config_dir: Path | None = None) -> "FromCsvGenesConfig":
        return cls(
            path=resolve_config_path(d["path"], config_dir),
            gene_col=d.get("gene_col", "Gene"),
        )


GenesConfigT = RandomGenesConfig | FromCsvGenesConfig


def parse_genes_config(d: Dict[str, Any], config_dir: Path | None = None) -> GenesConfigT:
    """Dispatch `genes:` block to RandomGenesConfig or FromCsvGenesConfig by `type` discriminator."""
    t = d.get("type")
    if t == "random":
        return RandomGenesConfig.from_dict(d)
    if t == "from_csv":
        return FromCsvGenesConfig.from_dict(d, config_dir=config_dir)
    raise ValueError(f"Unknown genes.type: {t!r}. Expected 'random' or 'from_csv'.")


@dataclass
class CrowdingConfig:
    n_trials: int = 5
    total_reads: int = 100_000
    cell_size_um: float = 100.0
    wavelength_nm: float = 500.0
    numerical_aperture: float = 1.4
    expansion_factor: float = 1.0
    diffraction_model: str = "abbe"
    neighborhood: str = "box"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrowdingConfig":
        return cls(
            n_trials=d.get("n_trials", 5),
            total_reads=d.get("total_reads", 100_000),
            cell_size_um=d.get("cell_size_um", 100.0),
            wavelength_nm=d.get("wavelength_nm", 500.0),
            numerical_aperture=d.get("numerical_aperture", 1.4),
            expansion_factor=d.get("expansion_factor", 1.0),
            diffraction_model=d.get("diffraction_model", "abbe"),
            neighborhood=d.get("neighborhood", "box"),
        )


@dataclass
class InitializationConfig:
    type: str
    path: Path | None = None
    sequence_col: str = "Sequence"
    noise_percent: float = 0.0

    def __post_init__(self):
        if self.type == "warm_start" and self.path is None:
            raise ValueError("InitializationConfig: path is required when type='warm_start'")

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config_dir: Path | None = None) -> "InitializationConfig":
        t = d.get("type", "random")
        if t == "random":
            return cls(type="random")
        if t == "warm_start":
            return cls(
                type="warm_start",
                path=resolve_config_path(d["path"], config_dir),
                sequence_col=d.get("sequence_col", "Sequence"),
                noise_percent=d.get("noise_percent", 0.0),
            )
        raise ValueError(f"Unknown initialization.type: {t!r}. Expected 'random' or 'warm_start'.")


@dataclass
class BaselineConfig:
    name: str
    path: Path
    gene_col: str = "Gene"
    sequence_col: str = "Sequence"

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config_dir: Path | None = None) -> "BaselineConfig":
        return cls(
            name=d["name"],
            path=resolve_config_path(d["path"], config_dir),
            gene_col=d.get("gene_col", "Gene"),
            sequence_col=d.get("sequence_col", "Sequence"),
        )

    @classmethod
    def parse_list(cls, raw, config_dir: Path | None = None) -> "List[BaselineConfig]":
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError(
                "baselines: must be a list of dicts. The pre-2026-05-25 struct "
                "(`baselines: {codebook_1: ..., codebook_2: ...}`) is no longer supported."
            )
        entries = [cls.from_dict(d, config_dir=config_dir) for d in raw]
        names = [e.name for e in entries]
        if len(names) != len(set(names)):
            dups = [n for n in set(names) if names.count(n) > 1]
            raise ValueError(f"baselines: duplicate name(s): {dups}")
        return entries


def _validate_csv_has_column(path: Path, column: str, field_label: str, col_param: str) -> None:
    """Validate that a CSV file exists and contains the expected column.

    Args:
        path: Path to the CSV file.
        column: Column name that must be present.
        field_label: Human-readable label used in error messages (e.g. "genes", "initialization").
        col_param: Name of the config parameter holding the column name (e.g. "gene_col",
            "sequence_col"). Included in the missing-column error message.
    """
    if not path.exists():
        raise ValueError(f"{field_label}.path does not exist: {path}")
    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception as e:
        raise ValueError(f"{field_label}.path: failed to read header: {e}") from e
    if column not in cols:
        raise ValueError(
            f"{field_label}.path {path} missing {col_param}={column!r} "
            f"(columns: {list(cols)})"
        )


@dataclass
class MerfishBenchmarkConfig:
    """Top-level MERFISH benchmark config (post-2026-05-25 schema)."""

    outdir: Path
    candidates: MerfishCandidatesConfig
    evaluator: EvaluatorConfig
    duet_optimizer: DuetOptimizerConfig
    duet_pep: EvaluatorConfig | None = None
    duet_cache_dir: Path | None = None
    duet_use_mmap: bool = True
    duet_device: str = "cpu"
    duet_force_rebuild: bool = False
    duet_sym_mem_budget_gb: float = 32.0
    eval_device: str = "cpu"          # device for the union ground-truth eval; CPU = today
    eval_mem_budget_gb: float = 20.0  # matmul batch-size budget for the union eval (NOT the CSR cache)
    device: str = "cpu"
    mem_budget_gb: float | None = None
    scratch_dir: str | None = None
    expression: ExpressionConfig | None = None
    genes: GenesConfigT | None = None
    crowding: CrowdingConfig | None = None
    initialization: InitializationConfig = field(default_factory=lambda: InitializationConfig(type="random"))
    baselines: List[BaselineConfig] = field(default_factory=list)
    cache_dir: Path | None = None
    seed: int | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "MerfishBenchmarkConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d, config_dir=Path(path).parent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config_dir: Path | None = None) -> "MerfishBenchmarkConfig":
        if "optical_crowding" in d:
            raise ValueError(
                "`optical_crowding` block removed in 2026-05-25 schema. "
                "Migrate to top-level `expression` + `genes` + `crowding`. "
                "See experiments/merfish_zhang2023_v2/config.yaml for the current layout."
            )
        if "eval_device" in d or "eval_mem_budget_gb" in d:
            raise ValueError(
                "top-level `eval_device`/`eval_mem_budget_gb` moved to "
                "`evaluator.device`/`evaluator.mem_budget_gb` in the 2026-06-21 schema."
            )
        duet = d["duet"]
        if "initialization" in duet:
            raise ValueError(
                "`duet.initialization` removed in 2026-05-25 schema. "
                "Move to top-level `initialization` block. "
                "See experiments/merfish_zhang2023_v2/config.yaml for the current layout."
            )
        if "optimizer" in duet:
            raise ValueError(
                "`duet.optimizer` removed in 2026-05-25 schema. "
                "Flatten optimizer fields directly under `duet`. "
                "See experiments/merfish_zhang2023_v2/config.yaml for the current layout."
            )
        reject_nested_eval_mem(duet)
        require_duet_pep(duet)
        reject_pep_runtime_keys(duet)

        top_device = normalized_top_device(d)
        top_mem = d.get("mem_budget_gb")
        top_scratch = d.get("scratch_dir")

        evaluator = EvaluatorConfig.from_dict(d["evaluator"])
        evaluator.scratch_dir = resolve_scratch(top_scratch, evaluator.scratch_dir)
        duet_pep = EvaluatorConfig.from_dict(duet["pep"])
        duet_pep.scratch_dir = resolve_scratch(top_scratch, duet_pep.scratch_dir)

        return cls(
            outdir=resolve_config_path(d["outdir"], config_dir),
            cache_dir=resolve_config_path(d["cache_dir"], config_dir) if d.get("cache_dir") else None,
            seed=d.get("seed"),
            candidates=MerfishCandidatesConfig.from_dict(d["candidates"]),
            evaluator=evaluator,
            duet_optimizer=DuetOptimizerConfig.from_dict(duet),
            duet_pep=duet_pep,
            duet_cache_dir=resolve_config_path(duet["cache_dir"], config_dir) if duet.get("cache_dir") else None,
            duet_use_mmap=duet.get("use_mmap", True),
            duet_device=resolve_step_device(top_device, duet.get("device"), step_name="DUET"),
            duet_force_rebuild=duet.get("force_rebuild", False),
            duet_sym_mem_budget_gb=resolve_step_mem(top_mem, duet.get("sym_mem_budget_gb"), DEFAULT_SYM_MEM_GB),
            device=top_device,
            mem_budget_gb=top_mem,
            scratch_dir=top_scratch,
            eval_device=resolve_step_device(top_device, evaluator.device, step_name="eval"),
            eval_mem_budget_gb=resolve_step_mem(top_mem, evaluator.mem_budget_gb, DEFAULT_EVAL_MEM_GB),
            expression=ExpressionConfig.from_dict(d["expression"], config_dir=config_dir) if d.get("expression") else None,
            genes=parse_genes_config(d["genes"], config_dir=config_dir) if d.get("genes") else None,
            crowding=CrowdingConfig.from_dict(d.get("crowding", {})) if d.get("crowding") else None,
            initialization=InitializationConfig.from_dict(
                d.get("initialization", {"type": "random"}), config_dir=config_dir,
            ),
            baselines=BaselineConfig.parse_list(d.get("baselines"), config_dir=config_dir),
        )

    def __post_init__(self):
        # Cross-block validation.
        if self.crowding is not None:
            if self.expression is None:
                raise ValueError("crowding: requires top-level `expression` block")
            if self.genes is None:
                raise ValueError("crowding: requires top-level `genes` block")
        else:
            # lambda weights decoding accuracy, so any lambda < 1 gives the
            # optical-crowding objective nonzero weight (1 - lambda). Without a
            # crowding block the runner substitutes a uniform placeholder
            # expression vector, which is only sound at zero crowding weight.
            non_decode = [lam for lam in self.duet_optimizer.lambda_ if lam < 1.0]
            if non_decode:
                raise ValueError(
                    f"duet lambda contains values below 1.0 ({non_decode}) but no "
                    "`crowding` block is configured. lambda weights decoding "
                    "accuracy, so lambda < 1.0 would optimize optical crowding "
                    "against a placeholder uniform expression vector. Add a "
                    "`crowding` block (with `expression` and `genes`) or use "
                    "lambda: [1.0] for decode-only optimization."
                )
        if isinstance(self.genes, FromCsvGenesConfig):
            _validate_csv_has_column(self.genes.path, self.genes.gene_col, "genes", "gene_col")
        if self.initialization.type == "warm_start":
            _validate_csv_has_column(
                self.initialization.path,
                self.initialization.sequence_col,
                "initialization",
                "sequence_col",
            )
