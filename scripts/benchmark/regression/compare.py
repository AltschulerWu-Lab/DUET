"""Tolerant numeric comparison of benchmark output tables against a frozen
reference. Pure / GPU-free so it can be unit-tested in CI.

See README.md in this folder for the tolerances and how the driver uses it.
"""
from __future__ import annotations

import dataclasses
import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RTOL = 1e-6
ATOL = 1e-9


@dataclass
class ColumnDiff:
    column: str
    kind: str
    n_mismatch: int
    max_abs: float
    max_rel: float


@dataclass
class FileComparison:
    name: str
    passed: bool
    n_reference: int = 0
    n_fresh: int = 0
    n_compared: int = 0
    only_in_reference: int = 0
    only_in_fresh: int = 0
    column_diffs: list[ColumnDiff] = field(default_factory=list)
    sample_mismatches: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class FixtureComparison:
    name: str
    passed: bool
    files: list[FileComparison] = field(default_factory=list)
    error: str | None = None


def values_close(ref, fresh, rtol: float = RTOL, atol: float = ATOL) -> bool:
    """True if two scalars match: numbers within tolerance, everything else exact.

    bool is treated as exact (not as the numbers 0/1).
    """
    if isinstance(ref, bool) or isinstance(fresh, bool):
        return type(ref) is type(fresh) and ref == fresh
    if isinstance(ref, (int, float)) and isinstance(fresh, (int, float)):
        a, b = float(ref), float(fresh)
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return ref == fresh


def _read_text(path) -> str:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    return path.read_text()


def read_table(path, str_cols=()) -> pd.DataFrame:
    """Read a CSV (optionally gzipped), forcing str_cols to string dtype so
    binary Sequence columns keep their leading zeros."""
    path = Path(path)
    dtype = {c: str for c in str_cols}
    kwargs = {"dtype": dtype}
    if path.suffix == ".gz":
        kwargs["compression"] = "gzip"
    return pd.read_csv(path, **kwargs)


def compare_csv(reference_path, fresh_path, key_cols, *, str_cols=(),
                exclude_cols=(), rtol: float = RTOL, atol: float = ATOL) -> FileComparison:
    """Compare two CSV tables aligned by key_cols. Float value columns are
    compared within tolerance; all others exactly. Added/removed keys fail."""
    name = Path(fresh_path).name
    read_str = tuple(set(key_cols) | set(str_cols))
    ref = read_table(reference_path, str_cols=read_str)
    fresh = read_table(fresh_path, str_cols=read_str)

    for label, df in (("reference", ref), ("fresh", fresh)):
        if df.duplicated(subset=key_cols).any():
            return FileComparison(name=name, passed=False,
                                  n_reference=len(ref), n_fresh=len(fresh),
                                  error=f"non-unique key {key_cols} in {label}")

    ref_i = ref.set_index(key_cols).sort_index()
    fresh_i = fresh.set_index(key_cols).sort_index()
    ref_keys, fresh_keys = set(ref_i.index), set(fresh_i.index)
    only_ref = ref_keys - fresh_keys
    only_fresh = fresh_keys - ref_keys
    common = sorted(ref_keys & fresh_keys)

    r = ref_i.loc[common]
    f = fresh_i.loc[common]
    value_cols = [c for c in ref_i.columns
                  if c not in exclude_cols and c in fresh_i.columns]

    def _safe_max(arr):
        return float(np.nanmax(arr)) if arr.size and not np.all(np.isnan(arr)) else 0.0

    column_diffs, samples = [], []
    for c in ref_i.columns:
        if c in exclude_cols:
            continue
        if c not in fresh_i.columns:
            column_diffs.append(ColumnDiff(c, "missing", 1, 0.0, 0.0))
    for c in value_cols:
        rc, fc_ = r[c], f[c]
        if pd.api.types.is_float_dtype(rc) or pd.api.types.is_float_dtype(fc_):
            a = rc.to_numpy(dtype=float)
            b = fc_.to_numpy(dtype=float)
            close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
            absd = np.abs(a - b)
            with np.errstate(divide="ignore", invalid="ignore"):
                reld = np.where(np.abs(a) > 0, absd / np.abs(a), 0.0)
            n = int((~close).sum())
            column_diffs.append(ColumnDiff(c, "float", n, _safe_max(absd), _safe_max(reld)))
            for bi in np.where(~close)[0][:5]:
                samples.append({"key": common[bi], "column": c,
                                "reference": float(a[bi]), "fresh": float(b[bi])})
        else:
            mism = rc.to_numpy() != fc_.to_numpy()
            n = int(mism.sum())
            column_diffs.append(ColumnDiff(c, "exact", n, 0.0, 0.0))
            for bi in np.where(mism)[0][:5]:
                samples.append({"key": common[bi], "column": c,
                                "reference": rc.iloc[bi], "fresh": fc_.iloc[bi]})

    passed = (not only_ref) and (not only_fresh) and all(
        cd.n_mismatch == 0 for cd in column_diffs)
    return FileComparison(name=name, passed=passed,
                          n_reference=len(ref), n_fresh=len(fresh),
                          n_compared=len(common),
                          only_in_reference=len(only_ref),
                          only_in_fresh=len(only_fresh),
                          column_diffs=column_diffs,
                          sample_mismatches=samples[:20])


def _diff_yaml(ref, fresh, rtol, atol, path=""):
    out = []
    if isinstance(ref, dict) and isinstance(fresh, dict):
        for k in sorted(set(ref) | set(fresh)):
            child = f"{path}.{k}" if path else k
            if k not in ref:
                out.append({"path": child, "reason": "only_in_fresh"})
            elif k not in fresh:
                out.append({"path": child, "reason": "only_in_reference"})
            else:
                out += _diff_yaml(ref[k], fresh[k], rtol, atol, child)
    elif isinstance(ref, list) and isinstance(fresh, list):
        if len(ref) != len(fresh):
            out.append({"path": path, "reason": f"len {len(ref)} vs {len(fresh)}"})
        else:
            for i, (a, b) in enumerate(zip(ref, fresh)):
                out += _diff_yaml(a, b, rtol, atol, f"{path}[{i}]")
    elif not values_close(ref, fresh, rtol, atol):
        out.append({"path": path or ".", "reference": ref, "fresh": fresh})
    return out


def compare_yaml(reference_path, fresh_path, rtol: float = RTOL,
                 atol: float = ATOL) -> FileComparison:
    """Compare two YAML docs recursively: floats within tolerance, else exact."""
    ref = yaml.safe_load(_read_text(reference_path))
    fresh = yaml.safe_load(_read_text(fresh_path))
    mm = _diff_yaml(ref, fresh, rtol, atol)
    return FileComparison(name=Path(fresh_path).name, passed=(len(mm) == 0),
                          sample_mismatches=mm[:20])


def render_report(fixtures) -> str:
    lines = []
    for fx in fixtures:
        lines.append(f"[{'PASS' if fx.passed else 'FAIL'}] {fx.name}")
        if fx.error:
            lines.append(f"    ERROR: {fx.error}")
        for fc in fx.files:
            tag = "ok  " if fc.passed else "DIFF"
            lines.append(
                f"    {tag} {fc.name}  (ref={fc.n_reference} fresh={fc.n_fresh} "
                f"+{fc.only_in_fresh}/-{fc.only_in_reference})")
            if fc.error:
                lines.append(f"         error: {fc.error}")
            for cd in fc.column_diffs:
                if cd.n_mismatch:
                    lines.append(
                        f"         {cd.column}: {cd.n_mismatch} mismatch, "
                        f"max_abs={cd.max_abs:.3g} max_rel={cd.max_rel:.3g}")
            for m in fc.sample_mismatches[:5]:
                lines.append(f"           e.g. {m}")
    overall = "PASS" if all(f.passed for f in fixtures) else "FAIL"
    lines.append(f"OVERALL: {overall}")
    return "\n".join(lines)


def to_json(fixtures) -> dict:
    return {"passed": all(f.passed for f in fixtures),
            "fixtures": [dataclasses.asdict(f) for f in fixtures]}
