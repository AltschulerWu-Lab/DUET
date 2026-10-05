import gzip
import math
import sys
from pathlib import Path

import pandas as pd
import yaml as _yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare


def test_values_close_floats_within_tol():
    assert compare.values_close(1.0, 1.0 + 1e-9)
    assert not compare.values_close(1.0, 1.0 + 1e-3)


def test_values_close_nan_equal():
    assert compare.values_close(float("nan"), float("nan"))


def test_values_close_bool_is_exact_not_numeric():
    # bool must not be treated as the number 1/0
    assert compare.values_close(True, True)
    assert not compare.values_close(True, 1.0)


def test_values_close_strings_exact():
    assert compare.values_close("ACGT", "ACGT")
    assert not compare.values_close("ACGT", "ACGA")


def _write_csv(path, df, gz=False):
    if gz:
        with gzip.open(path, "wt") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(path, index=False)


def test_compare_csv_identical_passes(tmp_path):
    df = pd.DataFrame(
        {"Method": ["A", "A"], "Index": [0, 1],
         "Sequence": ["0010", "1100"], "Decode accuracy": [0.81, 0.79],
         "Quota": [2, 2], "Valid": [True, True]}
    )
    ref, fresh = tmp_path / "ref.csv.gz", tmp_path / "fresh.csv"
    _write_csv(ref, df, gz=True)
    _write_csv(fresh, df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Method", "Index"],
                             str_cols=["Method", "Sequence"])
    assert fc.passed
    assert fc.n_compared == 2


def test_compare_csv_float_drift_within_tol_passes(tmp_path):
    base = pd.DataFrame({"Method": ["A"], "Index": [0], "Decode accuracy": [0.8]})
    drift = base.copy()
    drift["Decode accuracy"] = [0.8 + 1e-9]
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, base)
    _write_csv(fresh, drift)
    fc = compare.compare_csv(ref, fresh, key_cols=["Method", "Index"])
    assert fc.passed


def test_compare_csv_float_drift_beyond_tol_fails(tmp_path):
    base = pd.DataFrame({"Method": ["A"], "Index": [0], "Decode accuracy": [0.8]})
    drift = base.copy()
    drift["Decode accuracy"] = [0.9]
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, base)
    _write_csv(fresh, drift)
    fc = compare.compare_csv(ref, fresh, key_cols=["Method", "Index"])
    assert not fc.passed
    assert any(cd.column == "Decode accuracy" and cd.n_mismatch == 1
               for cd in fc.column_diffs)


def test_compare_csv_added_removed_key_fails(tmp_path):
    ref_df = pd.DataFrame({"Method": ["A", "A"], "Index": [0, 1], "x": [1.0, 2.0]})
    fresh_df = pd.DataFrame({"Method": ["A", "A"], "Index": [0, 2], "x": [1.0, 2.0]})
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, ref_df)
    _write_csv(fresh, fresh_df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Method", "Index"])
    assert not fc.passed
    assert fc.only_in_reference == 1 and fc.only_in_fresh == 1


def test_compare_csv_string_value_change_fails(tmp_path):
    ref_df = pd.DataFrame({"Index": [0], "Sequence": ["0010"]})
    fresh_df = pd.DataFrame({"Index": [0], "Sequence": ["1100"]})
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, ref_df)
    _write_csv(fresh, fresh_df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Index"], str_cols=["Sequence"])
    assert not fc.passed


def test_compare_csv_preserves_leading_zero_sequences(tmp_path):
    # str_cols must stop pandas inferring a binary Sequence as int (dropping 0s)
    df = pd.DataFrame({"Index": [0], "Sequence": ["0010010000001100"]})
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, df)
    _write_csv(fresh, df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Index"], str_cols=["Sequence"])
    assert fc.passed


def test_compare_csv_non_unique_key_errors(tmp_path):
    df = pd.DataFrame({"Index": [0, 0], "x": [1.0, 2.0]})
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, df)
    _write_csv(fresh, df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Index"])
    assert not fc.passed
    assert fc.error is not None and "non-unique" in fc.error


def test_compare_csv_exclude_cols_skipped(tmp_path):
    ref_df = pd.DataFrame({"Index": [0], "metric": [1.0], "noise": [1.0]})
    fresh_df = pd.DataFrame({"Index": [0], "metric": [1.0], "noise": [999.0]})
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, ref_df)
    _write_csv(fresh, fresh_df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Index"], exclude_cols=["noise"])
    assert fc.passed


def test_compare_csv_missing_column_fails(tmp_path):
    ref_df = pd.DataFrame({"Index": [0], "metric": [1.0]})
    fresh_df = pd.DataFrame({"Index": [0]})
    ref, fresh = tmp_path / "ref.csv", tmp_path / "fresh.csv"
    _write_csv(ref, ref_df)
    _write_csv(fresh, fresh_df)
    fc = compare.compare_csv(ref, fresh, key_cols=["Index"])
    assert not fc.passed
    assert any(cd.column == "metric" for cd in fc.column_diffs)


def _write_yaml(path, obj, gz=False):
    text = _yaml.dump(obj, default_flow_style=False)
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


def test_compare_yaml_float_within_tol_passes(tmp_path):
    ref_obj = {"initial_accuracy": 0.8148707142857142, "seed": 42,
               "lambdas": [0.0, 0.5, 1.0], "init_description": "warm_start"}
    fresh_obj = dict(ref_obj, initial_accuracy=0.8148707142857142 + 1e-12)
    ref, fresh = tmp_path / "ref.yaml.gz", tmp_path / "fresh.yaml"
    _write_yaml(ref, ref_obj, gz=True)
    _write_yaml(fresh, fresh_obj)
    fc = compare.compare_yaml(ref, fresh)
    assert fc.passed


def test_compare_yaml_int_or_string_change_fails(tmp_path):
    ref_obj = {"seed": 42, "init_description": "warm_start"}
    fresh_obj = {"seed": 7, "init_description": "warm_start"}
    ref, fresh = tmp_path / "ref.yaml", tmp_path / "fresh.yaml"
    _write_yaml(ref, ref_obj)
    _write_yaml(fresh, fresh_obj)
    fc = compare.compare_yaml(ref, fresh)
    assert not fc.passed


def test_compare_yaml_list_change_fails(tmp_path):
    ref_obj = {"lambdas": [0.0, 0.5, 1.0]}
    fresh_obj = {"lambdas": [0.0, 0.6, 1.0]}
    ref, fresh = tmp_path / "ref.yaml", tmp_path / "fresh.yaml"
    _write_yaml(ref, ref_obj)
    _write_yaml(fresh, fresh_obj)
    fc = compare.compare_yaml(ref, fresh)
    assert not fc.passed


def test_render_and_json_report():
    import json
    import numpy as np
    import pytest

    # compare_csv stores numpy scalars (e.g. np.int64) directly from the
    # DataFrame in sample_mismatches["reference"/"fresh"] for exact-match
    # columns.  Plain json.dumps cannot serialize numpy scalars; default=str
    # is required.  The "key" field holds a tuple from a multi-column index
    # (also non-standard, serialized as an array by default=str context).
    files = [
        compare.FileComparison(
            name="results.csv", passed=False,
            n_reference=10, n_fresh=10, only_in_fresh=1,
            column_diffs=[compare.ColumnDiff("Decode accuracy", "float", 2, 0.05, 0.06)],
            sample_mismatches=[{"key": ("Method_A", 3), "column": "Decode accuracy",
                                "reference": np.int64(9), "fresh": np.int64(8)}],
        )
    ]
    fx = [compare.FixtureComparison(name="ops_uniform", passed=False, files=files)]
    text = compare.render_report(fx)
    assert "ops_uniform" in text and "FAIL" in text and "Decode accuracy" in text
    blob = compare.to_json(fx)
    assert blob["passed"] is False
    assert blob["fixtures"][0]["name"] == "ops_uniform"
    # numpy scalar in sample_mismatches must make bare json.dumps raise …
    with pytest.raises(TypeError):
        json.dumps(blob)
    # … and succeed only when default=str is supplied
    json.dumps(blob, default=str)


def test_render_report_all_pass():
    fx = [compare.FixtureComparison(name="merfish_asymmetric", passed=True,
                                    files=[compare.FileComparison("metrics.csv", True)])]
    assert "OVERALL: PASS" in compare.render_report(fx)
