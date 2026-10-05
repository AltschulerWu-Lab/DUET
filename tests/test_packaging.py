"""Packaging behavior: version, light imports, optional GPU, cache provenance, data paths."""

import json
import os
import subprocess
import sys
from importlib.metadata import version
from unittest.mock import patch

import numpy as np
import pytest

import duet
from duet.evaluator_config import EvaluatorConfig
from duet.providers import PEPMatrixProvider

LIBRARY = ["ACGTAC", "ACGTTT", "TTGACA", "GGGCCA", "CATCAT", "ACGAAC"]


def _config(seed=3):
    return EvaluatorConfig.default_symmetric(0.1, num_samples=200, num_cpus=1, seed=seed)


def _run(code, env=None):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, **(env or {})}, check=True,
    ).stdout


def test_version_comes_from_distribution_metadata():
    assert duet.__version__ == version("duet-codebook")


def test_citation_version_matches_pyproject():
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    want = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    got = re.search(r"^version:\s*(\S+)\s*$", (root / "CITATION.cff").read_text(), re.M).group(1)
    assert got == want


def test_import_duet_does_not_load_optional_or_plotting_packages():
    out = _run(
        "import sys, duet; "
        "print(sorted(m for m in ('cupy', 'pymoo', 'matplotlib', 'seaborn', 'sklearn', 'yaml') "
        "if m in sys.modules))"
    )
    assert out.strip() == "[]"


def test_engine_modules_do_not_load_pymoo_matplotlib_or_cupy():
    out = _run(
        "import sys\n"
        "import duet.runner, duet.benchmark.metrics, duet.optical_crowding, duet.merfish_factory\n"
        "print(sorted(m for m in ('cupy', 'pymoo', 'matplotlib', 'seaborn') if m in sys.modules))"
    )
    assert out.strip() == "[]"


def test_evaluation_works_without_pymoo():
    # Simulate an install without the [benchmark] extra.
    out = _run(
        "import sys; sys.modules['pymoo'] = None\n"
        "from duet.benchmark.metrics import evaluate_codebooks_by_sequence, compute_metrics\n"
        "from duet.evaluator_config import EvaluatorConfig\n"
        "cfg = EvaluatorConfig.default_symmetric(0.1, num_samples=50, num_cpus=1, seed=1)\n"
        f"res = evaluate_codebooks_by_sequence([{LIBRARY[:3]!r}], eval_config=cfg, alphabet_size=4, n_jobs=1)\n"
        "print(round(compute_metrics(res[0].codeword_accuracy)['Mean decode accuracy'], 6))"
    )
    assert 0.0 <= float(out) <= 1.0


class TestGpuWithoutCupy:
    def test_pep_error_names_the_gpu_extra(self):
        from duet.evaluator_config import create_evaluator

        evaluator = create_evaluator(LIBRARY, _config(), alphabet_size=4)
        with patch("duet.gpu_utils.HAS_CUPY", False):
            with pytest.raises(ImportError, match=r"CuPy is required.*duet-codebook\[gpu\]"):
                evaluator.compute_pep_matrix(device="gpu:all")

    def test_evaluation_error_names_the_gpu_extra(self):
        from duet.evaluator_config import create_evaluator

        evaluator = create_evaluator(LIBRARY, _config(), alphabet_size=4)
        with patch("duet.gpu_utils.HAS_CUPY", False):
            with pytest.raises(ImportError, match=r"duet-codebook\[gpu\]"):
                evaluator.initialize_cache(device="gpu")

    def test_cached_pep_fails_before_allocating_the_matrix(self, tmp_path):
        provider = PEPMatrixProvider(tmp_path)
        with patch("duet.gpu_utils.HAS_CUPY", False):
            with pytest.raises(ImportError, match=r"duet-codebook\[gpu\]"):
                provider.get_mmap(LIBRARY, _config(), alphabet_size=4, n_jobs=1, device="gpu:all")
        assert list(tmp_path.iterdir()) == []


class TestPepCacheDeviceMetadata:
    def test_raw_and_symmetric_metadata_record_the_device(self, tmp_path):
        provider = PEPMatrixProvider(tmp_path)
        assert provider.cache_status(LIBRARY, _config(), 4) == {
            "fingerprint": provider._compute_fingerprint(LIBRARY, _config(), 4),
            "sym_hit": False, "raw_hit": False, "device": None,
        }
        provider.get_mmap(LIBRARY, _config(), alphabet_size=4, n_jobs=1, device="cpu")
        fp = provider._compute_fingerprint(LIBRARY, _config(), 4)
        raw = json.loads(provider.cache.get_path(fp, ".meta.json").read_text())
        sym = json.loads(provider.cache.get_path(fp, ".sym.meta.json").read_text())
        assert raw["device"] == "cpu" and sym["device"] == "cpu"
        status = provider.cache_status(LIBRARY, _config(), 4)
        assert status["sym_hit"] and status["raw_hit"] and status["device"] == "cpu"

    def test_device_is_not_part_of_the_cache_key(self, tmp_path):
        cpu_cfg, gpu_cfg = _config(), _config()
        gpu_cfg.device = "gpu:all"
        provider = PEPMatrixProvider(tmp_path)
        assert provider._compute_fingerprint(LIBRARY, cpu_cfg, 4) == \
            provider._compute_fingerprint(LIBRARY, gpu_cfg, 4)

    def test_symmetric_file_reports_the_device_of_reused_raw_counts(self, tmp_path):
        provider = PEPMatrixProvider(tmp_path)
        provider.get(LIBRARY, _config(), alphabet_size=4, n_jobs=1, device="cpu")
        fp = provider._compute_fingerprint(LIBRARY, _config(), 4)
        meta_path = provider.cache.get_path(fp, ".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["device"] = "gpu:all"  # as if another run had computed the counts on GPU
        meta_path.write_text(json.dumps(meta))
        # get_mmap reuses the raw counts and only builds the symmetric file.
        provider.get_mmap(LIBRARY, _config(), alphabet_size=4, n_jobs=1, device="cpu")
        sym = json.loads(provider.cache.get_path(fp, ".sym.meta.json").read_text())
        assert sym["device"] == "gpu:all"

    def test_cache_status_honours_force_rebuild(self, tmp_path):
        PEPMatrixProvider(tmp_path).get_mmap(LIBRARY, _config(), alphabet_size=4, n_jobs=1)
        status = PEPMatrixProvider(tmp_path, force_rebuild=True).cache_status(LIBRARY, _config(), 4)
        assert not status["sym_hit"] and not status["raw_hit"]


class TestDataDir:
    def test_default_is_the_source_checkout_path(self, monkeypatch):
        from duet.candidate_pool_factory import WeissmanCRISPRiFactory

        monkeypatch.delenv("DUET_DATA_DIR", raising=False)
        assert WeissmanCRISPRiFactory().csv_path == WeissmanCRISPRiFactory.DEFAULT_CSV_PATH

    def test_env_var_overrides_the_default(self, monkeypatch, tmp_path):
        from duet.candidate_pool_factory import WeissmanCRISPRaFactory
        from duet.dual_guide_factory import DualGuideWeissmanCRISPRiFactory

        monkeypatch.setenv("DUET_DATA_DIR", str(tmp_path))
        assert WeissmanCRISPRaFactory().csv_path == tmp_path / "processed/Horlbeck_2016/CRISPRa.csv"
        assert DualGuideWeissmanCRISPRiFactory().csv_path == \
            tmp_path / "processed/Horlbeck_2016/CRISPRi_v2_1.csv"

    def test_explicit_csv_path_wins(self, monkeypatch, tmp_path):
        from duet.candidate_pool_factory import WeissmanCRISPRiFactory

        monkeypatch.setenv("DUET_DATA_DIR", str(tmp_path))
        assert WeissmanCRISPRiFactory(csv_path=tmp_path / "x.csv").csv_path == tmp_path / "x.csv"

    def test_missing_default_file_says_where_it_comes_from(self, monkeypatch, tmp_path):
        from duet.candidate_pool_factory import WeissmanCRISPRiFactory

        monkeypatch.setenv("DUET_DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match=r"Horlbeck et al\. 2016.*DUET_DATA_DIR"):
            WeissmanCRISPRiFactory().create(num_groups=2, seed=0)

    def test_crispick_default_resolves_like_the_weissman_tables(self, monkeypatch, tmp_path):
        from duet.crispick_factory import DEFAULT_CSV_PATH, CRISPickFactory
        from duet.data_paths import SOURCE_DATA_DIR

        relpath = "processed/CRISPick/crispick_ensembl_aggrCFD_top20_candidates.csv"
        assert DEFAULT_CSV_PATH == SOURCE_DATA_DIR / relpath
        monkeypatch.delenv("DUET_DATA_DIR", raising=False)
        assert CRISPickFactory().csv_path == DEFAULT_CSV_PATH
        monkeypatch.setenv("DUET_DATA_DIR", str(tmp_path))
        assert CRISPickFactory().csv_path == tmp_path / relpath

    def test_missing_unshipped_tables_say_how_to_get_them(self, monkeypatch, tmp_path):
        from duet.candidate_pool_factory import WeissmanCRISPRaFactory
        from duet.crispick_factory import CRISPickFactory

        monkeypatch.setenv("DUET_DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError) as err:
            CRISPickFactory().create()
        msg = str(err.value)
        assert "not shipped with DUET" in msg and "Broad Institute GPP" in msg
        assert ("python scripts/data_processing/build_crispick_candidates.py "
                "--raw-dir data/raw/CRISPick") in msg
        with pytest.raises(FileNotFoundError) as err:
            WeissmanCRISPRaFactory().create(num_groups=2, seed=0)
        msg = str(err.value)
        assert "not shipped with DUET" in msg and "Supplementary Table S5" in msg
        assert "file data/" not in msg  # no promise of a file in the repository


def test_renamed_runtime_identifiers():
    from duet.plotting import delta_diverging_cmap, gain_sequential_cmap, reduction_sequential_cmap

    assert delta_diverging_cmap().name == "duet_delta"
    assert gain_sequential_cmap().name == "duet_gain"
    assert reduction_sequential_cmap().name == "duet_reduction"


def test_style_context_leaves_global_rcparams_unchanged():
    import matplotlib as mpl
    from duet.plotting import style_context

    with mpl.rc_context({"font.size": 23.0}):  # a value the stylesheet overrides
        before = dict(mpl.rcParams)
        with style_context():
            assert mpl.rcParams["font.size"] != 23.0
        assert dict(mpl.rcParams) == before
