import subprocess
import sys
import pytest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare
import fixtures
import run_regression as rr

from duet.ops_benchmark.config import OpsBenchmarkConfig
from duet.merfish_benchmark.config import MerfishBenchmarkConfig
from duet.evaluator_config import create_noise_channel, create_decoding_metric, create_decoding_rule


def test_ops_config_parses_on_current_schema():
    cfg = OpsBenchmarkConfig.from_yaml(fixtures.OPS_UNIFORM.config)
    assert cfg.trials == 3
    assert cfg.evaluator.noise_channel.type == "symmetric"
    assert cfg.evaluator.decoding_metric.type == "hamming"
    assert cfg.duet_optimizer is not None
    assert cfg.sivanandan is not None and cfg.feldman is not None


def test_merfish_config_parses_and_paths_resolve():
    cfg = MerfishBenchmarkConfig.from_yaml(fixtures.MERFISH_ASYMMETRIC.config)
    assert cfg.duet_force_rebuild is True
    assert cfg.evaluator.noise_channel.type == "asymmetric"
    # resolve_config_path made these absolute; existence confirms the ../ depth
    assert cfg.expression.path.exists()
    assert cfg.initialization.path.exists()
    assert cfg.baselines[0].path.exists()


def test_ops_evaluator_components_build():
    cfg = OpsBenchmarkConfig.from_yaml(fixtures.OPS_UNIFORM.config)
    create_noise_channel(cfg.evaluator.noise_channel, seq_length=10, alphabet_size=4)
    create_decoding_metric(cfg.evaluator.decoding_metric, seq_length=10, alphabet_size=4)
    create_decoding_rule(cfg.evaluator.decoding_rule)


def test_merfish_evaluator_components_build():
    cfg = MerfishBenchmarkConfig.from_yaml(fixtures.MERFISH_ASYMMETRIC.config)
    # binary MERFISH alphabet (2 symbols); inline channel_matrix is 2x2
    create_noise_channel(cfg.evaluator.noise_channel, seq_length=16, alphabet_size=2)
    create_decoding_metric(cfg.evaluator.decoding_metric, seq_length=16, alphabet_size=2)
    create_decoding_rule(cfg.evaluator.decoding_rule)


def test_fresh_outdir_resolves_against_config_dir(tmp_path):
    cfg = tmp_path / "fx" / "fx.yaml"
    cfg.parent.mkdir()
    cfg.write_text("outdir: ../out\n")
    fx = fixtures.Fixture(name="fx", config=cfg, runner=Path("r.py"),
                          visualizer=Path("v.py"))
    assert rr.fresh_outdir(fx) == (tmp_path / "out").resolve()


def test_ensure_runtime_dirs_creates_cache_and_scratch(tmp_path):
    # The runner's tempfile.mkdtemp(dir=scratch_dir) needs scratch_dir to exist.
    cfgdir = tmp_path / "fx"
    cfgdir.mkdir()
    cfg = cfgdir / "fx.yaml"
    cfg.write_text("cache_dir: ../out/cache\n"
                   "evaluator:\n  scratch_dir: ../out/scratch\n")
    fx = fixtures.Fixture(name="fx", config=cfg, runner=Path("r.py"),
                          visualizer=Path("v.py"))
    rr._ensure_runtime_dirs(fx)
    assert (tmp_path / "out" / "cache").is_dir()
    assert (tmp_path / "out" / "scratch").is_dir()


def test_run_logged_tees_output_and_propagates_failure(tmp_path):
    log = tmp_path / "out.log"
    # success: captures stdout to the log, no raise
    rr._run_logged([sys.executable, "-c", "print('hello-harness')"], tmp_path, log)
    assert "hello-harness" in log.read_text()
    # failure: non-zero exit must raise so main() can return exit code 2
    with pytest.raises(subprocess.CalledProcessError):
        rr._run_logged([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, log)


def test_build_run_command_uses_bare_python_and_config():
    fx = fixtures.OPS_UNIFORM
    cmd = rr.build_run_command(fx)
    assert cmd[0] == "python"
    assert str(fx.runner) in cmd
    assert "--config" in cmd and str(fx.config) in cmd
    assert "-vv" in cmd


def test_assert_figures_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError):
        rr.assert_figures(tmp_path)


def test_assert_figures_ok_when_svg_present(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a.svg").write_text("<svg/>")
    rr.assert_figures(tmp_path)  # no raise


def _toy_fixture(tmp_path):
    """A fixture whose outdir already holds canonical files, so we can exercise
    record/check without running any GPU experiment."""
    cfgdir = tmp_path / "fx"
    cfgdir.mkdir()
    cfg = cfgdir / "fx.yaml"
    cfg.write_text("outdir: ./out\n")
    out = cfgdir / "out"
    out.mkdir()
    pd.DataFrame({"Method": ["A", "A"], "Index": [0, 1],
                  "Sequence": ["0010", "1100"],
                  "Decode accuracy": [0.81, 0.79]}).to_csv(
        out / "results.csv", index=False)
    fx = fixtures.Fixture(
        name="fx", config=cfg, runner=Path("r.py"), visualizer=Path("v.py"),
        canonical=(fixtures.CanonicalFile(
            "results.csv", key_cols=("Method", "Index"),
            str_cols=("Method", "Sequence")),),
    )
    return fx, out


def test_record_then_check_passes(tmp_path):
    fx, _ = _toy_fixture(tmp_path)
    rr.record_fixture(fx, meta={"git_sha": "deadbeef"})
    ref = fx.config.parent / "reference"
    assert (ref / "results.csv.gz").exists()
    assert (ref / "manifest.json").exists()
    result = rr.check_fixture(fx)
    assert result.passed


def test_check_fails_after_value_mutation(tmp_path):
    fx, out = _toy_fixture(tmp_path)
    rr.record_fixture(fx, meta={"git_sha": "deadbeef"})
    pd.DataFrame({"Method": ["A", "A"], "Index": [0, 1],
                  "Sequence": ["0010", "1100"],
                  "Decode accuracy": [0.99, 0.79]}).to_csv(
        out / "results.csv", index=False)
    assert not rr.check_fixture(fx).passed


def test_check_fails_when_reference_file_missing_in_fresh(tmp_path):
    fx, out = _toy_fixture(tmp_path)
    rr.record_fixture(fx, meta={"git_sha": "deadbeef"})
    (out / "results.csv").unlink()
    result = rr.check_fixture(fx)
    assert not result.passed
    assert any("missing in fresh" in (f.error or "") for f in result.files)


def test_parse_args_defaults():
    ns = rr.parse_args([])
    assert ns.mode == "check"
    assert ns.only is None
    assert ns.rtol == compare.RTOL and ns.atol == compare.ATOL


def test_parse_args_record_and_only():
    ns = rr.parse_args(["--record", "--only", "ops_uniform"])
    assert ns.mode == "record"
    assert ns.only == "ops_uniform"


def test_select_fixtures_filters_by_only():
    sel = rr.select_fixtures("merfish_asymmetric")
    assert [f.name for f in sel] == ["merfish_asymmetric"]
    assert len(rr.select_fixtures(None)) == 2


def test_git_sha_returns_string():
    sha = rr.git_sha()
    assert isinstance(sha, str) and len(sha) >= 7


def test_record_keep_going_skips_failed_fixture(monkeypatch):
    # merfish fails to run; --keep-going continues. The failed fixture must NOT
    # be re-recorded (record_fixture wipes the reference), and exit must be 2.
    recorded = []

    def fake_run(fx):
        if fx.name == "merfish_asymmetric":
            raise subprocess.CalledProcessError(1, ["run"])

    monkeypatch.setattr(rr, "_run_experiment", fake_run)
    monkeypatch.setattr(rr, "record_fixture",
                        lambda fx, meta: recorded.append(fx.name) or ["results.csv"])
    rc = rr.main(["--record", "--keep-going"])
    assert recorded == ["ops_uniform"]
    assert rc == 2


def test_check_keep_going_surfaces_run_failure(monkeypatch, tmp_path):
    # merfish fails to run; ops passes its comparison. A swallowed run failure
    # must NOT be hidden: the failed fixture is not compared and exit is 2.
    checked = []

    def fake_run(fx):
        if fx.name == "merfish_asymmetric":
            raise subprocess.CalledProcessError(1, ["run"])

    def fake_check(fx, rtol, atol):
        checked.append(fx.name)
        return compare.FixtureComparison(
            name=fx.name, passed=True,
            files=[compare.FileComparison("results.csv", True)])

    monkeypatch.setattr(rr, "_run_experiment", fake_run)
    monkeypatch.setattr(rr, "check_fixture", fake_check)
    monkeypatch.setattr(rr, "fresh_outdir", lambda fx: tmp_path / fx.name)
    rc = rr.main(["--check", "--keep-going"])
    assert checked == ["ops_uniform"]
    assert rc == 2


def test_run_failure_without_keep_going_does_not_record(monkeypatch):
    def fake_run(fx):
        raise subprocess.CalledProcessError(1, ["run"])

    def must_not_record(*a, **k):
        raise AssertionError("record_fixture must not run after a run failure")

    monkeypatch.setattr(rr, "_run_experiment", fake_run)
    monkeypatch.setattr(rr, "record_fixture", must_not_record)
    assert rr.main(["--record"]) == 2
