"""Driver for the benchmark regression harness.

Runs each fixture's experiment + visualization, then compares (or records) the
canonical numeric outputs against a gzipped, git-committed reference.

The harness calls bare ``python``: activate the desired conda env first.
See README.md in this folder for how to run it and read the report.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import compare
import fixtures as fixtures_mod


def fresh_outdir(fixture) -> Path:
    """Absolute output dir for a fixture, resolved the way its runner does:
    relative to the config's directory."""
    raw = yaml.safe_load(fixture.config.read_text())["outdir"]
    return (fixture.config.parent / raw).resolve()


def _ensure_runtime_dirs(fixture) -> None:
    """Create the cache_dir and evaluator.scratch_dir the config references.

    The runner calls ``tempfile.mkdtemp(dir=evaluator.scratch_dir)``, which
    requires that directory to already exist (mkdtemp does not create parents).
    Paths are resolved relative to the config directory, matching ``fresh_outdir``
    and the subprocess cwd, so the dirs we create are the ones the runner uses.
    """
    cfg = yaml.safe_load(fixture.config.read_text())
    raw_dirs = [cfg.get("cache_dir"), cfg.get("evaluator", {}).get("scratch_dir")]
    for raw in raw_dirs:
        if raw:
            (fixture.config.parent / raw).resolve().mkdir(parents=True, exist_ok=True)


def build_run_command(fixture) -> list:
    return ["python", str(fixture.runner), "--config", str(fixture.config), "-vv"]


def build_visualize_command(fixture) -> list:
    return ["python", str(fixture.visualizer), "--config", str(fixture.config)]


def assert_figures(outdir) -> None:
    """Cheap guard that visualization produced something. Content is not diffed."""
    if not list(Path(outdir).rglob("*.svg")):
        raise RuntimeError(f"no figures (*.svg) produced under {outdir}")


def reference_dir(fixture) -> Path:
    return fixture.config.parent / "reference"


def _canonical_matches(fixture, outdir):
    """Yield (CanonicalFile, relpath) for every file under outdir matching a
    canonical glob."""
    outdir = Path(outdir)
    for cf in fixture.canonical:
        for fp in sorted(outdir.glob(cf.glob)):
            yield cf, fp.relative_to(outdir)


def record_fixture(fixture, meta: dict):
    """Freeze the fixture's current canonical outputs as the gzipped reference."""
    outdir = fresh_outdir(fixture)
    refdir = reference_dir(fixture)
    if refdir.exists():
        shutil.rmtree(refdir)
    refdir.mkdir(parents=True)

    written = []
    for _cf, rel in _canonical_matches(fixture, outdir):
        dest = refdir / (str(rel) + ".gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(outdir / rel, "rb") as src, gzip.open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        written.append(str(rel))

    (refdir / "manifest.json").write_text(
        json.dumps({**meta, "files": written}, indent=2, default=str))
    return written


def check_fixture(fixture, rtol: float = compare.RTOL,
                  atol: float = compare.ATOL) -> "compare.FixtureComparison":
    """Compare the fixture's fresh canonical outputs against its reference."""
    outdir = fresh_outdir(fixture)
    refdir = reference_dir(fixture)
    if not refdir.exists():
        return compare.FixtureComparison(
            name=fixture.name, passed=False,
            error=f"no reference at {refdir} (run --record first)")

    results = []
    fresh_rel = set()
    for cf, rel in _canonical_matches(fixture, outdir):
        fresh_rel.add(str(rel))
        ref = refdir / (str(rel) + ".gz")
        if not ref.exists():
            results.append(compare.FileComparison(
                name=str(rel), passed=False, error="missing in reference"))
            continue
        if cf.kind == "yaml":
            fc = compare.compare_yaml(ref, outdir / rel, rtol=rtol, atol=atol)
        else:
            fc = compare.compare_csv(
                ref, outdir / rel, list(cf.key_cols),
                str_cols=list(cf.str_cols), exclude_cols=list(cf.exclude_cols),
                rtol=rtol, atol=atol)
        fc.name = str(rel)
        results.append(fc)

    for refp in refdir.rglob("*.gz"):
        rel = str(refp.relative_to(refdir))[:-3]  # strip ".gz"
        if rel not in fresh_rel:
            results.append(compare.FileComparison(
                name=rel, passed=False,
                error="present in reference, missing in fresh run"))

    passed = bool(results) and all(f.passed for f in results)
    return compare.FixtureComparison(name=fixture.name, passed=passed, files=results)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=fixtures_mod.REGRESSION_DIR,
            text=True).strip()
    except Exception:
        return "unknown"


def select_fixtures(only):
    if only is None:
        return list(fixtures_mod.FIXTURES)
    return [f for f in fixtures_mod.FIXTURES if f.name == only]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark regression harness.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", dest="mode", action="store_const", const="check",
                      help="run + visualize + compare to reference (default)")
    mode.add_argument("--record", dest="mode", action="store_const", const="record",
                      help="run + visualize + freeze outputs as the new reference")
    mode.add_argument("--compare-only", dest="mode", action="store_const",
                      const="compare-only",
                      help="skip running; compare existing results to reference")
    p.set_defaults(mode="check")
    p.add_argument("--only", choices=[f.name for f in fixtures_mod.FIXTURES],
                   default=None, help="run a single fixture")
    p.add_argument("--rtol", type=float, default=compare.RTOL)
    p.add_argument("--atol", type=float, default=compare.ATOL)
    p.add_argument("--keep-going", action="store_true",
                   help="run the next fixture even if one fails to run")
    return p.parse_args(argv)


def _run_logged(cmd, cwd, log_path) -> None:
    """Run ``cmd``, streaming its combined stdout/stderr live to this terminal
    while also capturing it to ``log_path`` (tee). Raises CalledProcessError on
    a non-zero exit so the caller can fail the fixture.

    Output is teed through a pipe rather than a pty, so tqdm progress bars print
    as periodic lines instead of one in-place bar; the full ``-vv`` log streams
    live either way.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    with open(log_path, "w", errors="replace") as fh:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            fh.write(text)
            fh.flush()
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def _run_experiment(fixture):
    outdir = fresh_outdir(fixture)
    outdir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_dirs(fixture)
    log = outdir / "run.log"
    print(f"[{fixture.name}] running -> {outdir} (log: {log})", flush=True)
    _run_logged(build_run_command(fixture), fixture.config.parent, log)
    print(f"[{fixture.name}] visualizing", flush=True)
    _run_logged(build_visualize_command(fixture), fixture.config.parent,
                outdir / "visualize.log")
    assert_figures(outdir)


def main(argv=None) -> int:
    args = parse_args(argv)
    selected = select_fixtures(args.only)

    # In compare-only mode nothing runs, so every selected fixture is "ran".
    ran_ok = list(selected)
    run_failed = []
    if args.mode in ("check", "record"):
        ran_ok = []
        for fx in selected:
            try:
                _run_experiment(fx)
                ran_ok.append(fx)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                print(f"[{fx.name}] RUN FAILED: {e}")
                run_failed.append(fx)
                if not args.keep_going:
                    return 2

    if args.mode == "record":
        # Only freeze fixtures that actually ran: record_fixture wipes the
        # reference dir before re-freezing, so recording a failed fixture would
        # destroy its committed golden reference.
        meta = {"git_sha": git_sha(),
                "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "rtol": args.rtol, "atol": args.atol}
        for fx in ran_ok:
            files = record_fixture(fx, meta)
            print(f"[{fx.name}] recorded {len(files)} reference files")
        for fx in run_failed:
            print(f"[{fx.name}] SKIPPED recording: run failed (reference left intact)")
        if run_failed:
            return 2
        print("Reference recorded. Review the gzipped files + manifest, then commit.")
        return 0

    # check / compare-only: compare only fixtures whose outputs are fresh.
    comparisons = [check_fixture(fx, rtol=args.rtol, atol=args.atol)
                   for fx in ran_ok]
    print(compare.render_report(comparisons))
    for fx, cmp in zip(ran_ok, comparisons):
        report = fresh_outdir(fx) / "comparison_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(compare.to_json([cmp]), indent=2, default=str))
    if run_failed:
        names = ", ".join(fx.name for fx in run_failed)
        print(f"{len(run_failed)} fixture(s) failed to run: {names}")
        return 2
    return 0 if all(c.passed for c in comparisons) else 1


if __name__ == "__main__":
    sys.exit(main())
