"""Registry of regression fixtures: where each config lives, which scripts run
it, and which output files are gated (with their comparison keys)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REGRESSION_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = REGRESSION_DIR.parent  # scripts/benchmark


@dataclass(frozen=True)
class CanonicalFile:
    glob: str                      # glob under the fixture outdir, e.g. "trial_*/guides.csv"
    key_cols: tuple = ()           # tuple[str, ...] aligning rows
    str_cols: tuple = ()           # columns forced to string dtype on read
    exclude_cols: tuple = ()       # value columns to ignore
    kind: str = "csv"              # "csv" | "yaml"


@dataclass(frozen=True)
class Fixture:
    """A single regression fixture: config, scripts, and gated output files.

    cwd contract: the driver MUST invoke ``runner``/``visualizer`` with cwd set
    to ``config.parent``. OPS resolves outdir/cache/scratch relative to cwd;
    MERFISH resolves its paths relative to the config directory via
    resolve_config_path. Running with cwd=config.parent satisfies both.
    """

    name: str
    config: Path                   # absolute path to the fixture YAML
    runner: Path                   # absolute path to run_*.py
    visualizer: Path               # absolute path to visualize_*.py
    canonical: tuple = field(default_factory=tuple)  # tuple[CanonicalFile, ...]


OPS_UNIFORM = Fixture(
    name="ops_uniform",
    config=REGRESSION_DIR / "ops_uniform" / "ops_uniform.yaml",
    runner=BENCHMARK_DIR / "run_benchmark.py",
    visualizer=BENCHMARK_DIR / "visualize_benchmark.py",
    canonical=(
        CanonicalFile("results.csv", key_cols=("Method", "Trial", "Index"),
                      str_cols=("Method", "Group", "Sequence")),
        # guides.csv (the full candidate pool) is intentionally NOT gated: after
        # per-row candidates were introduced, (Group, Sequence) can collide within a
        # group, so it has no stable unique key for the comparator. results.csv
        # already flags pool-construction regressions via the selected
        # candidates' Index/Sequence/Activity score.
    ),
)

MERFISH_ASYMMETRIC = Fixture(
    name="merfish_asymmetric",
    config=REGRESSION_DIR / "merfish_asymmetric" / "merfish_asymmetric.yaml",
    runner=BENCHMARK_DIR / "run_merfish.py",
    visualizer=BENCHMARK_DIR / "visualize_merfish.py",
    canonical=(
        CanonicalFile("results.csv", key_cols=("Method", "Index"),
                      str_cols=("Method", "Gene", "Sequence")),
        CanonicalFile("metrics.csv", key_cols=("Method",), str_cols=("Method",)),
        CanonicalFile("summary.yaml", kind="yaml"),
        CanonicalFile("candidates.csv", key_cols=("Sequence",),
                      str_cols=("Group", "Sequence")),
        CanonicalFile("selected_codewords_lambda*.csv", key_cols=("Index",),
                      str_cols=("Sequence", "Gene")),
    ),
)

FIXTURES = (OPS_UNIFORM, MERFISH_ASYMMETRIC)
