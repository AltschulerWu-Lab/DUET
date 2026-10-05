"""The quickstart demos run and print the table recorded in their docstrings.

Each demo is run as a reviewer would (python examples/quickstart_*.py OUTDIR).
The OPS demo uses the Hamming metric, whose float32 costs are integers, so its
table must match exactly on any machine. The MERFISH demo uses an asymmetric
NLL metric, whose float32 BLAS sums can differ in the last bits between CPU
architectures (see the float32-tie note in tests/test_blas_decoding_metric.py),
so its numbers are compared within a tolerance.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _table_lines(text: str):
    """Rows of the printed result table: the header line and the rows after it."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.split()[:1] == ["codebook"])
    header = lines[start].split()
    rows = [header]
    for line in lines[start + 1:]:
        fields = line.split()
        if len(fields) != len(header) or fields[-1] not in ("True", "False"):
            break
        rows.append(fields)
    return rows


def _expected(script: Path):
    doc = ast.get_docstring(ast.parse(script.read_text()))
    return _table_lines(doc.split("Expected output", 1)[1])


def _run(script: Path, outdir: Path) -> str:
    proc = subprocess.run([sys.executable, str(script), str(outdir)],
                          capture_output=True, text=True, check=True)
    return proc.stdout


@pytest.mark.slow
def test_quickstart_ops_prints_the_recorded_table(tmp_path):
    script = EXAMPLES / "quickstart_ops.py"
    out = _run(script, tmp_path)
    assert _table_lines(out) == _expected(script)
    for name in ("table.csv", "codebooks.csv", "settings.json", "pareto_front.svg"):
        assert (tmp_path / name).exists()
    assert "toy scale" in out.lower() and "gpu" in out.lower()


@pytest.mark.slow
def test_quickstart_merfish_prints_the_recorded_table(tmp_path):
    script = EXAMPLES / "quickstart_merfish.py"
    out = _run(script, tmp_path)
    got, want = _table_lines(out), _expected(script)
    assert got[0] == want[0] and [r[0] for r in got] == [r[0] for r in want]
    header = got[0]
    tolerance = {"accuracy_mean": 0.02, "accuracy_p10": 0.05, "surrogate_accuracy": 0.05,
                 "crowding": 0.05}
    for g, w in zip(got[1:], want[1:]):
        for col, tol in tolerance.items():
            i = header.index(col)
            assert float(g[i]) == pytest.approx(float(w[i]), abs=tol), (g[0], col)
    for name in ("table.csv", "codebooks.csv", "settings.json", "pareto_front.svg"):
        assert (tmp_path / name).exists()
    assert "synthetic" in out.lower() or "toy scale" in out.lower()
