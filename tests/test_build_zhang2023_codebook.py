"""Tests for scripts/data_processing/build_zhang2023_codebook.py.

A tiny codebook in the Brain Image Library format (name, id, readout bits in
an unsorted order, blank barcodes) checks the conversion byte for byte. The
--download path is tested with a file:// URL, so no test uses the network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# The script is not an installed module (scripts/data_processing has no
# __init__.py); load it from its path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_processing" / "build_zhang2023_codebook.py"
_spec = importlib.util.spec_from_file_location("build_zhang2023_codebook", _SCRIPT)
bz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bz)

# Bit columns are not sorted by readout name, as in the real file; the
# sequence must follow the file's column order.
RAW = (
    "name,id,RS0015,RS0083,RS1334,RS0332\n"
    "Grp,nan,1,0,0,1\n"
    "blank-01,nan,0,1,1,0\n"
    "Endou,nan,0,1,1,0\n"
    "blank-02,nan,1,1,0,0\n"
    "Tfap2d,nan,0,0,1,1\n"
)
EXPECTED = b"Gene,Sequence\nGrp,1001\nEndou,0110\nTfap2d,0011\n"


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "src" / "codebook_32bit_v2.csv"
    path.parent.mkdir()
    path.write_text(RAW)
    return path


def test_convert_and_write_give_the_expected_bytes(raw_file, tmp_path):
    out = tmp_path / "out" / "codebook.csv"
    bz.write_codebook(bz.convert(pd.read_csv(raw_file)), out)
    assert out.read_bytes() == EXPECTED


def test_convert_rejects_non_binary_bits():
    raw = pd.DataFrame({"name": ["Grp"], "id": [None], "RS0001": [2], "RS0002": [0]})
    with pytest.raises(ValueError, match="only 0 and 1"):
        bz.convert(raw)


def test_main_converts_and_warns_that_the_input_is_not_the_papers(raw_file, tmp_path, capsys):
    out = tmp_path / "out.csv"
    assert bz.main(["--raw", str(raw_file), "--out", str(out)]) == 0
    assert out.read_bytes() == EXPECTED
    captured = capsys.readouterr()
    assert "3 genes" in captured.out
    assert "is not the file the paper used" in captured.err
    assert "differs from the paper's codebook" in captured.err


def test_main_without_the_raw_file_points_to_download(tmp_path, capsys):
    out = tmp_path / "out.csv"
    assert bz.main(["--raw", str(tmp_path / "absent.csv"), "--out", str(out)]) == 1
    assert "--download" in capsys.readouterr().err
    assert not out.exists()


def test_main_download_fetches_an_absent_raw_file(raw_file, tmp_path, monkeypatch):
    monkeypatch.setattr(bz, "RAW_URL", raw_file.as_uri())
    raw = tmp_path / "data" / "raw" / "codebook_32bit_v2.csv"
    out = tmp_path / "out.csv"
    assert bz.main(["--raw", str(raw), "--out", str(out), "--download"]) == 0
    assert raw.read_text() == RAW
    assert not raw.with_name(raw.name + ".part").exists()
    assert out.read_bytes() == EXPECTED


def test_failed_download_leaves_no_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bz, "RAW_URL", (tmp_path / "missing.csv").as_uri())
    raw = tmp_path / "raw" / "codebook_32bit_v2.csv"
    assert bz.main(["--raw", str(raw), "--out", str(tmp_path / "out.csv"), "--download"]) == 1
    assert "download failed" in capsys.readouterr().err
    assert list(raw.parent.iterdir()) == []


class _ResetAfterFirstChunk:
    """A response whose connection drops after the first chunk."""

    def __init__(self):
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        if self.sent:
            raise ConnectionResetError("connection reset")
        self.sent = True
        return b"name,id,RS0001\n"


def test_download_interrupted_midway_leaves_no_partial_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bz.urllib.request, "urlopen", lambda url, timeout: _ResetAfterFirstChunk())
    raw = tmp_path / "raw" / "codebook_32bit_v2.csv"
    assert bz.main(["--raw", str(raw), "--out", str(tmp_path / "out.csv"), "--download"]) == 1
    assert "connection reset" in capsys.readouterr().err
    assert list(raw.parent.iterdir()) == []
