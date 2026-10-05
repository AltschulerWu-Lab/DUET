"""The user documentation works as written.

Static checks (fast, in the default selection) read every page listed in
USER_DOCS:

- no em dashes, no home-directory paths and no manuscript figure numbers
  (they change in revision);
- in the private development repository, where the release tooling
  ``tools/release/export_public.py`` exists, also none of the private-material
  patterns that the export script scans for (its ``SCANS``). The public
  repository has no ``tools/`` folder, so this part is left out there;
- every relative link and heading anchor resolves, and the link target is
  part of the public release. Where ``tools/release/allowlist.tsv`` exists, the
  target must be a path the allowlist exports (EXPORT or PENDING). In the
  public repository the private files are absent, so a link to one fails as
  broken; links into local, untracked folders (``results/``, ``data/raw/``)
  are rejected everywhere.

Snippet checks (slow) run the code in each page as a reader would, from a
scratch directory that mirrors the repository root:

- the ``python`` blocks of a page run top to bottom in one interpreter, so a
  later block can use the variables of an earlier one;
- each ``bash`` block runs on its own with ``bash -e``, with this
  interpreter's ``python`` first on PATH.

A block is skipped when the last non-blank line before its fence is an HTML
comment ``<!-- docs-test: skip (reason) -->``: installation commands, commands
that need GPUs or hours, and signatures that are not meant to run. A block
marked ``<!-- docs-test: requires yaml -->`` runs only when that module is
importable (for example commands that need the ``benchmark`` extra). The checks
hide the GPUs (``CUDA_VISIBLE_DEVICES=""``), so ``device="auto"`` runs on CPU.

    python -m pytest tests/test_docs.py            # static checks
    python -m pytest tests/test_docs.py -m slow    # run every snippet (a few minutes on CPU)
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The public user documentation, relative to the repository root.
USER_DOCS = [
    "README.md",
    "docs/installation.md",
    "docs/design.md",
    "docs/design_ops.md",
    "docs/design_merfish.md",
    "docs/noise_channels.md",
    "docs/gpu_and_scaling.md",
    "docs/api.md",
    "docs/reproducing_the_paper.md",
]

# Release tooling of the private development repository (absent from the
# public repository). When present, it decides which paths the public release
# contains and which text counts as private material.
RELEASE_TOOLS = ROOT / "tools" / "release"


def _load_export_script():
    path = RELEASE_TOOLS / "export_public.py"
    if not path.is_file():
        return None
    name = "_duet_release_export_public"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _load_allowlist():
    if EXPORT_SCRIPT is None or not ALLOWLIST_PATH.is_file():
        return None
    try:
        return EXPORT_SCRIPT.parse_allowlist(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    except SystemExit as exc:  # how the export script reports a malformed row
        raise RuntimeError(str(exc)) from None


EXPORT_SCRIPT = _load_export_script()
ALLOWLIST_PATH = RELEASE_TOOLS / "allowlist.tsv"
ALLOWLIST = _load_allowlist()
EXPORTED = ("EXPORT", "PENDING")

# Folders of local, untracked files (outputs and downloads): never a link target.
LOCAL_PREFIXES = ("results/", "data/raw/")


FORBIDDEN_TEXT = {
    "em dash": "\u2014",
    "home directory path": re.compile(r"/(?:home|Users)/[A-Za-z]"),
    "manuscript figure number": re.compile(
        r"\b(Fig|Figure|Figs|Figures)\.?\s*\d|\bSupp\w*\.?\s*(Fig\w*\.?\s*)?S?\d|\bExtended Data Fig"),
}

FENCE = re.compile(r"^(?P<indent>[ ]*)(?P<fence>`{3,}|~{3,})(?P<info>[^`\n]*)$")
SKIP = re.compile(r"<!--\s*docs-test:\s*skip\b.*?-->")
REQUIRES = re.compile(r"<!--\s*docs-test:\s*requires\s+([\w., ]+?)\s*-->")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _doc_ids():
    return [d for d in USER_DOCS]


def _read(doc: str) -> str:
    path = ROOT / doc
    if not path.exists():
        pytest.fail(f"{doc} is listed in USER_DOCS but does not exist")
    return path.read_text(encoding="utf-8")


def code_blocks(text: str):
    """Fenced blocks as (first_code_line, language, code, skipped, required_modules)."""
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, lang = m.group("fence"), m.group("info").strip().split(" ")[0].lower()
        prev = next((l for l in reversed(lines[:i]) if l.strip()), "")
        skipped = bool(SKIP.search(prev))
        req = REQUIRES.search(prev)
        requires = tuple(m.strip() for m in req.group(1).split(",")) if req else ()
        body, j = [], i + 1
        while j < len(lines) and not lines[j].strip().startswith(fence):
            body.append(lines[j])
            j += 1
        blocks.append((i + 2, lang, textwrap.dedent("\n".join(body)) + "\n", skipped, requires))
        i = j + 1
    return blocks


def _prose(text: str) -> str:
    """The page without its fenced blocks, for link checks."""
    out, inside = [], None
    for line in text.splitlines():
        m = FENCE.match(line)
        if m and inside is None:
            inside = m.group("fence")
            continue
        if inside is not None:
            if line.strip().startswith(inside):
                inside = None
            continue
        out.append(line)
    return "\n".join(out)


def github_anchors(text: str) -> set:
    """Heading anchors as GitHub generates them."""
    anchors, seen = set(), {}
    for line in _prose(text).splitlines():
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(2))  # links -> text
        title = title.replace("`", "")
        slug = "".join(
            ch for ch in title.lower()
            if ch in " -_" or unicodedata.category(ch)[0] in ("L", "N")
        ).replace(" ", "-")
        n = seen.get(slug, 0)
        anchors.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    return anchors


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def _private_material(data: bytes):
    """Hits of the export script's private-material scans, as (line, scan name)."""
    if EXPORT_SCRIPT is None:
        return []
    hits = []
    for name, regex, _on_binary in EXPORT_SCRIPT.SCANS:
        for m in regex.finditer(data):
            hits.append((data.count(b"\n", 0, m.start()) + 1, name))
    return sorted(set(hits))


@pytest.mark.parametrize("doc", _doc_ids())
def test_no_forbidden_text(doc):
    text = _read(doc)
    problems = []
    for n, line in enumerate(text.splitlines(), 1):
        for what, pattern in FORBIDDEN_TEXT.items():
            hit = (pattern in line) if isinstance(pattern, str) else pattern.search(line)
            if hit:
                problems.append(f"{doc}:{n}: {what}: {line.strip()[:100]}")
    raw_lines = text.split("\n")  # the numbering of _private_material
    for n, what in _private_material(text.encode("utf-8")):
        problems.append(f"{doc}:{n}: private material ({what}): {raw_lines[n - 1].strip()[:100]}")
    assert not problems, "\n".join(problems)


def _exported(rel: str) -> bool:
    rule = EXPORT_SCRIPT.classify([rel], ALLOWLIST)[rel]
    return rule is not None and rule.decision in EXPORTED


def _stays_private(rel: str, dest: Path) -> bool:
    """True when the public release does not contain the link target."""
    if any(rel.startswith(p) or rel + "/" == p for p in LOCAL_PREFIXES):
        return True
    if ALLOWLIST is None:
        return False  # public copy: a private target is absent, so the link is broken
    if dest.is_dir():  # a folder is public when it holds an exported file
        return not any(
            f.is_file() and _exported(f.relative_to(ROOT).as_posix()) for f in dest.rglob("*")
        )
    return not _exported(rel)


@pytest.mark.parametrize("doc", _doc_ids())
def test_relative_links_resolve(doc):
    text = _read(doc)
    base = (ROOT / doc).parent
    problems = []
    for target in LINK.findall(_prose(text)):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
            continue
        path_part, _, anchor = target.partition("#")
        dest = (base / path_part).resolve() if path_part else (ROOT / doc).resolve()
        try:
            rel = dest.relative_to(ROOT).as_posix()
        except ValueError:
            problems.append(f"{doc}: link leaves the repository: {target}")
            continue
        if not dest.exists():
            problems.append(f"{doc}: broken link: {target}")
            continue
        if _stays_private(rel, dest):
            problems.append(f"{doc}: link to a file that stays private: {target}")
        if anchor and dest.suffix == ".md" and anchor not in github_anchors(dest.read_text(encoding="utf-8")):
            problems.append(f"{doc}: no heading for anchor #{anchor} in {rel}")
    assert not problems, "\n".join(problems)


def test_every_user_doc_is_in_the_index():
    """The Documentation section of README.md is the index of the user docs."""
    index = _read("README.md")
    missing = [d for d in USER_DOCS if d.startswith("docs/") and f"({d})" not in index]
    assert not missing, f"not linked from README.md: {missing}"


# ---------------------------------------------------------------------------
# Snippet checks
# ---------------------------------------------------------------------------

RUNNER = r'''
import json, sys
doc, blocks = sys.argv[1], json.load(open(sys.argv[2]))
ns = {"__name__": "__main__"}
for line, code in blocks:
    print(f"=== {doc}:{line}", flush=True)
    exec(compile(code, f"{doc}:{line}", "exec"), ns)
'''


def _scratch_root(tmp_path: Path) -> Path:
    """A directory that looks like the repository root but writes go to tmp_path."""
    work = tmp_path / "work"
    work.mkdir()
    for entry in ROOT.iterdir():
        if entry.name in (".git", "results", "build") or entry.name.startswith(".pytest"):
            continue
        (work / entry.name).symlink_to(entry)
    return work


def _env(work: Path) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # snippets run on CPU, even on a GPU machine
    env["MPLBACKEND"] = "Agg"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    return env


def _available(modules) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _blocks(doc, languages):
    return [(line, code) for line, lang, code, skipped, requires in code_blocks(_read(doc))
            if lang in languages and not skipped and _available(requires)]


@pytest.mark.slow
@pytest.mark.parametrize("doc", [d for d in USER_DOCS if _blocks(d, ("python", "py"))]
                         if all((ROOT / d).exists() for d in USER_DOCS) else USER_DOCS)
def test_python_snippets_run(doc, tmp_path):
    blocks = _blocks(doc, ("python", "py"))
    if not blocks:
        pytest.skip("no python blocks")
    work = _scratch_root(tmp_path)
    (tmp_path / "blocks.json").write_text(json.dumps(blocks))
    (tmp_path / "runner.py").write_text(RUNNER)
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "runner.py"), doc, str(tmp_path / "blocks.json")],
        cwd=work, env=_env(work), capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, f"{doc}\n--- stdout\n{proc.stdout[-4000:]}\n--- stderr\n{proc.stderr[-4000:]}"


@pytest.mark.slow
@pytest.mark.parametrize("doc", [d for d in USER_DOCS if _blocks(d, ("bash", "sh", "shell"))]
                         if all((ROOT / d).exists() for d in USER_DOCS) else USER_DOCS)
def test_bash_snippets_run(doc, tmp_path):
    blocks = _blocks(doc, ("bash", "sh", "shell"))
    if not blocks:
        pytest.skip("no bash blocks")
    work = _scratch_root(tmp_path)
    for line, code in blocks:
        proc = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", code],
            cwd=work, env=_env(work), capture_output=True, text=True, timeout=1800,
        )
        assert proc.returncode == 0, (
            f"{doc}:{line}\n{code}\n--- stdout\n{proc.stdout[-3000:]}\n--- stderr\n{proc.stderr[-3000:]}"
        )
