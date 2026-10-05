# src/duet/bostrom_codebook.py
"""Boström et al. (Sci. Adv. 2025, adr4026) codebook utilities.

Canonical home for loading Boström Set/Binary codebooks into binary strings
and assigning their codewords to a gene panel. Pure functions only — imports
nothing from duet.optical_crowding so the dependency stays one-directional
(a mutual import would be a real circular-import bug).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BOSTROM_NAME = re.compile(
    r"(?P<bits>\d+)Bit_HW(?P<hw>\d+)_HD(?P<hd>\d+)_finalsize(?P<n>\d+)(?P<fmt>Binary|Set)(?=[_.]|$)"
)


def parse_bostrom_filename(path: Union[str, Path]) -> dict:
    """Parse '<v>Bit_HW<k>_HD<d>_finalsize<N><Fmt>[_reordered<M>].csv'.

    Returns {'bits','hw','hd','finalsize','fmt'} where fmt is 'binary'|'set'
    (the Binary/Set token sits immediately after the finalsize digits). Returns
    {} when the name does not match. Reordered variants ship Binary-only.
    """
    m = _BOSTROM_NAME.search(Path(path).name)
    if not m:
        return {}
    return {
        "bits": int(m.group("bits")),
        "hw": int(m.group("hw")),
        "hd": int(m.group("hd")),
        "finalsize": int(m.group("n")),
        "fmt": m.group("fmt").lower(),
    }


def load_bostrom_codebook(
    path: Union[str, Path],
    barcode_length: int,
    fmt: str = "auto",
) -> list[str]:
    """Load a Boström codebook (Set or Binary CSV) into binary strings.

    Set format: whitespace-separated, header-less, HW columns of 1-indexed
    on-positions. Binary format: whitespace-separated, header-less,
    ``barcode_length`` columns of 0/1.

    ``fmt="auto"`` detects Binary when the maximum cell value is 1 (a real Set
    codebook with HW>=2 always has a position >= 2), else Set.

    Returns a list of '0'/'1' strings of length ``barcode_length``, in file order.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    arr = raw.to_numpy()
    if fmt == "auto":
        fmt = "binary" if int(np.nanmax(arr)) <= 1 else "set"
    if fmt == "binary":
        if arr.shape[1] != barcode_length:
            raise ValueError(
                f"Binary codebook has {arr.shape[1]} columns but "
                f"barcode_length={barcode_length}"
            )
        if not np.all(np.isin(arr, (0, 1))):
            raise ValueError("Binary codebook contains values other than 0 and 1")
        return ["".join(str(int(v)) for v in row) for row in arr]
    if fmt == "set":
        sequences = []
        for row in arr:
            bits = ["0"] * barcode_length
            for pos in row:
                if pd.isna(pos):
                    continue
                p = int(pos)
                if not (1 <= p <= barcode_length):
                    raise ValueError(
                        f"Set position {p} out of range 1..{barcode_length}"
                    )
                bits[p - 1] = "1"
            sequences.append("".join(bits))
        return sequences
    raise ValueError(f"Unknown fmt {fmt!r}; expected 'auto', 'set', or 'binary'")


def build_panel_codebook(
    genes_desc_expression: Sequence[str], sequences: Sequence[str]
) -> pd.DataFrame:
    """Assign panel genes to codewords by descending expression.

    Genes (already sorted by DESCENDING expression) are paired with the first
    rows of ``sequences``, so the highest-expression gene receives the first
    codeword, matching SimulatingCode.R.

    Raises:
        ValueError: If there are more genes than available codewords.
    """
    genes = list(genes_desc_expression)
    if len(genes) > len(sequences):
        raise ValueError(f"{len(genes)} genes but only {len(sequences)} codewords")
    # Surplus codewords beyond len(genes) are intentionally discarded; the
    # panel uses only the first len(genes) rows.
    return pd.DataFrame({"Gene": genes, "Sequence": list(sequences[: len(genes)])})


def assign_codewords_to_genes(
    codewords: Sequence[str],
    genes: Sequence[str],
    expression: Sequence[float],
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build a {Gene, Sequence} panel by Boström's policy.

    Genes are sorted by DESCENDING expression in BOTH policies, so the
    Gene-column row order is identical regardless of ``shuffle`` — a shuffled
    panel differs from its non-shuffled counterpart only in the Sequence column.

    shuffle=False (default): pair the highest-expression gene with the first
    codeword (via build_panel_codebook), dropping surplus codewords — the
    original Boström / SimulatingCode.R policy.

    shuffle=True: hold the codeword SET fixed (the first len(genes) codewords in
    file order, exactly the set the non-shuffled panel would use) but permute the
    gene->codeword pairing with ``np.random.default_rng(seed)``. This is an
    expression-agnostic (random) assignment imitating standard MERFISH. Requires
    an explicit ``seed`` for reproducibility.

    Raises:
        ValueError: if len(genes) != len(expression); if len(genes) >
            len(codewords); or if shuffle is True and seed is None.
    """
    genes = list(genes)
    expression = np.asarray(expression, dtype=float)
    if len(genes) != len(expression):
        raise ValueError(
            f"{len(genes)} genes but {len(expression)} expression values"
        )
    order = np.argsort(-expression, kind="stable")
    genes_desc = [genes[i] for i in order]
    if not shuffle:
        return build_panel_codebook(genes_desc, codewords)
    if seed is None:
        raise ValueError(
            "shuffle=True requires an explicit seed for reproducibility"
        )
    n = len(genes_desc)
    if len(codewords) < n:
        raise ValueError(f"{n} genes but only {len(codewords)} codewords")
    perm = np.random.default_rng(seed).permutation(n)
    chosen = [codewords[int(i)] for i in perm]
    return build_panel_codebook(genes_desc, chosen)


def verify_codebook(
    codewords: Sequence[str],
    expected_hw: int | None = None,
    check_min_hd: bool = True,
    max_n_for_hd: int = 2000,
) -> dict:
    """Guardrail: assert constant Hamming weight and (for small codebooks) minHD4.

    Run ONCE on the loaded codebook; a prefix of a verified minHD4 set is still
    minHD4, so do not re-verify a truncated panel. Above ``max_n_for_hd`` the
    O(N^2) HD check is skipped and logged (never silently). Returns
    {'hw', 'n', 'min_hd'?}.
    """
    if not list(codewords):
        raise ValueError("Empty codebook")
    arr = np.array([[1 if c == "1" else 0 for c in s] for s in codewords], dtype=np.uint8)
    weights = sorted({int(w) for w in arr.sum(axis=1)})
    if len(weights) != 1:
        raise ValueError(f"Non-constant Hamming weight across codebook: {weights}")
    hw = weights[0]
    if expected_hw is not None and hw != expected_hw:
        raise ValueError(f"Codebook Hamming weight {hw} != expected {expected_hw}")
    n = len(codewords)
    n_dup = n - len(set(codewords))
    if n_dup:
        logger.warning("Codebook has %d duplicate codewords", n_dup)
    summary = {"hw": hw, "n": n}
    if check_min_hd:
        if n < 2:
            # No pairs to compare; min Hamming distance is undefined.
            logger.info("min-HD check skipped: only %d codeword(s)", n)
        elif n <= max_n_for_hd:
            min_hd = min(
                int((arr[i + 1:] != arr[i]).sum(axis=1).min())
                for i in range(n - 1)
            )
            summary["min_hd"] = min_hd
            if min_hd < 4:
                raise ValueError(f"Codebook min Hamming distance {min_hd} < 4")
        else:
            logger.info(
                "min-HD check skipped: n=%d > max_n_for_hd=%d", n, max_n_for_hd
            )
    return summary
