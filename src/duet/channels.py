"""Noise channels for the public DUET API.

A channel describes how the readout corrupts each symbol of a codeword. Each
constructor returns a :class:`Channel` that also carries the matching
maximum-likelihood decoding metric, so you choose the channel and DUET chooses
the metric:

=====================================  =========================================
Constructor                            Decoding metric
=====================================  =========================================
:func:`symmetric`                      Hamming distance
:func:`position_varying`               position-varying negative log-likelihood
:func:`asymmetric`                     asymmetric negative log-likelihood
:func:`position_varying_asymmetric`    position-varying asymmetric NLL
=====================================  =========================================

**Symbol order.** Matrices are read as ``T[sent, read]``: row = transmitted
symbol, column = observed symbol, both in the order given by ``alphabet``
(default ``"ACGT"``; ``"01"`` for binary barcodes). DUET encodes DNA
internally as A, T, C, G and reorders your matrix to that order, so a matrix
written in A, C, G, T order is applied to the right bases. A labeled
:class:`pandas.DataFrame` (index = sent symbol, columns = read symbol) is
reordered by its labels.

Only square channels, where the read alphabet equals the sent alphabet, are
supported in this release; rectangular channels (such as dual-guide
chemistries) are rejected.

Examples
--------
>>> import pandas as pd, duet
>>> T = pd.DataFrame([[0.97, 0.01, 0.01, 0.01],
...                   [0.02, 0.94, 0.02, 0.02],
...                   [0.01, 0.01, 0.97, 0.01],
...                   [0.02, 0.02, 0.02, 0.94]],
...                  index=list("ACGT"), columns=list("ACGT"))
>>> ch = duet.channels.asymmetric(T)   # reordered to the internal A, T, C, G
>>> ch.params["channel_matrix"][1]     # the T row
[0.02, 0.94, 0.02, 0.02]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "Channel",
    "symmetric",
    "position_varying",
    "asymmetric",
    "position_varying_asymmetric",
]

# Symbol order of the engine's encoders (duet.codebook_evaluator): DNAEncoder
# maps A, T, C, G -> 0, 1, 2, 3 and BinaryEncoder maps "0", "1" -> 0, 1.
_INTERNAL_ORDER = {frozenset("ACGT"): "ATCG", frozenset("01"): "01"}


def check_alphabet(alphabet: str) -> str:
    """Validate an alphabet and return it.

    Supported: the four DNA bases in any order (for example ``"ACGT"``) and
    binary ``"01"``.
    """
    if not isinstance(alphabet, str) or not alphabet:
        raise ValueError(f"alphabet must be a string such as 'ACGT' or '01', got {alphabet!r}")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(f"alphabet {alphabet!r} lists a symbol more than once")
    if frozenset(alphabet) not in _INTERNAL_ORDER:
        raise ValueError(
            f"alphabet {alphabet!r} is not supported: use the four DNA bases in "
            "any order (for example 'ACGT') or binary '01'"
        )
    return alphabet


def internal_order(alphabet: str) -> str:
    """The engine's symbol order for ``alphabet``: ``'ATCG'`` or ``'01'``."""
    return _INTERNAL_ORDER[frozenset(check_alphabet(alphabet))]


@dataclass(frozen=True, eq=False)
class Channel:
    """A noise channel together with its maximum-likelihood decoding metric.

    Build one with :func:`symmetric`, :func:`position_varying`,
    :func:`asymmetric` or :func:`position_varying_asymmetric`. Parameters are
    stored in the engine's internal symbol order (A, T, C, G for DNA).

    Attributes
    ----------
    kind : str
        ``"symmetric"``, ``"position_varying"``, ``"asymmetric"`` or
        ``"position_varying_asymmetric"``.
    alphabet : str
        The alphabet the channel was built with. Its symbols must match the
        candidate sequences'.
    params : dict
        Channel parameters, in internal symbol order.

    Examples
    --------
    >>> import duet
    >>> ch = duet.channels.symmetric(0.1)
    >>> ch.kind, ch.alphabet_size
    ('symmetric', 4)
    """

    kind: str
    alphabet: str
    params: Dict[str, Any] = field(repr=False)

    @property
    def alphabet_size(self) -> int:
        """Number of symbols: 4 for DNA, 2 for binary.

        Examples
        --------
        >>> import duet
        >>> duet.channels.symmetric(0.05, alphabet="01").alphabet_size
        2
        """
        return len(self.alphabet)

    @property
    def seq_length(self) -> int | None:
        """Number of positions the channel covers, or None if it fits any length.

        Examples
        --------
        >>> import duet
        >>> duet.channels.position_varying([0.05, 0.1, 0.2]).seq_length
        3
        """
        if self.kind == "position_varying":
            return len(self.params["epsilon"])
        if self.kind == "position_varying_asymmetric":
            return len(self.params["channel_matrices"])
        return None

    def check_compatible(self, alphabet: str, seq_length: int) -> None:
        """Raise ValueError unless the channel fits sequences of this alphabet and length.

        Examples
        --------
        >>> import duet
        >>> duet.channels.symmetric(0.1).check_compatible("ACGT", 10)  # fits; returns None
        """
        if frozenset(alphabet) != frozenset(self.alphabet):
            raise ValueError(
                f"the channel was built for alphabet {self.alphabet!r} but the "
                f"sequences use {alphabet!r}; the pool and the channel must share "
                "an alphabet"
            )
        L = self.seq_length
        if L is not None and L != seq_length:
            raise ValueError(
                f"the {self.kind.replace('_', '-')} channel has {L} positions but "
                f"the sequences have length {seq_length} (after truncation to "
                "seq_length); give one entry per sequencing round"
            )

    def component_configs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """``(noise_channel, decoding_metric)`` dicts for :class:`duet.evaluator_config.EvaluatorConfig`.

        Examples
        --------
        >>> import duet
        >>> duet.channels.symmetric(0.1).component_configs()
        ({'type': 'symmetric', 'epsilon': 0.1}, {'type': 'hamming'})
        """
        p = self.params
        if self.kind == "symmetric":
            return {"type": "symmetric", "epsilon": p["epsilon"]}, {"type": "hamming"}
        if self.kind == "position_varying":
            eps = list(p["epsilon"])
            return (
                {"type": "position_varying", "epsilon": eps},
                {"type": "position_varying_nll", "epsilon": eps},
            )
        if self.kind == "asymmetric":
            T = p["channel_matrix"]
            return (
                {"type": "asymmetric", "channel_matrix": T},
                {"type": "asymmetric_nll", "channel_matrix": T},
            )
        T = p["channel_matrices"]
        return (
            {"type": "position_varying_asymmetric", "channel_matrices": T},
            {"type": "position_varying_asymmetric_nll", "channel_matrices": T},
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable description, used in result provenance.

        Examples
        --------
        >>> import duet
        >>> duet.channels.symmetric(0.1).to_dict()["internal_symbol_order"]
        'ATCG'
        """
        noise, metric = self.component_configs()
        return {
            "kind": self.kind,
            "alphabet": self.alphabet,
            "internal_symbol_order": internal_order(self.alphabet),
            "noise_channel": noise,
            "decoding_metric": metric,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Channel":
        """Rebuild a channel from :meth:`to_dict` output.

        Examples
        --------
        >>> import duet
        >>> ch = duet.channels.position_varying([0.1, 0.2])
        >>> duet.channels.Channel.from_dict(ch.to_dict()).params == ch.params
        True
        """
        noise = d["noise_channel"]
        key = {"symmetric": "epsilon", "position_varying": "epsilon",
               "asymmetric": "channel_matrix",
               "position_varying_asymmetric": "channel_matrices"}[d["kind"]]
        return cls(kind=d["kind"], alphabet=d["alphabet"], params={key: noise[key]})


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def symmetric(epsilon: float, alphabet: str = "ACGT") -> Channel:
    """Every symbol is misread with probability ``epsilon``, uniformly to the others.

    Decoded with the Hamming distance, which is maximum likelihood for this
    channel.

    Parameters
    ----------
    epsilon : float
        Per-symbol error probability, in [0, 1).
    alphabet : str, default "ACGT"
        ``"ACGT"`` (DNA, any order) or ``"01"`` (binary barcodes).

    Examples
    --------
    >>> import duet
    >>> duet.channels.symmetric(0.1)
    Channel(kind='symmetric', alphabet='ACGT')
    """
    check_alphabet(alphabet)
    return Channel("symmetric", alphabet, {"epsilon": _probability(epsilon, "epsilon")})


def position_varying(epsilon, alphabet: str = "ACGT") -> Channel:
    """A symmetric error rate per position (sequencing round).

    Parameters
    ----------
    epsilon : array-like of float, or path to a .npy file
        One error probability per position, length L = sequence length
        after truncation.
    alphabet : str, default "ACGT"
        ``"ACGT"`` (DNA, any order) or ``"01"`` (binary barcodes).

    Examples
    --------
    >>> import duet
    >>> ch = duet.channels.position_varying([0.05, 0.08, 0.1, 0.12])
    >>> ch.seq_length
    4
    """
    check_alphabet(alphabet)
    eps = _load_array(epsilon, "epsilon")
    if eps.ndim != 1 or eps.size == 0:
        raise ValueError(f"epsilon must be a 1-D array with one entry per position, got shape {eps.shape}")
    eps = [_probability(e, f"epsilon[{i}]") for i, e in enumerate(eps)]
    return Channel("position_varying", alphabet, {"epsilon": eps})


def asymmetric(matrix, alphabet: str = "ACGT") -> Channel:
    """One substitution matrix ``T[sent, read]`` shared by all positions.

    Parameters
    ----------
    matrix : array-like of shape (q, q), labeled DataFrame, or path to a .npy file
        Row-stochastic: ``T[a, b]`` is the probability of reading ``b`` when
        ``a`` was sent. Rows and columns follow ``alphabet``; a DataFrame is
        reordered by its index (sent) and column (read) labels.
    alphabet : str, default "ACGT"
        Symbol order of the matrix rows and columns: DNA bases in any order,
        or ``"01"``.

    Examples
    --------
    >>> import duet
    >>> ch = duet.channels.asymmetric([[0.98, 0.02], [0.06, 0.94]], alphabet="01")
    >>> ch.kind
    'asymmetric'
    """
    check_alphabet(alphabet)
    T = _matrix_in_internal_order(matrix, alphabet, "matrix")
    return Channel("asymmetric", alphabet, {"channel_matrix": T.tolist()})


def position_varying_asymmetric(matrices, alphabet: str = "ACGT") -> Channel:
    """One substitution matrix ``T[l, sent, read]`` per position.

    Parameters
    ----------
    matrices : array-like of shape (L, q, q), sequence of labeled DataFrames, or path to a .npy file
        ``matrices[l]`` is the row-stochastic matrix of position ``l``, rows
        and columns in ``alphabet`` order.
    alphabet : str, default "ACGT"
        Symbol order of each matrix's rows and columns.

    Examples
    --------
    >>> import numpy as np, duet
    >>> T = np.array([[[0.9, 0.1], [0.2, 0.8]]] * 6)
    >>> duet.channels.position_varying_asymmetric(T, alphabet="01").seq_length
    6
    """
    check_alphabet(alphabet)
    if isinstance(matrices, (str, Path)):
        matrices = _load_array(matrices, "matrices")
    if isinstance(matrices, np.ndarray) and matrices.ndim != 3:
        raise ValueError(
            f"matrices must have shape (L, q, q), one T[sent, read] per position; got {matrices.shape}"
        )
    per_position = [
        _matrix_in_internal_order(m, alphabet, f"matrices[{i}]") for i, m in enumerate(matrices)
    ]
    if not per_position:
        raise ValueError("matrices is empty; give one matrix per position")
    return Channel(
        "position_varying_asymmetric", alphabet,
        {"channel_matrices": [m.tolist() for m in per_position]},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probability(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be an error probability in [0, 1), got {value}")
    return value


def _load_array(value, name: str) -> np.ndarray:
    """Array from an array-like or from the path of a .npy file."""
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: no such file: {path}")
        return np.asarray(np.load(path), dtype=float)
    return np.asarray(value, dtype=float)


def _matrix_in_internal_order(value, alphabet: str, name: str) -> np.ndarray:
    """A square row-stochastic (q, q) matrix, reordered to internal symbol order."""
    import pandas as pd

    order = internal_order(alphabet)
    q = len(alphabet)
    if isinstance(value, pd.DataFrame):
        sent = [str(s) for s in value.index]
        read = [str(s) for s in value.columns]
        if len(read) != len(sent):
            raise ValueError(_rectangular_message(name, (len(sent), len(read))))
        for axis, labels in (("index (sent symbols)", sent), ("columns (read symbols)", read)):
            if sorted(labels) != sorted(alphabet):
                raise ValueError(
                    f"{name}: the DataFrame {axis} must be the symbols of alphabet "
                    f"{alphabet!r} once each, got {labels}"
                )
        relabeled = value.copy()
        relabeled.index, relabeled.columns = sent, read  # e.g. integer 0/1 labels -> "0"/"1"
        T = relabeled.loc[list(order), list(order)].to_numpy(dtype=float)
    else:
        T = _load_array(value, name)
        if T.ndim != 2:
            raise ValueError(f"{name} must be a 2-D matrix T[sent, read], got shape {T.shape}")
        if T.shape[0] != T.shape[1]:
            raise ValueError(_rectangular_message(name, T.shape))
        if T.shape != (q, q):
            raise ValueError(f"{name} has shape {T.shape} but alphabet {alphabet!r} has {q} symbols")
        perm = [alphabet.index(s) for s in order]
        T = T[np.ix_(perm, perm)]
    if not np.all(np.isfinite(T)) or np.any(T < 0) or np.any(T > 1):
        raise ValueError(f"{name}: entries must be probabilities in [0, 1]")
    sums = T.sum(axis=1)
    if not np.allclose(sums, 1.0):
        raise ValueError(
            f"{name}: each row T[sent, :] must sum to 1 (rows are the sent symbol); "
            f"row sums are {np.round(sums, 6).tolist()}"
        )
    return T


def _rectangular_message(name: str, shape) -> str:
    return (
        f"{name} has shape {tuple(shape)}: only square channels (the read alphabet "
        "equals the sent alphabet) are supported in this release. Rectangular "
        "channels, such as dual-guide chemistries, are not supported yet."
    )
