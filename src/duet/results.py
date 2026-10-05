"""Results of the public DUET API: :class:`DesignResult` and :class:`EvaluationResult`.

A :class:`DesignResult` holds either the Pareto front of a lambda sweep
(:func:`duet.design_ops`, :func:`duet.design_merfish`): one codebook per
lambda, each with its evaluated decoding accuracy and its secondary
objective, plus the initial codebook and the reference codebooks evaluated
alongside; or, for :func:`duet.design`, the initial and the designed
codebook. Choosing a codebook from a front is left to the user; the helpers
here (:meth:`DesignResult.pareto_front`, :meth:`DesignResult.operating_point`)
only narrow the choice.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "DesignResult",
    "EvaluationResult",
    "load",
    "non_dominated",
    "select_operating_point",
]

# Evaluated-accuracy columns, from duet.benchmark.metrics.compute_metrics.
ACCURACY_COLUMNS = ("accuracy_mean", "accuracy_p10", "accuracy_p95_p5_ratio")

_TABLE_FILE = "table.csv"
_CODEBOOKS_FILE = "codebooks.csv"
_SETTINGS_FILE = "settings.json"

# Secondary-objective column per design kind; a decoding-only design has none.
_SECONDARY_COLUMN = {"ops": "mean_score", "merfish": "crowding"}


def accuracy_summary(metrics: Mapping[str, float]) -> Dict[str, float]:
    """Map :func:`duet.benchmark.metrics.compute_metrics` output to table columns."""
    return {
        "accuracy_mean": metrics["Mean decode accuracy"],
        "accuracy_p10": metrics["10th percentile decode accuracy"],
        "accuracy_p95_p5_ratio": metrics["95th/5th percentile ratio"],
    }


def duplicate_count(sequences) -> int:
    """Number of entries whose sequence also appears elsewhere in the codebook.

    Entries that share a codeword cannot be told apart, so none of them
    decodes.

    Examples
    --------
    >>> duplicate_count(["AC", "AC", "GT"])
    2
    """
    counts = pd.Series(list(sequences), dtype=object).value_counts()
    return int(counts[counts > 1].sum())


def non_dominated(x, y) -> np.ndarray:
    """Boolean mask of the points that no other point dominates, maximizing both axes.

    A point is dominated when another point is at least as good on both axes
    and strictly better on one. Identical points do not dominate each other.

    Examples
    --------
    >>> non_dominated([0.9, 0.8, 0.7], [0.1, 0.3, 0.2]).tolist()
    [True, True, False]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominates_i = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        keep[i] = not dominates_i.any()
    return keep


def select_operating_point(
    lambdas,
    accuracy,
    mean_score,
    reference_mean_score: float,
    score_fraction: float = 0.975,
) -> float | None:
    """The paper's operating-point rule for optical pooled screens, as an optional helper.

    Among the lambda codebooks whose mean score is at least
    ``score_fraction * reference_mean_score`` (the maximum-score codebook's
    mean score), return the lambda with the highest mean evaluated accuracy.
    Ties go to the smallest lambda. Returns None when no codebook qualifies.

    Parameters
    ----------
    lambdas, accuracy, mean_score : array-like
        One entry per lambda codebook.
    reference_mean_score : float
        Mean score of the maximum-score codebook.
    score_fraction : float, default 0.975
        Fraction of the reference mean score a codebook must keep.

    Examples
    --------
    >>> select_operating_point([0.0, 0.1, 0.5], [0.80, 0.90, 0.95], [1.0, 0.99, 0.9], 1.0)
    0.1
    """
    lambdas = np.asarray(lambdas, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)
    mean_score = np.asarray(mean_score, dtype=float)
    threshold = score_fraction * reference_mean_score
    eligible = [i for i in np.argsort(lambdas, kind="stable") if mean_score[i] >= threshold]
    if not eligible:
        return None
    best = max(eligible, key=lambda i: (accuracy[i], -lambdas[i]))
    return float(lambdas[best])


def _dtype_name(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    return "str"


def _write_frame(df: pd.DataFrame, path: Path) -> Dict[str, str]:
    """Write ``df`` as CSV and return its column types, for :func:`_read_frame`."""
    df.to_csv(path, index=False)
    return {c: _dtype_name(df[c]) for c in df.columns}


def _read_frame(path: Path, dtypes: Mapping[str, str]) -> pd.DataFrame:
    """Read a CSV written by :func:`_write_frame`, restoring the column types.

    Text columns stay text (labels such as "007" or "NA" are not converted),
    and floats round-trip exactly.
    """
    floats = [c for c, t in dtypes.items() if t == "float64"]
    df = pd.read_csv(
        path,
        dtype={c: (str if t in ("str", "bool") else t) for c, t in dtypes.items() if t != "int64"},
        keep_default_na=False,
        na_values={c: [""] for c in floats},
        float_precision="round_trip",
    )
    for c, t in dtypes.items():
        if t == "bool":
            df[c] = df[c].map({"True": True, "False": False}).astype(bool)
        elif t == "int64":
            df[c] = df[c].astype("int64")
    return df


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


@dataclass
class DesignResult:
    """The outcome of :func:`duet.design`, :func:`duet.design_ops` or :func:`duet.design_merfish`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per codebook, in evaluation order: the initial codebook, the
        lambda codebooks in ascending lambda, then the references (for OPS the
        maximum-score codebook; for MERFISH the ``include`` codebooks).
        Columns:

        - ``codebook``: label (``"initial"``, ``"lambda=0.1"``, ``"max_score"``,
          an ``include`` name or ``"designed"``); ``kind``: ``"initial"``,
          ``"lambda"``, ``"max_score"``, ``"include"`` or ``"designed"``;
          ``lambda`` (NaN for non-lambda rows);
        - ``accuracy_mean``, ``accuracy_p10``, ``accuracy_p95_p5_ratio``: the
          evaluated decoding accuracy (independent Monte Carlo reads; NaN if
          ``eval_samples=0``). This is the accuracy the paper reports;
        - ``surrogate_accuracy``: the union-bound surrogate the optimizer
          maximizes (``1 - sum of pairwise error probabilities / |S|``). It is
          not an accuracy and can be negative;
        - the secondary objective: ``mean_score`` (OPS) or ``crowding``
          (MERFISH, ``1 - C(S)/C(S0)``);
        - ``n_codewords``; ``duplicate_codewords``: entries that share their
          codeword with another entry and so cannot decode;
        - ``on_pareto_front``: lambda codebooks that no other lambda codebook
          dominates on (accuracy, secondary objective).

        For :func:`duet.design` (kind ``"decoding"``): the rows ``initial``
        and ``designed`` and no ``lambda``, secondary-objective or
        ``on_pareto_front`` column.
    provenance : dict
        Version, seeds of every stage, devices, PEP source (computed or
        cached, and on which device), hash of the ordered candidate table,
        the lambda convention and all settings.

    Examples
    --------
    >>> res = duet.design_ops(pool, duet.channels.symmetric(0.1), seed=0)  # doctest: +SKIP
    >>> res.pareto_front()                                                 # doctest: +SKIP
    >>> res.codebook(0.25).head()                                          # doctest: +SKIP
    """

    kind: str
    table: pd.DataFrame
    codebooks: Dict[str, pd.DataFrame] = field(repr=False)
    provenance: Dict[str, Any] = field(repr=False)

    # -- choosing ----------------------------------------------------------

    @property
    def secondary_objective(self) -> str | None:
        """Name of the secondary-objective column.

        ``"mean_score"``, ``"crowding"``, or None for a decoding-only design.

        Examples
        --------
        >>> res.secondary_objective   # doctest: +SKIP
        'mean_score'
        """
        return _SECONDARY_COLUMN.get(self.kind)

    @property
    def front_axis(self) -> str | None:
        """Accuracy column the Pareto front is computed on.

        ``"accuracy_mean"``, or ``"surrogate_accuracy"`` when the evaluation
        stage was skipped (``eval_samples=0``). None when the result has no
        front (a decoding-only design).

        Examples
        --------
        >>> res.front_axis            # doctest: +SKIP
        'accuracy_mean'
        """
        if "on_pareto_front" not in self.table.columns:
            return None
        evaluated = self.table["accuracy_mean"].notna().any()
        return "accuracy_mean" if evaluated else "surrogate_accuracy"

    def _require_front(self, what: str) -> None:
        if "on_pareto_front" not in self.table.columns:
            raise ValueError(
                f"{what} needs a lambda sweep with a second objective; a decoding-only "
                "design (duet.design) has none: compare res.codebook('initial') with "
                "res.codebook(), or the two rows of res.table"
            )

    def pareto_front(self) -> pd.DataFrame:
        """The lambda codebooks on the Pareto front, the set to choose from.

        Non-domination is taken on :attr:`front_axis` (the evaluated mean
        accuracy, or ``surrogate_accuracy`` when ``eval_samples=0``) and the
        secondary objective. Only the accuracy axis carries Monte Carlo error;
        the secondary objective is exact. Raises ValueError for a decoding-only
        design (:func:`duet.design`).

        Examples
        --------
        >>> res.pareto_front()[["lambda", "accuracy_mean", "mean_score"]]  # doctest: +SKIP
        """
        self._require_front("pareto_front()")
        return self.table[self.table["on_pareto_front"]].reset_index(drop=True)

    def codebook(self, which=None) -> pd.DataFrame:
        """One codebook as a table.

        Parameters
        ----------
        which : float, str or None
            A lambda value from the sweep, or a row label of :attr:`table`
            (``"initial"``, ``"max_score"``, an ``include`` name or
            ``"designed"``). None (the default) returns the designed codebook
            of a :func:`duet.design` result; a lambda sweep needs ``which``.

        Returns
        -------
        pandas.DataFrame
            OPS: ``group``, ``sequence``, ``score``, ``accuracy`` and
            ``candidate`` (row index in the candidate table). MERFISH:
            ``gene``, ``barcode``, ``expression`` and ``accuracy``. decoding:
            ``group``, ``sequence``, ``accuracy``, ``candidate``. One row per
            codeword position.

        Examples
        --------
        >>> res.codebook(0.25)          # doctest: +SKIP
        >>> res.codebook("initial")     # doctest: +SKIP
        >>> res.codebook()              # doctest: +SKIP
        """
        return self.codebooks[self._label(which)].copy()

    def _label(self, which) -> str:
        if which is None:
            designed = self.table.loc[self.table["kind"] == "designed", "codebook"]
            if len(designed) == 1:
                return str(designed.iloc[0])
            raise TypeError(
                "codebook() needs a lambda value or a row label for a design with a lambda "
                f"sweep; labels: {list(self.codebooks)}"
            )
        if isinstance(which, str):
            if which not in self.codebooks:
                raise KeyError(f"no codebook labeled {which!r}; labels: {list(self.codebooks)}")
            return which
        if "lambda" not in self.table.columns:
            raise KeyError(
                "this design has no lambda values (duet.design optimizes decoding alone): "
                "res.codebook() returns the designed codebook, res.codebook('initial') the "
                "one it started from"
            )
        lam = float(which)
        rows = self.table[self.table["kind"] == "lambda"]
        match = rows[np.isclose(rows["lambda"].to_numpy(dtype=float), lam, rtol=0, atol=1e-12)]
        if match.empty:
            raise KeyError(
                f"lambda={lam} was not in the sweep; lambdas: "
                f"{rows['lambda'].tolist()}"
            )
        return str(match["codebook"].iloc[0])

    def operating_point(self, score_fraction: float = 0.975) -> pd.Series:
        """OPS only: the paper's operating-point rule, as an optional helper, not a recommendation.

        Among the lambda codebooks whose mean score is at least
        ``score_fraction`` times the maximum-score codebook's, return the row
        of :attr:`table` with the highest :attr:`front_axis` value (the
        evaluated mean accuracy unless ``eval_samples=0``). Raises ValueError
        for a decoding-only design (:func:`duet.design`).

        Examples
        --------
        >>> res.operating_point(0.975)["lambda"]  # doctest: +SKIP
        """
        self._require_front("operating_point()")
        if self.kind != "ops":
            raise ValueError("operating_point() applies to OPS designs (mean score) only")
        ref = self.table[self.table["kind"] == "max_score"]
        rows = self.table[self.table["kind"] == "lambda"]
        acc = self.front_axis
        lam = select_operating_point(
            rows["lambda"], rows[acc], rows["mean_score"],
            float(ref["mean_score"].iloc[0]), score_fraction,
        )
        if lam is None:
            raise ValueError(
                f"no lambda codebook keeps {score_fraction:.1%} of the maximum-score "
                "codebook's mean score; lower score_fraction or add smaller lambdas"
            )
        return self.table.loc[self.table["codebook"] == self._label(lam)].iloc[0]

    # -- display -------------------------------------------------------------

    def plot(self, ax=None, annotate: bool = True):
        """Plot the lambda codebooks, the Pareto front and the references.

        Uses the DUET stylesheet inside a style context, so global matplotlib
        settings are left unchanged. Imports matplotlib only when called.
        Raises ValueError for a decoding-only design (:func:`duet.design`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created when omitted.
        annotate : bool, default True
            Label each lambda codebook with its lambda.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ax = res.plot()                       # doctest: +SKIP
        >>> ax.figure.savefig("front.svg")        # doctest: +SKIP
        """
        self._require_front("plot()")
        import matplotlib.pyplot as plt

        from duet.plotting import FIGURE_WIDTHS, METHOD_PALETTE, OKABE_ITO, style_context

        acc = self.front_axis
        sec = self.secondary_objective
        with style_context():
            if ax is None:
                _, ax = plt.subplots(figsize=(FIGURE_WIDTHS["half_page"], 2.8))
            lam = self.table[self.table["kind"] == "lambda"].sort_values(acc)
            front = lam[lam["on_pareto_front"]]
            ax.plot(lam[sec], lam[acc], "o", color=METHOD_PALETTE["DUET"], alpha=0.35,
                    label="DUET (dominated)")
            ax.plot(front[sec], front[acc], "o-", color=METHOD_PALETTE["DUET"], label="DUET front")
            if annotate:
                for _, r in lam.iterrows():
                    ax.annotate(f"{r['lambda']:g}", (r[sec], r[acc]), textcoords="offset points",
                                xytext=(3, 3), fontsize=plt.rcParams["legend.fontsize"])
            others = self.table[self.table["kind"] != "lambda"]
            markers = {"initial": ("s", OKABE_ITO["black"]),
                       "max_score": ("^", METHOD_PALETTE.get("Maximum activity", OKABE_ITO["orange"])),
                       "include": ("D", OKABE_ITO["vermillion"])}
            for _, r in others.iterrows():
                if pd.isna(r[acc]) or pd.isna(r[sec]):
                    continue
                m, c = markers[r["kind"]]
                ax.plot(r[sec], r[acc], m, color=c, label=r["codebook"])
            ax.set_xlabel("Mean score" if sec == "mean_score" else "Crowding objective 1 - C(S)/C(S0)")
            ax.set_ylabel("Decoding accuracy" if acc == "accuracy_mean"
                          else "Surrogate accuracy (not evaluated)")
            ax.legend()
        return ax

    # -- persistence ---------------------------------------------------------

    def save(self, outdir) -> Path:
        """Write the table, every codebook and the settings to ``outdir``.

        Files: ``table.csv``, ``codebooks.csv`` (long format, one row per
        codeword of every codebook) and ``settings.json`` (provenance).
        :func:`duet.load` reads them back.

        Examples
        --------
        >>> res.save("my_design/")          # doctest: +SKIP
        >>> duet.load("my_design/").table   # doctest: +SKIP
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        table_dtypes = _write_frame(self.table, outdir / _TABLE_FILE)
        long = pd.concat(
            [cb.assign(codebook=label, position=np.arange(len(cb)))
             for label, cb in self.codebooks.items()],
            ignore_index=True,
        )
        cols = ["codebook", "position"] + [c for c in long.columns if c not in ("codebook", "position")]
        codebook_dtypes = _write_frame(long[cols], outdir / _CODEBOOKS_FILE)
        payload = {"format": "duet-design-result", "format_version": 1, "kind": self.kind,
                   "table_dtypes": table_dtypes, "codebook_dtypes": codebook_dtypes,
                   "provenance": self.provenance}
        (outdir / _SETTINGS_FILE).write_text(json.dumps(payload, indent=2, default=_json_default))
        return outdir


def load(outdir) -> DesignResult:
    """Read a :class:`DesignResult` written by :meth:`DesignResult.save`.

    Examples
    --------
    >>> res = duet.load("my_design/")   # doctest: +SKIP
    """
    outdir = Path(outdir)
    payload = json.loads((outdir / _SETTINGS_FILE).read_text())
    if payload.get("format") != "duet-design-result":
        raise ValueError(f"{outdir} does not hold a saved DUET design result")
    kind = payload["kind"]
    table = _read_frame(outdir / _TABLE_FILE, payload["table_dtypes"])
    long = _read_frame(outdir / _CODEBOOKS_FILE, payload["codebook_dtypes"])
    codebooks = {}
    for label in table["codebook"]:
        cb = long[long["codebook"] == label].sort_values("position")
        codebooks[label] = cb.drop(columns=["codebook", "position"]).reset_index(drop=True)
    return DesignResult(kind=kind, table=table, codebooks=codebooks,
                        provenance=payload["provenance"])


@dataclass
class EvaluationResult:
    """The outcome of :func:`duet.evaluate`.

    Attributes
    ----------
    per_codeword : pandas.DataFrame
        One row per codeword of every codebook: ``codebook``, ``position``,
        ``sequence``, ``accuracy`` (and ``group`` when the input carried one).
    summary : pandas.DataFrame
        One row per codebook: ``codebook``, ``n_codewords``, ``accuracy_mean``,
        ``accuracy_p10``, ``accuracy_p95_p5_ratio``, ``duplicate_codewords``.
    provenance : dict
        Version, seeds, device, channel, rule and sample count.

    Examples
    --------
    >>> acc = duet.evaluate({"mine": seqs}, duet.channels.symmetric(0.1))  # doctest: +SKIP
    >>> acc.summary                                                         # doctest: +SKIP
    """

    per_codeword: pd.DataFrame
    summary: pd.DataFrame
    provenance: Dict[str, Any] = field(repr=False)
