"""DUET: multi-objective codebook design for barcode-based assays.

The public API (docs/adr/0003-public-api.md):

- :class:`CandidatePool` (build with :meth:`CandidatePool.from_table`,
  summarize with :meth:`CandidatePool.describe`)
- :mod:`duet.channels`: noise channels
- :func:`design`: decoding accuracy alone (one search at lambda = 1) plus an
  independent evaluation, returning a :class:`DesignResult`
- :func:`design_ops`, :func:`design_merfish`: the lambda sweep plus an
  independent evaluation, returning a :class:`DesignResult`
- :func:`evaluate`: decoding accuracy of existing codebooks
- :func:`load`: read a saved :class:`DesignResult`
- :class:`UniqueMinimum`, :class:`MarginDecoding`: decoding rules for ``rule=``

GPUs are recommended for real libraries (install the ``gpu`` extra): they
compute the PEP matrix and the evaluation, while the lambda sweep runs on
CPU, one process per lambda.

Examples
--------
>>> import duet
>>> duet.__version__  # doctest: +SKIP
'0.2.0'
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    # Single source of truth: the version in pyproject.toml.
    __version__ = _version("duet-codebook")
except _PackageNotFoundError:  # imported from a source tree without installing
    __version__ = "0+unknown"

from duet import channels
from duet.api import design, design_merfish, design_ops, evaluate
from duet.candidate_pool import CandidatePool, PoolDescription
from duet.results import DesignResult, EvaluationResult, load

__all__ = [
    "__version__",
    "CandidatePool",
    "PoolDescription",
    "channels",
    "design",
    "design_ops",
    "design_merfish",
    "evaluate",
    "load",
    "DesignResult",
    "EvaluationResult",
    "UniqueMinimum",
    "MarginDecoding",
]


def __getattr__(name):
    # Decoding rules live in the engine module, which loads scipy; import it
    # only when a rule is asked for, so `import duet` stays light.
    if name in ("UniqueMinimum", "MarginDecoding"):
        from duet import codebook_evaluator

        return getattr(codebook_evaluator, name)
    raise AttributeError(f"module 'duet' has no attribute {name!r}")
