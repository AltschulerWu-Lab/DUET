"""Output tiers for visualizer figures: paper panels, diagnostics, debug plots.

Every visualizer sorts what it draws into three tiers:

1. Paper panels: files an Outputs table in ``experiments/`` names as a panel.
2. Diagnostics: the trial-aggregated figures a user needs to choose λ. Written
   by default.
3. Debug plots: per-trial copies, per-λ sweeps and other detail. Kept in code,
   but written only with ``--debug-plots``. The experiment drivers never pass it.

A paper panel that is a debug-tier file (one trial or one λ, picked by hand) is
listed in the experiment config, as paths relative to the visualizer's output
directory with or without the extension::

    visualization:
      paper_panels:
        - guide_level_comparisons/max_activity_97p5pct_trial_5_jointplot

:class:`PlotTiers` then writes it by default too, at its usual path.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

DEBUG_PLOTS_FLAG = "--debug-plots"

DEBUG_PLOTS_HELP = (
    "Also write the debug plots (per-trial copies, per-lambda sweeps and other "
    "detail). By default only paper panels and diagnostics are written."
)

# Suffixes stripped when matching a figure path to a paper_panels entry, so an
# entry names every format a figure is saved in.
FIGURE_SUFFIXES = (".svg", ".png", ".pdf")

# A file whose mtime is this many seconds before the run started still counts
# as written by the run (coarse filesystem timestamps). A stale copy from an
# earlier render is minutes or days older.
_MTIME_SLACK_S = 2.0


def add_debug_plots_arg(
    parser: argparse.ArgumentParser, help: str = DEBUG_PLOTS_HELP
) -> None:
    """Add the shared ``--debug-plots`` flag (store_true, default off).

    ``help`` overrides the shared help text for a script whose debug plots are
    something else, or that has none and takes the flag only for a uniform CLI.
    """
    parser.add_argument(DEBUG_PLOTS_FLAG, action="store_true", help=help)


def _figure_key(path: str | PurePosixPath) -> str:
    p = PurePosixPath(path)
    if p.suffix in FIGURE_SUFFIXES:
        p = p.with_suffix("")
    return p.as_posix()


def paper_panels_from_config(config: Mapping[str, Any] | None) -> frozenset[str]:
    """Return ``visualization.paper_panels`` from a raw config dict.

    Entries are output-relative POSIX paths; a figure suffix is dropped. A
    missing block gives an empty set.
    """
    if not config:
        return frozenset()
    visualization = config.get("visualization") or {}
    if not isinstance(visualization, Mapping):
        raise ValueError("`visualization` must be a mapping")
    panels = visualization.get("paper_panels") or []
    if isinstance(panels, str) or not all(isinstance(p, str) for p in panels):
        raise ValueError("`visualization.paper_panels` must be a list of paths")
    for p in panels:
        if PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts:
            raise ValueError(
                f"`visualization.paper_panels` entry {p!r} must be relative to "
                "the output directory"
            )
    return frozenset(_figure_key(p) for p in panels)


@dataclass(frozen=True)
class PlotTiers:
    """Which debug-tier figures a visualizer run writes.

    Paper panels and diagnostics are always written; call sites of debug-tier
    figures ask :meth:`writes_debug` first. Build it at the start of a run:
    ``started_at`` is when a listed panel must have been written by.
    """

    output_dir: Path
    debug: bool = False
    paper_panels: frozenset[str] = frozenset()
    started_at: float = field(default_factory=time.time, compare=False)

    @classmethod
    def from_config(
        cls, output_dir: Path, debug: bool, config: Mapping[str, Any] | None
    ) -> "PlotTiers":
        return cls(Path(output_dir), debug, paper_panels_from_config(config))

    def key(self, path: Path | str) -> str:
        """``path`` relative to the output directory, without figure suffix."""
        rel = Path(path).relative_to(self.output_dir)
        return _figure_key(PurePosixPath(rel.as_posix()))

    def writes_debug(self, path: Path | str) -> bool:
        """Whether the debug-tier figure at ``path`` is written in this run."""
        return self.debug or self.key(path) in self.paper_panels

    def missing_paper_panels(self) -> list[str]:
        """Listed paper panels that this run did not write, in any figure format.

        A file counts only if it was written after ``started_at``: the
        visualizers render in place and delete nothing, so a copy left by an
        earlier render must not hide a panel this run skipped. Call it after
        every figure is written; a non-empty result means a typo in the list or
        a trial/λ this run did not produce.
        """
        def written(k: str) -> bool:
            for s in FIGURE_SUFFIXES:
                path = self.output_dir / (k + s)
                if path.exists() and path.stat().st_mtime >= self.started_at - _MTIME_SLACK_S:
                    return True
            return False

        return sorted(k for k in self.paper_panels if not written(k))

    def report_missing_paper_panels(self) -> list[str]:
        """Print a warning naming each listed panel that was not written."""
        missing = self.missing_paper_panels()
        for k in missing:
            print(f"WARNING: paper panel {k!r} was not written under {self.output_dir}")
        return missing


def count_skipped(paths: Iterable[Path | str], tiers: PlotTiers) -> int:
    """How many of ``paths`` this run skips as debug plots."""
    return sum(not tiers.writes_debug(p) for p in paths)
