"""Tests for duet.plotting.tiers and duet.plotting.legend."""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from duet.plotting.legend import (  # noqa: E402
    LEGEND_HANDLER_MAP,
    LEGEND_SQUARE_MIN_SIZE,
    NotAvailable,
    square_legend_handles,
)
from duet.plotting.tiers import (  # noqa: E402
    PlotTiers,
    add_debug_plots_arg,
    count_skipped,
    paper_panels_from_config,
)


def test_flag_defaults_off_and_parses():
    parser = argparse.ArgumentParser()
    add_debug_plots_arg(parser)
    assert parser.parse_args([]).debug_plots is False
    assert parser.parse_args(["--debug-plots"]).debug_plots is True


def test_flag_help_can_be_overridden():
    parser = argparse.ArgumentParser()
    add_debug_plots_arg(parser, help="No debug plots here.")
    assert "No debug plots here." in parser.format_help()


def test_paper_panels_strip_suffix_and_default_empty():
    assert paper_panels_from_config(None) == frozenset()
    assert paper_panels_from_config({"outdir": "x"}) == frozenset()
    cfg = {"visualization": {"paper_panels": ["a/b_trial_3.svg", "figures/c_lambda0.80"]}}
    assert paper_panels_from_config(cfg) == {"a/b_trial_3", "figures/c_lambda0.80"}


@pytest.mark.parametrize("bad", ["a.svg", ["/abs/a.svg"], ["../a.svg"], [3]])
def test_paper_panels_rejects_bad_entries(bad):
    with pytest.raises(ValueError):
        paper_panels_from_config({"visualization": {"paper_panels": bad}})


def test_writes_debug(tmp_path):
    tiers = PlotTiers(tmp_path, debug=False, paper_panels=frozenset({"g/x_trial_5"}))
    assert tiers.writes_debug(tmp_path / "g" / "x_trial_5.svg")
    assert tiers.writes_debug(tmp_path / "g" / "x_trial_5.png")
    assert not tiers.writes_debug(tmp_path / "g" / "x_trial_4.svg")
    assert PlotTiers(tmp_path, debug=True).writes_debug(tmp_path / "g" / "x_trial_4.svg")
    assert count_skipped([tmp_path / "g" / f"x_trial_{k}.svg" for k in (4, 5)], tiers) == 1


def test_missing_paper_panels(tmp_path, capsys):
    cfg = {"visualization": {"paper_panels": ["g/x_trial_5", "g/y"]}}
    tiers = PlotTiers.from_config(tmp_path, False, cfg)
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "y.png").write_text("")
    assert tiers.report_missing_paper_panels() == ["g/x_trial_5"]
    assert "g/x_trial_5" in capsys.readouterr().out


def test_stale_copy_from_an_earlier_run_counts_as_missing(tmp_path):
    (tmp_path / "g").mkdir()
    stale = tmp_path / "g" / "x_trial_5.svg"
    stale.write_text("")
    tiers = PlotTiers.from_config(tmp_path, False, {"visualization": {"paper_panels": ["g/x_trial_5"]}})
    os.utime(stale, (tiers.started_at - 3600, tiers.started_at - 3600))
    assert tiers.missing_paper_panels() == ["g/x_trial_5"]
    stale.write_text("rewritten")  # this run writes it
    assert tiers.missing_paper_panels() == []


def test_square_legend_handles_and_na_render():
    handles, labels = square_legend_handles(
        [("DUET", "#0072B2"), ("Sivanandan et al.", None)], markersize=1.5
    )
    assert labels == ["DUET", "Sivanandan et al."]
    assert handles[0].get_marker() == "s"
    assert handles[0].get_markersize() == LEGEND_SQUARE_MIN_SIZE
    assert isinstance(handles[1], NotAvailable)

    fig, ax = plt.subplots()
    legend = ax.legend(handles, labels, handler_map=LEGEND_HANDLER_MAP)
    fig.canvas.draw()
    texts = [t.get_text() for t in legend.findobj(matplotlib.text.Text)]
    assert "N/A" in texts
    plt.close(fig)
