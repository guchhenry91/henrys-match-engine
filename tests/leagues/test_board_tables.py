"""The market-board tables and the Results "Latest settled" panel.

WHY. Settled player props were only reachable at the very bottom of the Results
summary, below the match record, the missed-fixtures list and every league's
tally -- present, and effectively impossible to find. Results now opens on the
newest graded picks, team and player props together, with a toggle between them.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _body(name):
    m = re.search(rf"function {name}\(.*?\)\{{(.*?)\n\}}", HTML, re.S)
    assert m, f"{name}() not found"
    return m.group(1)


def test_results_open_on_the_latest_settled_panel():
    view = _body("viewGrades")
    assert "latestSettled()" in view
    for ret in ("chips+latest+nflGradeSummary()", "chips+latest+uclGradeSummary()",
                "chips+latest+`<div"):
        assert ret in view, f"latest-settled panel missing from: {ret}"


def test_the_panel_holds_player_props_and_team_picks():
    rows = _body("settledRows")
    assert "gradedProps(gsport)" in rows and "gradedMatches(gsport)" in rows
    panel = _body("latestSettled")
    for kind in ('"all"', '"team"', '"prop"'):
        assert kind in panel


def test_a_voided_pick_never_reads_as_a_loss():
    assert "x.void?\"void\"" in _body("winChip")


def test_prop_price_is_never_invented():
    """No soccer prop price is fetched, so the column must say so."""
    assert 'valCell("Price","—","off")' in _body("propTableRow")


def test_edge_only_where_a_price_exists():
    body = _body("bestTableRow")
    assert "priced?(u.p_pick-1/odds)*100:null" in body


def test_leaving_a_board_clears_its_filters():
    body = _body("go")
    for s in ('fdate="all"', 'fleague="all"', 'rkind="all"'):
        assert s in body
