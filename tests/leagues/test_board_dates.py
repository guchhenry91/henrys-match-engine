"""Every board gets the match-date filter and the table layout.

NFL and Champions League fixtures carry `kickoff`, the soccer boards `date`; the
filter reads whichever exists, so no board silently loses its date chips.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _body(name):
    m = re.search(rf"function {name}\(.*?\)\{{(.*?)\n\}}", HTML, re.S)
    assert m, f"{name}() not found"
    return m.group(1)


def test_dates_read_kickoff_or_date():
    assert "const dateOf=p=>p.kickoff||p.date;" in HTML
    assert "dayKey(dateOf(p))" in _body("inBoard")


def test_every_board_has_the_date_filter():
    for view in ("viewBest", "viewPlayers", "viewNFL", "viewUCL", "viewSix", "viewParlay"):
        assert "boardFilters(" in _body(view), f"{view} has no match-date filter"


def test_nfl_and_ucl_render_tables():
    assert "nflGameTableRow" in _body("viewNFL") and "nflPropTableRow" in _body("viewNFL")
    assert "uclTableRow" in _body("viewUCL")
    assert "sixTableRow" in _body("viewSix")


def test_nfl_tiles_use_the_graders_strict_line():
    """features.py and the grader both settle on yards > line, strictly."""
    assert "x>p.line" in _body("nflPropTableRow")


def test_nfl_prop_price_is_never_invented():
    assert 'valCell("Price","—","off")' in _body("nflPropTableRow")
