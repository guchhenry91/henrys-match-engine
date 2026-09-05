"""The Grades tab must show every settled pick, not a sample of them.

WHAT IT USED TO SHOW. Aggregate tallies, plus the twelve most recent match picks.
Props -- 59 of them, all graded, all stored -- were itemised NOWHERE, and the
twelve-row cap on matches was invisible, so the list read as complete when it was
showing 12 of 74. A record that reports 12-7 asks to be trusted; a list naming
each pick and what actually happened can be checked, and that is the whole point
of the tab.

THE OTHER HALF IS THE PUBLISHERS. The NFL and UCL boards published a record and
not the picks behind it, so the moment their seasons start the tab would have
shown their totals with nothing underneath. Both now emit a settled list, derived
read-only from the picks log.
"""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    m = re.search(rf"function {name}\(.*?\)\{{(.*?)\n\}}", HTML, re.S)
    assert m, f"{name}() not found in index.html"
    return m.group(1)


def _code(name: str) -> str:
    """The function with its // comments stripped.

    Searching the raw body matched the COMMENT explaining a fix rather than any
    code doing it -- a test that fails on its own explanation.
    """
    return "\n".join(re.sub(r"//.*$", "", line)
                     for line in _fn(name).splitlines())


# A cap looks like slice(0, N). A bare .slice() is an array copy and is fine.
CAP = re.compile(r"slice\(\s*\d")


# --- the tab ------------------------------------------------------------------

@pytest.mark.parametrize("block", ["everyPropBlock", "everyMatchBlock"])
def test_the_itemised_blocks_are_wired_into_the_tab(block):
    view = _fn("viewGrades")
    assert f"{block}()" in view, f"{block} is defined but never rendered"


def test_every_settled_prop_is_collected():
    """THE ONE THAT WAS MISSING ENTIRELY. Props were graded, stored and never
    shown. The collection lives in gradedProps() so the chip row and the lists
    cannot disagree about what exists."""
    body = _code("gradedProps")
    assert "player_picks" in body and "settled" in body
    assert "DATA.nfl" in body, "NFL props must be included, not just soccer"
    assert not CAP.search(body)


def test_a_market_list_is_never_capped():
    """The chips exist so the reader picks a SHORT list, not so the list is
    trimmed for them. Once a market is chosen, every graded pick in it shows."""
    assert not CAP.search(_code("propMarketBlock"))


def test_every_market_with_graded_picks_gets_a_chip():
    """WITH ONE MARKET PER CHIP, AN UNLISTED MARKET IS UNREACHABLE. Its picks
    would be graded, stored, counted in the totals and impossible to open --
    strictly worse than the old long page. So membership is derived from the
    data and only the ORDER is curated."""
    body = _code("gradeMarketsPresent")
    assert "GRADE_MARKETS" in body
    assert "!GRADE_MARKETS.includes" in body, (
        "a market absent from GRADE_MARKETS must be appended, never dropped")
    for fn in ("gradeChips", "everyPropBlock"):
        assert "gradeMarketsPresent" in _code(fn), (
            f"{fn} must use the derived market list, not GRADE_MARKETS directly")


def test_the_summary_does_not_render_the_full_list():
    """The complaint that started this: every settled pick on one page came to
    133 rows. The summary shows tallies and a way in; the rows live behind a
    chip."""
    view = _code("viewGrades")
    assert 'filter==="matches"' in view
    assert "propMarketBlock(filter)" in view


def test_the_match_list_is_no_longer_capped():
    body = _fn("everyMatchBlock")
    assert not CAP.search(body), (
        "everyMatchBlock must not cap its list; the old block showed 12 of 74")


def test_the_old_capped_list_is_gone():
    """allPicksBlock used to render its own twelve-row list, which would now be a
    second, shorter copy of the same picks."""
    body = _fn("allPicksBlock")
    assert "slice(0,12)" not in body


def test_all_three_verdicts_are_rendered():
    body = _fn("verdictChip")
    for word in ("HIT", "MISS", "VOID"):
        assert word in body


def test_a_result_may_be_an_object_or_a_string():
    """Soccer stores {home_goals, away_goals}; the NFL log stores a formatted
    scoreline. Reading .home_goals off the string printed
    "undefined-undefined" on every NFL row."""
    body = _fn("gradedMatchRow")
    assert 'typeof r==="object"' in body
    assert 'typeof r==="string"' in body


def test_the_match_pick_is_read_off_the_season_entry():
    """The season array carries `pick` at the top level and no p_pick. Reading
    `m.prediction.pick` printed a bare dash on all 74 rows."""
    body = _code("everyMatchBlock")
    assert "m.prediction" not in body


# --- the publishers -----------------------------------------------------------

def test_nfl_settled_reads_the_log_and_writes_nothing(tmp_path, monkeypatch):
    from nfl import picks
    log = {
        picks.GAMES_KEY: {
            "2026_01_NE_SEA": {"pick": "SEA", "p_pick": 0.61, "graded": "correct",
                               "kickoff": "2026-09-10T00:20:00+00:00",
                               "home": "SEA", "away": "NE",
                               "result": "SEA 24-20 NE"},
            "_released": {"not": "a pick"},
            "2026_01_KC_BUF": {"pick": "KC", "p_pick": 0.55,
                               "kickoff": "2026-09-13T17:00:00+00:00"},  # pending
        },
        picks.PROPS_KEY: {
            "p1": {"market": "receiving_yards", "player": "Rico Dowdle",
                   "team": "PIT", "line": 45.5, "actual": 31.0, "graded": "wrong",
                   "kickoff": "2026-09-13T17:00:00+00:00"},
        },
    }
    before = json.dumps(log, sort_keys=True)
    out = picks.settled(log)

    assert json.dumps(log, sort_keys=True) == before, "settled() mutated the log"
    keys = {r["key"] for r in out}
    assert keys == {"2026_01_NE_SEA", "p1"}, (
        "settled must skip ungraded picks and the _released archive")
    game = next(r for r in out if r["kind"] == "game")
    assert game["home"] == "SEA" and game["result"] == "SEA 24-20 NE"
    prop = next(r for r in out if r["kind"] == "prop")
    assert prop["line"] == 45.5 and prop["actual"] == 31.0


def test_nfl_stores_the_teams_when_it_locks():
    """The log is keyed by game_id and grading writes only a score string, so
    without these a graded entry cannot name the fixture."""
    source = (ROOT / "nfl" / "picks.py").read_text(encoding="utf-8")
    assert 'entry["home"], entry["away"] = game.get("home"), game.get("away")' in source


@pytest.mark.parametrize("board,key", [("data/nfl/board.json", "settled"),
                                       ("data/ucl/board.json", "settled")])
def test_the_published_boards_carry_a_settled_list(board, key):
    payload = json.loads((ROOT / board).read_text(encoding="utf-8"))
    assert isinstance(payload.get(key), list), (
        f"{board} publishes a record but not the picks behind it")
