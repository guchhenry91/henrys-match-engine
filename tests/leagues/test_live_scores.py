"""Live in-play scores: display only, and never a guess.

TWO THINGS COULD GO WRONG HERE and both are worse than having no live scores.

  A SCORE ON THE WRONG FIXTURE. The board says "Newcastle United" and the feed
  says "Newcastle", so equality fails on the very first real match -- and loose
  matching is worse, because "Manchester" is a subset of both Manchester clubs.
  The join therefore requires BOTH teams to match, within the same league, and
  the pairing to be unique. Anything ambiguous shows nothing.

  A PROVISIONAL SCORE TREATED AS A RESULT. The record grades on FT/AET/PEN
  alone. Live data lives in a browser variable, is never written anywhere, and a
  fixture that already has a real result never shows a live one.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _fn(name):
    m = re.search(rf"function {name}\(.*?\)\{{(.*?)\n\}}", HTML, re.S)
    assert m, f"{name}() not found in index.html"
    return m.group(1)


def _code(name):
    return "\n".join(re.sub(r"//.*$", "", l) for l in _fn(name).splitlines())


# --- the join -----------------------------------------------------------------

def test_a_live_row_is_used_only_when_the_match_is_unambiguous():
    """THE ONE THAT PREVENTS A SCORE ON THE WRONG GAME."""
    body = _code("liveFor")
    assert "hits.length===1" in body, (
        "liveFor must refuse when 0 or >1 live rows match, never pick one")


def test_the_join_is_scoped_to_the_same_league():
    body = _code("liveFor")
    assert "l.league===leagueKey" in body, (
        "two clubs with similar names in different leagues must not collide")


def test_both_teams_must_match():
    body = _code("liveFor")
    assert body.count("liveTeamMatches") >= 2, (
        "matching on one team alone would pair the wrong fixture")


def test_team_matching_is_subset_not_equality():
    """The feed's "Newcastle" must reach the board's "Newcastle United"."""
    body = _code("liveTeamMatches")
    assert "subset" in body


def test_rows_carry_their_league_key():
    """liveFor cannot scope by league if the row does not know its own."""
    body = _code("todayFixtures")
    assert "lgKey" in body


# --- provisional never becomes a result ---------------------------------------

def test_a_finished_match_never_shows_a_live_score():
    """A live feed still calling a played match 90' must not overwrite the result
    the record is graded on."""
    body = _code("todayRow")
    assert re.search(r"score\s*\?\s*null\s*:\s*liveFor", body), (
        "a real result must take precedence over any live row")


def test_the_probability_is_labelled_pre_match_on_a_live_row():
    """The model does not condition on score, clock or red cards. Beside a live
    scoreline a reader would otherwise assume the number had updated."""
    assert "pre-match" in _fn("todayRow")


def test_live_data_is_never_persisted():
    """It lives in one variable and is written to no file. If this ever reaches
    the picks log it would settle a bet against a scoreline that had not
    happened, into an append-only record."""
    for name in ("pollLive", "liveFor", "todayRow"):
        body = _code(name)
        assert "localStorage" not in body
        assert "sessionStorage" not in body


# --- the fetch ----------------------------------------------------------------

def test_the_endpoint_is_one_configurable_constant():
    """The proxy may have to move hosts; that should be a one-line change."""
    found = re.findall(r'const LIVE_URL="([^"]+)"', HTML)
    assert len(found) == 1, f"expected exactly one LIVE_URL, found {found}"
    assert found[0].startswith("https://")


def test_a_dead_proxy_cannot_break_the_board():
    """Verified against the real suspended service: the board rendered all its
    rows with the fetch failing."""
    body = _code("pollLive")
    assert "catch" in body
    assert re.search(r"catch[\s\S]*matches:\s*\[\]", body), (
        "a failed fetch must clear live data, not leave stale scores on screen")


def test_polling_stops_when_nothing_can_be_live():
    """A timer running all night costs the reader battery and the proxy requests,
    to learn every minute that nothing is happening."""
    assert "liveExpected()" in _code("pollLive")
    assert "stopLive" in _code("pollLive")
    body = _code("liveExpected")
    assert "r.result" in body, "a finished fixture must not keep the poller alive"


def test_the_poll_interval_is_at_least_the_proxy_cache():
    """Polling faster than the cache only adds requests without adding data."""
    ms = int(re.search(r"const LIVE_POLL_MS=(\d+)", HTML).group(1))
    assert ms >= 30000
