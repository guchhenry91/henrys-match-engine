"""The NFL board must freeze and grade its picks like every other board.

It was the last one publishing predictions with no log, no freeze and no record.
These tests carry the load because week 1 has not been played: nothing here can be
proven by watching the live board until 10 September.

The grading definitions are mirrored from nfl/features.py -- `touchdowns > 0` and
`yards > line`. If those ever drift apart, the record starts describing a product
the release gate never validated, so they are asserted explicitly below.
"""
import pandas as pd
import pytest

from nfl import picks


KICKOFF = "2026-09-13T17:00:00+00:00"
GAME_ID = "2026_01_CHI_CAR"


def _at(hours_before):
    return pd.Timestamp(KICKOFF) - pd.Timedelta(hours=hours_before)


def _payload(**over):
    base = {
        "season": 2026,
        "games": [{"game_id": GAME_ID, "kickoff": KICKOFF, "home": "CAR",
                   "away": "CHI", "pick": "CAR", "p_pick": 0.62,
                   "gradeable": True}],
        "props": {"anytime_touchdown": {"released": True, "picks": [{
            "market": "anytime_touchdown", "game_id": GAME_ID, "kickoff": KICKOFF,
            "player": "Bryce Young", "player_id": "00-0038543", "team": "CAR",
            "line": None, "probability": 0.55, "availability": "fit",
            "club_source": "confirmed"}]}},
    }
    base.update(over)
    return base


def _stats(rows):
    return pd.DataFrame(rows or [], columns=[
        "player_id", "season", "week", "team", "touchdowns",
        "receiving_yards", "rushing_yards", "passing_yards"])


COVERING_STATS = _stats([{"player_id": "00-0038543", "season": 2026, "week": 1,
                          "team": "CAR", "touchdowns": 1, "receiving_yards": 0,
                          "rushing_yards": 12, "passing_yards": 245}])

RESULT_CAR_WON = {GAME_ID: {"home": "CAR", "away": "CHI", "home_score": 24,
                            "away_score": 17, "winner": "home"}}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(picks, "PICKS_LOG", tmp_path / "picks_log.json")


# --- freezing -----------------------------------------------------------------

def test_a_distant_game_is_not_frozen():
    """Freezing early is not caution: it locks a pick in before the inactives."""
    payload = _payload()
    rec = picks.freeze_and_grade(payload, now=_at(30), stats=_stats([]), results={})
    assert rec["team_winner"]["total"] == 0
    assert not payload["games"][0].get("locked")


def test_a_game_inside_the_window_freezes():
    payload = _payload()
    rec = picks.freeze_and_grade(payload, now=_at(1), stats=_stats([]), results={})
    assert rec["team_winner"]["pending"] == 1
    assert payload["games"][0]["locked"] is True


def test_the_board_shows_the_frozen_pick_not_a_fresh_one():
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    later = _payload()
    later["games"][0].update(pick="CHI", p_pick=0.71)      # model changed its mind
    picks.freeze_and_grade(later, now=_at(0.5), stats=_stats([]), results={})
    assert later["games"][0]["pick"] == "CAR"
    assert later["games"][0]["p_pick"] == 0.62


def test_locking_is_idempotent():
    picks.freeze_and_grade(_payload(), now=_at(2), stats=_stats([]), results={})
    log = picks.core.load_log(picks.PICKS_LOG)
    first = log["games"][GAME_ID]["locked_at"]
    picks.freeze_and_grade(_payload(), now=_at(0.1), stats=_stats([]), results={})
    log = picks.core.load_log(picks.PICKS_LOG)
    assert log["games"][GAME_ID]["locked_at"] == first


def test_a_game_with_no_real_kickoff_is_left_unlocked():
    """Every NFL kickoff used to be midnight, because only `gameday` was read."""
    payload = _payload()
    payload["games"][0]["kickoff"] = None
    rec = picks.freeze_and_grade(payload, now=_at(1), stats=_stats([]), results={})
    assert rec["team_winner"]["total"] == 0
    assert payload["games"][0]["lockable"] is False


def test_a_withheld_market_is_never_frozen():
    """The gate withheld it; the board must not quietly build a record on it."""
    payload = _payload()
    payload["props"]["anytime_touchdown"]["released"] = False
    rec = picks.freeze_and_grade(payload, now=_at(1), stats=_stats([]), results={})
    assert rec["props"]["total"] == 0


def test_an_ungradeable_game_is_never_frozen():
    payload = _payload()
    payload["games"][0]["gradeable"] = False
    rec = picks.freeze_and_grade(payload, now=_at(1), stats=_stats([]), results={})
    assert rec["team_winner"]["total"] == 0


def test_the_line_is_frozen_with_the_prop():
    """The line is the player's own entering MEDIAN and moves week to week.
    Grading against a later line would settle a bet nobody made."""
    payload = _payload()
    payload["props"]["anytime_touchdown"]["picks"][0].update(
        market="receiving_yards", line=45.5)
    payload["props"] = {"receiving_yards": payload["props"]["anytime_touchdown"]}
    picks.freeze_and_grade(payload, now=_at(1), stats=_stats([]), results={})
    log = picks.core.load_log(picks.PICKS_LOG)
    entry = next(iter(log["props"].values()))
    assert entry["line"] == 45.5


# --- grading ------------------------------------------------------------------

def test_a_played_game_grades():
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=COVERING_STATS,
                                 results=RESULT_CAR_WON)
    assert rec["team_winner"]["correct"] == 1
    assert rec["team_winner"]["hit_rate"] == 1.0


def test_a_losing_pick_grades_wrong():
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    lost = {GAME_ID: {"home": "CAR", "away": "CHI", "home_score": 10,
                      "away_score": 21, "winner": "away"}}
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=COVERING_STATS,
                                 results=lost)
    assert rec["team_winner"]["wrong"] == 1


def test_a_tie_is_void_not_a_loss():
    """The gate scores a tie 0.5. A win/loss record has no half, so it pushes --
    scoring it a loss would understate the model against its own measurement."""
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    tied = {GAME_ID: {"home": "CAR", "away": "CHI", "home_score": 17,
                      "away_score": 17, "winner": "tie"}}
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=COVERING_STATS,
                                 results=tied)
    assert rec["team_winner"]["void"] == 1
    assert rec["team_winner"]["correct"] == 0 and rec["team_winner"]["wrong"] == 0


def test_a_game_grades_even_after_it_leaves_the_board():
    """The board publishes only UPCOMING games, so a fixture vanishes the moment
    it is played. Grading off the board would skip it on the one run that could
    have settled it, and the record would read 0-0 forever."""
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    empty = {"season": 2026, "games": [], "props": {}}
    rec = picks.freeze_and_grade(empty, now=_at(-4), stats=COVERING_STATS,
                                 results=RESULT_CAR_WON)
    assert rec["team_winner"]["correct"] == 1


def test_a_touchdown_prop_grades_on_more_than_zero():
    """Mirrors nfl/features.py: outcome = touchdowns > 0."""
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=COVERING_STATS,
                                 results=RESULT_CAR_WON)
    assert rec["props_by_market"]["anytime_touchdown"]["correct"] == 1


def test_a_touchdown_prop_with_no_touchdown_grades_wrong():
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    scoreless = _stats([{"player_id": "00-0038543", "season": 2026, "week": 1,
                         "team": "CAR", "touchdowns": 0, "receiving_yards": 0,
                         "rushing_yards": 3, "passing_yards": 210}])
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=scoreless,
                                 results=RESULT_CAR_WON)
    assert rec["props_by_market"]["anytime_touchdown"]["wrong"] == 1


@pytest.mark.parametrize("actual,line,expected", [
    (46.0, 45.5, "correct"),
    (45.0, 45.5, "wrong"),
    (45.5, 45.5, "wrong"),      # strictly greater, mirroring features.py
])
def test_a_yards_prop_grades_strictly_over_the_line(actual, line, expected):
    entry = {"market": "receiving_yards", "line": line, "tainted": False}
    graded = picks.grade_prop(entry, {"receiving_yards": actual})
    assert graded["graded"] == expected


def test_a_prop_stays_pending_when_the_feed_has_not_filed_the_game():
    """THE 0-18 GUARD. The soccer board once graded 18 picks as losses against a
    feed that had published nothing for the season. A silent feed must produce
    pending picks, never fabricated losses."""
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    other_team_only = _stats([{"player_id": "00-0000001", "season": 2026,
                               "week": 1, "team": "SEA", "touchdowns": 2,
                               "receiving_yards": 0, "rushing_yards": 0,
                               "passing_yards": 0}])
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=other_team_only,
                                 results=RESULT_CAR_WON)
    assert rec["props"]["pending"] == 1
    assert rec["props"]["wrong"] == 0


def test_a_player_missing_from_a_covered_game_grades_wrong():
    """His side IS in the feed and he is not: he took no snap or recorded nothing.
    Either way he scored no touchdown, so both readings settle him under."""
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    teammate_only = _stats([{"player_id": "00-0009999", "season": 2026, "week": 1,
                             "team": "CAR", "touchdowns": 1, "receiving_yards": 0,
                             "rushing_yards": 0, "passing_yards": 0}])
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=teammate_only,
                                 results=RESULT_CAR_WON)
    assert rec["props"]["wrong"] == 1


def test_a_pick_first_seen_after_kickoff_is_void():
    payload = _payload()
    picks.freeze_and_grade(payload, now=_at(-0.5), stats=_stats([]), results={})
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=COVERING_STATS,
                                 results=RESULT_CAR_WON)
    assert rec["team_winner"]["void"] == 1
    assert rec["team_winner"]["correct"] == 0


def test_markets_are_recorded_separately():
    """A 52% team-winner pick and a 90% touchdown prop pooled give a number that
    describes neither."""
    picks.freeze_and_grade(_payload(), now=_at(1), stats=_stats([]), results={})
    rec = picks.freeze_and_grade(_payload(), now=_at(-4), stats=COVERING_STATS,
                                 results=RESULT_CAR_WON)
    assert rec["team_winner"]["settled"] == 1
    assert rec["props"]["settled"] == 1
    assert set(rec["props_by_market"]) == set(picks.config.MARKETS)


def test_season_week_is_read_off_the_game_id():
    assert picks.season_week("2026_01_NE_SEA") == (2026, 1)
    assert picks.season_week("2026_14_KC_BUF") == (2026, 14)
    assert picks.season_week(None) == (None, None)
