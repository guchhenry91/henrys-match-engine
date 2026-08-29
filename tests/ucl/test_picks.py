"""The Champions League board must freeze and grade its picks like every other.

It published predictions for weeks with no log, no freeze and no record, so
nothing stopped the displayed pick moving after kickoff. These tests exist because
the league phase opens on 8 September with ZERO fixtures published so far -- the
machinery cannot be proven by watching it, so it is proven here.
"""
import json

import pandas as pd
import pytest

from ucl import publish


KICKOFF = "2026-09-16T19:00:00+00:00"


def _match(fixture_id="1001", pick="Real Madrid", p=0.55, conf=3, best=False):
    return {"id": fixture_id, "kickoff": KICKOFF, "date": "2026-09-16",
            "matchday": "League Stage - 1", "home": "Real Madrid",
            "away": "Marseille", "pick": pick, "p_pick": p,
            "confidence": conf, "best_pick": best}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "PICKS_LOG", tmp_path / "picks_log.json")
    monkeypatch.setattr(publish.data, "results_by_id", lambda season=None: {})


def _at(hours_before_kickoff):
    return pd.Timestamp(KICKOFF) - pd.Timedelta(hours=hours_before_kickoff)


def test_a_distant_fixture_is_not_frozen():
    """Freezing early is not caution -- it locks in a number before team news."""
    matches = [_match()]
    log, _ = publish.freeze_and_grade(matches, now=_at(30))
    assert log == {}
    assert not matches[0].get("locked")


def test_a_fixture_inside_the_window_freezes():
    matches = [_match()]
    log, rec = publish.freeze_and_grade(matches, now=_at(1))
    assert log["1001"]["pick"] == "Real Madrid"
    assert log["1001"]["tainted"] is False
    assert matches[0]["locked"] is True
    assert rec["pending"] == 1


def test_the_board_shows_the_frozen_pick_not_a_fresh_one():
    """The whole point of a record: what is graded is what was displayed."""
    publish.freeze_and_grade([_match(pick="Real Madrid", p=0.55)], now=_at(1))
    later = [_match(pick="Marseille", p=0.61)]   # model changed its mind
    publish.freeze_and_grade(later, now=_at(0.5))
    assert later[0]["pick"] == "Real Madrid"
    assert later[0]["p_pick"] == 0.55


def test_locking_is_idempotent():
    """Runs every publish; a second call must not move locked_at forward and
    turn a well-timed lock into a late one."""
    publish.freeze_and_grade([_match()], now=_at(2))
    first = json.loads(publish.PICKS_LOG.read_text())["1001"]["locked_at"]
    publish.freeze_and_grade([_match()], now=_at(0.1))
    assert json.loads(publish.PICKS_LOG.read_text())["1001"]["locked_at"] == first


def test_a_pick_first_seen_after_kickoff_is_tainted_and_voided():
    matches = [_match()]
    log, rec = publish.freeze_and_grade(matches, now=_at(-0.5))
    assert log["1001"]["tainted"] is True
    publish.data.results_by_id = lambda season=None: {
        "1001": {"home": "Real Madrid", "away": "Marseille",
                 "home_goals": 2, "away_goals": 0}}
    _, rec = publish.freeze_and_grade([_match()], now=_at(-3))
    assert rec["void"] == 1 and rec["correct"] == 0


def test_a_fixture_with_no_kickoff_is_left_unlocked(monkeypatch):
    """An assumed kickoff either freezes hours early or taints a good pick."""
    match = _match(); match["kickoff"] = None
    log, _ = publish.freeze_and_grade([match], now=_at(1))
    assert log == {} and match["lockable"] is False


def test_a_fixture_with_no_id_is_left_unlocked():
    match = _match(); match["id"] = None
    log, _ = publish.freeze_and_grade([match], now=_at(1))
    assert log == {} and match["lockable"] is False


def test_a_played_fixture_grades(monkeypatch):
    publish.freeze_and_grade([_match(pick="Real Madrid")], now=_at(1))
    monkeypatch.setattr(publish.data, "results_by_id", lambda season=None: {
        "1001": {"home": "Real Madrid", "away": "Marseille",
                 "home_goals": 2, "away_goals": 0}})
    matches = [_match()]
    log, rec = publish.freeze_and_grade(matches, now=_at(-3))
    assert log["1001"]["graded"] == "correct"
    assert rec["correct"] == 1 and rec["hit_rate"] == 1.0
    assert matches[0]["result"] == "2-0"


def test_a_draw_grades_the_draw_pick():
    publish.freeze_and_grade([_match(pick="Draw")], now=_at(1))
    publish.data.results_by_id = lambda season=None: {
        "1001": {"home": "Real Madrid", "away": "Marseille",
                 "home_goals": 1, "away_goals": 1}}
    _, rec = publish.freeze_and_grade([_match()], now=_at(-3))
    assert rec["correct"] == 1


def test_a_finished_fixture_grades_even_after_it_leaves_the_board():
    """THE BUG THIS CAUGHT. `upcoming_fixtures()` publishes only unplayed games,
    so a match vanishes from `matches` the moment it finishes. Grading driven off
    the board would therefore skip every match on the one run it could have graded
    it, and the record would read 0-0 forever."""
    publish.freeze_and_grade([_match(pick="Real Madrid")], now=_at(1))
    publish.data.results_by_id = lambda season=None: {
        "1001": {"home": "Real Madrid", "away": "Marseille",
                 "home_goals": 3, "away_goals": 1}}
    log, rec = publish.freeze_and_grade([], now=_at(-3))   # board now EMPTY
    assert log["1001"]["graded"] == "correct"
    assert rec["correct"] == 1


def test_an_ungraded_result_that_never_arrives_stays_pending():
    """A missing result is not a loss. Understat's silence published 18 fabricated
    losses in August; the same mistake must not be available here."""
    publish.freeze_and_grade([_match()], now=_at(1))
    _, rec = publish.freeze_and_grade([], now=_at(-50))
    assert rec["pending"] == 1 and rec["wrong"] == 0


def test_the_released_archive_is_never_counted_as_a_pick():
    """A moved kickoff archives the old lock under _released. Counting that
    dictionary as a pick would inflate the record with entries that were retired."""
    publish.freeze_and_grade([_match()], now=_at(1))
    moved = _match()
    moved["kickoff"] = "2026-09-20T19:00:00+00:00"     # rescheduled, still future
    log, rec = publish.freeze_and_grade([moved],
                                        now=pd.Timestamp("2026-09-20T18:30:00Z"))
    assert publish.picks.RELEASED_KEY in log
    assert rec["total"] == 1, "the archived lock must not be counted"
