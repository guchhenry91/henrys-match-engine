import pandas as pd

from leagues.picks import lock_pick, grade, record, LATE_LOCK_HOURS

RESULT = {"home_goals": 2, "away_goals": 0, "home": "Arsenal", "away": "Fulham"}


def test_pick_locked_before_kickoff_is_graded_normally():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Arsenal", confidence=4, kickoff=ko,
              now=ko - pd.Timedelta(hours=3))
    out = grade(log["1"], RESULT)
    assert out["graded"] == "correct"
    assert out["void"] is False


def test_frozen_pick_is_graded_not_a_hindsight_pick():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Fulham", confidence=2, kickoff=ko,
              now=ko - pd.Timedelta(hours=3))
    # a later re-run must NOT overwrite the locked pick
    lock_pick(log, "1", pick="Arsenal", confidence=5, kickoff=ko,
              now=ko - pd.Timedelta(hours=1))
    assert log["1"]["pick"] == "Fulham"
    assert grade(log["1"], RESULT)["graded"] == "wrong"


def test_pick_locked_after_kickoff_is_void():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Arsenal", confidence=5, kickoff=ko,
              now=ko + pd.Timedelta(hours=LATE_LOCK_HOURS + 1))
    out = grade(log["1"], RESULT)
    assert out["void"] is True
    assert out["graded"] == "void"


def test_void_picks_are_excluded_from_the_record():
    entries = [
        {"graded": "correct", "confidence": 5},
        {"graded": "wrong", "confidence": 3},
        {"graded": "void", "confidence": 5},
    ]
    rec = record(entries)
    assert rec["correct"] == 1 and rec["wrong"] == 1
    assert rec["total"] == 2          # the void one does not count
    assert rec["void"] == 1
