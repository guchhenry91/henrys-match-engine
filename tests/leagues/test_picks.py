import pandas as pd

from leagues import picks
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


def test_even_one_second_after_kickoff_is_void():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Arsenal", confidence=5, kickoff=ko,
              now=ko + pd.Timedelta(seconds=1))
    out = grade(log["1"], RESULT)
    assert log["1"]["tainted"] is True
    assert out["void"] is True
    assert out["graded"] == "void"


def test_exactly_at_kickoff_is_not_late():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Arsenal", confidence=5, kickoff=ko, now=ko)
    assert log["1"]["tainted"] is False


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


# --- releasing a lock whose kickoff moved ------------------------------------
# Added after 2026-08-15, when a wrong feed time froze La Liga's openers against
# kickoffs 10 hours early -- one of them tainted, i.e. about to be voided.

def _locked(kickoff, locked_at, pick="Alaves"):
    log = {}
    picks.lock_pick(log, "2026:1", pick=pick, confidence=2,
                    kickoff=kickoff, now=locked_at, p_pick=0.47, board=False)
    return log


def test_release_drops_a_lock_when_the_kickoff_moved():
    log = _locked("2026-08-15T07:30:00Z", "2026-08-15T06:59:00Z")
    moved = picks.release_moved_lock(log, "2026:1", "2026-08-15T17:30:00Z",
                                     now="2026-08-15T10:30:00Z")
    assert moved is True and "2026:1" not in log


def test_release_keeps_a_lock_when_the_kickoff_is_unchanged():
    log = _locked("2026-08-21T19:00:00Z", "2026-08-21T18:30:00Z")
    kept = picks.release_moved_lock(log, "2026:1", "2026-08-21T19:00:00Z",
                                    now="2026-08-21T18:45:00Z")
    assert kept is False and "2026:1" in log


def test_release_refuses_once_the_new_kickoff_has_passed():
    """The safety condition: re-locking a started match is hindsight picking."""
    log = _locked("2026-08-15T07:30:00Z", "2026-08-15T06:59:00Z")
    moved = picks.release_moved_lock(log, "2026:1", "2026-08-15T17:30:00Z",
                                     now="2026-08-15T18:00:00Z")
    assert moved is False and "2026:1" in log


def test_release_refuses_an_already_graded_pick():
    log = _locked("2026-08-15T07:30:00Z", "2026-08-15T06:59:00Z")
    log["2026:1"]["graded"] = "wrong"
    moved = picks.release_moved_lock(log, "2026:1", "2026-08-16T17:30:00Z",
                                     now="2026-08-15T10:00:00Z")
    assert moved is False and log["2026:1"]["graded"] == "wrong"


def test_release_is_a_noop_for_an_unlocked_match():
    assert picks.release_moved_lock({}, "2026:1", "2026-08-15T17:30:00Z",
                                    now="2026-08-15T10:00:00Z") is False


def test_released_pick_relocks_cleanly_and_is_not_tainted():
    """The real repair: Sevilla was tainted (locked 13 min after a fake kickoff)
    and would have been voided. After release it locks properly and counts."""
    log = _locked("2026-08-15T09:30:00Z", "2026-08-15T09:43:00Z", pick="Sevilla")
    assert log["2026:1"]["tainted"] is True
    picks.release_moved_lock(log, "2026:1", "2026-08-15T19:30:00Z",
                             now="2026-08-15T10:30:00Z")
    picks.lock_pick(log, "2026:1", pick="Sevilla", confidence=2,
                    kickoff="2026-08-15T19:30:00Z", now="2026-08-15T18:50:00Z",
                    p_pick=0.37, board=False)
    assert log["2026:1"]["tainted"] is False
    assert picks.grade(log["2026:1"],
                       {"home": "Sevilla", "away": "Vallecano",
                        "home_goals": 2, "away_goals": 0})["graded"] == "correct"


def test_release_archives_the_entry_instead_of_erasing_it():
    """sanity_check enforces an append-only log against git HEAD -- correctly,
    since a silent removal is how a `wrong` becomes a `correct`. A release must
    therefore ARCHIVE the entry verbatim, not delete it."""
    log = _locked("2026-08-15T07:30:00Z", "2026-08-15T06:59:00Z")
    original = dict(log["2026:1"])
    picks.release_moved_lock(log, "2026:1", "2026-08-15T17:30:00Z",
                             now="2026-08-15T10:30:00Z")
    archived = log[picks.RELEASED_KEY]["2026:1"][-1]
    assert archived["entry"] == original          # verbatim, byte for byte
    assert archived["was_kickoff"].startswith("2026-08-15T07:30")
    assert archived["now_kickoff"].startswith("2026-08-15T17:30")
    assert archived["reason"]


def test_released_key_cannot_collide_with_a_match_key():
    """Every log consumer filters on the season tag, so the archive must not look
    like one."""
    assert picks.RELEASED_KEY.startswith("_")
    assert ":" not in picks.RELEASED_KEY
