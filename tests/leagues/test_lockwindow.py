"""The adaptive lock window must close the gap that lost four played fixtures.

On 2026-08-28 the locker ran at 15:23 and 21:18. Every fixture kicking off between
18:30 and 19:30 -- four of them, across four leagues -- was shown with a pick,
played, and then absent from the record entirely. The record was not wrong, it was
a biased sample: it described the fixtures that happened to kick off near a
workflow run.
"""
from datetime import datetime, timedelta, timezone

import pytest

from leagues import lockwindow
from leagues.config import LOCK_WINDOW_HOURS


@pytest.fixture
def beat_path(tmp_path):
    return tmp_path / "heartbeat.json"


def _at(text):
    return datetime.fromisoformat(text)


def test_with_no_heartbeat_it_is_the_plain_floor(beat_path):
    """First run ever, or an unreadable file: behave exactly as before."""
    assert lockwindow.window(path=beat_path) == LOCK_WINDOW_HOURS


def test_frequent_runs_leave_the_window_at_the_floor(beat_path):
    """Nothing changes while the scheduler is healthy."""
    lockwindow.beat(now=_at("2026-08-28T15:00:00+00:00"), path=beat_path)
    assert lockwindow.window(now=_at("2026-08-28T15:15:00+00:00"),
                             path=beat_path) == LOCK_WINDOW_HOURS


def test_a_long_gap_widens_the_window_to_cover_it(beat_path):
    """THE ACTUAL INCIDENT. Runs at 15:23 and 21:18; Bayern kicked off at 18:30.

    Under the old fixed 2-hour window the 21:18 run saw a fixture 2.8 hours in the
    past and froze nothing, so the match left no pick at all. The window must now
    cover the 5.9 hours that actually elapsed.
    """
    lockwindow.beat(now=_at("2026-08-28T15:23:00+00:00"), path=beat_path)
    w = lockwindow.window(now=_at("2026-08-28T21:18:00+00:00"), path=beat_path)
    assert w == pytest.approx(5.92, abs=0.05)
    # A 18:30 kickoff is 2.8h ahead of the 15:41 point where the previous run's
    # own window ran out -- inside the widened window, so it would have frozen.
    assert w > 3.1


def test_the_widening_is_capped(beat_path):
    """After a multi-day outage the first run back must not freeze an entire
    matchweek at once on days-stale numbers."""
    lockwindow.beat(now=_at("2026-08-20T12:00:00+00:00"), path=beat_path)
    assert lockwindow.window(now=_at("2026-08-28T12:00:00+00:00"),
                             path=beat_path) == lockwindow.MAX_WINDOW_HOURS


def test_it_never_goes_below_the_floor(beat_path):
    lockwindow.beat(now=_at("2026-08-28T15:00:00+00:00"), path=beat_path)
    for minutes in (1, 10, 59):
        w = lockwindow.window(now=_at("2026-08-28T15:00:00+00:00")
                              + timedelta(minutes=minutes), path=beat_path)
        assert w >= LOCK_WINDOW_HOURS


def test_clock_skew_falls_back_to_the_floor(beat_path):
    """A heartbeat from the future must not produce a negative window."""
    lockwindow.beat(now=_at("2026-08-28T20:00:00+00:00"), path=beat_path)
    assert lockwindow.window(now=_at("2026-08-28T15:00:00+00:00"),
                             path=beat_path) == LOCK_WINDOW_HOURS


def test_a_corrupt_heartbeat_degrades_to_the_floor(beat_path):
    """A broken file must never widen the window, and never crash a locker."""
    beat_path.write_text("{ not json", encoding="utf-8")
    assert lockwindow.window(path=beat_path) == LOCK_WINDOW_HOURS


def test_the_beat_round_trips(beat_path):
    when = _at("2026-08-29T13:45:00+00:00")
    lockwindow.beat(now=when, path=beat_path)
    assert lockwindow.last_run(beat_path) == when


def test_widening_never_relaxes_the_late_lock_rule():
    """The window governs how EARLY a pick may freeze. Freezing after kickoff is
    still tainted and still void -- that rule is not touched here."""
    from leagues.picks import LATE_LOCK_HOURS
    assert LATE_LOCK_HOURS == 0.0


# --- publishing the gap -------------------------------------------------------

def test_unrecorded_lists_played_fixtures_with_no_frozen_pick():
    """The four that vanished on 2026-08-28 must appear, not disappear."""
    import pandas as pd
    from leagues import publish
    played = pd.DataFrame([
        {"match_id": 1, "date": "2026-08-28T18:30:00+00:00", "home": "Bayern Munich",
         "away": "Stuttgart", "home_goals": 5, "away_goals": 1},
        {"match_id": 2, "date": "2026-08-22T14:00:00+00:00", "home": "Everton",
         "away": "Crystal Palace", "home_goals": 2, "away_goals": 0},
    ])
    log = {"2026:2": {"pick": "Everton", "graded": "correct"}}
    out = publish.unrecorded_fixtures(played, log, lambda mid: f"2026:{mid}")
    assert len(out) == 1
    assert out[0]["home"] == "Bayern Munich" and out[0]["score"] == "5-1"


def test_unrecorded_is_empty_when_every_played_fixture_was_frozen():
    import pandas as pd
    from leagues import publish
    played = pd.DataFrame([
        {"match_id": 2, "date": "2026-08-22T14:00:00+00:00", "home": "Everton",
         "away": "Crystal Palace", "home_goals": 2, "away_goals": 0}])
    log = {"2026:2": {"pick": "Everton", "graded": "correct"}}
    assert publish.unrecorded_fixtures(played, log, lambda mid: f"2026:{mid}") == []
