

# --- freshness vs a match in play -------------------------------------------
# On 2026-08-23 the freshness rule blocked every publish and deploy for the
# duration of Rennes v Paris SG. It was the last fixture of Ligue 1's matchweek,
# so the published window held exactly one match; the moment it kicked off the
# file had zero FUTURE fixtures and was branded a stale snapshot. It was 27
# minutes old.

import datetime as dt


def _current(dates, now, in_play_hours=3.5):
    """The rule as it now stands: a fixture counts while it could still be on."""
    return sum(1 for k in dates if k > now - dt.timedelta(hours=in_play_hours))


NOW = dt.datetime(2026, 8, 23, 19, 12, tzinfo=dt.timezone.utc)


def test_a_kicked_off_match_still_counts_as_current():
    kickoff = dt.datetime(2026, 8, 23, 18, 45, tzinfo=dt.timezone.utc)
    assert kickoff < NOW                      # already started
    assert _current([kickoff], NOW) == 1      # ...but not stale


def test_last_weeks_leftovers_are_still_caught():
    """The rule must still do the job it was written for."""
    old = dt.datetime(2026, 8, 16, 18, 45, tzinfo=dt.timezone.utc)
    assert _current([old], NOW) == 0


def test_a_match_finished_hours_ago_is_stale():
    finished = dt.datetime(2026, 8, 23, 12, 0, tzinfo=dt.timezone.utc)
    assert _current([finished], NOW) == 0


def test_a_future_fixture_obviously_counts():
    later = dt.datetime(2026, 8, 24, 18, 45, tzinfo=dt.timezone.utc)
    assert _current([later], NOW) == 1


# --- a frozen pick may trail the fresh argmax -----------------------------------
# On 2026-09-19 Newcastle v Hull was frozen as Hull (34.7% vs a 34.3% draw). A
# later refit put the draw a fraction ahead, and the "pick must be the most likely
# outcome" rule failed the whole refresh over a pick that could no longer change.

import json

import pytest

from scripts import sanity_check


def _league(tmp_path, monkeypatch, provisional):
    match = {"home": "Newcastle United", "away": "Hull", "date": "2026-09-19T14:00:00+00:00",
             "prediction": {"p_home": 0.31, "p_draw": 0.346, "p_away": 0.344,
                            "pick": "Hull", "pick_type": "away",
                            "provisional": provisional}}
    payload = {"matches": [match], "table": [], "backtest": {"rps": 0.2}}
    monkeypatch.setattr(sanity_check, "load", lambda path: json.loads(json.dumps(payload)))
    failures = []
    monkeypatch.setattr(sanity_check, "fail", lambda where, msg: failures.append(msg))
    monkeypatch.setattr(sanity_check, "warn", lambda where, msg: None)
    try:
        sanity_check.check_league("pl", "PL", 0, 0)
    except Exception:
        pass                      # only the argmax rule is under test here
    return [f for f in failures if "most likely" in f]


def test_a_frozen_pick_may_trail_the_fresh_probabilities(tmp_path, monkeypatch):
    assert _league(tmp_path, monkeypatch, provisional=False) == []


def test_a_provisional_pick_must_still_be_the_most_likely(tmp_path, monkeypatch):
    assert _league(tmp_path, monkeypatch, provisional=True)
