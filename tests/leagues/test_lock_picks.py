"""The fast locker must freeze the published pick, and nothing else.

Written against the path that does the work rather than the one that reports it.
An hour before this file existed I "verified" a UI change by calling the render
function directly, which bypassed the dispatch function where the bug actually
was, and shipped it broken. A locker that prints "nothing in the window" proves
only that nothing was in the window.
"""
import json

import pandas as pd
import pytest

from leagues import picks
from scripts import lock_picks


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A published payload with one fixture, and empty logs to write into."""
    out = tmp_path / "data"
    raw = tmp_path / "raw"
    (out).mkdir(parents=True)
    (raw / "pl").mkdir(parents=True)
    monkeypatch.setattr(lock_picks, "OUT", out)
    monkeypatch.setattr(lock_picks, "PICKS_DIR", raw)
    monkeypatch.setattr(lock_picks, "FILES", {"PL": "pl"})
    monkeypatch.setattr(lock_picks, "_season_tag", lambda league: "2026")
    return out, raw


def _write(out, kickoff, *, suspect=False, pick="Arsenal", p=0.72, best=True):
    (out / "pl.json").write_text(json.dumps({"matches": [{
        "id": 7, "home": "Arsenal", "away": "Chelsea",
        "date": pd.Timestamp(kickoff).isoformat(),
        "time_suspect": suspect,
        "prediction": {"pick": pick, "p_pick": p, "confidence": 5,
                       "best_pick": best},
    }]}), encoding="utf-8")


NOW = pd.Timestamp("2026-08-27T18:00:00Z")


def test_a_fixture_inside_the_window_is_frozen_with_the_published_number(board):
    out, raw = board
    _write(out, "2026-08-27T18:40:00Z")          # 40 minutes out
    frozen = lock_picks.lock_matches(NOW)
    assert len(frozen) == 1
    log = picks.load_log(raw / "pl" / "picks_log.json")
    entry = log["2026:7"]
    assert entry["pick"] == "Arsenal"
    assert entry["p_pick"] == 0.72               # the PUBLISHED number, not a new one
    assert entry["board"] is True
    assert entry["tainted"] is False


def test_a_fixture_beyond_the_window_is_left_alone(board):
    out, raw = board
    _write(out, "2026-08-27T23:00:00Z")          # 5 hours out
    assert lock_picks.lock_matches(NOW) == []
    assert picks.load_log(raw / "pl" / "picks_log.json") == {}


def test_a_fixture_that_has_kicked_off_is_never_frozen(board):
    """Locking after kickoff is the failure this whole job exists to prevent."""
    out, raw = board
    _write(out, "2026-08-27T17:45:00Z")          # started 15 minutes ago
    assert lock_picks.lock_matches(NOW) == []
    assert picks.load_log(raw / "pl" / "picks_log.json") == {}


def test_a_suspect_kickoff_time_is_never_frozen(board):
    """Freezing against a time the feed does not believe is what
    release_moved_lock exists to undo. Do not create the problem here."""
    out, raw = board
    _write(out, "2026-08-27T18:40:00Z", suspect=True)
    assert lock_picks.lock_matches(NOW) == []


def test_running_again_does_not_overwrite_an_earlier_lock(board):
    """It runs every five minutes. A second pass must never move locked_at
    forward -- that is precisely how a good pick becomes a late one."""
    out, raw = board
    _write(out, "2026-08-27T18:40:00Z")
    lock_picks.lock_matches(NOW)
    first = picks.load_log(raw / "pl" / "picks_log.json")["2026:7"]["locked_at"]

    later = NOW + pd.Timedelta(minutes=35)       # now only 5 minutes out
    assert lock_picks.lock_matches(later) == []
    again = picks.load_log(raw / "pl" / "picks_log.json")["2026:7"]["locked_at"]
    assert again == first


def test_it_never_invents_a_pick_the_board_did_not_show(board):
    """It computes nothing. A payload with no probability yields no lock."""
    out, raw = board
    (out / "pl.json").write_text(json.dumps({"matches": [{
        "id": 7, "home": "Arsenal", "away": "Chelsea",
        "date": "2026-08-27T18:40:00Z", "time_suspect": False,
        "prediction": {"pick": "Arsenal", "p_pick": None, "confidence": 5},
    }]}), encoding="utf-8")
    assert lock_picks.lock_matches(NOW) == []


def test_a_missing_payload_is_survivable(board):
    """A league that failed to publish must not stop the others locking."""
    out, raw = board
    assert lock_picks.lock_matches(NOW) == []


def test_the_window_it_uses_is_the_one_publish_uses(board):
    """Two different windows would mean two different answers to the same
    question, and the one that fired first would silently win."""
    from leagues import publish
    assert lock_picks.LOCK_WINDOW_HOURS == publish.LOCK_WINDOW_HOURS
