"""What is frozen with a pick at lock (leagues/lockstamp.py), via the fast locker.

Before 2026-10-04 the fast locker froze soccer picks with no price and no value
verdict, and release.mark then labelled every one of them "tracked" for good -- so
no soccer pick it froze could ever be an official bet.
"""
import json

import pandas as pd
import pytest

from leagues import lockstamp, odds, picks, prop_calibration, prop_odds, shot_odds
from scripts import lock_picks
from tracking import release, trust

NOW = pd.Timestamp("2026-10-04T12:00:00Z")
KICK = "2026-10-04T13:30:00Z"


def _market(book="bet365"):
    return pd.DataFrame([{"home": "Arsenal", "away": "Chelsea", "m_home": 0.60, "m_draw": 0.22,
                          "m_away": 0.18, "b_home": 1.80, "b_draw": 3.80, "b_away": 4.60, "book": book}])


@pytest.fixture
def board(tmp_path, monkeypatch):
    out, raw = tmp_path / "data", tmp_path / "raw"
    out.mkdir()
    (raw / "pl").mkdir(parents=True)
    monkeypatch.setattr(lock_picks, "OUT", out)
    monkeypatch.setattr(lock_picks, "PICKS_DIR", raw)
    monkeypatch.setattr(lock_picks, "FILES", {"PL": "pl"})
    monkeypatch.setattr(odds, "fetch_fixture_odds", lambda league: _market())
    # a fixed trust weight, so the test does not depend on the committed record
    monkeypatch.setattr(trust, "_SOCCER", {"winner": {"w": 0.2}, "prop": {"w": 0.25}})
    return out, raw


def _write_match(out, p=0.62, suspect=False):
    (out / "pl.json").write_text(json.dumps({"matches": [{
        "id": 7, "home": "Arsenal", "away": "Chelsea", "date": KICK, "time_suspect": suspect,
        "prediction": {"pick": "Arsenal", "p_pick": p, "confidence": 62, "best_pick": False}}]}))


def test_fast_locker_freezes_the_bet365_price_and_value(board):
    out, raw = board
    _write_match(out)
    lock_picks.lock_matches(NOW, window=2.0)
    log = picks.load_log(raw / "pl" / "picks_log.json")
    entry = next(v for k, v in log.items() if not k.startswith("_"))
    assert entry["odds"] == 1.80
    # judged on 0.60 + 0.2 x (0.62 - 0.60) = 0.604; x 1.80 - 1 = +8.7% at bet365: value
    assert entry["value"]["basis"] == "bet365" and entry["value"]["value"] is True
    assert "release" in entry           # decided AFTER the price was there


def test_value_pick_in_an_earned_market_is_official(board, monkeypatch):
    out, raw = board
    _write_match(out)
    monkeypatch.setattr(release, "status_map", lambda sport: {"match_winner": "promising"})
    lock_picks.lock_matches(NOW, window=2.0)
    log = picks.load_log(raw / "pl" / "picks_log.json")
    entry = next(v for k, v in log.items() if not k.startswith("_"))
    assert entry["release"] == "official"


def test_a_suspect_kickoff_is_never_locked(board):
    out, raw = board
    _write_match(out, suspect=True)
    assert lock_picks.lock_matches(NOW, window=2.0) == []


def test_no_price_means_no_stamp_but_still_a_verdict(board, monkeypatch):
    out, raw = board
    monkeypatch.setattr(odds, "fetch_fixture_odds", lambda league: pd.DataFrame())
    _write_match(out)
    lock_picks.lock_matches(NOW, window=2.0)
    entry = next(v for k, v in picks.load_log(raw / "pl" / "picks_log.json").items() if not k.startswith("_"))
    assert entry.get("odds") is None and entry["value"]["value"] is None


def test_a_prop_freezes_what_the_board_showed_and_the_raw_model(board, monkeypatch):
    out, raw = board
    monkeypatch.setattr(prop_odds, "load", lambda: {})
    monkeypatch.setattr(shot_odds, "load", lambda: {})
    (out / "player_picks.json").write_text(json.dumps({"upcoming": [
        {"league_key": "PL", "id": 7, "date": KICK, "market": "goal", "player": "Saka", "team": "Arsenal",
         "p_pick": 0.44, "p_model": 0.47, "confidence": 44},
        {"league_key": "PL", "id": 8, "date": KICK, "market": "goal", "player": "Palmer", "team": "Chelsea",
         "p_pick": 0.45, "p_model": 0.45, "confidence": 45, "time_suspect": True}]}))
    lock_picks.lock_players(NOW, window=2.0)
    log = picks.load_log(raw / "pl" / "player_picks_log.json")
    saka = next(v for v in log.values() if isinstance(v, dict) and v.get("player") == "Saka")
    assert (saka["p_pick"], saka["p_model"]) == (0.44, 0.47)
    assert not any(isinstance(v, dict) and v.get("player") == "Palmer" for v in log.values())


def test_a_frozen_prop_is_not_recalibrated_again_on_display():
    upcoming = [{"market": "goal", "p_pick": 0.44, "p_model": 0.47, "provisional": False},
                {"market": "goal", "p_pick": 0.47, "p_model": 0.47, "provisional": True}]
    settled = [{"market": "goal", "graded": "correct" if i % 3 == 0 else "wrong", "p_pick": 0.47}
               for i in range(400)]
    prop_calibration.apply(upcoming, settled)
    assert upcoming[0]["p_pick"] == 0.44          # frozen value kept
    assert upcoming[1]["p_pick"] != 0.47          # provisional one recalibrated


def test_stamp_only_on_the_run_that_locked():
    entry = {"pick": "Arsenal", "p_pick": 0.62, "locked_at": "2026-10-03T10:00:00Z"}
    lockstamp.stamp_match(entry, _market(), "Arsenal", "Chelsea", NOW)
    assert "odds" not in entry and "value" not in entry
