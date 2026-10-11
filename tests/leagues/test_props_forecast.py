import pandas as pd

from leagues import forecast as fc
from leagues import forecast_board as fb


def _players():
    base = {"match_id": 1, "datetime": "2025-08-15 19:00:00", "home": "A", "away": "B",
            "home_goals": 1, "away_goals": 0, "home_xg": 1.5, "away_xg": 0.7,
            "own_goals": 0, "yellow_card": 0, "red_card": 0, "roster_in": 0, "roster_out": 0,
            "key_passes": 1, "assists": 0, "xA": 0.1, "xGChain": 0, "xGBuildup": 0}
    return pd.DataFrame([
        {**base, "id": 1, "goals": 1, "shots": 3, "xG": 0.9, "time": 90, "player_id": 10,
         "team_id": 1, "position": "FW", "player": "Striker", "h_a": "h", "positionOrder": 15},
        {**base, "id": 2, "goals": 0, "shots": 0, "xG": 0.0, "time": 15, "player_id": 11,
         "team_id": 1, "position": "Sub", "player": "Benchman", "h_a": "h", "positionOrder": 17},
        {**base, "id": 3, "goals": 0, "shots": 1, "xG": 0.1, "time": 90, "player_id": 20,
         "team_id": 2, "position": "MC", "player": "Mid", "h_a": "a", "positionOrder": 9}])


def _shots():
    return pd.DataFrame([
        {"match_id": 1, "player_id": 10, "result": "Goal", "situation": "Penalty", "xG": 0.76},
        {"match_id": 1, "player_id": 10, "result": "SavedShot", "situation": "OpenPlay", "xG": 0.1},
        {"match_id": 1, "player_id": 10, "result": "BlockedShot", "situation": "OpenPlay", "xG": 0.04},
        {"match_id": 1, "player_id": 20, "result": "OwnGoal", "situation": "OpenPlay", "xG": 0.0},
        {"match_id": 1, "player_id": 20, "result": "MissedShots", "situation": "OpenPlay", "xG": 0.1}])


def test_trim_counts_on_target_penalties_and_roles():
    t = fc.trim(_players(), _shots(), "PL", 2025).set_index("player")
    assert t.loc["Striker", "sot"] == 2 and t.loc["Striker", "pens"] == 1
    assert t.loc["Striker", "pen_goals"] == 1
    assert abs(t.loc["Striker", "npxg"] - 0.14) < 1e-9          # penalty xG left out
    assert t.loc["Benchman", "started"] == 0 and t.loc["Striker", "started"] == 1
    assert t.loc["Mid", "team"] == "B" and t.loc["Mid", "is_home"] == 0
    assert t.loc["Mid", "sot"] == 0                               # own goals are not his shots
    assert t.loc["Striker", "date"] == "2025-08-15"


def test_apply_replaces_numbers_and_drops_unread_players():
    squad = [{"team": "A", "player": "Striker", "anytime_pct": 60.0, "p_shots_2plus": 90.0,
              "p_sot_1plus": 80.0, "appearance_pct": 100.0, "expected_minutes": 90.0,
              "penalty_taker": False, "exp_shots": 3.0, "exp_sot": 1.0, "position": "FW"},
             {"team": "A", "player": "New Signing", "anytime_pct": 30.0}]
    fcast = [{"team": "A", "player": "Striker", "anytime_pct": 41.0, "p_shots_2plus": 63.0,
              "p_sot_1plus": 55.0, "appearance_pct": 92.0, "expected_minutes": 75.0,
              "penalty_taker": True, "start_pct": 85.0, "engine": fb.ENGINE}]
    out = fb.apply(squad, fcast)
    assert len(out) == 1 and out[0]["anytime_pct"] == 41.0 and out[0]["engine"] == fb.ENGINE
    assert out[0]["exp_shots"] is None and out[0]["position"] == "FW"


def test_match_forecast_none_when_model_not_prepared(monkeypatch):
    monkeypatch.setattr(fb, "_STATE", {})
    assert fb.match_forecast("PL", "A", "B", "2026-10-12 19:00") is None
