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


def test_second_opinion_logs_and_stays_alone_until_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "WEIGHTS", tmp_path / "w.json")
    monkeypatch.setattr(fb, "PRICED", [])
    rows = [{"player": "Striker", "player_id": 10, "anytime_pct": 45.0, "understat_match": 99},
            {"player": "Unpriced", "player_id": 11, "anytime_pct": 20.0, "understat_match": 99}]
    fb.second_opinion(rows, lambda n: 2.0 if n == "Striker" else None, "PL", "2026-10-12T19:00:00+00:00")
    assert rows[0]["b365_p"] == 0.5 and rows[0]["anytime_pct"] == 45.0     # forecast alone
    assert len(fb.PRICED) == 1 and fb.PRICED[0]["game_id"] == "PL:99"
    import json
    (tmp_path / "w.json").write_text(json.dumps({"goal": {"w": [0.5, 0.5, 0.0]}}))
    fb.second_opinion(rows[:1], lambda n: 2.0, "PL", "2026-10-12T19:00:00+00:00")
    assert rows[0]["combined_with_bet365"] and 40 < rows[0]["anytime_pct"] < 50


def test_grade_void_when_player_absent(tmp_path):
    import json
    from datetime import datetime, timezone
    from tracking import combine_fit
    log = tmp_path / "l.json"
    log.write_text(json.dumps({"a": {"line": 0.5, "y": None, "player_id": 1},
                               "b": {"line": 0.5, "y": None, "player_id": 2}}))
    combine_fit.grade(log, lambda e: "void" if e["player_id"] == 1 else 1.0)
    out = json.loads(log.read_text())
    assert out["a"]["y"] == "void" and out["b"]["y"] == 1


def test_did_not_play_is_void():
    from leagues import picks
    e = {"market": "goal", "player": "X"}
    assert picks.void_did_not_play(e)["graded"] == "void"
    assert picks.grade_prop(e, {"goals": 0, "minutes": 0})["graded"] == "void"
    assert picks.grade_prop(e, None)["graded"] == "wrong"          # played, no shot event
