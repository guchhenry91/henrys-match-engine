"""Grading finished NBA games between NBA runs, from the free stats.nba.com log."""
import json

import pandas as pd

from nba import current
from nba import picks as nba_picks
from scripts import grade_nba

HEAD = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
        "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "FG3M",
        "FGA", "FTA", "FG3A", "TOV", "OREB", "DREB"]


def _row(pid, team, opp, home, pts, reb=5):
    m = f"{team} vs. {opp}" if home else f"{team} @ {opp}"
    return ["22026", pid, f"P{pid}", 1, team, "0022600099", "2026-10-22", m, "W", 34,
            pts, reb, 3, 1, 15, 4, 5, 2, 1, 4]


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "DIR", tmp_path / "current")
    monkeypatch.setattr(grade_nba, "BOARD", tmp_path / "board.json")
    monkeypatch.setattr(grade_nba, "ATTEMPT", tmp_path / "attempt.json")
    log_path = tmp_path / "picks_log.json"
    monkeypatch.setattr(nba_picks.load_log, "__defaults__", (log_path,))
    monkeypatch.setattr(nba_picks.save_log, "__defaults__", (log_path,))
    (tmp_path / "board.json").write_text(json.dumps({"record": {}}))
    log = {
        "0029000020:winner": {"kind": "winner", "game_id": "0029000020", "tipoff": "2026-10-22T23:10:00Z",
                              "home": "PHI", "away": "CLE", "pick": "PHI", "p_pick": 0.55, "locked_at": "x"},
        "0029000020:points:1": {"kind": "prop", "game_id": "0029000020", "tipoff": "2026-10-22T23:10:00Z",
                                "market": "points", "player": "P1", "player_id": 1, "team": "PHI",
                                "opponent": "CLE", "line": 20.5, "side": "under", "probability": 0.56,
                                "engine": "forecast-v1", "locked_at": "x"},
        "0029000021:points:9": {"kind": "prop", "game_id": "0029000021", "tipoff": "2026-10-23T01:40:00Z",
                                "market": "points", "player": "P9", "player_id": 9, "team": "OKC",
                                "opponent": "DEN", "line": 25.5, "side": "over", "probability": 0.56,
                                "engine": "forecast-v1", "locked_at": "x"}}
    log_path.write_text(json.dumps(log))
    return log_path


def test_a_finished_game_grades_from_the_free_log_and_an_unfiled_one_waits(tmp_path, monkeypatch):
    log_path = _setup(tmp_path, monkeypatch)
    rows = [_row(1, "PHI", "CLE", True, 18), _row(2, "PHI", "CLE", True, 90),
            _row(3, "CLE", "PHI", False, 100),                     # PHI 108 - CLE 100
            _row(7, "BOS", "NYK", True, 30)]                       # another game: not kept
    fetch = lambda season_type: {"resultSets": [{"headers": HEAD, "rowSet": rows}]}
    grade_nba.main(now=pd.Timestamp("2026-10-23T06:30:00Z"), fetch=fetch)
    log = json.loads(log_path.read_text())
    assert log["0029000020:points:1"]["graded"] == "correct"        # 18 under 20.5
    assert log["0029000020:winner"]["graded"] == "correct"          # PHI won
    assert log["0029000021:points:9"].get("graded") is None         # not in the log yet: pending
    kept = current.player_games()
    assert set(kept["GAME_ID"].astype(str)) == {"0029000020"}       # no stray NBA-id copies
    board = json.loads((tmp_path / "board.json").read_text())
    assert board["record"]["props"]["correct"] == 1 and board["record"]["props"]["pending"] == 1


def test_nothing_is_fetched_before_a_game_can_be_over(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    def boom(_):
        raise AssertionError("fetched too early")
    grade_nba.main(now=pd.Timestamp("2026-10-23T01:00:00Z"), fetch=boom)   # 1h50 after the first tip
