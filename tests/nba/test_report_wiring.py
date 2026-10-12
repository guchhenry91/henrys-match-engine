"""Teammates-out reads the official pre-game report live, as it was trained."""
import pandas as pd

from nba import forecast as fc
from nba import forecast_board, injury_report, publish


def _run(monkeypatch, vacated):
    seen = {}
    monkeypatch.setattr(forecast_board, "VACATED", vacated)
    monkeypatch.setattr(publish, "evidence", lambda: {})
    monkeypatch.setattr(publish, "load_player_news", lambda path: {"Joel Embiid": {"status": "out"}})
    monkeypatch.setattr(publish.news_edge, "remember", lambda path, d: {})
    monkeypatch.setattr(injury_report, "refresh", lambda season, now: ["Tatum, Jayson"])

    def fake_build(sched, now, released, ruled_out=()):
        seen["ruled_out"] = list(ruled_out)
        return {"games": [], "props": {}, "held_back": {}, "odds_checked_at": None}
    monkeypatch.setattr(publish.board, "build", fake_build)
    sched = pd.DataFrame({"stage": [], "status": []})
    publish.build(sched=sched)
    return seen["ruled_out"]


def test_report_outs_join_the_news_when_teammates_out_is_on(monkeypatch):
    assert _run(monkeypatch, True) == ["Joel Embiid", "Jayson Tatum"]


def test_no_report_fetch_while_it_is_off(monkeypatch):
    assert _run(monkeypatch, False) == ["Joel Embiid"]


def test_report_outs_map_names_to_ids(tmp_path):
    pd.DataFrame([{"date": "2026-01-15", "player": "Tatum, Jayson", "status": "Out"},
                  {"date": "2026-01-15", "player": "Brown, Jaylen", "status": "Questionable"}]).to_csv(
        tmp_path / "2026.csv.gz", index=False, compression="gzip")
    rows = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Jayson Tatum", "Jaylen Brown"]})
    assert fc.report_outs(rows, tmp_path) == {(pd.Timestamp("2026-01-15"), 1)}


def test_teammates_out_is_on_where_the_gate_passed():
    assert forecast_board.VACATED_STATS == {"PTS", "REB", "AST"}
    assert forecast_board.VACATED
