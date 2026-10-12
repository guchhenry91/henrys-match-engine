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
    report = {"ok": True, "day": "2026-10-20", "slot": "05_30PM", "out": ["Tatum, Jayson"]}
    monkeypatch.setattr(injury_report, "fetch_today", lambda season, now: dict(report, ok=report_ok))

    def fake_build(sched, now, released, ruled_out=(), vacated_out=None):
        seen["ruled_out"], seen["vacated_out"] = list(ruled_out), vacated_out
        return {"games": [], "props": {}, "held_back": {}, "odds_checked_at": None}
    monkeypatch.setattr(publish.board, "build", fake_build)
    sched = pd.DataFrame({"stage": [], "status": []})
    seen["payload"] = publish.build(sched=sched)
    return seen


report_ok = True


def test_report_outs_join_the_news_when_teammates_out_is_on(monkeypatch):
    seen = _run(monkeypatch, True)
    assert seen["ruled_out"] == ["Joel Embiid", "Jayson Tatum"]
    # teammates-out reads the REPORT only: training never saw the news file
    assert seen["vacated_out"] == ["Jayson Tatum"]
    assert seen["payload"]["injury_report"]["teammates_out"] is True


def test_no_report_fetch_while_it_is_off(monkeypatch):
    seen = _run(monkeypatch, False)
    assert seen["ruled_out"] == ["Joel Embiid"] and seen["vacated_out"] is None


def test_no_report_today_turns_teammates_out_off(monkeypatch):
    """Third review: a failed fetch must not read as 'nobody is out'."""
    global report_ok
    report_ok = False
    try:
        seen = _run(monkeypatch, True)
    finally:
        report_ok = True
    assert seen["vacated_out"] is None
    assert seen["payload"]["injury_report"]["teammates_out"] is False


def test_report_outs_map_names_to_ids(tmp_path):
    pd.DataFrame([{"date": "2026-01-15", "player": "Tatum, Jayson", "status": "Out"},
                  {"date": "2026-01-15", "player": "Brown, Jaylen", "status": "Questionable"}]).to_csv(
        tmp_path / "2026.csv.gz", index=False, compression="gzip")
    rows = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Jayson Tatum", "Jaylen Brown"]})
    assert fc.report_outs(rows, tmp_path) == {(pd.Timestamp("2026-01-15"), 1)}


def test_teammates_out_is_on_where_the_gate_passed():
    assert forecast_board.VACATED_STATS == {"PTS", "REB", "AST"}
    assert forecast_board.VACATED
