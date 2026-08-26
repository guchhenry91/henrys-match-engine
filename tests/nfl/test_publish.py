"""The board must never publish something it cannot stand behind."""
import json

import pandas as pd
import pytest

from nfl import publish


def test_a_withheld_market_publishes_no_picks(monkeypatch):
    monkeypatch.setattr(publish, "_released", lambda: {"team_winner"})
    assert "anytime_touchdown" not in publish._released()


def test_evidence_survives_a_missing_report(monkeypatch, tmp_path):
    monkeypatch.setattr(publish, "REPORT", tmp_path / "absent.json")
    assert publish._evidence() == {}
    assert publish._released() == set()


def test_evidence_survives_a_corrupt_report(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(publish, "REPORT", bad)
    assert publish._evidence() == {}


def test_upcoming_is_the_earliest_unplayed_week_only():
    schedule = pd.DataFrame([
        {"season": 2026, "week": 1, "gameday": pd.Timestamp("2026-09-10"),
         "home_team": "AAA", "away_team": "BBB", "played": False},
        {"season": 2026, "week": 1, "gameday": pd.Timestamp("2026-09-13"),
         "home_team": "CCC", "away_team": "DDD", "played": False},
        {"season": 2026, "week": 2, "gameday": pd.Timestamp("2026-09-17"),
         "home_team": "EEE", "away_team": "FFF", "played": False},
        {"season": 2025, "week": 18, "gameday": pd.Timestamp("2026-01-04"),
         "home_team": "GGG", "away_team": "HHH", "played": True},
    ])
    out = publish.upcoming_games(schedule)
    assert list(out["week"].unique()) == [1]
    assert len(out) == 2
    assert out.iloc[0]["home_team"] == "AAA", "not ordered by kickoff"


def test_an_all_played_schedule_yields_nothing_rather_than_crashing():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 18, "gameday": pd.Timestamp("2026-01-04"),
         "home_team": "AAA", "away_team": "BBB", "played": True}])
    assert publish.upcoming_games(schedule).empty


def test_the_published_board_matches_what_the_gate_measured():
    """A market on the board must be one the release file actually released."""
    from pathlib import Path
    board = Path("data/nfl/board.json")
    report = Path("data-raw/nfl/backtest_report.json")
    if not board.exists() or not report.exists():
        pytest.skip("board has not been built in this checkout")
    published = json.loads(board.read_text(encoding="utf-8"))
    released = set(json.loads(report.read_text(encoding="utf-8"))["released_markets"])
    for market, block in published["props"].items():
        if block["picks"]:
            assert market in released, f"{market} has picks but was never released"
        assert block["released"] == (market in released)


def test_every_pick_carries_its_last_five():
    from pathlib import Path
    board = Path("data/nfl/board.json")
    if not board.exists():
        pytest.skip("board has not been built in this checkout")
    published = json.loads(board.read_text(encoding="utf-8"))
    for market, block in published["props"].items():
        for pick in block["picks"]:
            assert "last_five" in pick, f"{market}/{pick['player']} has no last five"
            assert len(pick["last_five"]) <= 5
            if pick["last_five"]:
                assert pick["last_five_average"] == pytest.approx(
                    sum(pick["last_five"]) / len(pick["last_five"]), abs=0.05)
