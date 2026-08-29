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
         "game_id": "2026_01_BBB_AAA",
         "kickoff": pd.Timestamp("2026-09-11T00:20:00Z"),
         "home_team": "AAA", "away_team": "BBB", "played": False},
        {"season": 2026, "week": 1, "gameday": pd.Timestamp("2026-09-13"),
         "game_id": "2026_01_DDD_CCC",
         "kickoff": pd.Timestamp("2026-09-13T17:00:00Z"),
         "home_team": "CCC", "away_team": "DDD", "played": False},
        {"season": 2026, "week": 2, "gameday": pd.Timestamp("2026-09-17"),
         "game_id": "2026_02_FFF_EEE",
         "kickoff": pd.Timestamp("2026-09-17T17:00:00Z"),
         "home_team": "EEE", "away_team": "FFF", "played": False},
        {"season": 2025, "week": 18, "gameday": pd.Timestamp("2026-01-04"),
         "game_id": "2025_18_HHH_GGG",
         "kickoff": pd.Timestamp("2026-01-04T18:00:00Z"),
         "home_team": "GGG", "away_team": "HHH", "played": True},
    ])
    out = publish.upcoming_games(schedule)
    assert list(out["week"].unique()) == [1]
    assert len(out) == 2
    assert out.iloc[0]["home_team"] == "AAA", "not ordered by kickoff"


def test_an_all_played_schedule_yields_nothing_rather_than_crashing():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 18, "gameday": pd.Timestamp("2026-01-04"),
         "game_id": "2025_18_BBB_AAA",
         "kickoff": pd.Timestamp("2026-01-04T18:00:00Z"),
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


# --- availability -------------------------------------------------------------

def test_a_ruled_out_player_is_removed_from_the_board(monkeypatch, tmp_path):
    """His last five look exactly as good as anyone's right up until he is
    inactive, which is why publishing him is the most misleading thing here."""
    import pandas as pd
    from nfl import features, model

    weeks = pd.DataFrame([{
        "player_id": "P1", "player_display_name": "Ruled Out", "position": "WR",
        "team": "AAA", "season": 2025, "week": w, "season_type": "REG",
        "opponent_team": "ZZZ", "passing_yards": 0.0, "rushing_yards": 0.0,
        "receiving_yards": 60.0, "receptions": 5.0, "carries": 0.0, "targets": 8.0,
        "attempts": 0.0, "completions": 0.0, "passing_tds": 0.0,
        "rushing_tds": 0.0, "receiving_tds": 0.0, "touchdowns": 0, "touches": 5.0,
    } for w in range(1, 14)])
    upcoming = pd.DataFrame([{"season": 2026, "week": 1,
                              "gameday": pd.Timestamp("2026-09-13"),
                              # Every prop card carries the id it will be frozen
                              # and settled under, and the REAL kickoff rather
                              # than the bare date the board used to publish.
                              "game_id": "2026_01_BBB_AAA",
                              "kickoff": pd.Timestamp("2026-09-13T17:00:00Z"),
                              "home_team": "AAA", "away_team": "BBB"}])

    monkeypatch.setattr(model.PropModel, "fit", lambda self, f: self)
    monkeypatch.setattr(model.PropModel, "predict",
                        lambda self, f: [0.9] * len(f))

    healthy = publish.player_projections(weeks, None, "receiving_yards", upcoming,
                                         injuries={})
    assert [p["player"] for p in healthy] == ["Ruled Out"]
    assert healthy[0]["availability"] == "not reported"

    gone = publish.player_projections(weeks, None, "receiving_yards", upcoming,
                                      injuries={"Ruled Out": {"status": "out"}})
    assert gone == [], "a player who will not dress was still published"


def test_a_doubtful_player_stays_but_carries_the_flag(monkeypatch):
    """Dropping him would hide a real pick; hiding the doubt would mislead."""
    import pandas as pd
    from nfl import model

    weeks = pd.DataFrame([{
        "player_id": "P1", "player_display_name": "Doubtful Man", "position": "WR",
        "team": "AAA", "season": 2025, "week": w, "season_type": "REG",
        "opponent_team": "ZZZ", "passing_yards": 0.0, "rushing_yards": 0.0,
        "receiving_yards": 60.0, "receptions": 5.0, "carries": 0.0, "targets": 8.0,
        "attempts": 0.0, "completions": 0.0, "passing_tds": 0.0,
        "rushing_tds": 0.0, "receiving_tds": 0.0, "touchdowns": 0, "touches": 5.0,
    } for w in range(1, 14)])
    upcoming = pd.DataFrame([{"season": 2026, "week": 1,
                              "gameday": pd.Timestamp("2026-09-13"),
                              # Every prop card carries the id it will be frozen
                              # and settled under, and the REAL kickoff rather
                              # than the bare date the board used to publish.
                              "game_id": "2026_01_BBB_AAA",
                              "kickoff": pd.Timestamp("2026-09-13T17:00:00Z"),
                              "home_team": "AAA", "away_team": "BBB"}])
    monkeypatch.setattr(model.PropModel, "fit", lambda self, f: self)
    monkeypatch.setattr(model.PropModel, "predict", lambda self, f: [0.9] * len(f))

    out = publish.player_projections(
        weeks, None, "receiving_yards", upcoming,
        injuries={"Doubtful Man": {"status": "doubt", "detail": "hamstring"}})
    assert len(out) == 1
    assert out[0]["availability"] == "doubt"
    assert out[0]["injury_note"] == "hamstring"


def test_a_missing_injury_file_means_not_reported_never_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(publish, "ROOT", tmp_path)
    assert publish.availability() == {}
