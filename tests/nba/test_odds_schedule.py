"""The NBA schedule from The Odds API, used while the NBA's CDN refuses
(nba/current.odds_schedule, scripts/sync_nba_odds.refresh_odds_schedule)."""
import urllib.error

import pandas as pd
import pytest

from nba import book_lines as bl
from nba import current


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "ODDS_SCHEDULE", tmp_path / "odds_schedule.json")


EVENTS = [
    {"id": "aaa", "commence_time": "2026-10-20T23:30:00Z", "home_team": "Detroit Pistons", "away_team": "Boston Celtics"},
    {"id": "bbb", "commence_time": "2026-10-21T02:00:00Z", "home_team": "Golden State Warriors", "away_team": "Los Angeles Lakers"},
    {"id": "pre", "commence_time": "2026-10-15T23:00:00Z", "home_team": "Utah Jazz", "away_team": "Phoenix Suns"},
]
NOW = pd.Timestamp("2026-10-20T12:00:00Z")


def test_rows_have_the_cdn_shape_and_tricodes():
    current.update_odds_schedule(EVENTS, [], bl.TEAM_CODES, NOW)
    df = current.odds_schedule().set_index("home_team")
    assert list(current.odds_schedule().columns) == current.SCHED_COLUMNS
    assert df.loc["GSW", "away_team"] == "LAL"
    # 02:00 UTC on the 21st is the evening of the 20th in New York
    assert df.loc["GSW", "game_date"] == "2026-10-20"
    assert df.loc["DET", "stage"] == current.REGULAR and df.loc["UTA", "stage"] == current.PRESEASON
    assert (df["status"] == 1).sum() == 2           # the preseason game is already past
    assert all(str(g).startswith("0029") and str(g).isdigit() for g in df["game_id"])


def test_ids_never_change_between_runs():
    current.update_odds_schedule(EVENTS[:1], [], bl.TEAM_CODES, NOW)
    first = current.odds_schedule().iloc[0]["game_id"]
    current.update_odds_schedule(list(reversed(EVENTS)), [], bl.TEAM_CODES, NOW)
    df = current.odds_schedule()
    assert df[df["home_team"] == "DET"].iloc[0]["game_id"] == first
    assert df["game_id"].is_unique


def test_scores_mark_a_game_final():
    current.update_odds_schedule(EVENTS, [], bl.TEAM_CODES, NOW)
    scores = [{**EVENTS[0], "completed": True,
               "scores": [{"name": "Detroit Pistons", "score": "112"}, {"name": "Boston Celtics", "score": "108"}]}]
    current.update_odds_schedule([], scores, bl.TEAM_CODES, pd.Timestamp("2026-10-21T06:00:00Z"))
    row = current.odds_schedule().set_index("home_team").loc["DET"]
    assert (row["status"], row["home_score"], row["away_score"]) == (current.FINAL, 112, 108)


def test_schedule_falls_back_when_the_cdn_refuses():
    current.update_odds_schedule(EVENTS, [], bl.TEAM_CODES, NOW)

    def refused(*a, **k):
        raise urllib.error.HTTPError("u", 403, "Access Denied", {}, None)
    assert len(current.schedule(opener=refused)) == 3


def test_no_fallback_still_raises():
    def refused(*a, **k):
        raise urllib.error.HTTPError("u", 403, "Access Denied", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        current.schedule(opener=refused)


def test_box_sync_never_asks_the_cdn_for_odds_api_games(monkeypatch, tmp_path):
    monkeypatch.setattr(current, "DIR", tmp_path)
    sched = pd.DataFrame([{"game_id": "0029000001", "stage": current.REGULAR, "status": current.FINAL,
                           "tipoff": "2026-10-20T23:30:00Z", "game_date": "2026-10-20"}])
    asked = []
    current.sync(sched, opener=lambda *a, **k: asked.append(1), sleeper=lambda s: None)
    assert asked == []
