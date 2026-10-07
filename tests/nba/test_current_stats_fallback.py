"""stats.nba.com fills the box scores the NBA's CDN will not serve (nba.current)."""
import pandas as pd

from nba import current

HEAD = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
        "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "FG3M"]


def _log(rows):
    return {"resultSets": [{"headers": HEAD, "rowSet": rows}]}


def test_preseason_rows_are_rekeyed_to_the_schedule_game_and_dnp_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "DIR", tmp_path)
    sched = pd.DataFrame([{"game_id": "0029000001", "stage": current.PRESEASON, "tipoff": "x",
                           "game_date": "2026-10-04", "home_team": "BOS", "away_team": "NYK",
                           "home_score": 110, "away_score": 100, "status": current.FINAL,
                           "neutral": False}])
    rows = [["22026", 1628369, "Jayson Tatum", 1, "BOS", 12600003, "2026-10-04", "BOS vs. NYK",
             "W", 28, 25, 7, 4, 3],
            ["22026", 1, "Bench Guy", 1, "BOS", 12600003, "2026-10-04", "BOS vs. NYK",
             "W", 0, 0, 0, 0, 0],
            ["22026", 2, "Elsewhere", 2, "LAL", 12600004, "2026-10-04", "LAL @ GSW",
             "L", 30, 20, 5, 5, 2]]
    asked = []
    out = current.sync(sched, opener=lambda *a, **k: None, sleeper=lambda s: None,
                       fetch=lambda st: asked.append(st) or _log(rows), today="2026-10-21")
    assert asked == ["Pre Season"] and out["preseason_games_added"] == 2
    got = current._load(current.PRESEASON)
    tatum = got[got["PLAYER_NAME"] == "Jayson Tatum"].iloc[0]
    assert tatum["GAME_ID"] == "0029000001"                 # the id picks grade on
    assert "Bench Guy" not in set(got["PLAYER_NAME"])       # did not play
    assert set(got["GAME_ID"]) == {"0029000001", "0012600004"}
    assert current.current_teams()[1628369] == "BOS"
    # Covered now: a second run asks nothing more.
    current.sync(sched, opener=lambda *a, **k: None, sleeper=lambda s: None,
                 fetch=lambda st: asked.append(st) or _log(rows), today="2026-10-21")
    assert asked == ["Pre Season"]


def test_a_stand_in_opener_never_reaches_the_network(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "DIR", tmp_path)
    sched = pd.DataFrame([{"game_id": "0029000001", "stage": current.REGULAR, "tipoff": "x",
                           "game_date": "2026-10-21", "home_team": "BOS", "away_team": "NYK",
                           "home_score": 1, "away_score": 0, "status": current.FINAL,
                           "neutral": False}])
    from nba import data
    monkeypatch.setattr(data, "_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    current.sync(sched, opener=lambda *a, **k: None, sleeper=lambda s: None)


def test_preseason_is_asked_for_even_when_the_schedule_lists_none(tmp_path, monkeypatch):
    """The Odds API schedule carries regular-season games only; before opening
    night the preseason log is fetched anyway, keeping the NBA's own ids."""
    monkeypatch.setattr(current, "DIR", tmp_path)
    sched = pd.DataFrame([{"game_id": "0029000001", "stage": current.REGULAR, "tipoff": "x",
                           "game_date": "2026-10-20", "home_team": "BOS", "away_team": "NYK",
                           "home_score": None, "away_score": None, "status": 1,
                           "neutral": False}])
    rows = [["22026", 1628369, "Jayson Tatum", 1, "BOS", 12600003, "2026-10-04",
             "BOS vs. NYK", "W", 28, 25, 7, 4, 3]]
    asked = []
    out = current.sync(sched, opener=lambda *a, **k: None, sleeper=lambda s: None,
                       fetch=lambda st: asked.append(st) or _log(rows), today="2026-10-06")
    assert asked == ["Pre Season"] and out["preseason_games_added"] == 1
    assert current._load(current.PRESEASON)["GAME_ID"].tolist() == ["0012600003"]
    assert current.current_teams()[1628369] == "BOS"
    # From opening night on, an empty preseason schedule asks nothing.
    current.sync(sched, opener=lambda *a, **k: None, sleeper=lambda s: None,
                 fetch=lambda st: asked.append(st) or _log(rows), today="2026-10-20")
    assert asked == ["Pre Season"]


class _Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def test_league_log_asks_the_stats_proxy_first(monkeypatch):
    import json
    monkeypatch.setenv("STATE_URL", "https://worker.example/")
    seen = []

    def opener(req, timeout):
        seen.append(req.full_url)
        return _Reply(json.dumps(_log([])).encode())
    out = current.league_log("Pre Season", opener=opener)
    assert out["resultSets"][0]["rowSet"] == []
    assert seen == ["https://worker.example/api/nba/leaguegamelog?season=2026-27"
                    "&season_type=Pre+Season"]


def test_league_log_falls_back_to_stats_nba_com(monkeypatch):
    from nba import data
    asked = []
    monkeypatch.setattr(data, "_fetch", lambda *a, **k: asked.append(k) or _log([]))

    def opener(req, timeout):
        return _Reply(b'{"error": "Not Found"}')
    current.league_log("Pre Season", opener=opener)
    assert asked and asked[0]["season_type"] == "Pre Season" and asked[0]["attempts"] == 1
