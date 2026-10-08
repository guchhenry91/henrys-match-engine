"""Current NBA rosters (nba.rosters) and how current_teams uses them."""
import json
from datetime import datetime, timezone

import pandas as pd

from nba import current, rosters

NOW = datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)


def _league(extra=()):
    """A complete 30-team roster: 14 filler players a team, plus `extra`."""
    players = {f"p {t.lower()} {i}": t for t in rosters.TEAMS for i in range(14)}
    players.update({rosters.norm_name(n): t for n, t in extra})
    return players


def test_sleeper_parse_keeps_active_players_and_drops_shared_names():
    payload = {
        "1": {"full_name": "Kevin Durant", "team": "HOU", "active": True},
        "2": {"first_name": "Trae", "last_name": "Young", "team": "WAS", "active": True},
        "3": {"full_name": "Retired Guy", "team": None, "active": False},
        "4": {"full_name": "Same Name", "team": "BOS", "active": True},
        "5": {"full_name": "Same Name", "team": "LAL", "active": True},
        "6": {"full_name": "Free Agent", "team": None, "active": True},
    }
    assert rosters.parse_sleeper(payload) == {"kevin durant": "HOU", "trae young": "WAS"}


def test_a_roster_contradicting_the_box_scores_is_refused():
    box = {f"p {t.lower()} {i}": t for t in rosters.TEAMS for i in range(3)}   # 90 shared
    order = sorted(rosters.TEAMS)
    shift = {t: order[(i + 1) % len(order)] for i, t in enumerate(order)}
    wrong = {k: shift[v] for k, v in _league().items()}    # complete, but every team wrong
    ok, why = rosters.check(wrong, box)
    assert not ok and "agrees on 0%" in why
    ok, why = rosters.check(_league(), box)
    assert ok, why


def test_a_partial_roster_is_refused():
    thin = {k: v for k, v in _league().items() if v != "BOS"}
    ok, why = rosters.check(thin, {})
    assert not ok and "BOS" in why


def test_sync_falls_back_to_espn_and_stays_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(rosters, "STORE", tmp_path / "rosters.json")
    calls = []

    def sleeper():
        calls.append("sleeper")
        raise OSError("refused")

    def espn():
        calls.append("espn")
        return _league([("Kevin Durant", "HOU")])
    out = rosters.sync({}, now=NOW, fetchers=(("sleeper", sleeper), ("espn", espn)))
    assert out["status"] == "updated" and out["source"] == "espn" and calls == ["sleeper", "espn"]
    saved = json.loads((tmp_path / "rosters.json").read_text())
    assert saved["players"]["kevin durant"] == "HOU" and saved["source"] == "espn"
    # Within MAX_AGE_HOURS nothing is asked again.
    assert rosters.sync({}, now=NOW, fetchers=(("sleeper", sleeper),))["status"] == "fresh"
    assert calls == ["sleeper", "espn"]


def test_a_refused_roster_keeps_the_previous_one(tmp_path, monkeypatch):
    monkeypatch.setattr(rosters, "STORE", tmp_path / "rosters.json")
    rosters.sync({}, now=NOW, fetchers=(("sleeper", lambda: _league()),))
    later = datetime(2026, 10, 10, tzinfo=timezone.utc)
    out = rosters.sync({}, now=later, fetchers=(("sleeper", lambda: {"x": "BOS"}),))
    assert out["status"] == "kept previous" and "sleeper refused" in out["reason"]
    assert rosters.load()["fetched_at"].startswith("2026-10-08")


def test_current_teams_roster_beats_older_box_scores_not_newer(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "DIR", tmp_path)
    monkeypatch.setattr(rosters, "STORE", tmp_path / "rosters.json")
    rows = pd.DataFrame([
        # Durant last played for PHX (last season's team, in a box score before the roster).
        {"PLAYER_ID": 201142, "PLAYER_NAME": "Kevin Durant", "TEAM_ABBREVIATION": "PHX",
         "GAME_ID": "0012600001", "GAME_DATE": "2026-10-04"},
        # Traded AFTER the roster was fetched: his newer box score wins.
        {"PLAYER_ID": 2, "PLAYER_NAME": "Late Trade", "TEAM_ABBREVIATION": "CHI",
         "GAME_ID": "0012600009", "GAME_DATE": "2026-10-09"},
    ], columns=current.COLUMNS)
    rows.to_csv(current._path(current.PRESEASON), index=False)
    rosters.STORE.write_text(json.dumps({"fetched_at": "2026-10-08T14:00:00+00:00",
                                         "players": {"kevin durant": "HOU", "late trade": "BOS",
                                                     "never played": "SAS"}}))
    names = {"kevin durant": 201142, "late trade": 2, "never played": 3}
    assert current.current_teams(names) == {201142: "HOU", 2: "CHI", 3: "SAS"}
    assert current.current_teams() == {201142: "PHX", 2: "CHI"}   # no names: box scores only
    assert current.box_name_teams() == {"kevin durant": "PHX", "late trade": "CHI"}
