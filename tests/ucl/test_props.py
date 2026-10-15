"""Champions League player props (ucl/props.py): selection, locking, grading."""
import pandas as pd
import pytest

from ucl import props

KICK = "2026-10-14T19:00:00+00:00"
FIX = {"fid": "77", "mid": -77, "kickoff": KICK, "date": KICK, "home": "Manchester City",
       "away": "Paris Saint Germain", "home_us": "Manchester City", "away_us": "Paris Saint Germain",
       "hours": 1.0}


def _row(name, goal, sot, start=90.0, team="Manchester City"):
    return {"team": team, "player": name, "player_id": 1, "anytime_pct": goal,
            "p_sot_1plus": sot, "start_pct": start, "appearance_pct": 95.0}


@pytest.fixture(autouse=True)
def _paths(tmp_path, monkeypatch):
    for k in ("LOG", "CACHE", "STATS"):
        monkeypatch.setattr(props, k, tmp_path / f"{k}.json")


def test_only_likely_starters_above_the_bar():
    rows = [_row("A", 50, 80), _row("B", 50, 80, start=60), _row("C", 20, 40)]
    out = props.candidates(FIX, rows, {})
    assert {p["player"] for p in out} == {"A"}
    assert all(p["pick_kind"] == "proving" for p in out)      # no prices: proving only


def test_scorer_price_too_short_is_dropped():
    store = {"fixtures": {"77": {"anytime": {"A": 1.5}}}}
    out = props.candidates(FIX, [_row("A", 50, 10)], store)
    assert out == []


def test_eligible_needs_two_big_five_clubs():
    now = pd.Timestamp("2026-10-14T12:00:00Z")
    ms = [{"id": 1, "kickoff": KICK, "home": "Manchester City", "away": "Paris Saint Germain"},
          {"id": 2, "kickoff": KICK, "home": "Slovan Bratislava", "away": "VfB Stuttgart"}]
    assert [f["fid"] for f in props.eligible(ms, now)] == ["1"]


def test_build_locks_one_pick_per_player_and_grades(monkeypatch):
    rows = [_row("Erling Haaland", 48, 77)]
    monkeypatch.setattr(props, "forecast", lambda *a, **k: {"77": rows})
    match = {"id": 77, "kickoff": KICK, "home": "Manchester City", "away": "Paris Saint Germain"}
    out = props.build([match], {}, now="2026-10-14T18:00:00Z", odds_store={})
    assert [(p["player"], p["market"], p.get("locked")) for p in out["upcoming"]] == \
        [("Erling Haaland", "sot", True)]

    class Client:
        def get(self, path, **kw):
            return [{"team": {"name": "Manchester City"}, "players": [{"player": {"id": 9, "name": "E. Haaland"},
                     "statistics": [{"games": {"minutes": 90}, "shots": {"total": 4, "on": 2}, "goals": {"total": 1}}]}]}]
    out = props.build([], {}, now="2026-10-15T00:00:00Z", odds_store={}, client=Client())
    assert out["record"]["sot"]["correct"] == 1
    assert out["upcoming"] == []


def test_a_player_who_did_not_play_is_void(monkeypatch):
    monkeypatch.setattr(props, "forecast", lambda *a, **k: {"77": [_row("Erling Haaland", 48, 77)]})
    match = {"id": 77, "kickoff": KICK, "home": "Manchester City", "away": "Paris Saint Germain"}
    props.build([match], {}, now="2026-10-14T18:00:00Z", odds_store={})

    class Client:
        def get(self, path, **kw):
            return [{"team": {"name": "Manchester City"}, "players": [{"player": {"id": 9, "name": "E. Haaland"},
                     "statistics": [{"games": {"minutes": None}, "shots": {}, "goals": {}}]}]}]
    out = props.build([], {}, now="2026-10-15T00:00:00Z", odds_store={}, client=Client())
    assert out["record"]["sot"]["void"] == 1


def test_shots_pick_is_graded_from_espn_blocked_shots_included(monkeypatch):
    import io
    import json as _json
    from ucl import espn
    monkeypatch.setattr(espn, "PACE", 0)
    rows = [_row("Erling Haaland", 30, 50) | {"p_shots_2plus": 80.0}]
    monkeypatch.setattr(props, "forecast", lambda *a, **k: {"77": rows})
    match = {"id": 77, "kickoff": KICK, "home": "Manchester City", "away": "Paris Saint Germain"}
    out = props.build([match], {}, now="2026-10-14T18:00:00Z", odds_store={})
    assert [(p["market"], p.get("locked")) for p in out["upcoming"]] == [("shots", True)]

    def opener(req, timeout=30):
        url = req.full_url
        if "scoreboard" in url:
            body = {"events": [{"id": "5", "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": "Manchester City"}},
                {"homeAway": "away", "team": {"displayName": "Paris Saint-Germain"}}]}]}]}
        else:
            body = {"rosters": [{"team": {"displayName": "Manchester City"}, "roster": [
                {"athlete": {"displayName": "Erling Haaland"}, "starter": True,
                 "stats": [{"name": "appearances", "value": 1}, {"name": "totalShots", "value": 3},
                           {"name": "shotsOnTarget", "value": 1}, {"name": "totalGoals", "value": 0}]}]}]}
        return io.BytesIO(_json.dumps(body).encode())
    out = props.build([], {}, now="2026-10-15T00:00:00Z", odds_store={}, espn_opener=opener)
    assert out["record"]["shots"]["correct"] == 1


def test_a_different_first_name_still_finds_the_player_and_regrades_a_wrong_void(monkeypatch):
    """Understat's 'Mathis Cherki' is ESPN's 'Rayan Cherki': he started, so the pick is
    graded on his line -- and an earlier 'did not play' void is corrected."""
    import io
    import json as _json
    from ucl import espn
    monkeypatch.setattr(espn, "PACE", 0)
    log = {"77:shots:Mathis Cherki": {"market": "shots", "player": "Mathis Cherki",
                                      "team": "Manchester City", "fixture_id": "77", "kickoff": KICK,
                                      "home": "Manchester City", "away": "Paris Saint Germain",
                                      "p_pick": 0.63, "graded": "void", "void": True,
                                      "did_not_play": True, "void_reason": "did not play"}}

    def opener(req, timeout=30):
        if "scoreboard" in req.full_url:
            body = {"events": [{"id": "5", "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": "Manchester City"}},
                {"homeAway": "away", "team": {"displayName": "Paris Saint-Germain"}}]}]}]}
        else:
            body = {"rosters": [{"team": {"displayName": "Manchester City"}, "roster": [
                {"athlete": {"displayName": "Rayan Cherki"}, "starter": True,
                 "stats": [{"name": "appearances", "value": 1}, {"name": "totalShots", "value": 0}]}]}]}
        return io.BytesIO(_json.dumps(body).encode())
    n = props.grade(log, pd.Timestamp("2026-10-15T00:00:00Z"), espn_opener=opener)
    e = log["77:shots:Mathis Cherki"]
    assert n == 1 and e["graded"] == "wrong" and not e.get("void") and e["matched_as"] == "Rayan Cherki"


def test_two_squad_players_sharing_a_surname_are_not_guessed():
    squad = [{"player": "Inaki Williams"}, {"player": "Nico Williams"}]
    assert props.find_player("Someone Williams", squad) is None


def test_far_matches_rebuild_twice_a_day_near_ones_often():
    now = pd.Timestamp("2026-10-15T00:00:00Z")
    far = [{"fid": "1", "hours": 100.0}]
    near = [{"fid": "1", "hours": 20.0}]
    cache = {"built_at": (now - pd.Timedelta(hours=5)).isoformat(), "fixtures": {"1": []}}
    assert not props.due(cache, far, now)
    assert props.due(cache, near, now)
