"""MLB live board: bookmaker lines only, two checks a game, frozen before first
pitch, graded from the box score."""
from datetime import datetime, timezone

from mlb import book_lines as bl
from mlb import picks as mp
from scripts import sync_mlb_odds as sync


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_every_mlb_team_name_maps_to_a_retrosheet_code():
    assert len(set(bl.TEAM_CODES.values())) == 30


def test_home_runs_one_sided_over_is_kept_with_its_margin_estimated():
    payload = {"bookmakers": [{"key": "draftkings", "markets": [{"key": "batter_home_runs",
               "outcomes": [{"name": "Over", "description": "Aaron Judge", "point": 0.5, "price": 3.2}]}]}]}
    q = bl.parse_event(payload)["hr"]["Aaron Judge"]
    assert q["fair"] is False and abs(q["over"] - (1 / 3.2) / bl.ONE_SIDED_OVERROUND) < 1e-3


def test_two_sided_home_run_quote_is_devigged_normally():
    payload = {"bookmakers": [{"key": "pinnacle", "markets": [{"key": "batter_home_runs",
               "outcomes": [{"name": "Over", "description": "Ben Rice", "point": 0.5, "price": 5.6},
                            {"name": "Under", "description": "Ben Rice", "point": 0.5, "price": 1.12}]}]}]}
    q = bl.parse_event(payload)["hr"]["Ben Rice"]
    assert q.get("fair", True) and abs(q["over"] + q["under"] - 1) < 1e-3


def test_team_totals_are_devigged_per_team():
    payload = {"bookmakers": [{"key": "pinnacle", "markets": [{"key": "team_totals", "outcomes": [
        {"name": "Over", "description": "New York Yankees", "point": 4.5, "price": 1.9},
        {"name": "Under", "description": "New York Yankees", "point": 4.5, "price": 1.9}]}]}]}
    got = bl.parse_team_totals(payload, "New York Yankees", "Baltimore Orioles")
    assert got["home"]["line"] == 4.5 and abs(got["home"]["over"] - 0.5) < 1e-6


def test_odds_event_matches_the_closest_mlb_game_of_a_series():
    games = [{"game_pk": 1, "home_team": "NYA", "away_team": "BAL", "start": "2026-09-26T17:05:00Z"},
             {"game_pk": 2, "home_team": "NYA", "away_team": "BAL", "start": "2026-09-27T17:05:00Z"}]
    ev = {"home_team": "New York Yankees", "away_team": "Baltimore Orioles",
          "commence_time": "2026-09-27T17:10:00Z"}
    assert sync.match_event(ev, games)["game_pk"] == 2


def test_grading_void_for_a_non_starter_and_pending_without_a_box_score():
    log = {"G1:hits:abc": {"kind": "prop", "game_id": "G1", "market": "hits",
                           "player_id": "abc", "line": 0.5, "tipoff": "x"},
           "G1:team_runs:NYA": {"kind": "prop", "game_id": "G1", "market": "team_runs",
                                "player_id": "NYA", "line": 4.5, "tipoff": "x"}}
    mp.grade(log, {}, {}, set())
    assert not any(e.get("graded") for e in log.values())
    mp.grade(log, {}, {("G1", "NYA"): {"RUNS": 6}}, {"G1"})
    assert log["G1:hits:abc"]["graded"] == "void"
    assert log["G1:team_runs:NYA"]["graded"] == "correct"


# --- who can actually play -------------------------------------------------------

def test_availability_drops_inactive_benched_and_non_starting_players():
    from types import SimpleNamespace
    from mlb import board
    g = SimpleNamespace(game_pk=7, home_team="NYA", away_team="BAL",
                        home_pitcher=None, away_pitcher=None)
    avail = {"active": {"NYA": {"judga001", "ricbe001", "colega001"}},
             "lineups": {(7, "NYA"): {"judga001"}}}
    assert board.unavailable("batter", "stanc001", "NYA", g, avail) == "not on the active roster"
    assert board.unavailable("batter", "ricbe001", "NYA", g, avail) == "not in the posted lineup"
    assert board.unavailable("batter", "judga001", "NYA", g, avail) is None
    # BAL: roster not fetched and no lineup yet -> not filtered, never guessed
    assert board.unavailable("batter", "anyone01", "BAL", g, avail) is None
    assert board.unavailable("batter", "stanc001", "NYA", g, None) is None


def test_only_the_probable_starter_gets_a_strikeout_prop(monkeypatch):
    from types import SimpleNamespace
    from mlb import board, current
    monkeypatch.setattr(current, "retro_id", lambda m: {1: "colega001"}.get(m, f"mlbam:{m}"))
    avail = {"active": {"NYA": {"colega001", "rodoc001"}}, "lineups": {}}
    g = SimpleNamespace(game_pk=7, home_team="NYA", away_team="BAL",
                        home_pitcher=1, away_pitcher=None)
    assert board.unavailable("pitcher", "colega001", "NYA", g, avail) is None
    assert board.unavailable("pitcher", "rodoc001", "NYA", g, avail) == "not the probable starter"


def test_availability_reads_rosters_and_posted_lineups(monkeypatch):
    import io
    import json as _json
    import pandas as pd
    from mlb import current
    monkeypatch.setattr(current, "retro_id", lambda m: f"r{m}")
    monkeypatch.setattr(current, "PAUSE", 0)

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        url = req.full_url
        if "teams/147/roster" in url:
            body = {"roster": [{"person": {"id": 1}}, {"person": {"id": 2}}]}
        elif "teams/110/roster" in url:
            raise OSError("down")
        else:
            body = {"dates": [{"games": [{"gamePk": 7, "lineups": {"homePlayers": [{"id": 1}]}}]}]}
        return R(_json.dumps(body).encode())
    games = pd.DataFrame([{"game_pk": 7, "home_team": "NYA", "away_team": "BAL"}])
    got = current.availability(games, opener)
    assert got["active"] == {"NYA": {"r1", "r2"}}             # BAL failed: not filtered
    assert got["lineups"] == {(7, "NYA"): {"r1"}}
