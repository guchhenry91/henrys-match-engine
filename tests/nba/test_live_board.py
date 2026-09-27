"""The live NBA board: bookmaker lines only, two checks a game, frozen before
tip-off, graded from the box score -- never from missing data."""
import io
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from nba import book_lines as bl
from nba import config, current, features
from nba import picks as nba_picks
from scripts import sync_nba_odds as sync

UTC = timezone.utc


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# --- when to spend and when to freeze --------------------------------------------

def test_a_2330_utc_tip_is_checked_at_15_then_locked_at_2130():
    tip = "2026-10-21T23:30:00+00:00"
    assert bl.due([], tip, _t("2026-10-21T15:05")) == "board"
    assert bl.due(["2026-10-21T15:05:00+00:00"], tip, _t("2026-10-21T21:35")) == "lock"
    assert bl.due(["a", "b"], tip, _t("2026-10-21T21:35")) is None


def test_a_matinee_is_locked_at_the_15_utc_run():
    tip = "2026-12-25T17:00:00+00:00"
    assert bl.is_last_run_before(tip, _t("2026-12-25T15:05"))
    assert not bl.is_last_run_before("2026-12-25T23:00:00+00:00", _t("2026-12-25T15:05"))


def test_picks_freeze_on_the_last_run_or_inside_two_hours():
    last = bl.is_last_run_before
    assert nba_picks.should_freeze("2026-10-21T23:30:00Z", _t("2026-10-21T21:35"), last)
    assert not nba_picks.should_freeze("2026-10-21T23:30:00Z", _t("2026-10-21T15:05"), last)
    assert nba_picks.should_freeze("2026-10-21T23:30:00Z", _t("2026-10-21T22:00"), last)
    assert not nba_picks.should_freeze("2026-10-21T23:30:00Z", _t("2026-10-21T23:40"), last)


# --- parsing and matching --------------------------------------------------------

def test_parse_nba_props_prefers_pinnacle_and_devigs():
    def book(key, rows):
        return {"key": key, "markets": [{"key": "player_points", "outcomes": [
            {"name": s, "description": n, "point": pt, "price": pr} for n, s, pt, pr in rows]}]}
    payload = {"bookmakers": [
        book("fanduel", [("Cade Cunningham", "Over", 25.5, 1.87), ("Cade Cunningham", "Under", 25.5, 1.95)]),
        book("pinnacle", [("Cade Cunningham", "Over", 26.5, 1.95), ("Cade Cunningham", "Under", 26.5, 1.91)])]}
    q = bl.parse_event(payload)["points"]["Cade Cunningham"]
    assert q["line"] == 26.5 and q["source"] == "pinnacle"
    assert q["over"] + q["under"] == pytest.approx(1.0, abs=1e-3)


def test_an_odds_event_matches_the_cdn_game_by_teams_and_time():
    sched = [{"game_id": "0022600001", "home_team": "DET", "away_team": "BOS",
              "tipoff": "2026-10-20T19:00:00Z"},
             {"game_id": "0022600050", "home_team": "DET", "away_team": "BOS",
              "tipoff": "2026-11-20T00:00:00Z"}]
    ev = {"home_team": "Detroit Pistons", "away_team": "Boston Celtics",
          "commence_time": "2026-10-20T19:10:00Z"}
    assert sync.match_event(ev, sched)["game_id"] == "0022600001"
    ev["commence_time"] = "2026-10-25T19:00:00Z"
    assert sync.match_event(ev, sched) is None


def test_every_nba_team_name_maps_to_a_tricode():
    assert len(bl.TEAM_CODES) == 30 and len(set(bl.TEAM_CODES.values())) == 30


# --- box scores --------------------------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _box():
    def player(pid, name, played, mins, pts):
        return {"personId": pid, "name": name, "played": played,
                "statistics": {"minutes": mins, "points": pts, "reboundsTotal": 5,
                               "assists": 4, "threePointersMade": 2}}
    return {"game": {
        "homeTeam": {"teamId": 1, "teamTricode": "DET", "score": 110, "players": [
            player(10, "Cade Cunningham", "1", "PT36M30.00S", 31),
            player(11, "Bench Guy", "0", "PT00M00.00S", 0)]},
        "awayTeam": {"teamId": 2, "teamTricode": "BOS", "score": 104, "players": [
            player(20, "Jaylen Brown", "1", "PT34M00.00S", 22)]}}}


def test_box_rows_are_gamelog_shaped_and_skip_players_who_did_not_play():
    opener = lambda req, timeout=None: _Resp(json.dumps(_box()).encode())
    rows = current.box_rows("0022600001", "2026-10-20", opener)
    assert {r["PLAYER_NAME"] for r in rows} == {"Cade Cunningham", "Jaylen Brown"}
    cade = next(r for r in rows if r["PLAYER_ID"] == 10)
    assert cade["MIN"] == 36.5 and cade["WL"] == "W" and cade["MATCHUP"] == "DET vs. BOS"
    brown = next(r for r in rows if r["PLAYER_ID"] == 20)
    assert brown["MATCHUP"] == "BOS @ DET" and brown["WL"] == "L"


def test_current_teams_come_from_the_newest_box_score(tmp_path, monkeypatch):
    monkeypatch.setattr(current, "DIR", tmp_path)
    cols = current.COLUMNS
    pre = pd.DataFrame([dict(zip(cols, ["2", 7, "Traded Man", 1, "OLD", "0012600001",
                                        "2026-10-05", "OLD vs. X", "W", 20, 1, 1, 1, 1]))])
    reg = pd.DataFrame([dict(zip(cols, ["2", 7, "Traded Man", 2, "NEW", "0022600003",
                                        "2026-10-22", "NEW vs. Y", "W", 30, 1, 1, 1, 1]))])
    pre.to_csv(tmp_path / f"preseason_{config.CURRENT_SEASON}.csv", index=False)
    reg.to_csv(tmp_path / f"player_{config.CURRENT_SEASON}.csv", index=False)
    assert current.current_teams() == {7: "NEW"}


# --- grading --------------------------------------------------------------------------

def _payload():
    return {"games": [{"game_id": "G1", "tipoff": "2026-10-21T23:30:00Z", "home": "DET",
                       "away": "BOS", "pick": "DET", "p_pick": 0.6}],
            "props": {"points": {"picks": [
                {"game_id": "G1", "tipoff": "2026-10-21T23:30:00Z", "market": "points",
                 "player": "Cade Cunningham", "player_id": 10, "team": "DET",
                 "opponent": "BOS", "line": 26.5, "line_source": "pinnacle",
                 "book": "Pinnacle", "probability": 0.55},
                {"game_id": "G1", "tipoff": "2026-10-21T23:30:00Z", "market": "points",
                 "player": "Scratched", "player_id": 99, "team": "DET",
                 "opponent": "BOS", "line": 10.0, "line_source": "fanduel",
                 "book": "FanDuel", "probability": 0.6}]}}}


def test_freeze_once_then_grade_from_the_box_score():
    log = {}
    n = nba_picks.freeze(_payload(), log, _t("2026-10-21T21:35"), bl.is_last_run_before)
    assert n == 3
    # A later run with a moved line must not rewrite the frozen pick.
    moved = _payload()
    moved["props"]["points"]["picks"][0]["line"] = 30.5
    assert nba_picks.freeze(moved, log, _t("2026-10-21T22:00"), bl.is_last_run_before) == 0
    # Box score not held yet: everything stays pending.
    nba_picks.grade(log, {}, {}, set())
    assert not any(e.get("graded") for e in log.values())
    nba_picks.grade(log, {"G1": "home"}, {("G1", 10): {"PTS": 31}}, {"G1"})
    assert log["G1:winner"]["graded"] == "correct"
    assert log["G1:points:10"]["graded"] == "correct" and log["G1:points:10"]["line"] == 26.5
    assert log["G1:points:99"]["graded"] == "void"            # did not play
    rec = nba_picks.record(log)
    assert rec["team_winner"]["correct"] == 1 and rec["props"]["void"] == 1


def test_a_whole_number_line_landing_exactly_is_a_push():
    log = {}
    nba_picks.freeze(_payload(), log, _t("2026-10-21T21:35"), bl.is_last_run_before)
    nba_picks.grade(log, {}, {("G1", 99): {"PTS": 10}}, {"G1"})
    assert log["G1:points:99"]["graded"] == "void"


# --- the line spread the model is trained and gated on ---------------------------------

def test_augment_lines_spreads_and_resettles_and_never_goes_below_the_floor():
    frame = pd.DataFrame({"base_line": [22.5, 7.5], "line": [22.5, 7.5], "PTS": [25, 9]})
    out = features.augment_lines(frame, "points")
    assert set(out["line_step"]) == set(config.LINE_STEPS["points"])
    assert out["line"].min() >= config.MIN_LINE["points"]
    row = out[(out["base_line"] == 22.5) & (out["line_step"] == 3)].iloc[0]
    assert row["line"] == 25.5 and row["outcome"] == 0.0         # 25 is not over 25.5


def test_the_slate_before_opening_night_is_the_first_game_date_within_a_week():
    from nba import board
    sched = pd.DataFrame([
        {"game_id": "0022600001", "stage": "002", "status": 1, "tipoff": "2026-10-20T23:00:00Z",
         "game_date": "2026-10-20", "home_team": "DET", "away_team": "BOS", "neutral": False},
        {"game_id": "0022600002", "stage": "002", "status": 1, "tipoff": "2026-10-21T23:00:00Z",
         "game_date": "2026-10-21", "home_team": "NYK", "away_team": "PHI", "neutral": False},
        {"game_id": "0012600001", "stage": "001", "status": 1, "tipoff": "2026-10-15T23:00:00Z",
         "game_date": "2026-10-15", "home_team": "MIA", "away_team": "TOR", "neutral": False}])
    assert list(board.slate(sched, _t("2026-10-16T12:00"))["game_id"]) == ["0022600001"]
    assert board.slate(sched, _t("2026-10-01T12:00")).empty           # more than a week out
