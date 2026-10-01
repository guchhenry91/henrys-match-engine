"""Shots / on-target prices from US books (leagues/shot_odds.py)."""
from datetime import datetime, timezone

from leagues import shot_odds
from scripts import sync_soccer_shot_odds as sync
from tracking import performance, value


def _o(name, player, point, price):
    return {"name": name, "description": player, "point": point, "price": price}


PAYLOAD = {"bookmakers": [
    {"key": "fanduel", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            _o("Over", "Bukayo Saka", 0.5, 1.6), _o("Under", "Bukayo Saka", 0.5, 2.25)]}]},
    {"key": "draftkings", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            _o("Over", "Bukayo Saka", 0.5, 1.55),              # one-sided: loses to FanDuel
            _o("Over", "Martin Odegaard", 1.5, 3.0)]},         # wrong line: ignored
        {"key": "player_shots", "outcomes": [
            _o("Over", "Bukayo Saka", 1.5, 1.5), _o("Under", "Bukayo Saka", 1.5, 2.5),
            _o("Over", "Bukayo Saka", 2.5, 2.6)]}]}]}


def test_parse_keeps_the_picks_own_line_and_prefers_a_two_sided_quote():
    got = shot_odds.parse(PAYLOAD)
    sot = got["sot"]["Bukayo Saka"]
    assert sot["book"] == "FanDuel" and sot["line"] == 0.5
    assert abs(sot["over"] + sot["under"] - 1) < 1e-6
    assert got["shots"]["Bukayo Saka"]["line"] == 1.5
    assert "Martin Odegaard" not in got["sot"]


def test_one_sided_over_is_kept_with_its_margin_estimated():
    p = {"bookmakers": [{"key": "draftkings", "markets": [{"key": "player_shots_on_target",
         "outcomes": [_o("Over", "Erling Haaland", 0.5, 1.25)]}]}]}
    q = shot_odds.parse(p)["sot"]["Erling Haaland"]
    assert q["fair"] is False and abs(q["over"] - 0.8 / shot_odds.ONE_SIDED_OVERROUND) < 1e-3


def _store():
    first = shot_odds.parse(PAYLOAD)
    later = shot_odds.parse(PAYLOAD)
    later["sot"]["Bukayo Saka"] = dict(later["sot"]["Bukayo Saka"], over=0.66, odd_over=1.5)
    return {"fixtures": {"PL#7": {"checks": ["2026-10-09T10:00:00+00:00",
                                            "2026-10-10T08:00:00+00:00"],
                                  "open": first, "props": later}}}


def test_attach_and_freeze_give_price_open_and_a_clv_pair():
    store = _store()
    up = [{"market": "sot", "league_key": "PL", "id": 7, "player": "Bukayo Saka", "p_pick": 0.7},
          {"market": "goal", "league_key": "PL", "id": 7, "player": "Bukayo Saka", "p_pick": 0.4}]
    assert shot_odds.attach(up, store) == 1
    assert up[0]["book_price"] == 1.5 and up[0]["book"] == "FanDuel"
    v = value.annotate_soccer_player(up[0])
    assert v["basis"] == "estimate"                  # bet365 estimated from the fair chance
    entry = {"market": "sot", "player": "Bukayo Saka", "graded": "correct",
             "line": "1+ shot on target"}            # settled picks carry the LABEL
    assert shot_odds.freeze(entry, "PL", 7, store)
    assert entry["open"]["price"] == 1.6 and "line" not in entry["open"]
    assert performance.clv(entry) == ("ours", round(0.66 - entry["open"]["p"], 4))
    assert performance.summarize([entry])["bets"] == 1


def test_two_checks_at_most_and_the_second_only_near_kickoff():
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    ko = "2026-10-10T11:30:00+00:00"
    assert sync.due([], ko, now) == "board"
    assert sync.due(["2026-10-09T11:00:00+00:00"], ko, now) is None
    late = datetime(2026, 10, 10, 8, tzinfo=timezone.utc)
    assert sync.due(["2026-10-09T11:00:00+00:00"], ko, late) == "lock"
    assert sync.due(["a", "b"], ko, late) is None
    assert sync.due([], "2026-10-12T11:30:00+00:00", now) is None   # too far out


def test_events_join_only_when_both_clubs_map():
    events = [{"id": "x", "home_team": "Arsenal", "away_team": "Leeds United",
               "commence_time": "2026-10-10T11:30:00Z"},
              {"id": "y", "home_team": "Nowhere FC", "away_team": "Leeds United",
               "commence_time": "2026-10-10T11:30:00Z"}]
    ev = sync.match_event(events, "PL", "2026-10-10T11:30:00+00:00", "Arsenal", "Leeds")
    assert ev["id"] == "x"
    assert sync.match_event(events[1:], "PL", "2026-10-10T11:30:00+00:00", "Arsenal", "Leeds") is None
