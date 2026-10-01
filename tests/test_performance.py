"""Profit at the frozen price and closing-line value (tracking/performance.py)."""
import json

from oddsapi import props as shared
from tracking import performance as pf


def _bet(graded, price, **kw):
    return {"kind": "prop", "market": "hits", "graded": graded, "book_price": price, **kw}


def test_profit_is_scored_at_the_frozen_price_one_unit_a_bet():
    s = pf.summarize([_bet("correct", 2.0), _bet("wrong", 2.0), _bet("correct", 1.5)])
    assert s["bets"] == 3 and s["wins"] == 2
    assert s["profit"] == 0.5                      # +1, -1, +0.5
    assert abs(s["roi"] - 0.5 / 3) < 1e-4
    assert abs(s["break_even"] - (0.5 + 0.5 + 1 / 1.5) / 3) < 1e-4


def test_voids_and_unpriced_picks_never_count_as_bets():
    s = pf.summarize([_bet("void", 2.0), _bet("correct", 2.0, void=True),
                      _bet("correct", None), _bet("wrong", 1.9)])
    assert s["bets"] == 1 and s["voids"] == 2 and s["profit"] == -1.0


def test_a_high_hit_rate_on_short_prices_can_still_lose():
    s = pf.summarize([_bet("correct", 1.3)] * 6 + [_bet("wrong", 1.3)] * 4)
    assert s["hit_rate"] == 0.6 and s["profit"] < 0


def test_clv_same_line_uses_the_fair_probability_of_the_pick_side():
    e = {"market": "hits", "line": 0.5, "side": "under", "book_p": 0.40,
         "open": {"line": 0.5, "p": 0.35, "at": "2026-10-01T10:00:00+00:00"},
         "close_at": "2026-10-01T16:00:00+00:00"}
    assert pf.clv(e) == ("ours", 0.05)
    e["book_p"] = 0.352
    assert pf.clv(e)[0] == "flat"


def test_clv_on_a_moved_line_is_scored_by_direction_for_each_side():
    over = {"market": "receiving_yards", "line": 52.5, "side": "over", "book_p": 0.5,
            "open": {"line": 48.5, "p": 0.5, "at": "a"}, "close_at": "b"}
    assert pf.clv(over) == ("ours", None)          # line rose: market agreed with over
    under = dict(over, side="under")
    assert pf.clv(under) == ("against", None)


def test_one_check_only_gives_no_clv_pair():
    e = {"market": "hits", "line": 0.5, "book_p": 0.6,
         "open": {"line": 0.5, "p": 0.5, "at": "2026-10-01T16:00:00+00:00"},
         "close_at": "2026-10-01T16:00:00+00:00"}
    assert pf.clv(e) == (None, None)


def test_status_needs_evidence_before_any_verdict():
    assert pf.summarize([_bet("correct", 2.0)] * 30)["status"] == "testing"
    losing = [_bet("correct", 1.8)] * 45 + [_bet("wrong", 1.8)] * 60
    assert pf.summarize(losing)["status"] == "losing"
    big = [_bet("correct", 2.0)] * 140 + [_bet("wrong", 2.0)] * 100
    assert pf.summarize(big)["status"] == "proven"   # +16.7% over 240: beyond luck
    small_edge = [_bet("correct", 2.0)] * 30 + [_bet("wrong", 2.0)] * 28
    assert pf.summarize(small_edge)["status"] == "promising"


def test_remember_open_keeps_only_the_first_quote():
    entry = {"props": {"hits": {"A": {"line": 0.5, "over": 0.6}}},
             "moneyline": {"home": 0.55, "away": 0.45}}
    shared.remember_open(entry, "t1")
    entry["props"] = {"hits": {"A": {"line": 0.5, "over": 0.7}, "B": {"line": 0.5, "over": 0.5}}}
    entry["moneyline"] = {"home": 0.6, "away": 0.4}
    shared.remember_open(entry, "t2")
    assert entry["open"]["hits"]["A"] == {"line": 0.5, "over": 0.6, "at": "t1"}
    assert entry["open"]["hits"]["B"]["at"] == "t2"
    assert entry["open"]["moneyline"]["home"] == 0.55


def test_moneyline_keeps_the_raw_prices():
    ev = {"bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Home FC", "price": 1.8}, {"name": "Away FC", "price": 2.1}]}]}]}
    line = shared.parse_moneyline(ev, ["pinnacle"], "Home FC", "Away FC")
    assert line["odd_home"] == 1.8 and line["odd_away"] == 2.1


def test_stamp_line_log_prices_winners_and_finds_props_and_team_totals(tmp_path):
    store = {"games": {"7": {
        "home": "NYA", "away": "BAL", "checks": ["2026-10-01T10:00:00+00:00", "2026-10-01T16:00:00+00:00"],
        "moneyline": {"home": 0.6, "away": 0.4, "overround": 0.04, "odd_home": 1.6,
                      "at": "2026-10-01T16:00:00+00:00"},
        "open": {"moneyline": {"home": 0.55, "away": 0.45, "odd_home": 1.75, "at": "2026-10-01T10:00:00+00:00"},
                 "hits": {"Aaron Judge": {"line": 0.5, "over": 0.62, "under": 0.38,
                                          "odd_over": 1.55, "at": "2026-10-01T10:00:00+00:00"}},
                 "team_totals": {"home": {"line": 4.5, "over": 0.5, "under": 0.5,
                                          "at": "2026-10-01T10:00:00+00:00"}}}}}}
    path = tmp_path / "odds.json"
    path.write_text(json.dumps(store))
    log = {
        "7:winner": {"kind": "winner", "game_id": "7", "home": "NYA", "away": "BAL", "pick": "NYA",
                     "locked_at": "2026-10-01T16:05:00+00:00"},
        "7:hits:j": {"kind": "prop", "game_id": "7", "market": "hits", "player": "Aaron Judge",
                     "side": "over", "line": 0.5, "book_p": 0.66, "book_price": 1.48,
                     "locked_at": "2026-10-01T16:05:00+00:00"},
        "7:team_runs:NYA": {"kind": "prop", "game_id": "7", "market": "team_runs", "team": "NYA",
                            "side": "over", "line": 4.5, "book_p": 0.47,
                            "locked_at": "2026-10-01T16:05:00+00:00"},
    }
    assert pf.stamp_line_log(log, path) == 3
    w = log["7:winner"]
    assert w["book_price"] == 1.6 and pf.clv(w) == ("ours", 0.05)
    assert pf.clv(log["7:hits:j"]) == ("ours", 0.04)
    assert pf.clv(log["7:team_runs:NYA"]) == ("against", -0.03)
    assert pf.stamp_line_log(log, path) == 0          # never re-stamped


def test_old_odds_files_give_an_estimated_winner_price(tmp_path):
    path = tmp_path / "odds.json"
    path.write_text(json.dumps({"games": {"7": {"home": "NYA", "moneyline": {
        "home": 0.5, "away": 0.5, "overround": 0.04}}}}))
    log = {"w": {"kind": "winner", "game_id": "7", "home": "NYA", "away": "BAL", "pick": "BAL"}}
    pf.stamp_line_log(log, path)
    assert log["w"]["price_estimated"] and abs(log["w"]["book_price"] - 1.923) < 1e-3


def test_nfl_props_find_their_game_by_team_pair(tmp_path):
    path = tmp_path / "nfl.json"
    path.write_text(json.dumps({"games": {"PIT|CLE": {"checks": ["a", "c"], "open": {
        "receiving_yards": {"DK Metcalf": {"line": 40.5, "over": 0.5, "at": "a"}}}}}}))
    props = {"k": {"market": "receiving_yards", "player": "DK Metcalf", "team": "PIT",
                   "away": "CLE", "side": "over", "line": 42.5, "locked_at": "c"}}
    assert pf.stamp_nfl_log(props, path) == 1
    assert pf.clv(props["k"]) == ("ours", None)


def test_nfl_props_find_their_game_from_the_game_id(tmp_path):
    """A prop's "home" field is a flag and "away" is his opponent; the nflverse id
    (season_week_AWAY_HOME) names the game whichever side he plays for."""
    path = tmp_path / "nfl.json"
    path.write_text(json.dumps({"games": {"NYG|TEN": {"checks": ["a", "c"], "open": {
        "rushing_yards": {"Tony Pollard": {"line": 60.5, "over": 0.5, "at": "a"}}}}}}))
    props = {"k": {"market": "rushing_yards", "player": "Tony Pollard", "team": "TEN",
                   "home": False, "away": "NYG", "game_id": "2026_03_TEN_NYG",
                   "side": "over", "line": 60.5, "locked_at": "c"}}
    assert pf.stamp_nfl_log(props, path) == 1 and props["k"]["open"]["line"] == 60.5


def test_nfl_moneyline_keeps_the_decimal_prices_as_quoted():
    from nfl import odds
    book = {"name": "Bet365", "bets": [{"name": "Home/Away", "values": [
        {"value": "Home", "odd": "2.25"}, {"value": "Away", "odd": "1.66"}]}]}
    line = odds.moneyline(book, "CLE", "PIT")
    assert line["odd_home"] == 2.25 and line["odd_away"] == 1.66
    assert abs(line["home"] + line["away"] - 1) < 1e-3


def test_nfl_winner_lock_freezes_price_and_open(monkeypatch):
    from datetime import datetime, timezone
    from nfl import picks
    now = datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(picks.lockwindow, "window", lambda now: 2.0)
    game = {"game_id": "2026_05_PIT_CLE", "kickoff": "2026-10-01T23:30:00+00:00",
            "gradeable": True, "pick": "PIT", "p_pick": 0.57, "home": "CLE", "away": "PIT",
            "book": "Bet365", "book_p_pick": 0.5754, "book_price": 1.66,
            "book_at": "2026-10-01T21:00:00+00:00",
            "book_open": {"p": 0.55, "price": 1.74, "book": "Bet365",
                          "at": "2026-09-29T11:00:00+00:00"}}
    log = {}
    picks._lock_games({"games": [game]}, log, now)
    e = log["2026_05_PIT_CLE"]
    assert e["book_price"] == 1.66 and e["open"]["price"] == 1.74
    assert pf.clv(e) == ("ours", 0.0254)
