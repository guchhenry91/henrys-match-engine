"""bet365 prices that beat the Betfair Exchange (leagues/mispricing.py)."""
import pandas as pd

from leagues import mispricing as mp


def test_flags_a_bet365_price_above_the_exchanges_fair_price(monkeypatch):
    monkeypatch.setattr(mp, "canonical", lambda name, league: name)
    row = {"Div": "E0", "Date": "10/10/2026", "Time": "15:00", "HomeTeam": "Arsenal", "AwayTeam": "Leeds",
           "B365H": 1.40, "B365D": 5.0, "B365A": 9.0, "BFEH": 1.42, "BFED": 5.2, "BFEA": 8.0,
           "B365>2.5": 1.60, "B365<2.5": 2.40, "BFE>2.5": 1.62, "BFE<2.5": 2.60,
           "AHh": -1.25, "B365AHH": 1.95, "B365AHA": 1.95, "BFEAHH": 2.00, "BFEAHA": 1.98}
    found = mp.scan(pd.DataFrame([row]))
    sel = {(f["market"], f["selection"]) for f in found}
    assert ("match_result", "away") in sel                 # 9.0 vs ~8.1 fair
    assert ("match_result", "home") not in sel
    away = next(f for f in found if f["selection"] == "away" and f["market"] == "match_result")
    assert away["ev"] >= mp.MIN_EV and away["exchange"] == 8.0


def test_illiquid_exchange_rows_are_not_trusted(monkeypatch):
    monkeypatch.setattr(mp, "canonical", lambda name, league: name)
    row = {"Div": "E0", "Date": "10/10/2026", "Time": "15:00", "HomeTeam": "A", "AwayTeam": "B",
           "B365H": 3.0, "B365D": 3.0, "B365A": 3.0, "BFEH": 2.0, "BFED": 2.0, "BFEA": 2.0}
    assert mp.scan(pd.DataFrame([row])) == []              # exchange book adds to 150%


def test_asian_handicap_quarter_lines_split_the_stake():
    e = {"market": "asian_handicap", "selection": "home", "line": -0.75, "b365": 2.0}
    assert mp.settle(e, 1, 0) == 0.5                       # -0.5 wins, -1.0 pushes
    assert mp.settle(e, 2, 0) == 1.0
    assert mp.settle(e, 0, 0) == -1.0
    away = {"market": "asian_handicap", "selection": "away", "line": -0.75, "b365": 2.0}
    assert mp.settle(away, 1, 0) == -0.5
    assert mp.settle({"market": "over_under_2.5", "selection": "under", "b365": 2.4}, 1, 1) == 1.4


def test_nfl_same_line_and_line_adjusted_flags():
    from nfl import mispricing as nm
    b365 = {"PIT|CLE": {"passing_yards": {
        "A": {"_name": "A", "line": 214.5, "odd_over": 2.10, "odd_under": 1.70},
        "B": {"_name": "B", "line": 181.5, "odd_over": 1.86, "odd_under": 1.86}}}}
    from datetime import datetime, timedelta, timezone
    fresh = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    sharp = {"PIT|CLE": {"checks": [fresh], "props": {"passing_yards": {
        "A": {"_name": "A", "line": 214.5, "odd_over": 1.93, "odd_under": 1.89, "book": "Pinnacle"},
        "B": {"_name": "B", "line": 185.5, "odd_over": 1.91, "odd_under": 1.91, "book": "Pinnacle"}}}}}
    found = {(f["player"], f["side"]): f for f in nm.scan(b365, sharp)}
    a = found[("A", "over")]
    assert a["basis"] == "same line" and a["ev"] > 0.03
    b = found.get(("B", "over"))                   # 4 yards lower than Pinnacle's fair 50% line
    assert b is None or b["basis"] == "line-adjusted"
    # a Pinnacle price older than MAX_SHARP_AGE_H is not compared at all
    sharp["PIT|CLE"]["checks"] = [(datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()]
    assert nm.scan(b365, sharp) == []


def test_anytime_td_yes_no_is_de_vigged_like_a_line():
    from oddsapi import props as shared
    payload = {"bookmakers": [{"key": "pinnacle", "markets": [{"key": "player_anytime_td", "outcomes": [
        {"name": "Yes", "description": "Derrick Henry", "price": 1.48},
        {"name": "No", "description": "Derrick Henry", "price": 2.68}]}]}]}
    q = shared.parse_event(payload, {"player_anytime_td": "anytime_touchdown"}, ("pinnacle",), {"pinnacle": "Pinnacle"})
    henry = q["anytime_touchdown"]["Derrick Henry"]
    io, iu = 1 / 1.48, 1 / 2.68
    assert henry["line"] == 0.5 and abs(henry["over"] - io / (io + iu)) < 1e-3


def test_bet365_td_price_beating_pinnacle_fair_is_flagged_and_graded():
    from datetime import datetime, timedelta, timezone
    from nfl import mispricing as nm
    now = datetime.now(timezone.utc)
    b365 = {"TEN|BAL": {"anytime_touchdown": {"Derrick Henry": {"_name": "Derrick Henry", "odd": 1.80, "book": "Bet365"},
                                              "Zay Flowers": {"_name": "Zay Flowers", "odd": 2.00, "book": "Bet365"}}}}
    sharp = {"TEN|BAL": {"checks": [(now - timedelta(hours=1)).isoformat()], "props": {"anytime_touchdown": {
        "Derrick Henry": {"_name": "Derrick Henry", "line": 0.5, "over": 0.64, "book": "Pinnacle"},
        "Zay Flowers": {"_name": "Zay Flowers", "line": 0.5, "over": 0.40, "book": "Pinnacle"}}}}}
    found = nm.scan(b365, sharp)
    assert [(f["player"], f["side"]) for f in found] == [("Derrick Henry", "yes")]   # 0.64 x 1.80 = +15%
    assert abs(found[0]["ev"] - 0.152) < 1e-6
