"""The MLB board build, end to end, on the real committed history (two teams).

The 2026-10-04 outage was a NameError deep inside props_and_totals: every unit
test passed and every live MLB run crashed at publish for two days. This runs the
same function the live job runs -- every market, the model fits, the last-five
strip, the ladders -- so a crash there fails the tests instead of the board.
"""
import pandas as pd

from mlb import board, config, current

TEAMS = ("ATL", "PHI")


def _two_teams(frame):
    return frame[frame["TEAM"].isin(TEAMS)] if not frame.empty else frame


def _quote(line, over=0.5):
    return {"line": line, "over": over, "under": round(1 - over, 4), "odd_over": 1.9,
            "odd_under": 1.9, "book": "Pinnacle", "source": "pinnacle"}


def test_every_mlb_market_builds_on_real_history(monkeypatch):
    real_rows, real_load = board._rows, current.load
    monkeypatch.setattr(board, "_rows", lambda kind: _two_teams(real_rows(kind)))
    monkeypatch.setattr(current, "load", lambda kind: _two_teams(real_load(kind)))

    bat = current.load("batting").sort_values("game_date")
    pit = current.load("pitching").sort_values("game_date")
    batters = bat.groupby("PLAYER_ID").tail(1)
    batters = batters[batters["PLAYER_ID"].isin(
        bat.groupby("PLAYER_ID").size().loc[lambda s: s >= 60].index)].head(6)
    starter = pit[pit["BFP"] >= 18].groupby("PLAYER_ID").tail(1).tail(1)
    assert len(batters) >= 2 and len(starter) == 1, "fixture data missing from the repo"

    props = {"hits": {n: _quote(0.5, 0.62) for n in batters["NAME"]},
             "hrr": {n: _quote(1.5) for n in batters["NAME"]},
             "hr": {n: _quote(0.5, 0.12) for n in batters["NAME"]},
             "rbi": {n: _quote(0.5, 0.3) for n in batters["NAME"]},
             "strikeouts": {starter["NAME"].iloc[0]: _quote(4.5)}}
    odds = {"games": {"1": {"props": props, "alt": {},
                            "team_totals": {"home": _quote(3.5), "away": _quote(3.5)}}}}
    games = pd.DataFrame([{"game_pk": 1, "home_team": "ATL", "away_team": "PHI",
                           "home_name": "Atlanta Braves", "away_name": "Philadelphia Phillies",
                           "start": "2026-10-06T23:00:00Z", "game_date": "2026-10-06",
                           "status": "Scheduled", "game_type": "D",
                           "home_pitcher": None, "away_pitcher": None,
                           "home_pitcher_name": None, "away_pitcher_name": None}])

    collect = {}
    out = board.props_and_totals(games, odds, list(config.MARKETS), None, collect=collect)

    assert set(out) == set(config.MARKETS)
    priced = collect.get("1") or {}
    for market in config.MARKETS:            # every market priced at least one quote
        assert priced.get(market), f"{market}: nothing priced"
        for p in priced[market]:
            assert 0.0 < p["probability"] < 1.0
            assert len(p["last_five"]) == 5  # the card's strip
