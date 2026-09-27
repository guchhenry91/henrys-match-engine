"""bet365 anytime-scorer prices on the soccer player picks (API-Football)."""
from datetime import datetime, timezone

from leagues import prop_odds
from scripts import sync_soccer_prop_odds as sync


def _rows():
    return [{"bookmakers": [
        {"id": 8, "name": "Bet365", "bets": [
            {"id": 92, "name": "Anytime Goal Scorer", "values": [
                {"value": "Viktor Gyokeres", "odd": "2.05"},
                {"value": "No Goalscorer", "odd": "13.00"},
                {"value": "Nico Elvedi", "odd": "23.00"}]},
            {"id": 275, "name": "Away Player Shots On Target Total", "values": [
                {"value": "Viktor Gyokeres", "odd": "9.50"}]}]},
        {"id": 11, "name": "Other", "bets": [
            {"id": 92, "values": [{"value": "Viktor Gyokeres", "odd": "1.50"}]}]}]}]


def test_only_bet365_anytime_scorer_is_read():
    got = prop_odds.parse_anytime(_rows())
    assert got == {"Viktor Gyokeres": 2.05, "Nico Elvedi": 23.0}


def test_attach_matches_accents_and_skips_long_or_missing_prices():
    store = {"fixtures": {"PL#12": {"anytime": prop_odds.parse_anytime(_rows()),
                                    "checked_at": "2026-10-09T12:00:00+00:00"}}}
    picks = [
        {"market": "goal", "league_key": "PL", "id": 12, "player": "Viktor Gyökeres", "p_pick": 0.52},
        {"market": "goal", "league_key": "PL", "id": 12, "player": "Nico Elvedi", "p_pick": 0.41},
        {"market": "goal", "league_key": "PL", "id": 12, "player": "Nobody", "p_pick": 0.45},
        {"market": "shots", "league_key": "PL", "id": 12, "player": "Viktor Gyökeres", "p_pick": 0.8}]
    assert prop_odds.attach(picks, store) == 1
    g = picks[0]
    assert g["book"] == "bet365" and g["book_price"] == 2.05
    assert abs(g["edge"] - (0.52 - 1 / 2.05)) < 1e-3
    assert "book_price" not in picks[1]          # 23.00 is past MAX_ODDS
    assert "book_price" not in picks[3]          # no bet365 shots market


def test_only_scorer_fixtures_inside_the_window_and_not_recently_checked_are_due():
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    board = {"upcoming": [
        {"market": "goal", "league_key": "PL", "id": 1, "date": "2026-10-10T11:30:00+00:00", "home": "Arsenal", "away": "Leeds"},
        {"market": "goal", "league_key": "PL", "id": 1, "date": "2026-10-10T11:30:00+00:00", "home": "Arsenal", "away": "Leeds"},
        {"market": "shots", "league_key": "PL", "id": 2, "date": "2026-10-10T14:00:00+00:00", "home": "Chelsea", "away": "Bournemouth"},
        {"market": "goal", "league_key": "PL", "id": 3, "date": "2026-10-20T14:00:00+00:00", "home": "A", "away": "B"},
        {"market": "goal", "league_key": "PL", "id": 4, "date": "2026-10-10T14:00:00+00:00", "home": "C", "away": "D"}]}
    store = {"fixtures": {"PL#4": {"checked_at": "2026-10-09T06:00:00+00:00"}}}
    due = sync.due_fixtures(board, store, now)
    assert [d[1] for d in due] == [1]
