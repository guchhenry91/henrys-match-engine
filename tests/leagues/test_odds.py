import io

from leagues.odds import BOOK_AVG, BOOK_B365, market_for, parse_fixture_odds

CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    "E0,22/08/2026,15:00,Arsenal,Man City,2.15,3.30,3.40,2.10,3.40,3.30\n"
    "SP1,22/08/2026,20:00,Barcelona,Sevilla,1.53,4.10,6.20,1.50,4.20,6.00\n"  # other league
    "E0,22/08/2026,17:30,Nott'm Forest,Brighton,,,,,,\n"          # no odds posted yet
)

# bet365 has not priced this one but the market average has: the row must still
# publish, falling back to the average and SAYING which book it is showing.
CSV_NO_B365 = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    "E0,22/08/2026,15:00,Arsenal,Man City,,,,2.10,3.40,3.30\n"
)


def test_devigged_market_probs_sum_to_one_and_map_names():
    df = parse_fixture_odds(io.StringIO(CSV), "PL")
    assert len(df) == 1                          # only the priced E0 row
    row = df.iloc[0]
    assert row["home"] == "Arsenal" and row["away"] == "Manchester City"   # canonical
    s = row["m_home"] + row["m_draw"] + row["m_away"]
    assert abs(s - 1.0) < 1e-9                    # overround removed
    assert row["m_home"] > row["m_away"]          # 2.10 shorter than 3.30


def test_no_rows_for_a_league_is_empty_not_an_error():
    df = parse_fixture_odds(io.StringIO(CSV), "BUNDESLIGA")     # no D1 rows
    assert df.empty
    assert list(df.columns) == ["date", "home", "away", "m_home", "m_draw", "m_away",
                                "b_home", "b_draw", "b_away", "book"]


def test_unpriced_fixture_is_skipped_not_crashed():
    df = parse_fixture_odds(io.StringIO(CSV), "PL")
    assert "Nottingham Forest" not in set(df["home"])           # the blank-odds row dropped


# --- bet365 prices ------------------------------------------------------------

def test_bet365_price_is_carried_through_undevigged():
    """The displayed price must be the price as OFFERED. De-vigging it, or using
    the average in its place, would misstate what you would actually be quoted."""
    df = parse_fixture_odds(io.StringIO(CSV), "PL")
    row = df.iloc[0]
    assert (row["b_home"], row["b_draw"], row["b_away"]) == (2.15, 3.30, 3.40)
    assert row["book"] == BOOK_B365
    # de-vigged probabilities still come from the AVERAGE, not from bet365, so
    # the published edge stays comparable with the backtest
    assert abs(row["m_home"] + row["m_draw"] + row["m_away"] - 1.0) < 1e-9
    assert row["b_home"] != row["m_home"]


def test_missing_bet365_falls_back_to_the_average_and_says_so():
    df = parse_fixture_odds(io.StringIO(CSV_NO_B365), "PL")
    row = df.iloc[0]
    assert (row["b_home"], row["b_draw"], row["b_away"]) == (2.10, 3.40, 3.30)
    assert row["book"] == BOOK_AVG          # never implies a bet365 quote it lacks


def test_market_for_returns_prices_and_the_book():
    df = parse_fixture_odds(io.StringIO(CSV), "PL")
    m = market_for(df, "Arsenal", "Manchester City")
    assert m["odds"] == {"home": 2.15, "draw": 3.30, "away": 3.40}
    assert m["book"] == BOOK_B365


def test_market_for_is_none_when_the_fixture_is_not_priced():
    df = parse_fixture_odds(io.StringIO(CSV), "PL")
    assert market_for(df, "Arsenal", "Chelsea") is None
