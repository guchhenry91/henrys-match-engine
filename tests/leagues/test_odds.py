import io

from leagues.odds import parse_fixture_odds

CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
    "E0,22/08/2026,15:00,Arsenal,Man City,2.10,3.40,3.30\n"
    "SP1,22/08/2026,20:00,Barcelona,Sevilla,1.50,4.20,6.00\n"    # different league
    "E0,22/08/2026,17:30,Nott'm Forest,Brighton,,,\n"            # no odds posted yet
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
    assert list(df.columns) == ["date", "home", "away", "m_home", "m_draw", "m_away"]


def test_unpriced_fixture_is_skipped_not_crashed():
    df = parse_fixture_odds(io.StringIO(CSV), "PL")
    assert "Nottingham Forest" not in set(df["home"])           # the blank-odds row dropped
