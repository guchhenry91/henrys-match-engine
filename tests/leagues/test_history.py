import io
import pandas as pd
from leagues.history import parse_history

CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgCH,AvgCD,AvgCA\n"
    "E0,16/08/2025,Liverpool,Man Utd,2,1,H,1.80,3.60,4.50\n"
    "E0,17/08/2025,Arsenal,Chelsea,0,0,D,2.10,3.40,3.60\n"
)


def test_parse_history_normalizes_and_types():
    df = parse_history(io.StringIO(CSV), "PL", season="2526")
    assert len(df) == 2
    assert df.loc[0, "home"] == "Liverpool"
    assert df.loc[0, "away"] == "Manchester United"
    assert df.loc[0, "home_goals"] == 2
    assert df.loc[0, "season"] == "2526"
    assert isinstance(df.loc[0, "date"], pd.Timestamp)


def test_parse_history_keeps_closing_odds():
    df = parse_history(io.StringIO(CSV), "PL", season="2526")
    assert df.loc[0, "odds_h"] == 1.80
    assert df.loc[0, "odds_d"] == 3.60
    assert df.loc[0, "odds_a"] == 4.50
