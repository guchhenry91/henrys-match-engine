"""bet365 pre-match and closing prices stamped onto graded match picks."""
import pandas as pd

from leagues import closing

CSV = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,B365CH,B365CD,B365CA\n"
       "E0,21/08/2026,Arsenal,Coventry,3,0,1.2,7,13,1.18,7.5,13\n"
       "E0,20/09/2026,Bournemouth,Liverpool,0,1,3.4,3.6,2.1,3.6,3.7,2.0\n")
PLAYED = pd.DataFrame([{"match_id": 1, "home": "Arsenal", "away": "Coventry"},
                       {"match_id": 2, "home": "Bournemouth", "away": "Liverpool"}])


def test_graded_picks_get_bet365_price_open_and_close():
    rows = closing.b365_rows("PL", CSV)
    log = {"2026:1": {"pick": "Arsenal", "graded": "correct"},
           "2026:2": {"pick": "Liverpool", "graded": "correct", "odds": 2.05},
           "2026:3": {"pick": "Draw", "graded": None}}
    assert closing.stamp(log, PLAYED, rows, "2026") == 2
    a = log["2026:1"]
    assert a["odds"] == 1.2 and a["open"]["price"] == 1.2 and a["close_price"] == 1.18
    assert a["close_p"] > a["open"]["p"]          # the market moved toward Arsenal
    # A price frozen at lock is never overwritten.
    assert log["2026:2"]["odds"] == 2.05 and log["2026:2"]["close_price"] == 2.0
    assert "odds" not in log["2026:3"]            # ungraded: untouched


def test_a_draw_pick_reads_the_draw_column_and_reruns_change_nothing():
    rows = closing.b365_rows("PL", CSV)
    log = {"2026:2": {"pick": "Draw", "graded": "wrong"}}
    closing.stamp(log, PLAYED, rows, "2026")
    assert log["2026:2"]["odds"] == 3.6
    assert closing.stamp(log, PLAYED, rows, "2026") == 0
