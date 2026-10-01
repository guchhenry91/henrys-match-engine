"""Last-five strips (tracking/recent.py)."""
import pandas as pd

from tracking import recent


def test_last_five_is_oldest_first_and_skips_games_he_did_not_play():
    rows = pd.DataFrame({
        "PLAYER_ID": [1] * 7 + [2],
        "game_date": pd.to_datetime(["2026-01-0%d" % d for d in range(1, 8)] + ["2026-01-01"]),
        "PTS": [10, 11, 12, 13, 0, 15, 16, 30],
        "MIN": [30, 30, 30, 30, 0, 30, 30, 25],
    })
    got = recent.last_n(rows, [1], "PTS", played="MIN")
    assert got == {1: [11.0, 12.0, 13.0, 15.0, 16.0]}     # the DNP is not a zero
    assert recent.last_n(rows, [3], "PTS") == {}
