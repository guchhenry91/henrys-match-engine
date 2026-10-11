"""The rebuilt NBA forecast's features never see the game being forecast."""
import pandas as pd

from nba import forecast as fc


def _rows():
    out = []
    for i, (date, pts, mins) in enumerate([("2025-01-01", 10, 30), ("2025-01-03", 20, 32),
                                           ("2025-01-04", 40, 34)]):
        for team, opp, h in (("BOS", "NYK", "vs."), ("NYK", "BOS", "@")):
            out.append({"SEASON_ID": "22024", "PLAYER_ID": 1 if team == "BOS" else 2,
                        "PLAYER_NAME": "A" if team == "BOS" else "B", "TEAM_ABBREVIATION": team,
                        "GAME_ID": f"00224000{i}", "GAME_DATE": date, "MATCHUP": f"{team} {h} {opp}",
                        "WL": "W", "MIN": mins, "PTS": pts, "REB": 5, "AST": 3, "FG3M": 1,
                        "FGA": 10, "FTA": 2, "FG3A": 3, "TOV": 1, "OREB": 1, "DREB": 4, "season": 2025})
    return pd.DataFrame(out)


def test_features_are_entering_only():
    d = fc.build(_rows())
    a = d[d["PLAYER_ID"] == 1].sort_values("game_date").reset_index(drop=True)
    assert pd.isna(a.loc[0, "PTS_last"]) and a.loc[1, "PTS_last"] == 10 and a.loc[2, "PTS_last"] == 20
    # The third game's 40 points never reach its own features.
    assert a.loc[2, "PTS_pm10"] < 40 / 34
    assert list(a["rest_days"])[1:] == [2, 1] and a.loc[2, "b2b"] == 1.0
    assert a.loc[0, "home"] == 1.0


def _team_rows():
    """BOS: A and B play two games; game 3 A plays, B absent; game 4 upcoming."""
    out = []
    games = [("g1", "2025-01-01"), ("g2", "2025-01-03"), ("g3", "2025-01-05")]
    for gid, date in games:
        for pid, name in ((1, "A"), (2, "B")):
            if gid == "g3" and pid == 2:
                continue
            out.append({"PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ABBREVIATION": "BOS",
                        "GAME_ID": gid, "GAME_DATE": date, "MATCHUP": "BOS vs. NYK", "MIN": 30,
                        "PTS": 20, "REB": 5, "AST": 3, "FG3M": 1, "FGA": 15, "FTA": 5, "FG3A": 4,
                        "TOV": 2, "OREB": 1, "DREB": 4, "season": 2025})
    out.append({"PLAYER_ID": 1, "PLAYER_NAME": "A", "TEAM_ABBREVIATION": "BOS", "GAME_ID": "NEXTg4",
                "GAME_DATE": "2025-01-07", "MATCHUP": "BOS vs. NYK", "MIN": None, "season": 2025,
                "is_next": True})
    return pd.DataFrame(out)


def test_an_absent_rotation_player_vacates_his_usage():
    d = fc.build(_team_rows())
    g = d.drop_duplicates("GAME_ID").set_index("GAME_ID")
    assert g.loc["g2", "vac_share"] == 0.0
    assert abs(g.loc["g3", "vac_share"] - 0.5) < 1e-9      # B, half the rotation's usage, missing
    # Upcoming game: nobody's row exists, so only the ruled-out count as absent.
    assert fc.build(_team_rows()).set_index("GAME_ID").loc["NEXTg4", "vac_share"] == 0.0
    ruled = fc.build(_team_rows(), {"BOS": {2}}).set_index("GAME_ID").loc["NEXTg4", "vac_share"]
    assert ruled > 0.4
