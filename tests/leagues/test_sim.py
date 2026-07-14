import pandas as pd

from leagues.sim import final_table, rank_teams


def test_tiebreakers_points_then_gd_then_gf():
    rows = [
        {"team": "A", "points": 10, "gd": 5, "gf": 12},
        {"team": "B", "points": 10, "gd": 5, "gf": 15},   # same pts+gd, more GF -> above A
        {"team": "C", "points": 11, "gd": 0, "gf": 3},    # more points -> top
        {"team": "D", "points": 10, "gd": 2, "gf": 20},   # worse gd -> below A and B
    ]
    assert rank_teams(pd.DataFrame(rows)) == ["C", "B", "A", "D"]


def test_played_results_are_locked_in():
    """A team that already won 3-0 keeps those points in every simulation."""
    played = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 3, "away_goals": 0, "played": True},
    ])
    remaining = pd.DataFrame(columns=["home", "away"])
    table = final_table(played, remaining, sample=lambda h, a: (0, 0)).set_index("team")
    assert table.loc["A", "points"] == 3 and table.loc["A", "gd"] == 3
    assert table.loc["B", "points"] == 0 and table.loc["B", "gd"] == -3


def test_remaining_fixtures_are_sampled():
    played = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals", "played"])
    remaining = pd.DataFrame([{"home": "A", "away": "B"}])
    table = final_table(played, remaining, sample=lambda h, a: (2, 1)).set_index("team")
    assert table.loc["A", "points"] == 3 and table.loc["A", "gf"] == 2
