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


import numpy as np
from leagues.sim import order_teams


def _one_sim(pts, gd, gf):
    """Shape [1, T] arrays for a single simulated season."""
    return (np.array([pts], dtype=np.int32), np.array([gd], dtype=np.int32),
            np.array([gf], dtype=np.int32))


def _h2h(T, pairs):
    """h2h_pts/h2h_gd [T,T,1]; pairs maps (i,j) -> (points_i_took, gd_i)."""
    hp = np.zeros((T, T, 1), dtype=np.int32)
    hg = np.zeros((T, T, 1), dtype=np.int32)
    for (i, j), (p, g) in pairs.items():
        hp[i, j, 0] = p; hg[i, j, 0] = g
        hp[j, i, 0] = 6 - p if p in (0, 3, 6) else p  # mirror for the simple cases
        hg[j, i, 0] = -g
    return hp, hg


def test_gd_league_breaks_ties_on_goal_difference():
    """PL/Bundesliga/Ligue 1: points -> goal difference -> goals for."""
    pts, gd, gf = _one_sim([70, 70], [20, 5], [60, 60])
    hp, hg = _h2h(2, {(0, 1): (0, -4)})       # team 0 LOST the h2h badly...
    order = order_teams(pts, gd, gf, hp, hg, "gd")
    assert list(order[0]) == [0, 1]           # ...but its better GD still wins


def test_h2h_league_breaks_ties_on_head_to_head_first():
    """La Liga: points -> head-to-head -> goal difference. The club with the worse
    overall GD still finishes above if it won the head-to-head."""
    pts, gd, gf = _one_sim([70, 70], [20, 5], [60, 60])
    hp, hg = _h2h(2, {(0, 1): (0, -4)})       # team 0 lost the h2h (0 pts of 6)
    order = order_teams(pts, gd, gf, hp, hg, "h2h")
    assert list(order[0]) == [1, 0]           # team 1 goes above despite worse GD


def test_h2h_falls_back_to_goal_difference_when_head_to_head_is_level():
    pts, gd, gf = _one_sim([70, 70], [20, 5], [60, 60])
    hp, hg = _h2h(2, {(0, 1): (3, 0)})        # h2h level on points and goals
    order = order_teams(pts, gd, gf, hp, hg, "h2h")
    assert list(order[0]) == [0, 1]           # better overall GD decides


def test_teams_not_tied_on_points_are_never_reordered_by_h2h():
    pts, gd, gf = _one_sim([71, 70], [0, 30], [50, 60])
    hp, hg = _h2h(2, {(0, 1): (0, -5)})       # team 0 lost h2h, worse GD
    order = order_teams(pts, gd, gf, hp, hg, "h2h")
    assert list(order[0]) == [0, 1]           # more points wins outright
