"""MLB engine: entering-only features, the line spread, and the gate's shape."""
import numpy as np
import pandas as pd

from mlb import config, features


def _rows(n=40, stat_values=None):
    stat_values = stat_values or [i % 3 for i in range(n)]
    return pd.DataFrame({
        "PLAYER_ID": "p1", "TEAM": "NYA", "GAME_ID": [f"G{i:03d}" for i in range(n)],
        "game_date": pd.date_range("2024-04-01", periods=n), "season": 2024,
        "opponent": "BOS", "is_home": 1.0, "PA": 4, "H": stat_values,
    })


def test_fifteen_seasons_scored_after_a_four_season_burn_in():
    assert len(config.SCORED_SEASONS) == 15
    assert config.SCORED_SEASONS[0] == 2011 and config.SCORED_SEASONS[-1] == 2025


def test_features_never_see_the_game_they_predict():
    rows = _rows(40, [0] * 39 + [9])                 # a freak last game
    built = features.build(rows, "hits")
    last = built.iloc[-1]
    assert last["form5"] == 0 and last["hist_rate"] == 0     # its own 9 is invisible


def test_lines_are_on_the_half_and_inside_what_books_quote():
    built = features.augment_lines(features.build(_rows(), "hits"), "hits")
    assert ((built["line"] * 2) % 2 == 1).all()
    assert built["line"].between(config.MIN_LINE["hits"], config.MAX_LINE["hits"]).all()
    assert set(built["line_step"]) <= set(config.LINE_STEPS["hits"])


def test_outcome_is_resettled_at_each_line():
    built = features.augment_lines(features.build(_rows(), "hits"), "hits")
    assert (built["outcome"] == (built["H"] > built["line"]).astype(float)).all()


def test_every_requested_market_is_configured():
    assert set(config.MARKETS) == {"hits", "hrr", "hr", "rbi", "strikeouts", "team_runs"}


def test_a_floored_median_never_counts_one_game_twice():
    rows = _rows(40, [0] * 40)                          # median 0 -> base line -0.5
    rows = rows.rename(columns={"H": "RBI"})
    built = features.augment_lines(features.build(rows, "rbi"), "rbi")
    assert not built.duplicated(subset=["PLAYER_ID", "GAME_ID", "line"]).any()
