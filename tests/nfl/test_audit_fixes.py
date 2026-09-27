"""The 2026-09-27 audit fixes: frozen feature scaling, the published-picks
correction, two-sided picks, and under grading."""
import numpy as np
import pandas as pd

from nfl import model as m
from nfl import picks as nfl_picks
from nfl import selection


def test_opponent_scaling_is_frozen_from_training_not_the_batch():
    frame = pd.DataFrame({"opp_allowed": [10.0, 20.0, 30.0], "line": 5.0, "hist_rate": 1.0,
                          "form5": 1.0, "form10": 1.0, "opp5": 1.0, "games_before": 9})
    stats = m.frame_stats(frame)
    one = m.frame_features(frame.iloc[[2]], "receiving_yards", stats)
    assert abs(one["opp_allowed_edge"].iloc[0] - 1.0) < 1e-9       # (30-20)/10, not 0
    alone = m.frame_features(frame.iloc[[2]], "receiving_yards")
    assert alone["opp_allowed_edge"].iloc[0] == 0.0                 # the old batch behaviour


def test_anytime_touchdown_is_calibrated_smoothly():
    assert "anytime_touchdown" in m.SMOOTH_CALIBRATION


def test_shrink_is_fitted_on_the_published_subset():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.6, 0.8, 5000)
    o = (rng.uniform(size=5000) < 0.5 + 0.8 * (p - 0.5)).astype(float)   # overstated by 20%
    s = selection.fit_shrink(p, o)
    assert 0.75 <= s <= 0.85


def test_sides_produce_an_under_at_the_books_under_price():
    quote = {"over": 0.52, "under": 0.48, "odd_over": 1.85, "odd_under": 1.95}
    over, under = selection.sides({"probability": 0.40, "book_p": 0.52, "book_price": 1.85}, quote)
    assert over["side"] == "over" and abs(over["probability"] - 0.40) < 1e-9
    assert under["side"] == "under" and abs(under["probability"] - 0.60) < 1e-9
    assert under["book_price"] == 1.95 and abs(under["edge"] - 0.12) < 1e-9


def test_an_under_wins_below_the_line_and_pushes_on_it():
    base = {"market": "rushing_yards", "line": 40.5, "side": "under", "tainted": False}
    assert nfl_picks.grade_prop(base, {"rushing_yards": 30.0})["graded"] == "correct"
    assert nfl_picks.grade_prop(base, {"rushing_yards": 55.0})["graded"] == "wrong"
    whole = dict(base, line=40.0)
    assert nfl_picks.grade_prop(whole, {"rushing_yards": 40.0})["graded"] == "void"


def test_an_under_is_its_own_frozen_bet():
    pick = {"game_id": "G", "market": "rushing_yards", "player_id": "P"}
    assert nfl_picks.prop_key(pick) != nfl_picks.prop_key(dict(pick, side="under"))
