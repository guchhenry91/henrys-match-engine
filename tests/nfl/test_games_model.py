"""The Elo model must be causal, handle ties, and respect neutral sites."""
import pandas as pd
import pytest

from nfl import games_model


def _games(rows):
    frame = pd.DataFrame(rows)
    frame["played"] = frame["home_score"].notna()
    frame["winner"] = None
    frame.loc[frame["home_score"] > frame["away_score"], "winner"] = "home"
    frame.loc[frame["home_score"] < frame["away_score"], "winner"] = "away"
    frame.loc[frame["home_score"] == frame["away_score"], "winner"] = "tie"
    frame["gameday"] = pd.to_datetime("2024-09-08")
    return frame


def _row(week, home, away, hs, as_, season=2024, location="Home"):
    return {"season": season, "week": week, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_, "location": location}


def test_expected_is_symmetric_and_even_at_equal_ratings():
    assert games_model.expected(1500, 1500) == pytest.approx(0.5)
    assert (games_model.expected(1600, 1500)
            == pytest.approx(1 - games_model.expected(1500, 1600)))


def test_a_win_raises_the_winner_and_lowers_the_loser_equally():
    out = games_model.run_elo(_games([_row(1, "AAA", "BBB", 30, 10)]),
                              k=20, home_edge=25, regression=0.33)
    assert len(out) == 1
    # zero-sum: what one side gains the other loses
    assert out.iloc[0]["rating_home"] == pytest.approx(1500.0)


def test_the_probability_for_a_game_cannot_contain_that_game():
    """Causality: the first meeting is priced from nothing, not from its result."""
    frame = _games([_row(1, "AAA", "BBB", 50, 0), _row(2, "AAA", "BBB", 50, 0)])
    out = games_model.run_elo(frame, k=20, home_edge=0, regression=0.0)
    assert out.iloc[0]["prob_home"] == pytest.approx(0.5), "the result leaked backwards"
    assert out.iloc[1]["prob_home"] > 0.5, "the first result never reached the second game"


def test_a_neutral_site_gets_no_home_edge():
    home = games_model.run_elo(_games([_row(1, "AAA", "BBB", None, None)]),
                               k=20, home_edge=100, regression=0.0)
    neutral = games_model.run_elo(
        _games([_row(1, "AAA", "BBB", None, None, location="Neutral")]),
        k=20, home_edge=100, regression=0.0)
    assert home.iloc[0]["prob_home"] > 0.5
    assert neutral.iloc[0]["prob_home"] == pytest.approx(0.5)


def test_a_tie_moves_ratings_toward_each_other_not_to_one_side():
    frame = _games([_row(1, "AAA", "BBB", 17, 17), _row(2, "AAA", "BBB", None, None)])
    out = games_model.run_elo(frame, k=20, home_edge=0, regression=0.0)
    # a tie between equals should leave the second meeting still even
    assert out.iloc[1]["prob_home"] == pytest.approx(0.5, abs=0.02)


def test_between_seasons_ratings_regress_toward_the_mean():
    frame = _games([_row(1, "AAA", "BBB", 60, 0, season=2023),
                    _row(1, "AAA", "BBB", None, None, season=2024)])
    strong = games_model.run_elo(frame, k=40, home_edge=0, regression=0.0)
    regressed = games_model.run_elo(frame, k=40, home_edge=0, regression=1.0)
    assert strong.iloc[1]["prob_home"] > regressed.iloc[1]["prob_home"]
    assert regressed.iloc[1]["prob_home"] == pytest.approx(0.5)


def test_margin_multiplier_damps_blowouts_by_heavy_favourites():
    """Otherwise good teams inflate forever by beating bad ones."""
    even = games_model.mov_multiplier(28, 0)
    lopsided = games_model.mov_multiplier(28, 600)
    assert lopsided < even
