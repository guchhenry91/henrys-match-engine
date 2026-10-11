"""The rebuilt NBA board's decision rule (nba/forecast_board.py)."""
from nba import forecast_board as fb


def test_combine_leans_on_both_and_is_neutral_at_even():
    w = (0.5, 0.5, 0.0)
    assert abs(fb.combine(0.5, 0.5, w) - 0.5) < 1e-9
    both = fb.combine(0.7, 0.7, w)
    assert 0.65 < both < 0.75
    assert fb.combine(0.7, 0.4, w) < both          # a disagreeing book pulls it down


def _p(name, prob, p_model, game="g1"):
    return {"player": name, "probability": prob, "p_model": p_model, "game_id": game}


def test_select_needs_the_bar_and_the_forecast_on_the_same_side():
    got = fb.select([_p("A", 0.70, 0.66), _p("B", 0.68, 0.42),    # B: the book's pick, not ours
                     _p("C", 0.60, 0.70), _p("D", 0.64, 0.55)])
    assert [p["player"] for p in got] == ["A", "D"]


def test_select_caps_each_game():
    priced = [_p(f"P{i}", 0.70 + i / 100, 0.6) for i in range(5)]
    assert len(fb.select(priced)) == fb.TOP_PER_MARKET
