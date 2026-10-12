from tracking import combine_fit, pricecheck


def test_break_even_min_price_and_ev_at_bet365():
    p = {"probability": 0.6, "book_price": 1.80, "book": "bet365"}
    c = pricecheck.check(p)
    assert c["ok"] and c["break_even_price"] == 1.67 and c["min_price"] == 1.72
    assert abs(c["ev"] - 0.08) < 1e-9


def test_plus_three_percent_is_the_bar():
    """EV +2% used to pass (bar 0); it no longer does (third review)."""
    c = pricecheck.check({"probability": 0.6, "book_price": 1.70, "book": "bet365"})
    assert not c["ok"] and c["reason"] == "price too short"


def test_estimate_uses_a_realistic_bet365_margin():
    p = {"probability": 0.6, "book_p": 0.5, "book": "pinnacle", "market": "points"}
    c = pricecheck.check(p)
    assert c["basis"] == "estimate" and c["ref_price"] == round(1 / (0.5 * 1.075), 2)
    one = pricecheck.check({"probability": 0.6, "book_p": 0.5, "book": "fanduel", "market": "shots"})
    assert one["ref_price"] == round(1 / (0.5 * 1.12), 2)
    assert not pricecheck.check({"probability": 0.7})["ok"]
    assert pricecheck.check({"probability": 0.7})["reason"] == "no price"


def test_forecast_far_from_the_book_is_not_published():
    p = {"probability": 0.62, "p_model": 0.80, "book_p": 0.60, "book_price": 1.80, "book": "bet365"}
    c = pricecheck.check(p)
    assert not c["ok"] and c["reason"] == "forecast too far from the book" and c["gap"] == 0.2


def test_agreeing_with_the_book_creates_no_edge():
    """The old stacked prior turned a forecast EQUAL to a 72% book into 75.6% and so
    into 'value' at bet365. Shrunk toward the book it stays 72%: no pick."""
    w = combine_fit.prior((0.2, 1.0, 0.0))
    assert w == (0.2, 0.8, 0.0)
    p = combine_fit._apply(w, combine_fit.np.array([[combine_fit._logit(0.72), combine_fit._logit(0.72)]]))[0]
    assert abs(p - 0.72) < 1e-9
    price = 1 / (0.72 * 1.075)                       # bet365's two-way price on it
    assert not pricecheck.check({"probability": p, "p_model": 0.72, "book_p": 0.72,
                                 "book_price": price, "book": "bet365"})["ok"]


def test_one_pick_per_player_keeps_the_highest_stated_chance():
    """Ranking on expected return first favoured the biggest disagreement (third review)."""
    a = {"game_id": "g", "player_id": 9, "probability": 0.7, "price_check": {"ev": 0.04}}
    b = {"game_id": "g", "player_id": 9, "probability": 0.65, "price_check": {"ev": 0.10}}
    c = {"game_id": "g", "player_id": 8, "probability": 0.65, "price_check": {"ev": 0.04}}
    out = pricecheck.one_per_player({"shots": [a, c], "goal": [b]})
    assert out == {"shots": [a, c], "goal": []} and b["on_board"] is False
