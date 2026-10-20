"""The rebuilt NBA board's decision rule (nba/forecast_board.py)."""
from nba import forecast_board as fb


def test_combine_leans_on_both_and_is_neutral_at_even():
    w = (0.5, 0.5, 0.0)
    assert abs(fb.combine(0.5, 0.5, w) - 0.5) < 1e-9
    both = fb.combine(0.7, 0.7, w)
    assert 0.65 < both < 0.75
    assert fb.combine(0.7, 0.4, w) < both          # a disagreeing book pulls it down


def _p(name, prob, p_model, game="g1"):
    return {"player": name, "probability": prob, "p_model": p_model, "game_id": game,
            "book_price": 2.0, "book_p": p_model, "book": "bet365"}


def test_select_needs_the_bar_and_the_forecast_on_the_same_side():
    got = fb.select([_p("A", 0.70, 0.66), _p("B", 0.68, 0.42),    # B: the book's pick, not ours
                     _p("C", 0.60, 0.70), _p("D", 0.64, 0.55)])
    assert [p["player"] for p in got] == ["A", "D"]


def test_select_caps_each_game():
    priced = [_p(f"P{i}", 0.70 + i / 100, 0.6) for i in range(5)]
    assert len(fb.select(priced)) == fb.TOP_PER_MARKET


def test_select_keeps_a_short_priced_contender_as_proving_only():
    """70% at 1.30 loses money (needs 1.43): the price check keeps it off the board."""
    short = dict(_p("short", 0.70, 0.70), book_price=1.30, book="bet365", book_p=0.66)   # model backs it
    fair = dict(_p("fair", 0.70, 0.70), book_price=1.60, book="bet365")
    out = fb.select([short, fair])
    # Owner 2026-10-12: a likely pick the model backs is published as a PROVING pick --
    # frozen and graded, never a bet at that price. Only the fair-priced one is value.
    assert sorted(p["player"] for p in out) == ["fair", "short"]
    assert short["pick_kind"] == "proving" and fair["pick_kind"] == "value"
    assert short["price_check"]["reason"] == "price too short"
    assert short["price_check"]["break_even_price"] == 1.43


def test_no_under_on_a_player_whose_minutes_dipped_late_last_season():
    """Opening-night audit, 2026-10-19: such players were under-forecast by 1.7 points
    over four openers -- an under on him is the model's error, not an edge."""
    from nba import forecast_board as fb
    base = {"market": "rebounds", "player": "X", "player_id": 1, "game_id": "g", "line": 5.5,
            "probability": 0.60, "p_model": 0.70, "book_p": 0.57, "book_price": 1.75,
            "season_games_before": 0, "min_dip": 8.0}
    assert fb._dip_under({**base, "side": "under"})
    assert not fb._dip_under({**base, "side": "over"})
    assert not fb._dip_under({**base, "side": "under", "season_games_before": 1})
    assert not fb._dip_under({**base, "side": "under", "min_dip": 2.0})
    assert fb.select([{**base, "side": "under"}]) == []
