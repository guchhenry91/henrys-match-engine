from tracking import pricecheck


def test_break_even_and_ev_at_bet365():
    p = {"probability": 0.6, "book_price": 1.80, "book": "bet365"}
    c = pricecheck.check(p)
    assert c["ok"] and c["break_even_price"] == 1.67 and abs(c["ev"] - 0.08) < 1e-9


def test_estimate_from_another_book_and_no_price():
    p = {"probability": 0.6, "book_price": 1.95, "book_p": 0.5, "book": "pinnacle"}
    c = pricecheck.check(p)
    assert c["basis"] == "estimate" and c["ok"]            # 0.6 x 1.90 - 1 > 0
    q = pricecheck.check({"probability": 0.7})
    assert not q["ok"] and q["reason"] == "no price"


def test_one_pick_per_player_per_game_keeps_the_best_return():
    a = {"game_id": "g", "player_id": 9, "probability": 0.7, "price_check": {"ev": 0.02}}
    b = {"game_id": "g", "player_id": 9, "probability": 0.65, "price_check": {"ev": 0.10}}
    c = {"game_id": "g", "player_id": 8, "probability": 0.65, "price_check": {"ev": 0.01}}
    out = pricecheck.one_per_player({"shots": [a, c], "goal": [b]})
    assert out == {"shots": [c], "goal": [b]} and a["on_board"] is False
