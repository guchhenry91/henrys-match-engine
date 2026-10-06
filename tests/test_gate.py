"""Each kind of pick earns its place from its own record (tracking/gate.py)."""
from tracking import gate


def _picks(market, side, n, won, p, price):
    return [{"kind": "prop", "market": market, "side": side, "probability": p,
             "book_price": price, "graded": "correct" if i < won else "wrong"} for i in range(n)]


def test_a_group_winning_far_below_its_stated_chances_is_held_back():
    rep = gate.report(_picks("hits", "over", 42, 19, 0.64, 1.6))
    assert rep["hits|over"]["held"] and "won 19 of 42" in rep["hits|over"]["reason"]


def test_long_shots_losing_as_expected_are_not_held_back():
    rep = gate.report(_picks("hr", "over", 45, 5, 0.135, 7.0))
    assert not rep["hr|over"]["held"]


def test_too_few_picks_are_never_judged():
    rep = gate.report(_picks("strikeouts", "over", 19, 5, 0.6, 1.9))
    assert not rep["strikeouts|over"]["held"]


def test_held_picks_stay_on_the_board_but_lose_their_value_verdict():
    rep = gate.report(_picks("hits", "over", 42, 19, 0.64, 1.6))
    board = {"props": {"hits": {"picks": [
        {"market": "hits", "side": "over", "value": {"value": True, "stake": 0.01}},
        {"market": "hits", "side": "under", "value": {"value": True, "stake": 0.01}}]}},
        "game_props": {"1": {"hits": [{"market": "hits", "side": "over", "value": None}]}}}
    assert gate.apply(board, rep) == 2
    over, under = board["props"]["hits"]["picks"]
    assert over["held_back"] and over["value"]["value"] is False and over["value"]["stake"] == 0
    assert "held_back" not in under and under["value"]["value"] is True
    assert board["gate"]["held_back"] == ["hits|over"]
