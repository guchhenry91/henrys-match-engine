"""Worth betting at bet365, and how much (tracking/value.py)."""
from tracking import value as v


def test_min_price_is_where_the_pick_returns_three_percent():
    a = v.assess(0.55)
    assert a["min_price"] == round(1.03 / 0.55, 2)       # 1.87
    assert a["value"] is None and a["stake"] == 0.0        # no price to judge at


def test_bet365_price_is_used_directly_when_we_have_it():
    a = v.assess(0.55, fair_p=0.50, price=2.00, price_is_bet365=True)
    assert a["basis"] == "bet365" and a["ref_price"] == 2.0
    assert a["value"] is True and abs(a["ev"] - 0.10) < 1e-9


def test_without_bet365_the_price_is_estimated_from_the_fair_chance_less_a_margin():
    a = v.assess(0.55, fair_p=0.50)
    assert a["basis"] == "estimate"
    assert a["ref_price"] == round(1 / (0.50 * 1.05), 2)   # 1.90
    assert a["value"] is True                               # 0.55 * 1.905 - 1 = +4.8%
    thin = v.assess(0.52, fair_p=0.50)                      # +0.4%: not enough
    assert thin["value"] is False and thin["stake"] == 0.0


def test_a_us_book_price_is_never_taken_as_bet365s():
    a = v.assess(0.55, fair_p=0.53, price=2.20, price_is_bet365=False)
    assert a["basis"] == "estimate" and a["value"] is False


def test_stake_is_quarter_kelly_capped_at_two_percent():
    assert v.stake(0.55, 2.0) == 0.02                       # kelly 10% -> 2.5%, capped
    assert v.stake(0.52, 2.0) == 0.01                       # kelly 4% -> 1%
    assert v.stake(0.50, 2.0) == 0.0


def test_board_annotation_marks_bet365_lines_and_counts_value():
    board = {"games": [{"p_pick": 0.6, "book_p_pick": 0.5}],
             "props": {"rushing_yards": {"picks": [
                 {"probability": 0.56, "book_p": 0.5, "book_price": 1.95, "line_source": "bet365"},
                 {"probability": 0.51, "book_p": 0.5, "book_price": 1.95, "line_source": "pinnacle"}]}}}
    assert v.annotate_line_board(board) == 2
    picks = board["props"]["rushing_yards"]["picks"]
    assert picks[0]["value"]["basis"] == "bet365" and picks[1]["value"]["value"] is False
