"""Model trust: pull picks toward the book by the weight the live record supports."""
from tracking import trust


def test_zero_trust_shows_the_books_own_chance_and_keeps_the_raw_model():
    board = {"games": [{"p_pick": 0.60, "book_p_pick": 0.50}],
             "props": {"hits": {"picks": [{"probability": 0.68, "book_p": 0.59},
                                          {"probability": 0.70, "book_p": 0.60, "locked": True}]}}}
    trust.apply_line_board(board, {"prop": {"w": 0.0}, "winner": {"w": 0.25}})
    p0, p1 = board["props"]["hits"]["picks"]
    assert p0["probability"] == 0.59 and p0["p_model"] == 0.68 and p0["edge"] == 0.0
    assert p1["probability"] == 0.70                       # frozen picks are never re-priced
    g = board["games"][0]
    assert g["p_pick"] == 0.525 and g["p_model"] == 0.60


def test_rows_use_the_raw_model_not_the_published_number():
    entries = [{"kind": "prop", "graded": "correct", "probability": 0.55, "p_model": 0.70, "book_p": 0.5},
               {"kind": "prop", "graded": "void", "probability": 0.55, "book_p": 0.5},
               {"kind": "winner", "graded": "wrong", "p_pick": 0.6, "book_p_pick": 0.55}]
    assert trust.rows(entries, "prop") == [(0.70, 0.5, 1.0)]
    assert trust.rows(entries, "winner") == [(0.6, 0.55, 0.0)]
