"""Pulling the overconfident NFL yardage model toward the bookmaker's price."""
from nfl import market_blend as mb


def test_blend_keeps_only_the_fitted_share_of_the_disagreement():
    assert mb.blend(0.80, 0.50, 0.0) == 0.50
    assert mb.blend(0.80, 0.50, 1.0) == 0.80
    assert abs(mb.blend(0.80, 0.50, 0.1) - 0.53) < 1e-9


def test_fit_finds_no_trust_for_a_model_that_is_confidently_wrong():
    # Says 70% every time, hits 40%: the book (50%) is the better number.
    rows = [(0.70, 0.50, 1)] * 40 + [(0.70, 0.50, 0)] * 60
    assert mb.fit(rows)["w"] == 0.0


def test_fit_trusts_a_model_that_is_right():
    rows = ([(0.70, 0.50, 1)] * 70 + [(0.70, 0.50, 0)] * 30
            + [(0.30, 0.50, 1)] * 30 + [(0.30, 0.50, 0)] * 70)
    assert mb.fit(rows)["w"] == 1.0


def test_old_median_line_picks_anchor_at_one_half_and_book_picks_at_the_book():
    log = {"a": {"market": "rushing_yards", "graded": "correct", "p_pick": 0.6,
                 "line_source": "model"},
           "b": {"market": "rushing_yards", "graded": "wrong", "p_pick": 0.7,
                 "p_model": 0.75, "line_source": "pinnacle", "book_p": 0.52},
           "c": {"market": "anytime_touchdown", "graded": "correct", "p_pick": 0.5}}
    assert sorted(mb.rows_from_log(log)) == [(0.6, 0.5, 1), (0.75, 0.52, 0)]


def test_a_small_market_uses_the_pooled_weight():
    log = {str(i): {"market": "passing_yards", "graded": "wrong", "p_pick": 0.7,
                    "line_source": "model"} for i in range(10)}
    out = mb.fit_all(log)
    assert out["passing_yards"]["source"] == "pooled"


def test_missing_weights_file_trusts_the_book(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "WEIGHTS", tmp_path / "nope.json")
    assert set(mb.weights().values()) == {0.0}
