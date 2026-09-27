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


# --- the "70%+ line": the highest bookmaker alternate rung still likely to hit ----

def test_parse_alternates_keeps_one_books_over_ladder_per_player():
    from nfl import book_lines as bl
    payload = {"bookmakers": [
        {"key": "fanduel", "markets": [{"key": "player_rush_yds_alternate", "outcomes": [
            {"name": "Over", "description": "James Cook", "point": 39.5, "price": 1.07}]}]},
        {"key": "draftkings", "markets": [{"key": "player_rush_yds_alternate", "outcomes": [
            {"name": "Over", "description": "James Cook", "point": 59.5, "price": 1.32},
            {"name": "Over", "description": "James Cook", "point": 49.5, "price": 1.17},
            {"name": "Under", "description": "James Cook", "point": 49.5, "price": 4.0}]}]}]}
    got = bl.parse_alternates(payload)["rushing_yards"]["James Cook"]
    assert got["source"] == "draftkings" and got["ladder"] == [[49.5, 1.17], [59.5, 1.32]]


def test_safest_is_the_highest_rung_still_at_seventy_percent():
    ladder = {"book": "DraftKings", "rungs": [
        {"line": 39.5, "price": 1.08, "raw": 0.95},
        {"line": 49.5, "price": 1.17, "raw": 0.90},
        {"line": 59.5, "price": 1.32, "raw": 0.80},
        {"line": 69.5, "price": 1.54, "raw": 0.70}]}
    pick = {"book_price": 1.91, "book_p": 0.5}          # main overround ~1.047
    s = mb.safest(ladder, pick, w=0.1)
    # 1/1.32/1.047 = 0.72 anchor -> blended 0.73 >= 0.70; 69.5 -> 0.62 < 0.70
    assert s["line"] == 59.5 and s["book"] == "DraftKings" and 0.70 <= s["p"] < 0.75


def test_a_rung_outside_the_trained_range_is_judged_on_the_price_alone():
    ladder = {"book": "FanDuel", "rungs": [{"line": 14.5, "price": 1.07, "raw": None}]}
    s = mb.safest(ladder, {"book_price": 1.9, "book_p": 0.5}, w=1.0)
    assert s["p"] == round(min((1 / 1.07) / ((1 / 1.9) / 0.5), 0.99), 4)


def test_no_ladder_means_no_safe_line():
    assert mb.safest(None, {}, 0.1) is None
