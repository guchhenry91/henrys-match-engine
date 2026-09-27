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


def test_ladder_up_lists_higher_rungs_with_falling_chances():
    ladder = {"book": "DraftKings", "rungs": [
        {"line": 39.5, "price": 1.30, "raw": 0.80},
        {"line": 55.5, "price": 2.30, "raw": 0.45},
        {"line": 65.5, "price": 3.05, "raw": 0.30},
        {"line": 75.5, "price": 4.10, "raw": None},
        {"line": 150.5, "price": 60.0, "raw": None}]}
    pick = {"line": 46.5, "probability": 0.52, "book_price": 1.91, "book_p": 0.5}
    up = mb.ladder_up(ladder, pick, w=0.1)
    assert [r["line"] for r in up] == [55.5, 65.5, 75.5]      # none below, 1.6% one cut
    assert up[0]["p"] > up[1]["p"] > up[2]["p"] and up[0]["book"] == "DraftKings"
    assert all(r["p"] <= 0.52 for r in up)


def test_no_ladder_means_nothing_to_show():
    assert mb.ladder_up(None, {"line": 20.5}, 0.1) == []
