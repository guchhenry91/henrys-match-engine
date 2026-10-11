import json

import numpy as np

from nfl import forecast_board as fb
from tracking import combine_fit


def _p(market="receiving_yards", game="2026_06_A_B", prob=0.7, p_model=0.8, side="over", player="x"):
    return {"market": market, "game_id": game, "probability": prob, "p_model": p_model,
            "side": side, "player": player}


def test_select_needs_bar_and_agreement():
    picks = [_p(prob=0.70, p_model=0.8, player="a"), _p(prob=0.61, player="b"),
             _p(prob=0.75, p_model=0.45, player="c")]
    assert [p["player"] for p in fb.select(picks)] == ["a"]


def test_select_td_yes_side_only_and_top_per_game():
    picks = [_p("anytime_touchdown", side="under", prob=0.8, player="no")]
    picks += [_p(prob=0.9 - i / 100, player=str(i)) for i in range(5)]
    out = fb.select(picks)
    assert "no" not in [p["player"] for p in out]
    assert len(out) == fb.TOP_PER_MARKET


def test_combine_book_weight_one_and_zero_pull_is_the_book():
    assert abs(fb.combine(0.9, 0.55, (0.0, 1.0, 0.0)) - 0.55) < 1e-9
    assert fb.combine(0.9, 0.5, (0.2, 1.0, 0.0)) > 0.5


def test_adopted_weights_win(tmp_path, monkeypatch):
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"markets": {"passing_yards": {"combined": {"weights_model_book_const": [0.1, 1.0, 0.0]}}}}))
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"passing_yards": {"w": [0.3, 1.0, 0.05]}}))
    monkeypatch.setattr(fb, "REPORT", report)
    monkeypatch.setattr(fb, "WEIGHTS", w)
    assert fb.weights()["passing_yards"] == (0.3, 1.0, 0.05)


def test_fit_offset_stable_when_book_is_flat():
    rng = np.random.default_rng(0)
    pm = rng.uniform(0.2, 0.8, 3000)
    y = (rng.uniform(size=3000) < 0.5 + 0.3 * (pm - 0.5)).astype(float)
    a, b, c = combine_fit.fit_offset(pm, np.full(3000, 0.5), y)
    assert b == 1.0 and 0.0 < a < 1.5 and abs(c) < 0.2


def test_fit_offset_never_reverses_the_forecast():
    rng = np.random.default_rng(1)
    pm = rng.uniform(0.2, 0.8, 2000)
    y = (rng.uniform(size=2000) < 1 - pm).astype(float)
    assert combine_fit.fit_offset(pm, np.full(2000, 0.5), y)[0] == 0.0


class _Spread:
    """P(over line) falls linearly from 1 at 0 to 0 at 2*mu."""
    def p_over(self, mu, line):
        return np.clip(1 - np.asarray(line, float) / (2 * np.asarray(mu, float)), 0.001, 0.999)


class _Model:
    spread = _Spread()


def test_alt_rung_takes_highest_line_the_forecast_backs():
    pick = {"line": 60.5, "book_price": 1.91, "book_p": 0.5, "book": "fanduel",
            "ladder": {"book": "fanduel", "rungs": [{"line": 30.5, "price": 1.15},
                                                     {"line": 40.5, "price": 1.30},
                                                     {"line": 50.5, "price": 1.55},
                                                     {"line": 70.5, "price": 2.6}]}}
    a = fb._alt_rung(pick, _Model(), 80.0, (0.2, 1.0, 0.0))
    assert a is not None and a["line"] == 40.5          # 50.5's chance is under the bar
    assert a["p_model"] >= a["book_p"] and a["probability"] >= fb.ALT_BAR


def test_alt_rung_none_when_forecast_less_sure_than_book():
    pick = {"line": 60.5, "book_price": 1.91, "book_p": 0.5,
            "ladder": {"book": "x", "rungs": [{"line": 30.5, "price": 1.10}]}}
    assert fb._alt_rung(pick, _Model(), 40.0, (0.2, 1.0, 0.0)) is None


def test_select_alt_skips_players_with_a_main_pick():
    a = {"line": 40.5, "price": 1.3, "book": "x", "probability": 0.75, "p_model": 0.8, "book_p": 0.72}
    priced = [dict(_p(player="main"), player_id=1, alt_candidate=a),
              dict(_p(player="alt"), player_id=2, alt_candidate=a)]
    out = fb.select_alt(priced, [priced[0]])
    assert [p["player"] for p in out] == ["alt"] and out[0]["alt"] and out[0]["line"] == 40.5
    assert "alt_candidate" not in out[0]


def test_alt_pick_has_its_own_record_key():
    from nfl import picks
    base = {"game_id": "g", "market": "receiving_yards", "player_id": 1}
    assert picks.prop_key(dict(base, alt=True)) != picks.prop_key(base)
