import numpy as np
import pandas as pd

from mlb import forecast as fc
from mlb import forecast_board as fb


def _p(market="hits", game="1", prob=0.7, p_model=0.7, side="over", player="x"):
    return {"market": market, "game_id": game, "probability": prob, "p_model": p_model,
            "side": side, "player": player}


def test_select_bar_agreement_and_hr_yes_only():
    picks = [_p(prob=0.70, player="a"), _p(prob=0.61, player="b"),
             _p(prob=0.75, p_model=0.45, player="c"),
             _p("hr", prob=0.9, side="under", player="d")]
    assert [p["player"] for p in fb.select(picks)] == ["a"]


def test_select_top_per_game():
    picks = [_p(prob=0.9 - i / 100, player=str(i)) for i in range(5)]
    assert len(fb.select(picks)) == fb.TOP_PER_MARKET


def test_rules_hold_hrr_and_hr():
    assert fb.RULE["hrr"] == "held" and fb.RULE["hr"] == "held"
    assert fb.RULE["hits"] == "combined" and fb.RULE["strikeouts"] == "held"


def test_count_model_reads_any_line_off_the_distribution():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(0, 1, n)
    rows = pd.DataFrame({c: 0.0 for c in fc.FEATURES["pitcher"]}, index=range(n))
    rows["krate_e20"] = x
    rows["games_before"] = 10
    rows["is_next"] = False
    rows["K"] = rng.poisson(2 + 8 * x)
    m = fc.CountModel("strikeouts").fit(rows)
    hi, lo = rows.iloc[[0]].assign(krate_e20=0.95), rows.iloc[[0]].assign(krate_e20=0.05)
    assert m.p_over(hi, 5.5)[0] > m.p_over(lo, 5.5)[0]
    p = m.p_over(pd.concat([hi] * 3), np.array([3.5, 5.5, 7.5]))
    assert p[0] >= p[1] >= p[2]                          # higher line, lower chance
    assert abs(m.pmf(hi).sum() - 1) < 1e-9
