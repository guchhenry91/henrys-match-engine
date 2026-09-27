"""Soccer props recalibrated from their own record, penalised toward no change."""
from leagues import prop_calibration as pc


def _rows(market, p, hit, n):
    return [{"market": market, "p_pick": p, "graded": "correct" if i < hit else "wrong"}
            for i in range(n)]


def test_an_underconfident_market_is_moved_up_but_not_all_the_way():
    settled = _rows("shots", 0.65, 45, 50)                 # 65% stated, 90% landed
    ups = [{"market": "shots", "p_pick": 0.65}]
    fits = pc.apply(ups, settled)
    assert "shots" in fits and 0.65 < ups[0]["p_pick"] < 0.90
    assert ups[0]["p_model"] == 0.65


def test_a_thin_market_is_left_alone():
    ups = [{"market": "goal", "p_pick": 0.40}]
    pc.apply(ups, _rows("goal", 0.40, 20, 30))
    assert ups[0]["p_pick"] == 0.40 and "p_model" not in ups[0]
