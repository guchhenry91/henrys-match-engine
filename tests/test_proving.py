from tracking import proving


def _e(p, g, market="hits", engine="forecast-v1"):
    return {"market": market, "p_pick": p, "graded": g, "engine": engine}


def test_unproven_then_on_target_and_voids_counted():
    es = [_e(0.7, "correct")] * 7 + [_e(0.7, "wrong")] * 3 + [_e(0.7, "void")] + [_e(0.7, "correct", engine=None)]
    s = proving.summary(es)["hits"]
    assert s["n"] == 10 and s["landed"] == 0.7 and s["said"] == 0.7
    assert s["voids"] == 1 and s["status"] == "unproven" and s["needed"] == 290
    big = [_e(0.68, "correct")] * 210 + [_e(0.68, "wrong")] * 100
    assert proving.summary(big)["hits"]["status"] == "on target"


def test_behind_early_when_clearly_missing():
    es = [_e(0.7, "correct")] * 15 + [_e(0.7, "wrong")] * 35
    assert proving.summary(es)["hits"]["status"] == "behind (early)"


def _priced(p, g, price, close_p=None):
    e = dict(_e(p, g), book_price=price, book="bet365",
             price_check={"ref_price": price, "basis": "bet365"})
    if close_p is not None:
        e.update(close_p=close_p, close_price=price)
    return e


def test_on_target_is_not_bettable_without_profit():
    """Calibrated at 68% but priced at 1.45 (needs 69%): on target, not bettable."""
    es = [_priced(0.68, "correct", 1.45)] * 211 + [_priced(0.68, "wrong", 1.45)] * 99
    s = proving.summary(es)["hits"]
    assert s["status"] == "on target" and s["roi"] < 0 and s["bettable"] is False


def test_bettable_needs_a_real_profit_edge():
    win = [_priced(0.62, "correct", 1.80)] * 220 + [_priced(0.62, "wrong", 1.80)] * 100
    s = proving.summary(win)["hits"]
    assert s["status"] == "on target" and s["roi_lo95"] > 0 and s["bettable"] is True


def test_bettable_by_beating_the_close():
    """Profit not yet significant, but the price taken beats the close clearly."""
    es = ([_priced(0.60, "correct", 1.70, close_p=0.62)] * 190
          + [_priced(0.60, "wrong", 1.70, close_p=0.63)] * 120)
    s = proving.summary(es)["hits"]
    assert s["roi"] >= 0 and s["clv_z"] >= 2 and s["bettable"] is True
