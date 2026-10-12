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
