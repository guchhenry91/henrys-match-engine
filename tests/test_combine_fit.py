"""Re-fitting forecast-vs-book weights on real graded lines (tracking/combine_fit.py)."""
from datetime import datetime, timezone

import numpy as np

from tracking import combine_fit as cf

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _pick(i, p_model, p_book, side="over", tip="2026-10-21T23:30:00Z"):
    return {"game_id": f"g{i}", "player_id": i, "market": "points", "line": 20.5,
            "tipoff": tip, "side": side, "p_model": p_model, "book_p": p_book}


def test_record_keeps_the_last_quote_before_tipoff_and_stores_the_over(tmp_path):
    path = tmp_path / "priced.json"
    cf.record(path, [_pick(1, 0.6, 0.55)], NOW)
    cf.record(path, [_pick(1, 0.7, 0.52, side="under")], NOW)        # newer quote, under side
    e = cf._load(path)["g1:1:points"]
    assert abs(e["p_model"] - 0.3) < 1e-9 and abs(e["p_book"] - 0.48) < 1e-9
    late = datetime(2026, 10, 22, tzinfo=timezone.utc)
    cf.record(path, [_pick(1, 0.9, 0.9)], late)                       # after tip-off: ignored
    assert abs(cf._load(path)["g1:1:points"]["p_model"] - 0.3) < 1e-9


def test_grade_pushes_and_results(tmp_path):
    path = tmp_path / "priced.json"
    cf.record(path, [_pick(1, 0.6, 0.5), _pick(2, 0.6, 0.5), _pick(3, 0.6, 0.5)], NOW)
    actual = {"g1": 25, "g2": 20.5, "g3": None}
    cf.grade(path, lambda e: actual[e["game_id"]])
    log = cf._load(path)
    assert log["g1:1:points"]["y"] == 1 and log["g2:2:points"]["y"] == "push"
    assert log["g3:3:points"]["y"] is None


def test_refit_waits_then_adopts_only_when_better(tmp_path):
    rng = np.random.default_rng(0)
    path, wpath = tmp_path / "priced.json", tmp_path / "w.json"
    picks, truth = [], {}
    for i in range(1200):
        pm, pb = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        # Truth: the BOOK carries nearly everything.
        p = 1 / (1 + np.exp(-(0.1 * np.log(pm / (1 - pm)) + 1.0 * np.log(pb / (1 - pb)))))
        picks.append(_pick(i, pm, pb))
        truth[f"g{i}"] = 25 if rng.uniform() < p else 10
    cf.record(path, picks[:100], NOW)
    cf.grade(path, lambda e: truth[e["game_id"]])
    out = cf.refit(path, wpath, {"points": (0.9, 0.1, 0.0)}, NOW)
    assert out["_waiting"]["points"] == 100 and not cf.adopted(wpath)
    cf.record(path, picks, NOW)
    cf.grade(path, lambda e: truth[e["game_id"]])
    out = cf.refit(path, wpath, {"points": (0.9, 0.1, 0.0)}, NOW)
    assert out["points"]["adopted"] is True
    a, b, _ = cf.adopted(wpath)["points"]
    assert b > a                                   # it learned to lean on the book
    # Already-good weights are NOT replaced by a fit that is no better out of fold.
    out = cf.refit(path, wpath, {"points": (0.1, 1.0, 0.0)}, NOW)
    assert out["points"]["logloss_candidate_oof"] >= out["points"]["logloss_current"] - 0.002
