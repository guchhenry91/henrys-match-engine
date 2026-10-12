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
    # A free fit that STACKS the book (weight > 1) inflates favourites; its would-publish
    # picks land below their stated rate, so it is refused even though log-loss improves.
    free = cf.refit(path, wpath, {"points": (0.9, 0.1, 0.0)}, NOW)["points"]
    assert free["candidate"][1] > 1.0 and free["adopted"] is False
    # The shrink-form fit (book moved part way to the forecast) is adopted.
    wpath2 = tmp_path / "w2.json"
    out = cf.refit(path, wpath2, {"points": (0.9, 0.1, 0.0)}, NOW, offset=True)
    assert out["points"]["adopted"] is True, out["points"]["why"]
    a, b, _ = cf.adopted(wpath2)["points"]
    assert b > a and abs(a + b - 1) < 1e-9        # it learned to lean on the book
    # Already-good weights are NOT replaced by a fit that is no better out of fold.
    out = cf.refit(path, tmp_path / "w3.json", {"points": (0.1, 1.0, 0.0)}, NOW)
    assert out["points"]["logloss_candidate_oof"] >= out["points"]["logloss_current"] - 0.002


def test_refit_refuses_a_gain_inside_the_noise(tmp_path):
    """The book is right and in force; a re-fit on pure noise must not be adopted."""
    rng = np.random.default_rng(3)
    path, wpath = tmp_path / "priced.json", tmp_path / "w.json"
    picks, truth = [], {}
    for i in range(400):
        pm, pb = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        picks.append(_pick(i, pm, pb))
        truth[f"g{i}"] = 25 if rng.uniform() < pb else 10
    cf.record(path, picks, NOW)
    cf.grade(path, lambda e: truth[e["game_id"]])
    out = cf.refit(path, wpath, {"points": (0.0, 1.0, 0.0)}, NOW)
    assert out["points"]["adopted"] is False and not cf.adopted(wpath)
    assert out["points"]["why"].startswith(("gain inside the noise", "too few would-publish",
                                            "would-publish picks landed below"))


def test_folds_keep_a_game_together():
    es = [{"game_id": "g1"}, {"game_id": "g1"}, {"game_id": "g2"}]
    f = cf._folds(es)
    assert f[0] == f[1]


def test_record_keeps_nfl_string_player_ids(tmp_path):
    """The 2026-10-12 05:00 NFL publish crashed on int('00-0039875')."""
    from datetime import datetime, timezone
    from tracking import combine_fit as cf
    path = tmp_path / "lines.json"
    picks = [{"game_id": "2026_06_BUF_NYJ", "player_id": "00-0039875", "market": "rushing_yards",
              "line": 55.5, "tipoff": "2026-10-18T17:00:00Z", "side": "over",
              "p_model": 0.6, "book_p": 0.5, "probability": 0.55},
             {"game_id": "0022600001", "player_id": 1630596, "market": "points",
              "line": 20.5, "tipoff": "2026-10-20T23:00:00Z", "side": "over",
              "p_model": 0.6, "book_p": 0.5, "probability": 0.55}]
    now = datetime(2026, 10, 12, 5, 0, tzinfo=timezone.utc)
    cf.record(path, picks, now)
    import json
    ids = sorted(str(e["player_id"]) for e in json.loads(path.read_text()).values()
                 if isinstance(e, dict) and "player_id" in e)
    assert "00-0039875" in ids


def test_refit_waits_for_new_lines_and_revokes_when_the_book_wins(tmp_path):
    """Third review: no re-test until REFIT_STEP new lines; adopted weights that the
    book alone now clearly beats are dropped."""
    rng = np.random.default_rng(5)
    path, wpath = tmp_path / "priced.json", tmp_path / "w.json"
    picks, truth = [], {}
    for i in range(600):
        pm, pb = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        picks.append(_pick(i, pm, pb))
        truth[f"g{i}"] = 25 if rng.uniform() < pb else 10       # the book is right
    cf.record(path, picks[:400], NOW)
    cf.grade(path, lambda e: truth[e["game_id"]])
    # Pretend a strongly model-leaning weight had been adopted earlier on 360 lines.
    cf._save(wpath, {"points": {"n": 360, "w": [0.9, 0.1, 0.0]}})
    out = cf.refit(path, wpath, {"points": (0.9, 0.1, 0.0)}, NOW)
    assert out["points"]["n"] == 360                  # 400 < 360 + 50: not looked at again
    cf.record(path, picks, NOW)
    cf.grade(path, lambda e: truth[e["game_id"]])
    # A candidate that adds nothing (it IS the weights in force) is not adopted; the
    # backstop then finds the book alone clearly better and revokes the old weights.
    monkey = cf.fit_offset
    cf.fit_offset = lambda pm, pb, y, l2=1.0: (0.9, 0.1, 0.0)
    try:
        out = cf.refit(path, wpath, {"points": (0.9, 0.1, 0.0)}, NOW, offset=True)
    finally:
        cf.fit_offset = monkey
    assert out["points"]["n"] == 600 and out["points"]["adopted"] is False
    assert "revoked" in out["points"] and "points" not in cf.adopted(wpath)


def test_record_skips_a_row_with_no_line_and_keeps_the_rest(tmp_path):
    """Live NFL, 2026-10-12: one anytime-TD row with line None sank the whole log."""
    path = tmp_path / "priced.json"
    bad = dict(_pick(1, 0.6, 0.5), line=None)
    n = cf.record(path, [bad, _pick(2, 0.6, 0.5)], NOW)
    assert n == 1 and list(cf._load(path)) == ["g2:2:points"]


def test_nfl_td_rows_are_logged_as_a_half_line(monkeypatch):
    from nfl import publish
    seen = []

    def capture(path, priced, now):
        seen.extend(priced)
        raise RuntimeError("stop here")              # the caller's guard returns
    monkeypatch.setattr(publish, "REBUILT", True)
    monkeypatch.setattr(cf, "record", capture)
    td = {"engine": "forecast-v1", "market": "anytime_touchdown", "line": None,
          "kickoff": "2026-10-13T00:15:00+00:00", "p_model": 0.3, "book_p": 0.35}
    yd = {"engine": "forecast-v1", "market": "rushing_yards", "line": 55.5,
          "kickoff": "2026-10-13T00:15:00+00:00", "p_model": 0.5, "book_p": 0.5}
    publish._rebuilt_weights({"game_props": {"g": {"anytime_touchdown": [td], "rushing_yards": [yd]}}})
    assert sorted(p["line"] for p in seen) == [0.5, 55.5]
    assert all(p["tipoff"] == "2026-10-13T00:15:00+00:00" for p in seen)
