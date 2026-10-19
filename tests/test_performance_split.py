"""The current model is judged on its own record (owner, 2026-10-19)."""
from tracking import performance


def _e(market, won, engine=None, kind=None):
    e = {"market": market, "graded": "correct" if won else "wrong", "book_price": 2.0}
    if engine:
        e["engine"] = engine
    if kind:
        e["kind"] = kind
    return e


def test_retired_props_are_scored_apart():
    entries = [_e("hits", True, "forecast-v1"), _e("hits", False),
               _e("hr", False), _e(None, True, kind="winner")]
    now, old = performance.split_by_model(entries, performance.market_of)
    assert now["all"]["bets"] == 2 and now["hits"]["bets"] == 1
    assert "hr" not in now
    assert old["bets"] == 2 and old["wins"] == 0


def test_no_retired_picks_means_no_retired_line():
    now, old = performance.split_by_model([_e("hits", True, "forecast-v1")], performance.market_of)
    assert old is None and now["all"]["bets"] == 1
