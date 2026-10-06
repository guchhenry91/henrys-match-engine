"""NFL price edges against Pinnacle's last pre-kickoff price (nfl/mispricing.py)."""
from nfl import mispricing as m
from tracking import performance


def _edge(**kw):
    return {"game": "NO|ATL", "market": "rushing_yards", "player": "Bijan Robinson",
            "side": "over", "line": 70.5, "b365": 1.95, "book_price": 1.95, "book": "bet365",
            "fair_p": 0.54, "ev": 0.053, "kickoff": "2026-10-06T00:15:00+00:00",
            "first_seen": "2026-10-05T09:00:00+00:00", **kw}


def _sharp(checks, over=1.80, under=2.05, line=70.5):
    return {"checks": checks, "props": {"rushing_yards": {"Bijan Robinson": {
        "line": line, "odd_over": over, "odd_under": under, "book": "Pinnacle"}}}}


def test_the_last_check_before_kickoff_is_the_close():
    e = _edge()
    assert m.stamp_close(e, _sharp(["2026-10-05T09:00:00+00:00", "2026-10-05T22:30:00+00:00"]),
                         {"rushing_yards": {"Bijan Robinson": {"odd_over": 1.87}}})
    assert 0.53 < e["close_p"] < 0.54 and e["close_price"] == 1.87
    assert performance.clv(e)[0] == "ours"          # 1.95 still beat Pinnacle's close


def test_the_flag_check_itself_is_never_the_close():
    e = _edge()
    assert not m.stamp_close(e, _sharp(["2026-10-05T09:00:00+00:00"]), None)
    assert "close_p" not in e


def test_a_check_after_kickoff_is_not_a_close():
    e = _edge()
    assert not m.stamp_close(e, _sharp(["2026-10-06T01:00:00+00:00"]), None)
