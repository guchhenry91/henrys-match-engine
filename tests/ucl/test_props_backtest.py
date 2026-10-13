from scripts import ucl_props_backtest as bt


def test_batches_never_hold_a_club_twice():
    fx = [{"date": "2024-10-01", "home_us": "A", "away_us": "B"},
          {"date": "2024-10-02", "home_us": "C", "away_us": "D"},
          {"date": "2024-10-22", "home_us": "B", "away_us": "C"},
          {"date": "2024-10-23", "home_us": "E", "away_us": "F"}]
    out = bt.batches(fx)
    assert [len(b) for b in out] == [2, 2]
    for b in out:
        clubs = [c for f in b for c in (f["home_us"], f["away_us"])]
        assert len(clubs) == len(set(clubs))


def test_verdict_needs_sample_and_calibration():
    assert not bt.verdict({"n": 10, "said": 0.5, "landed": 0.5})["pass"]
    assert bt.verdict({"n": 200, "said": 0.66, "landed": 0.65})["pass"]
    assert not bt.verdict({"n": 200, "said": 0.66, "landed": 0.55})["pass"]
