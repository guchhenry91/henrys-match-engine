"""The release gate must refuse things, including things I would like to ship.

A gate that passes everything is decoration. These tests assert it still bites:
on sample size, on failing to beat the baseline, and on calibration -- and that
it judges EVERY season rather than an average that a good year can carry.
"""
import numpy as np
import pandas as pd
import pytest

from nfl import backtest, config


def _predictions(seasons, n_per_season, prob, outcome_rate, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for _ in range(n_per_season):
            rows.append({"season": season, "prob": prob,
                         "baseline": 0.5,
                         "outcome": float(rng.random() < outcome_rate)})
    return pd.DataFrame(rows)


def test_ece_is_zero_for_a_perfectly_calibrated_constant():
    prob = np.full(2000, 0.3)
    outcome = np.concatenate([np.ones(600), np.zeros(1400)])
    assert backtest.ece(prob, outcome) < 0.01


def test_ece_catches_a_confidently_wrong_model():
    prob = np.full(2000, 0.9)
    outcome = np.concatenate([np.ones(200), np.zeros(1800)])
    assert backtest.ece(prob, outcome) > 0.5


def test_the_null_grows_as_the_sample_shrinks():
    """Why a fixed 0.04 cannot be right for both a big market and a small one."""
    rng = np.random.default_rng(7)
    big = rng.uniform(0.2, 0.8, 12000)
    small = rng.uniform(0.2, 0.8, 500)
    assert backtest.ece_null(small, draws=60) > backtest.ece_null(big, draws=60)


def test_a_market_that_never_beats_its_baseline_is_refused():
    preds = _predictions([2022, 2023], 800, prob=0.5, outcome_rate=0.5)
    preds["baseline"] = 0.5
    preds["prob"] = 0.9              # confidently wrong, baseline is honest
    result = backtest.evaluate(preds, "test")
    assert not result["released"]
    assert "baseline" in result["reason"]


def test_a_market_too_small_to_judge_is_refused():
    preds = _predictions([2022], 50, prob=0.5, outcome_rate=0.5)
    result = backtest.evaluate(preds, "test")
    assert not result["released"]
    assert "predictions" in result["reason"]


def test_one_bad_season_sinks_a_good_average():
    """The rule that matters: a market is not carried by its best year."""
    good = _predictions([2022, 2023, 2024], 800, prob=0.5, outcome_rate=0.5, seed=2)
    good["prob"] = 0.5
    good["baseline"] = 0.6           # model beats baseline in these seasons
    bad = _predictions([2025], 800, prob=0.5, outcome_rate=0.5, seed=3)
    bad["prob"] = 0.95               # and is a disaster in this one
    bad["baseline"] = 0.5
    result = backtest.evaluate(pd.concat([good, bad], ignore_index=True), "test")
    assert not result["released"]
    assert "2025" in result["reason"]


def test_empty_predictions_are_refused_not_crashed():
    result = backtest.evaluate(pd.DataFrame(), "test")
    assert not result["released"]
