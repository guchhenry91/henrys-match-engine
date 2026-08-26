"""The model must not select, must stay in bounds, and must degrade safely."""
import numpy as np
import pandas as pd
import pytest

from nfl import model as M


def _frame(n=600, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "season": np.repeat([2020, 2021, 2022], n // 3),
        "week": np.tile(np.arange(1, n // 3 + 1), 3),
        "hist_rate": rng.uniform(20, 80, n),
        "form5": rng.uniform(20, 80, n),
        "form10": rng.uniform(20, 80, n),
        "opp5": rng.uniform(2, 10, n),
        "opp_allowed": rng.uniform(150, 300, n),
        "games_before": rng.integers(6, 60, n),
        "is_home": rng.integers(0, 2, n).astype(float),
        "rest_days": np.full(n, 7.0),
        "share5": rng.uniform(0.05, 0.4, n),
        "eff5": rng.uniform(4, 12, n),
        "team_scored5": rng.uniform(14, 30, n),
        "team_allowed5": rng.uniform(14, 30, n),
        "opp_scored5": rng.uniform(14, 30, n),
        "opp_allowed5": rng.uniform(14, 30, n),
        "line": rng.uniform(30, 70, n),
        "outcome": rng.integers(0, 2, n).astype(float),
    })


def test_it_fits_every_candidate_rather_than_choosing_one():
    """Selection variance was the dominant error; averaging removes the choice."""
    fitted = M.PropModel("receiving_yards").fit(_frame())
    assert len(fitted.sets) == len(M.CANDIDATES)
    assert len(fitted.models) == len(M.CANDIDATES)
    assert not hasattr(fitted, "_choose_features"), "the selector is back"


def test_probabilities_stay_in_bounds():
    frame = _frame()
    fitted = M.PropModel("receiving_yards").fit(frame)
    prob = fitted.predict(frame)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    assert len(prob) == len(frame)


def test_a_single_class_falls_back_to_the_baseline_not_a_guess():
    frame = _frame()
    frame["outcome"] = 1.0
    fitted = M.PropModel("receiving_yards").fit(frame)
    assert not fitted.fitted
    assert np.allclose(fitted.predict(frame), M.empirical_baseline(frame, "receiving_yards"))


def test_too_little_data_means_no_calibration_rather_than_in_sample_calibration():
    """The bug that flattered an earlier run: calibrating on memorised rows."""
    fitted = M.PropModel("receiving_yards").fit(_frame(n=90))
    assert fitted.calibrator is None


def test_thin_markets_use_platt_not_isotonic():
    """Isotonic carves step functions out of noise on small samples."""
    from sklearn.isotonic import IsotonicRegression
    fitted = M.PropModel("receiving_yards").fit(_frame(n=900))
    assert fitted.calibrator is None or not isinstance(fitted.calibrator,
                                                       IsotonicRegression)
