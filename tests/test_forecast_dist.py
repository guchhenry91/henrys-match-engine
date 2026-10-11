"""The rebuild's shared distribution core (forecast/dist.py)."""
import numpy as np

from forecast import dist


def test_p_over_falls_as_the_line_rises_and_matches_a_simulation():
    rng = np.random.default_rng(0)
    mu, alpha = 18.0, 0.08
    n, p = dist.nb_params(mu, alpha)
    draws = rng.negative_binomial(n, p, 200_000)
    assert abs(draws.mean() - mu) < 0.1
    assert abs(draws.var() - (mu + alpha * mu * mu)) < 1.0
    lines = np.array([10.5, 15.5, 17.5, 20.5, 25.5])
    ps = dist.p_over(mu, alpha, lines)
    assert np.all(np.diff(ps) < 0)
    for line, pr in zip(lines, ps):
        assert abs(pr - (draws > line).mean()) < 0.01


def test_dispersion_recovers_the_spread_it_was_fitted_on():
    rng = np.random.default_rng(1)
    mu = rng.uniform(2, 30, 60_000)
    n, p = dist.nb_params(mu, 0.15)
    y = rng.negative_binomial(n, p)
    disp = dist.Dispersion().fit(mu, y)
    assert np.all(np.abs(disp(np.array([5.0, 15.0, 25.0])) - 0.15) < 0.04)
    back = dist.Dispersion.from_dict(disp.to_dict())
    assert np.allclose(back(np.array([5.0, 25.0])), disp(np.array([5.0, 25.0])))


def test_calibration_of_a_perfect_forecast_is_near_zero():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, 50_000) < p).astype(float)
    assert dist.calibration(p, y)["ece"] < 0.01
