import numpy as np
from leagues.backtest import accuracy, brier, devig, outcome_index, rps


def test_outcome_index_maps_results():
    assert outcome_index(2, 1) == 0
    assert outcome_index(1, 1) == 1
    assert outcome_index(0, 3) == 2


def test_devig_removes_overround_and_sums_to_one():
    p = devig(2.0, 3.5, 4.0)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] and p[0] > p[2]


def test_rps_is_zero_for_perfect_forecast():
    assert rps(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0


def test_rps_penalizes_distance_ordering():
    far = rps(np.array([[0.0, 0.0, 1.0]]), np.array([0]))
    near = rps(np.array([[0.0, 1.0, 0.0]]), np.array([0]))
    assert far > near


def test_rps_of_uniform_forecast_is_known_value():
    # uniform (1/3,1/3,1/3), home win: cum=(1/3,2/3); obs cum=(1,1)
    # rps = ((1/3-1)^2 + (2/3-1)^2)/2 = (0.4444+0.1111)/2 = 0.2778
    val = rps(np.array([[1/3, 1/3, 1/3]]), np.array([0]))
    assert abs(val - 0.2778) < 1e-3


def test_accuracy_counts_argmax_hits():
    p = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    assert accuracy(p, np.array([0, 2])) == 1.0
    assert accuracy(p, np.array([1, 2])) == 0.5


def test_brier_is_zero_for_perfect_forecast():
    assert brier(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0
