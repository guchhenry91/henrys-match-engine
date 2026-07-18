import numpy as np
import pandas as pd
import pytest

from leagues.model import Calibrator, LeagueModel, blend_probs, dc_tau, scoreline_grid


def test_dc_tau_lifts_draws_and_trims_1_0():
    assert dc_tau(0, 0, 1.4, 1.2, rho=-0.1) > 1.0
    assert dc_tau(1, 1, 1.4, 1.2, rho=-0.1) > 1.0
    assert dc_tau(1, 0, 1.4, 1.2, rho=-0.1) < 1.0
    assert dc_tau(3, 2, 1.4, 1.2, rho=-0.1) == 1.0


def test_scoreline_grid_is_a_normalized_distribution():
    g = scoreline_grid(1.5, 1.1, rho=-0.1, max_goals=10)
    assert abs(g.sum() - 1.0) < 1e-9
    assert (g >= 0).all()


def test_grid_gives_higher_home_win_prob_for_stronger_home_team():
    strong = scoreline_grid(2.2, 0.8, rho=-0.1)
    weak = scoreline_grid(0.8, 2.2, rho=-0.1)
    assert np.tril(strong, -1).sum() > 0.5 > np.tril(weak, -1).sum()


def _toy_matches(n=300):
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 1.8, "B": 1.4, "C": 1.0, "D": 0.7}
    rows = []
    start = pd.Timestamp("2025-08-01")
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hg = int(rng.poisson(strength[h] * 1.15))
        ag = int(rng.poisson(strength[a] * 0.85))
        rows.append({"date": start + pd.Timedelta(days=i), "home": h, "away": a,
                     "home_goals": hg, "away_goals": ag,
                     "home_xg": float(hg), "away_xg": float(ag)})
    return pd.DataFrame(rows)


def test_model_fits_and_ranks_teams_correctly():
    m = LeagueModel().fit(_toy_matches(), ref=pd.Timestamp("2026-06-20"))
    p = m.predict("A", "D")
    assert p["p_home"] > p["p_away"]
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1.0) < 1e-6
    assert 0 < p["p_draw"] < 0.45


def test_predict_unknown_team_raises():
    m = LeagueModel().fit(_toy_matches(), ref=pd.Timestamp("2026-06-20"))
    with pytest.raises(KeyError):
        m.predict("A", "ZZ")


def test_blend_probs_averages_and_normalizes():
    out = blend_probs((0.5, 0.3, 0.2), (0.7, 0.2, 0.1), weight=0.5)
    assert abs(sum(out) - 1.0) < 1e-9
    assert abs(out[0] - 0.6) < 1e-9


def test_calibrator_preserves_discrimination_and_normalizes():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2, 1, 2], size=500)
    y = np.array([rng.choice(3, p=row) for row in p])
    cal = Calibrator().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.corrcoef(out[:, 0], p[:, 0])[0, 1] > 0.9


def test_score_for_outcome_agrees_with_the_picked_result():
    """The displayed scoreline must not contradict the pick. 1-1 is often the
    single most likely EXACT score even when one side is clearly favoured, so the
    card must show the most likely score *within* the picked outcome."""
    import numpy as np
    from leagues.model import scoreline_grid, outcome_probs, score_for_outcome
    # a fixture where the away side is favoured overall
    grid = scoreline_grid(0.9, 1.8, rho=-0.05)
    ph, pd_, pa = outcome_probs(grid)
    assert pa > ph and pa > pd_                      # away win is the pick

    home_s = score_for_outcome(grid, "home")
    draw_s = score_for_outcome(grid, "draw")
    away_s = score_for_outcome(grid, "away")
    hh, ha = (int(x) for x in home_s.split("-"))
    dh, da = (int(x) for x in draw_s.split("-"))
    ah, aa = (int(x) for x in away_s.split("-"))
    assert hh > ha, f"home pick gave {home_s}"       # a home win scoreline
    assert dh == da, f"draw pick gave {draw_s}"      # a level scoreline
    assert aa > ah, f"away pick gave {away_s}"       # an away win scoreline
