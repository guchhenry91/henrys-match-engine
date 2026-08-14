"""The gate's own gate.

These cover the parts of `leagues.tune` that decide what SHIPS -- the holdout
boundary, the promotion rule, and the tier stats publish.py reads -- none of
which run in a normal publish, so nothing else would catch them breaking.
"""
import numpy as np
import pandas as pd
import pytest

from leagues import config, tune
from leagues.second_tier import (feeder_season, promotion_season, season_code,
                                 season_code_start_year, season_start_year)


# --- season arithmetic: what makes the backtest causal -----------------------

def test_season_code_round_trips():
    assert season_code(2025) == "2526"
    assert season_code_start_year("2526") == 2025


def test_season_boundary_is_july_not_january():
    """August and the following May belong to the SAME season."""
    assert season_start_year("2026-08-15") == 2026
    assert season_start_year("2027-05-01") == 2026
    assert season_start_year("2026-06-30") == 2025


def test_feeder_season_is_one_behind_the_cutoff_season():
    """A club promoted for 2026-27 played the second tier in 2025-26. Seeding it
    from the 2026-27 table would be reading a season that has not happened."""
    assert feeder_season("2026-08-15") == "2526"
    assert feeder_season("2022-01-10") == "2021"


def test_promotion_season_tracks_config_not_a_literal():
    """The regression this replaced: a hardcoded '2526' stays silently wrong for a
    year after the seasons roll forward."""
    for lg in config.LEAGUES:
        assert promotion_season(lg) == config.get(lg).history_seasons[-1]


def test_holdout_starts_at_the_last_history_season():
    for lg in config.LEAGUES:
        want = season_code_start_year(config.get(lg).history_seasons[-1])
        assert tune.holdout_start(lg) == pd.Timestamp(f"{want}-07-01")


# --- the promotion rule ------------------------------------------------------

def test_paired_bootstrap_promotes_a_genuinely_better_challenger():
    ci = tune.paired_bootstrap(np.full(400, 0.18), np.full(400, 0.20))
    assert ci["hi"] < 0
    assert ci["mean"] == pytest.approx(-0.02)


def test_paired_bootstrap_refuses_a_noise_difference():
    """The whole point: a challenger that merely LOOKS better must not ship."""
    rng = np.random.default_rng(7)
    inc = rng.normal(0.20, 0.15, 400)
    d = rng.normal(0.0, 0.15, 400)
    # Centre the difference exactly, so the sample carries real per-match spread
    # but zero average edge. Drawing an uncentred sample would leave the verdict
    # to the seed: at n=400 the SE is ~0.0075, so a "no edge" draw lands two SE
    # from zero often enough to make such a test flap.
    chal = inc + (d - d.mean())
    ci = tune.paired_bootstrap(chal, inc)
    assert ci["lo"] < 0 < ci["hi"]                 # straddles zero -> no promotion


def test_paired_bootstrap_is_reproducible():
    a, b = np.linspace(0.1, 0.3, 200), np.linspace(0.12, 0.28, 200)
    assert tune.paired_bootstrap(a, b) == tune.paired_bootstrap(a, b)


def test_paired_bootstrap_handles_an_empty_holdout():
    assert tune.paired_bootstrap(np.array([]), np.array([]))["n"] == 0


def test_incumbent_is_inside_the_sweep_grid():
    """If the shipped config is not a candidate there is nothing to promote
    against, and tune() raises rather than comparing to the runner-up."""
    assert tune.INCUMBENT["xi"] in tune.XIS
    assert tune.INCUMBENT["xg_weight"] in tune.XGWS


# --- the numbers the board is billed on --------------------------------------

def test_tier_hit_rates_filter_by_top_probability():
    res = pd.DataFrame({
        "p_home": [0.90, 0.66, 0.50],
        "p_draw": [0.05, 0.20, 0.30],
        "p_away": [0.05, 0.14, 0.20],
        "outcome": [0, 2, 0],          # confident right, mid wrong, low right
    })
    by = {t["min_prob"]: t for t in tune.tier_hit_rates(res)}
    assert by[0.0]["n"] == 3 and by[0.0]["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert by[0.70]["n"] == 1 and by[0.70]["hit_rate_pct"] == 100.0
    assert by[0.65]["n"] == 2 and by[0.65]["hit_rate_pct"] == 50.0


def test_tier_hit_rate_is_none_when_a_tier_is_empty():
    res = pd.DataFrame({"p_home": [0.4], "p_draw": [0.35], "p_away": [0.25],
                        "outcome": [0]})
    assert {t["min_prob"]: t["hit_rate_pct"] for t in tune.tier_hit_rates(res)}[0.70] is None


def test_pooled_weights_by_sample_and_reports_the_spread():
    """A pooled headline hides the per-league range, which is what oversells it."""
    rep = {"PL": {"tiers_full": [{"min_prob": 0.65, "n": 30, "hit_rate_pct": 80.0}]},
           "LIGUE1": {"tiers_full": [{"min_prob": 0.65, "n": 10, "hit_rate_pct": 60.0}]}}
    t = tune.pooled(rep)["tiers"][0]
    assert t["n"] == 40
    assert t["hit_rate_pct"] == pytest.approx(75.0)      # 30/40 weighted, not 70.0
    assert (t["league_min_pct"], t["league_max_pct"]) == (60.0, 80.0)


def test_release_policy_carries_every_league_and_its_evidence():
    rep = {"PL": {"xi": 0.003, "xg_weight": 0.75, "selection": "incumbent retained",
                  "holdout": {"rps": 0.19, "n": 380}}}
    pol = tune.release_policy(rep)["leagues"]["PL"]
    assert pol["xi"] == 0.003 and pol["xg_weight"] == 0.75
    assert pol["holdout_rps"] == 0.19 and pol["holdout_n"] == 380
