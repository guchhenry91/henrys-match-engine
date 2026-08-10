"""Shared helpers for the correct-score audit (scripts/correct_score_*.py).

fit_production_model() replicates EXACTLY the fit recipe leagues/publish.py
uses in build() (base fit -> second-tier priors for promoted/no-history teams
-> promoted_priors fallback -> final fit with priors), so every diagnostic and
benchmark script here scores the SAME model production actually runs, not a
simplified stand-in.
"""
import pandas as pd

from leagues import dataset, fixtures, second_tier
from leagues.model import LeagueModel, promoted_priors


def fit_production_model(league: str, ref: pd.Timestamp | None = None):
    """Returns (model, fx) using production's exact fit recipe.

    fx is the fixtures.fetch_fixtures(league) frame (played + remaining), so
    callers can find upcoming fixtures and squad_teams the same way build() does.
    """
    matches = dataset.build_matches(league)
    fx = fixtures.fetch_fixtures(league)
    played = fx[fx["played"]].copy()

    if not played.empty:
        add = played[["date", "home", "away", "home_goals", "away_goals"]].copy()
        add["date"] = pd.to_datetime(add["date"]).dt.tz_localize(None)
        matches = pd.concat([matches, add], ignore_index=True)

    ref = ref or pd.Timestamp.now("UTC")
    squad_teams = sorted(set(fx["home"]) | set(fx["away"]))

    base = LeagueModel().fit(matches, ref=ref)
    no_history = [t for t in squad_teams if t not in base.attack]
    priors = {}
    prior_source = {}  # team -> "second_tier" | "weakest_fallback"
    if no_history:
        try:
            priors = second_tier.second_tier_priors(base, league, no_history)
            for t in priors:
                prior_source[t] = "second_tier"
        except Exception:
            priors = {}
        still_missing = [t for t in no_history if t not in priors]
        if still_missing:
            priors.update(promoted_priors(base, still_missing))
            for t in still_missing:
                prior_source[t] = "weakest_fallback"

    model = LeagueModel().fit(matches, ref=ref, priors=priors)
    model.prior_source = prior_source  # stashed for diagnostics, not used by predict()
    return model, fx


def upcoming_fixtures(fx: pd.DataFrame) -> pd.DataFrame:
    return fx[~fx["played"]].copy()


def near_term_fixtures(fx: pd.DataFrame, matchweeks_ahead: int = 1) -> pd.DataFrame:
    """Same window leagues/publish.py actually ships (MATCHWEEKS_AHEAD=1): only
    the next round, not the whole remaining season. This is what the site's
    "25 of 38" figure is counted over, so the diagnostic table should match it."""
    remaining = fx[~fx["played"]].copy()
    if remaining.empty:
        return remaining
    next_round = int(remaining.sort_values("date").iloc[0]["round"])
    return remaining[remaining["round"] < next_round + matchweeks_ahead]
