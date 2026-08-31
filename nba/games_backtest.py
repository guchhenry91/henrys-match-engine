"""Team-winner validation: Elo, walked forward, against home-court advantage.

THE BASELINE IS HOME COURT, and in basketball that is a much stronger opponent
than it is in football: the home side wins about 55-58% of NBA games outright,
against 53% in the NFL. A team-winner model that cannot beat "always pick the
home team" has demonstrated nothing at all, so that is what it is scored against.

NEUTRAL-VENUE GAMES GET NO HOME EDGE. The NBA plays a handful abroad each season
and the feed marks both sides away; granting home advantage on a court neither
team owns would be inventing an effect. `nba.data.games` flags them.
"""
import numpy as np
import pandas as pd

from nba import config
from nfl.backtest import brier
from nfl.games_model import fit_parameters, run_elo


def walk_forward(games: pd.DataFrame) -> pd.DataFrame:
    """Rate every scored season from the seasons before it."""
    # Column names are left exactly as nfl.games_model expects them -- renaming
    # them to home/away was the first attempt and simply broke the shared Elo.
    played = games[games["played"] & games["winner"].notna()].copy()
    seasons = sorted(played["season"].unique())
    scored = seasons[config.BURN_IN_SEASONS:]
    out = []
    for season in scored:
        train = played[played["season"] < season]
        test = played[played["season"] == season]
        if train.empty or test.empty:
            continue
        params = fit_parameters(train)
        priced = run_elo(pd.concat([train, test], ignore_index=True),
                         k=params["k"], home_edge=params["home_edge"],
                         regression=params["regression"])
        priced = priced[priced["season"] == season]
        out.append(pd.DataFrame({
            "season": season,
            "prob_home": priced["prob_home"].to_numpy(dtype=float),
            "outcome": (priced["winner"] == "home").astype(float).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"market": "team_winner", "released": False, "reason": "no predictions"}
    prob = predictions["prob_home"].to_numpy(dtype=float)
    outcome = predictions["outcome"].to_numpy(dtype=float)

    per_season, failures = {}, []
    for season, block in predictions.groupby("season"):
        p = block["prob_home"].to_numpy(dtype=float)
        o = block["outcome"].to_numpy(dtype=float)
        # The baseline for THIS season is its own home-win rate, applied flat --
        # the strongest form of "always pick the home team", since it is allowed
        # to know how strong home court was that year.
        home_rate = float(o.mean())
        stats = {"n": int(len(block)),
                 "brier": round(brier(p, o), 4),
                 "baseline": round(brier(np.full_like(o, home_rate), o), 4),
                 "accuracy": round(float(((p >= 0.5) == (o == 1)).mean()), 4),
                 "home_rate": round(home_rate, 4)}
        per_season[int(season)] = stats
        if stats["brier"] >= stats["baseline"]:
            failures.append(f"season {int(season)} did not beat home court")

    home_rate = float(outcome.mean())
    return {
        "market": "team_winner",
        "released": not failures,
        "failures": failures,
        "n": int(len(predictions)),
        "seasons_scored": sorted(int(s) for s in per_season),
        "brier": round(brier(prob, outcome), 4),
        "baseline_brier": round(brier(np.full_like(outcome, home_rate), outcome), 4),
        "accuracy": round(float(((prob >= 0.5) == (outcome == 1)).mean()), 4),
        "home_win_rate": round(home_rate, 4),
        "per_season": per_season,
    }
