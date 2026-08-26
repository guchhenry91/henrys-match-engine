"""Walk-forward validation for the team-winner model.

Elo parameters are refitted before each scored season on the seasons before it,
then frozen. Nothing that happens in a scored season can influence the ratings or
the parameters used to predict it.

THE BASELINE IS HOME ADVANTAGE, which in this sample is worth 53.3%. That is a
real strategy anyone can follow without a model, so a game model that cannot beat
it has produced nothing. The props side's baseline is per-player form; this is its
equivalent -- the best you can do while knowing nothing about the teams.
"""
import numpy as np
import pandas as pd

from nfl import config
from nfl.backtest import brier, ece, ece_null
from nfl.games_model import fit_parameters, run_elo

ACTUAL = {"home": 1.0, "away": 0.0, "tie": 0.5}


def walk_forward(games: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(games["season"].unique())
    scored = seasons[config.BURN_IN_SEASONS:]
    frames = []
    for season in scored:
        history = games[games["season"] < season]
        if history.empty:
            continue
        params = fit_parameters(history)
        # Ratings must carry INTO the scored season, so Elo is run across history
        # plus that season and only that season's rows are kept. Restarting at
        # 1500 each year would throw away everything the model knows in September.
        through = games[games["season"] <= season]
        out = run_elo(through, params["k"], params["home_edge"], params["regression"])
        out = out[(out["season"] == season) & out["played"] & out["winner"].notna()].copy()
        # The naive comparator, learned from the SAME history: how often the home
        # side won before this season.
        past = games[(games["season"] < season) & games["played"]]
        home_rate = float(past["winner"].map(ACTUAL).mean())
        out["baseline"] = home_rate
        out["outcome"] = out["winner"].map(ACTUAL)
        out["k"] = params["k"]
        out["home_edge"] = params["home_edge"]
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"market": "team_winner", "released": False,
                "reason": "no walk-forward predictions were produced"}

    def accuracy(frame):
        """Ties count as neither right nor wrong -- the model named a side and the
        game did not produce one, so scoring it either way would be a fiction."""
        decisive = frame[frame["outcome"] != 0.5]
        if decisive.empty:
            return None
        picked_home = decisive["prob_home"] >= 0.5
        correct = (picked_home == (decisive["outcome"] == 1.0))
        return round(float(correct.mean()), 4)

    per_season = {}
    for season, group in predictions.groupby("season"):
        per_season[int(season)] = {
            "n": int(len(group)),
            "brier": round(brier(group["prob_home"], group["outcome"]), 6),
            "baseline_brier": round(brier(group["baseline"], group["outcome"]), 6),
            "ece": round(ece(group["prob_home"], group["outcome"]), 6),
            "ece_null_95": round(ece_null(group["prob_home"]), 6),
            "accuracy": accuracy(group),
            "home_win_rate": round(float((group["outcome"] == 1.0).mean()), 4),
        }

    overall = {
        "n": int(len(predictions)),
        "brier": round(brier(predictions["prob_home"], predictions["outcome"]), 6),
        "baseline_brier": round(brier(predictions["baseline"], predictions["outcome"]), 6),
        "ece": round(ece(predictions["prob_home"], predictions["outcome"]), 6),
        "ece_null_95": round(ece_null(predictions["prob_home"]), 6),
        "accuracy": accuracy(predictions),
    }

    failures = []
    if overall["brier"] >= overall["baseline_brier"]:
        failures.append(f"overall Brier {overall['brier']} does not beat home-advantage "
                        f"baseline {overall['baseline_brier']}")
    if overall["ece"] > max(config.MAX_ECE, overall["ece_null_95"]):
        failures.append(f"overall ECE {overall['ece']} exceeds both {config.MAX_ECE} "
                        f"and the calibrated-null 95th centile {overall['ece_null_95']}")
    for season, stats in sorted(per_season.items()):
        if stats["brier"] >= stats["baseline_brier"]:
            failures.append(f"{season}: Brier {stats['brier']} does not beat baseline "
                            f"{stats['baseline_brier']}")
        if stats["ece"] > max(config.MAX_ECE, stats["ece_null_95"]):
            failures.append(f"{season}: ECE {stats['ece']} exceeds both "
                            f"{config.MAX_ECE} and {stats['ece_null_95']}")

    return {"market": "team_winner", "released": not failures,
            "reason": "; ".join(failures) if failures else "passed every gate",
            "overall": overall, "per_season": per_season}
