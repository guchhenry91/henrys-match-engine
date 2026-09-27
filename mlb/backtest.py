"""The MLB release gate: walk-forward over fifteen scored seasons.

Every scored season is predicted by a model fitted only on the seasons before it
(capped at config.TRAIN_SEASONS). A market is released only if it beats its
baseline in EVERY scored season, is calibrated overall, and is calibrated at
EVERY step of the line spread -- the board asks about bookmakers' lines, which
sit either side of a player's median.

Team winner: Elo (nfl.games_model), parameters fitted ONCE on the burn-in
seasons and held, against a home-field baseline.
"""
import numpy as np
import pandas as pd

from mlb import config
from nfl.backtest import brier, ece, ece_null
from nfl.games_model import fit_parameters, run_elo
from nfl.model import PropModel, empirical_baseline


def walk_forward(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    out = []
    for season in config.SCORED_SEASONS:
        train = frame[(frame["season"] < season)
                      & (frame["season"] >= season - config.TRAIN_SEASONS)]
        test = frame[frame["season"] == season]
        if train.empty or test.empty:
            continue
        model = PropModel(market).fit(train)
        out.append(pd.DataFrame({
            "season": season,
            "prob": model.predict(test),
            "outcome": test["outcome"].to_numpy(),
            "baseline": empirical_baseline(test, market),
            "line_step": test["line_step"].to_numpy() if "line_step" in test else 0,
        }))
        print(f"    {market} {season}: n={len(test)}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(pred: pd.DataFrame, market: str) -> dict:
    if pred.empty:
        return {"market": market, "released": False, "failures": ["no predictions"], "n": 0}
    p, o, b = (pred[c].to_numpy(dtype=float) for c in ("prob", "outcome", "baseline"))
    per_season, failures = {}, []
    for season, blk in pred.groupby("season"):
        ps, os_, bs = (blk[c].to_numpy(dtype=float) for c in ("prob", "outcome", "baseline"))
        st = {"n": int(len(blk)), "brier": round(brier(ps, os_), 4),
              "baseline": round(brier(bs, os_), 4),
              "accuracy": round(float(((ps >= 0.5) == (os_ == 1)).mean()), 4)}
        per_season[int(season)] = st
        if st["brier"] >= st["baseline"]:
            failures.append(f"season {int(season)} did not beat the baseline")
        if st["n"] < config.MIN_PREDICTIONS_PER_SEASON:
            failures.append(f"season {int(season)} has only {st['n']} predictions")
    cal = ece(p, o)
    bar = max(config.MAX_ECE, ece_null(p))
    if cal > bar:
        failures.append(f"ECE {cal:.3f} above the bar {bar:.3f}")
    if len(pred) < config.MIN_PREDICTIONS_TOTAL:
        failures.append(f"only {len(pred)} predictions in total")
    by_step = {}
    if pred["line_step"].nunique() > 1:
        for step, blk in pred.groupby("line_step"):
            ps, os_, bs = (blk[c].to_numpy(dtype=float) for c in ("prob", "outcome", "baseline"))
            e, e_bar = ece(ps, os_), max(config.MAX_ECE, ece_null(ps))
            by_step[str(int(step))] = {"n": int(len(blk)), "ece": round(e, 4),
                                       "predicted": round(float(ps.mean()), 4),
                                       "landed": round(float(os_.mean()), 4),
                                       "brier": round(brier(ps, os_), 4),
                                       "baseline": round(brier(bs, os_), 4)}
            if e > e_bar:
                failures.append(f"line step {int(step):+d}: ECE {e:.3f} above the bar {e_bar:.3f}")
    return {"market": market, "released": not failures, "failures": failures,
            "n": int(len(pred)), "seasons_scored": sorted(per_season),
            "brier": round(brier(p, o), 4), "baseline_brier": round(brier(b, o), 4),
            "accuracy": round(float(((p >= 0.5) == (o == 1)).mean()), 4),
            "base_rate": round(float(o.mean()), 4), "ece": round(cal, 4),
            "ece_bar": round(bar, 4), "per_season": per_season, "by_line_step": by_step}


def team_winner(games: pd.DataFrame) -> tuple:
    """(report, params). Elo params from the burn-in seasons only, then held."""
    played = games[games["played"] & games["winner"].notna()].copy()
    burn = played[played["season"] < config.SCORED_SEASONS[0]]
    params = fit_parameters(burn, grid=config.ELO_GRID)
    walked = run_elo(played, params["k"], params["home_edge"], params["regression"])
    walked = walked[walked["season"].isin(config.SCORED_SEASONS)]
    prob = walked["prob_home"].to_numpy(dtype=float)
    outcome = (walked["winner"] == "home").astype(float).to_numpy()
    per_season, failures = {}, []
    for season, blk in walked.groupby("season"):
        ps = blk["prob_home"].to_numpy(dtype=float)
        os_ = (blk["winner"] == "home").astype(float).to_numpy()
        home_rate = float(os_.mean())
        st = {"n": int(len(blk)), "brier": round(brier(ps, os_), 4),
              "baseline": round(brier(np.full_like(os_, home_rate), os_), 4),
              "accuracy": round(float(((ps >= 0.5) == (os_ == 1)).mean()), 4),
              "home_rate": round(home_rate, 4)}
        per_season[int(season)] = st
        if st["brier"] >= st["baseline"]:
            failures.append(f"season {int(season)} did not beat home field")
    home_rate = float(outcome.mean())
    return ({"market": "team_winner", "released": not failures, "failures": failures,
             "n": int(len(walked)), "seasons_scored": sorted(per_season),
             "brier": round(brier(prob, outcome), 4),
             "baseline_brier": round(brier(np.full_like(outcome, home_rate), outcome), 4),
             "accuracy": round(float(((prob >= 0.5) == (outcome == 1)).mean()), 4),
             "home_win_rate": round(home_rate, 4), "per_season": per_season,
             "elo_params": params}, params)
