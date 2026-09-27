"""Walk-forward validation over fifteen seasons, and the release gate.

EVERY SCORED SEASON IS ONE THE MODEL NEVER SAW. For each season in turn the model
is fitted on every earlier season only and predicts that one, so no fold is ever
graded on rows it trained on. The first `BURN_IN_SEASONS` are never scored at all
-- the model needs history before its first honest prediction, and grading a
season it partly trained on measures nothing.

WHAT IS REUSED, AND WHY. `nfl.model.PropModel` is a logistic ensemble blended with
an empirical baseline, cross-fitted for calibration, isotonic above 5,000 rows and
Platt below. None of that is football-specific -- it takes a feature frame and an
outcome column. The metrics (`brier`, `ece`, `ece_null`) are likewise arithmetic.
Copying either into this package would have created a second implementation of the
most subtle code in the repo, which is exactly the drift the engine avoids
elsewhere by keeping ONE `leagues.picks`. The import direction is admittedly
awkward -- `nba` importing `nfl` -- and the honest alternative was a shared package
neither sport owns; that refactor touches a live, validated NFL board and was not
worth doing on the eve of its week 1.

THE BASELINE IS THE PLAYER'S OWN ENTERING HIT RATE against his own line. It is
dumb but never overconfident, and beating it is the minimum claim this board has
to make: if the model cannot improve on "what he has done before", the model is
not adding anything a reader could not do with an average.
"""
import numpy as np
import pandas as pd

from nba import config
from nfl.backtest import brier, ece, ece_null
from nfl.model import PropModel, empirical_baseline, fit_predict


def walk_forward(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    """Predict each scored season from the seasons before it.

    Training is capped at the most recent `config.TRAIN_SEASONS` -- see that
    constant for the measurement. Briefly: an unbounded window anchors the model
    to a decade-old scoring environment, and points lost to its baseline in three
    seasons purely because it was calibrated to one the league had left behind.
    """
    seasons = sorted(frame["season"].unique())
    scored = seasons[config.BURN_IN_SEASONS:]
    out = []
    for season in scored:
        train = frame[(frame["season"] < season)
                      & (frame["season"] >= season - config.TRAIN_SEASONS)]
        test = frame[frame["season"] == season]
        if train.empty or test.empty:
            continue
        predicted = (fit_predict(train, test, market, per_step=True)
                     if market in config.PER_STEP_MODELS
                     else PropModel(market).fit(train).predict(test))
        out.append(pd.DataFrame({
            "season": season,
            "group": (test["GAME_ID"].to_numpy() if "GAME_ID" in test.columns
                      else np.arange(len(test))),
            "prob": predicted,
            "outcome": test["outcome"].to_numpy(),
            "baseline": empirical_baseline(test, market),
            # WHETHER THE LINE IS REALLY HIS MEDIAN. MIN_LINE floors a line that
            # would otherwise be too low to quote, and for assists and threes that
            # floor binds on ~80% of rows -- so most of this market is "will a
            # rotation player clear a CONSTANT", which is a far more predictable
            # question than a balanced prop. Carried through so the report can
            # separate the two rather than publish one flattering average.
            "at_floor": (test["line"] <= config.MIN_LINE[market] + 1e-9).to_numpy(),
            # Which rung of the line spread this row was asked at (0 = his own
            # median). Absent when the frame was not augmented.
            "line_step": (test["line_step"].to_numpy() if "line_step" in test.columns
                          else 0),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(predictions: pd.DataFrame, market: str) -> dict:
    """Score the walk-forward, and decide release.

    The gate is deliberately the same shape as the NFL's: beat the baseline, stay
    calibrated, and do it in EVERY scored season rather than on average. A market
    that works in twelve seasons and fails in three is not a market, it is a coin
    that landed twelve times.
    """
    if predictions.empty:
        return {"market": market, "released": False, "reason": "no predictions"}
    prob = predictions["prob"].to_numpy(dtype=float)
    outcome = predictions["outcome"].to_numpy(dtype=float)
    base = predictions["baseline"].to_numpy(dtype=float)

    per_season, failures = {}, []
    for season, block in predictions.groupby("season"):
        p = block["prob"].to_numpy(dtype=float)
        o = block["outcome"].to_numpy(dtype=float)
        b = block["baseline"].to_numpy(dtype=float)
        stats = {"n": int(len(block)), "brier": round(brier(p, o), 4),
                 "baseline": round(brier(b, o), 4),
                 "accuracy": round(float(((p >= 0.5) == (o == 1)).mean()), 4)}
        per_season[int(season)] = stats
        if stats["brier"] >= stats["baseline"]:
            failures.append(f"season {int(season)} did not beat the baseline")
        if stats["n"] < config.MIN_PREDICTIONS_PER_SEASON:
            failures.append(f"season {int(season)} has only {stats['n']} predictions")

    calibration = ece(prob, outcome)
    # PRACTICALLY CALIBRATED, OR INDISTINGUISHABLE FROM PERFECT AT THIS SAMPLE
    # SIZE. A bare constant punishes a small market for being small; a bare null
    # test punishes a large one for being measurable enough to fail it.
    bar = max(config.MAX_ECE, ece_null(prob))
    if calibration > bar:
        failures.append(f"ECE {calibration:.3f} above the bar {bar:.3f}")
    if len(predictions) < config.MIN_PREDICTIONS_TOTAL:
        failures.append(f"only {len(predictions)} predictions in total")

    # CALIBRATED AT EVERY LINE, not just on average. The board asks the model
    # about a bookmaker's number, which can sit a few points either side of the
    # player's median; a model calibrated overall but 8 points off at +2 would
    # be wrong exactly where the book puts its line.
    by_step = {}
    if "line_step" in predictions.columns and predictions["line_step"].nunique() > 1:
        for step, block in predictions.groupby("line_step"):
            p = block["prob"].to_numpy(dtype=float)
            o = block["outcome"].to_numpy(dtype=float)
            b = block["baseline"].to_numpy(dtype=float)
            step_ece = ece(p, o)
            step_bar = max(config.MAX_ECE, ece_null(p))
            by_step[str(int(step))] = {
                "n": int(len(block)), "brier": round(brier(p, o), 4),
                "baseline": round(brier(b, o), 4), "ece": round(step_ece, 4),
                "predicted": round(float(p.mean()), 4),
                "landed": round(float(o.mean()), 4)}
            if step_ece > step_bar:
                failures.append(f"line step {int(step):+d}: ECE {step_ece:.3f} "
                                f"above the bar {step_bar:.3f}")

    # The honest split: how the model does where the line is genuinely the
    # player's own median, which is the only part comparable to a book's number.
    above = predictions[~predictions["at_floor"]]
    subset = None
    if len(above) >= 500:
        ap = above["prob"].to_numpy(dtype=float)
        ao = above["outcome"].to_numpy(dtype=float)
        ab = above["baseline"].to_numpy(dtype=float)
        subset = {"n": int(len(above)),
                  "brier": round(brier(ap, ao), 4),
                  "baseline_brier": round(brier(ab, ao), 4),
                  "accuracy": round(float(((ap >= 0.5) == (ao == 1)).mean()), 4),
                  "base_rate": round(float(ao.mean()), 4)}

    return {
        "market": market,
        "released": not failures,
        "floor_share": round(float(predictions["at_floor"].mean()), 4),
        "above_floor": subset,
        "failures": failures,
        "n": int(len(predictions)),
        "seasons_scored": sorted(int(s) for s in per_season),
        "brier": round(brier(prob, outcome), 4),
        "baseline_brier": round(brier(base, outcome), 4),
        "accuracy": round(float(((prob >= 0.5) == (outcome == 1)).mean()), 4),
        "base_rate": round(float(outcome.mean()), 4),
        "ece": round(calibration, 4),
        "ece_bar": round(bar, 4),
        "per_season": per_season,
        "by_line_step": by_step,
    }
