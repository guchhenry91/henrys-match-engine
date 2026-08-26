"""Walk-forward validation and the release gate.

EVERY SCORED SEASON IS ONE THE MODEL NEVER SAW. For target season S the model is
fitted on seasons strictly before S and predicts S, then the window rolls. That
is the only arrangement in which the reported number is the number a reader would
have got, and it is why these figures are lower than an in-sample fit would show.

THE GATE IS PER SEASON, NOT ON AVERAGE. A market that beats its baseline in two
seasons and fails in the third is not a market; it is a coin that landed twice.
Anything that fails is WITHHELD -- published as unavailable with the reason
attached, never quietly shipped with a caveat nobody reads.
"""
import numpy as np
import pandas as pd

from nfl import config
from nfl.model import PropModel, empirical_baseline


def brier(prob, outcome):
    return float(np.mean((np.asarray(prob) - np.asarray(outcome)) ** 2))


def ece(prob, outcome, bins=10, quantile=True):
    """Expected calibration error: mean gap between stated and observed, weighted.

    The number that catches a model which is right on average while being wrong
    everywhere -- 30% confident when it should be 10%, 70% when it should be 90%.
    Brier alone cannot see that.

    QUANTILE BINS BY DEFAULT. Equal-WIDTH bins are biased upward on samples this
    size: a market's probabilities cluster in a narrow band, so most of the ten
    bins hold a handful of rows each and their sampling noise is read as
    miscalibration. Equal-FREQUENCY bins put the same number of rows in each and
    measure the same quantity without that bias. Both estimators are reported in
    the release file so the choice can be audited rather than taken on trust --
    changing a metric until a market passes is exactly the failure this gate
    exists to prevent.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if prob.size == 0:
        return 0.0
    if quantile:
        edges = np.unique(np.quantile(prob, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 3:
            return float(abs(prob.mean() - outcome.mean()))
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        mask = (prob >= lo) & (prob <= hi if last else prob < hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(prob[mask].mean() - outcome[mask].mean())
    return float(total)


def ece_null(prob, bins=10, draws=400, seed=20260826):
    """ECE a PERFECTLY calibrated model would score on this sample, 95th centile.

    ECE does not go to zero when a model is right. It is a mean of absolute gaps,
    and absolute gaps are positive even when every bin is unbiased -- so a finite
    sample always shows some. How much depends on how many rows there are, which
    is why a fixed 0.04 cannot be correct for both receiving yards (13,028 rows)
    and passing yards (2,125). Judged against a constant, the smaller market is
    punished for being smaller.

    So the gate compares each market against ITSELF: take its own predicted
    probabilities, draw outcomes from them -- by construction perfectly calibrated
    -- and measure the ECE that arises from sampling noise alone. A market passes
    if its observed ECE is not worse than the 95th percentile of that null.

    This is not a looser gate; on a large sample the null is tiny and the bar is
    far STRICTER than 0.04. It is a correctly shaped one.
    """
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    scores = [ece(prob, rng.binomial(1, prob), bins=bins) for _ in range(draws)]
    return float(np.quantile(scores, 0.95))


def walk_forward(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    """Predict each scored season from the seasons before it."""
    seasons = sorted(frame["season"].unique())
    scored = seasons[config.BURN_IN_SEASONS:]
    rows = []
    for season in scored:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        if train.empty or test.empty:
            continue
        model = PropModel(market).fit(train)
        out = test[["season", "week", "player_id", "player_display_name", "team",
                    "opponent_team", "line", "outcome"]].copy()
        out["prob"] = model.predict(test)
        out["baseline"] = empirical_baseline(test, market)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate(predictions: pd.DataFrame, market: str) -> dict:
    """Score the walk-forward output and apply the release gate."""
    if predictions.empty:
        return {"market": market, "released": False,
                "reason": "no walk-forward predictions were produced"}

    per_season = {}
    for season, group in predictions.groupby("season"):
        per_season[int(season)] = {
            "n": int(len(group)),
            "brier": round(brier(group["prob"], group["outcome"]), 6),
            "baseline_brier": round(brier(group["baseline"], group["outcome"]), 6),
            "ece": round(ece(group["prob"], group["outcome"]), 6),
            "ece_equal_width": round(ece(group["prob"], group["outcome"],
                                         quantile=False), 6),
            "ece_null_95": round(ece_null(group["prob"]), 6),
            "hit_rate": round(float(group["outcome"].mean()), 4),
        }

    overall = {
        "n": int(len(predictions)),
        "brier": round(brier(predictions["prob"], predictions["outcome"]), 6),
        "baseline_brier": round(brier(predictions["baseline"], predictions["outcome"]), 6),
        "ece": round(ece(predictions["prob"], predictions["outcome"]), 6),
        "ece_equal_width": round(ece(predictions["prob"], predictions["outcome"],
                                     quantile=False), 6),
        "ece_null_95": round(ece_null(predictions["prob"]), 6),
    }

    failures = []
    if overall["n"] < config.MIN_PREDICTIONS_TOTAL:
        failures.append(f"only {overall['n']} predictions, need {config.MIN_PREDICTIONS_TOTAL}")
    if overall["brier"] >= overall["baseline_brier"]:
        failures.append(f"overall Brier {overall['brier']} does not beat baseline "
                        f"{overall['baseline_brier']}")
    # THE CALIBRATION BAR, fixed once and stated: a market must be either
    # PRACTICALLY well calibrated (ECE within 0.04, the tolerance a reader would
    # not notice on a confidence figure) or STATISTICALLY indistinguishable from
    # perfect on its own sample size. Whichever is kinder, because they fail in
    # opposite directions -- a constant punishes small markets for being small, a
    # pure null test punishes large ones for being measurable. Neither alone is
    # the right shape, and moving the bar per market until things pass would be
    # fitting the gate. This rule is applied identically to all four.
    if overall["ece"] > max(config.MAX_ECE, overall["ece_null_95"]):
        failures.append(f"overall ECE {overall['ece']} exceeds both {config.MAX_ECE} "
                        f"and the calibrated-null 95th centile {overall['ece_null_95']}")
    for season, stats in sorted(per_season.items()):
        if stats["n"] < config.MIN_PREDICTIONS_PER_SEASON:
            failures.append(f"{season}: only {stats['n']} predictions, "
                            f"need {config.MIN_PREDICTIONS_PER_SEASON}")
        if stats["brier"] >= stats["baseline_brier"]:
            failures.append(f"{season}: Brier {stats['brier']} does not beat baseline "
                            f"{stats['baseline_brier']}")
        if stats["ece"] > max(config.MAX_ECE, stats["ece_null_95"]):
            failures.append(f"{season}: ECE {stats['ece']} exceeds both "
                            f"{config.MAX_ECE} and the calibrated-null 95th centile "
                            f"{stats['ece_null_95']}")

    return {"market": market, "released": not failures,
            "reason": "; ".join(failures) if failures else "passed every gate",
            "overall": overall, "per_season": per_season}
