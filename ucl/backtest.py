"""Walk-forward validation for the Champions League model.

EVERY SCORED SEASON IS ONE THE MODEL NEVER SAW. For target season S the model is
fitted only on matches played BEFORE S began, then predicts S. 2011-2015 are
burn-in and never scored, because a club's strength on the first scored matchday
has to come from somewhere.

THE BASELINE IS HOME ADVANTAGE, learned from the same history: the home win rate
up to that point, applied to every match. It is a real strategy anyone can follow
knowing nothing about the clubs, so a model that cannot beat it has produced
nothing. Ranked Probability Score is the measure, because a three-way market
rewards being right about HOW likely a draw is, not just picking a side.

THE 2024 FORMAT CHANGE IS REPORTED SEPARATELY. Before it the competition was eight
groups and a knockout; since, a 36-team league phase, which is why fixture counts
jump from ~215 to ~280. Team strength carries across -- Real Madrid did not become
a different club -- but if the model behaves differently under the new format that
is worth seeing rather than averaging away.
"""
import numpy as np
import pandas as pd

from leagues.model import LeagueModel
from ucl import config, data

OUTCOMES = ("home", "draw", "away")


def actual_outcome(row) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "home"
    if row["home_goals"] < row["away_goals"]:
        return "away"
    return "draw"


def rps(probs: dict, outcome: str) -> float:
    """Ranked Probability Score over an ORDERED three-way market.

    Ordered home > draw > away, so predicting an away win when the home side wins
    is penalised more than predicting a draw. A plain Brier treats those as
    equally wrong, which for a football result they are not.
    """
    order = ["home", "draw", "away"]
    cumulative_p, cumulative_o, total = 0.0, 0.0, 0.0
    for key in order[:-1]:
        cumulative_p += probs.get(key, 0.0)
        cumulative_o += 1.0 if outcome == key else 0.0
        total += (cumulative_p - cumulative_o) ** 2
    return total / (len(order) - 1)


def fit_for(train: pd.DataFrame) -> LeagueModel:
    """Fit on everything played before the target season.

    xg_weight is ZERO: there is no xG for this competition, and the leagues engine
    defaults to blending 75% xG. Left at the default it would silently weight a
    column that does not exist.
    """
    model = LeagueModel(xi=config.XI_PER_DAY, xg_weight=0.0,
                        prior_strength=config.PRIOR_STRENGTH)
    return model.fit(train[["date", "home", "away", "home_goals", "away_goals"]],
                     ref=train["date"].max())


def walk_forward(frame: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(frame["season"].unique())
    scored = seasons[config.BURN_IN_SEASONS:]
    rows = []
    for season in scored:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        if train.empty or test.empty:
            continue
        try:
            model = fit_for(train)
        except Exception as exc:
            print(f"  {season}: fit failed ({exc})")
            continue
        # The baseline knows only what the history says about home advantage.
        home_rate = float((train["home_goals"] > train["away_goals"]).mean())
        draw_rate = float((train["home_goals"] == train["away_goals"]).mean())
        base = {"home": home_rate, "draw": draw_rate,
                "away": max(1e-6, 1.0 - home_rate - draw_rate)}

        for _, match in test.iterrows():
            try:
                pred = model.predict(match["home"], match["away"])
            except Exception:
                # A club with NO prior European match cannot be priced by a model
                # fitted on clubs that have one. Skipped and counted rather than
                # given the average, which would read as knowledge.
                rows.append({"season": season, "skipped": True})
                continue
            probs = {"home": pred["p_home"], "draw": pred["p_draw"],
                     "away": pred["p_away"]}
            truth = actual_outcome(match)
            pick = max(probs, key=probs.get)
            rows.append({
                "season": season, "skipped": False,
                "date": match["date"], "home": match["home"], "away": match["away"],
                "round": match.get("round"),
                "p_home": probs["home"], "p_draw": probs["draw"], "p_away": probs["away"],
                "outcome": truth, "pick": pick, "correct": int(pick == truth),
                "rps": rps(probs, truth), "rps_base": rps(base, truth),
                "swiss": season >= config.SWISS_FROM,
            })
    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {"released": False, "reason": "no walk-forward predictions"}
    skipped = int(preds.get("skipped", pd.Series(dtype=bool)).sum())
    scored = preds[~preds["skipped"].astype(bool)] if "skipped" in preds else preds
    if scored.empty:
        return {"released": False, "reason": "every match was skipped"}

    def block(frame):
        return {
            "n": int(len(frame)),
            "rps": round(float(frame["rps"].mean()), 5),
            "rps_baseline": round(float(frame["rps_base"].mean()), 5),
            "accuracy": round(float(frame["correct"].mean()), 4),
            "home_rate": round(float((frame["outcome"] == "home").mean()), 4),
        }

    per_season = {int(s): block(g) for s, g in scored.groupby("season")}
    overall = block(scored)
    eras = {
        "groups_2016_2023": block(scored[~scored["swiss"]]) if (~scored["swiss"]).any() else None,
        "swiss_2024_on": block(scored[scored["swiss"]]) if scored["swiss"].any() else None,
    }

    failures = []
    if overall["rps"] >= overall["rps_baseline"]:
        failures.append(f"overall RPS {overall['rps']} does not beat home-advantage "
                        f"baseline {overall['rps_baseline']}")
    for season, stats in sorted(per_season.items()):
        if stats["rps"] >= stats["rps_baseline"]:
            failures.append(f"{season}: RPS {stats['rps']} does not beat "
                            f"{stats['rps_baseline']}")

    return {"released": not failures,
            "reason": "; ".join(failures) if failures else "passed every gate",
            "overall": overall, "per_season": per_season, "eras": eras,
            "skipped_no_history": skipped}
