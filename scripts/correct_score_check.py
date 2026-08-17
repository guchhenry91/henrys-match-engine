"""Does the correct-score engine actually predict, or does it hedge?

The question this answers is not "what is the accuracy" -- that number already
exists -- but the sharper one: is the model picking scorelines because the match
suggests them, or is it defaulting to whatever score is safest in general?

Three things are measured, all on the HELD-OUT season only (leagues.tune's
holdout boundary), so nothing here is scored on data that chose a parameter:

  1. TOP-1 and TOP-3 accuracy for both candidate scorelines --
     grid mode (the likeliest score overall) and pick-conditional (the likeliest
     score given the 1X2 pick).
  2. The same for an ALWAYS 1-1 baseline. A model that has learned nothing
     useful about scorelines will not beat this, because 1-1 is the single most
     common result in all four leagues.
  3. The DISTRIBUTION of predicted scorelines. This is the hedging test: if the
     engine emits 1-1 for most fixtures it is not predicting, it is abstaining,
     and its accuracy would be the baseline's wearing a model's clothes.

Strictly causal: reuses leagues.backtest.walk_forward's own refit-weekly loop and
second-tier priors, so it scores the model that ships.
"""
import collections
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import poisson

from leagues import backtest, config, dataset, publish, second_tier, tune
from leagues.model import LeagueModel, score_for_outcome

BASELINE = "1-1"


def _walk(matches: pd.DataFrame, league: str, xi: float, xgw: float,
          min_train: int = 760, step_days: int = 7) -> pd.DataFrame:
    """walk_forward, but recording SCORELINES rather than 1X2 probabilities."""
    df = (matches.dropna(subset=["home_goals", "away_goals"])
                 .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) &
                  (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xgw).fit(train, ref=cutoff)
            priors = backtest._cutoff_priors(model, league, cutoff,
                                             set(test["home"]) | set(test["away"]))
            if priors:
                model = LeagueModel(xi=xi, xg_weight=xgw).fit(
                    train, ref=cutoff, priors=priors)
        except Exception as exc:
            print(f"  skip {cutoff.date()}: {exc}")
            continue
        for _, m in test.iterrows():
            p = model.predict(m["home"], m["away"])
            grid = p["grid"]
            flat = grid.ravel()
            order = np.argsort(-flat)[:3]
            top3 = []
            for k in order:
                h, a = np.unravel_index(k, grid.shape)
                top3.append(f"{int(h)}-{int(a)}")
            probs = {"home": p["p_home"], "draw": p["p_draw"], "away": p["p_away"]}
            outcome = max(probs, key=probs.get)
            lh, la = p["lambda_home"], p["lambda_away"]

            # CANDIDATE SELECTORS. rho is fitted by MLE and earns its place in the
            # 1X2 model, but tau multiplies the 1-1 cell by (1 - rho): at PL's
            # rho = -0.105 that is a 10.5% lift, enough to make 1-1 the argmax on
            # 85% of fixtures and wipe out any correct-score discrimination. These
            # ask whether a different way of READING the same fitted model picks
            # scorelines better, without touching the model itself.
            raw = np.outer(poisson.pmf(np.arange(grid.shape[0]), lh),
                           poisson.pmf(np.arange(grid.shape[1]), la))
            rh, ra = np.unravel_index(np.argmax(raw), raw.shape)
            rows.append({
                "date": m["date"],
                "actual": f"{int(m['home_goals'])}-{int(m['away_goals'])}",
                "grid": top3[0], "top3": top3,
                "conditional": score_for_outcome(grid, outcome),
                # argmax of the SAME lambdas with the tau correction removed
                "no_tau": f"{int(rh)}-{int(ra)}",
                # the blunt baseline: round each expected-goals number
                "rounded": f"{int(round(lh))}-{int(round(la))}",
                "p_top": float(flat[order[0]]),
            })
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, league: str) -> dict:
    n = len(res)
    grid_hit = (res["grid"] == res["actual"]).mean()
    cond_hit = (res["conditional"] == res["actual"]).mean()
    notau_hit = (res["no_tau"] == res["actual"]).mean()
    round_hit = (res["rounded"] == res["actual"]).mean()
    base_hit = (res["actual"] == BASELINE).mean()
    top3_hit = res.apply(lambda r: r["actual"] in r["top3"], axis=1).mean()

    dist = collections.Counter(res["grid"])
    top_share = dist.most_common(1)[0][1] / n
    distinct = len(dist)

    out = {
        "league": league, "n": n,
        "grid_top1_pct": round(100 * grid_hit, 2),
        "conditional_top1_pct": round(100 * cond_hit, 2),
        "no_tau_top1_pct": round(100 * notau_hit, 2),
        "rounded_top1_pct": round(100 * round_hit, 2),
        "no_tau_lift_pp": round(100 * (notau_hit - base_hit), 2),
        "no_tau_distinct": len(set(res["no_tau"])),
        "no_tau_top_share_pct": round(100 * max(collections.Counter(res["no_tau"]).values()) / n, 1),
        "always_1_1_pct": round(100 * base_hit, 2),
        "grid_top3_pct": round(100 * top3_hit, 2),
        "lift_over_baseline_pp": round(100 * (grid_hit - base_hit), 2),
        "distinct_scorelines_predicted": distinct,
        "most_predicted": dist.most_common(1)[0][0],
        "most_predicted_share_pct": round(100 * top_share, 1),
        "mean_top_score_prob_pct": round(100 * float(res["p_top"].mean()), 1),
        "distribution": {k: v for k, v in dist.most_common(8)},
    }
    print(f"\n=== {league} (holdout, n={n}) ===")
    print(f"  grid mode   top-1 {out['grid_top1_pct']:.2f}%   top-3 {out['grid_top3_pct']:.2f}%")
    print(f"  conditional top-1 {out['conditional_top1_pct']:.2f}%")
    print(f"  NO-TAU      top-1 {out['no_tau_top1_pct']:.2f}%  (lift {out['no_tau_lift_pp']:+.2f}pp, "
          f"{out['no_tau_distinct']} distinct, top score on {out['no_tau_top_share_pct']}%)")
    print(f"  rounded-xG  top-1 {out['rounded_top1_pct']:.2f}%")
    print(f"  always 1-1  top-1 {out['always_1_1_pct']:.2f}%   -> lift {out['lift_over_baseline_pp']:+.2f}pp")
    print(f"  predicts {distinct} distinct scorelines; most common "
          f"{out['most_predicted']} on {out['most_predicted_share_pct']}% of fixtures")
    print(f"  mean probability it assigns its own top score: {out['mean_top_score_prob_pct']}%")
    print(f"  distribution: {out['distribution']}")
    return out


def main():
    leagues = sys.argv[1:] or list(config.LEAGUES)
    pol = publish._read_raw("release_policy.json").get("leagues", {})
    allres, out = [], []
    for lg in leagues:
        p = pol.get(lg, {})
        xi, xgw = p.get("xi", 0.003), p.get("xg_weight", 0.75)
        print(f"\n--- {lg} (xi={xi}, xg_weight={xgw}) ---")
        res = _walk(dataset.build_matches(lg), lg, xi, xgw)
        held = res[pd.to_datetime(res["date"]) >= tune.holdout_start(lg)]
        if held.empty:
            print(f"  no holdout rows for {lg}")
            continue
        out.append(report(held, lg))
        allres.append(held)

    if allres:
        pooled = pd.concat(allres, ignore_index=True)
        out.append(report(pooled, "POOLED"))
        # What the 6 Scores Challenge would actually return, from the pooled rate.
        p = (pooled["grid"] == pooled["actual"]).mean()
        from math import comb
        print(f"\n=== 6 Scores Challenge, at a {100*p:.2f}% per-match rate ===")
        print(f"  expected correct per week: {6*p:.2f}")
        for k in range(0, 7):
            pk = comb(6, k) * p**k * (1 - p)**(6 - k)
            atleast = sum(comb(6, j) * p**j * (1-p)**(6-j) for j in range(k, 7))
            print(f"   exactly {k}: {100*pk:6.2f}%    at least {k}: {100*atleast:6.2f}%")

    path = "data-raw/leagues/correct_score_check.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
