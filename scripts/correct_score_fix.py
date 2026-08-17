"""Can the Premier League's correct-score prediction be made to beat 1-1?

WHY THIS IS NOT OBVIOUS. The engine is not broken. For most PL fixtures 1-1
genuinely IS the single likeliest scoreline -- with mean lambdas of 1.55 and
1.29, both marginals peak at one goal -- and the model assigns it about 12%,
which is roughly what it then hits. The model is CALIBRATED and still useless
here, because always writing 1-1 also scores ~12.4%. Beating that needs the
scoreline choice to discriminate BETWEEN fixtures, not to be individually
well-calibrated.

So before trying fixes, this establishes the CEILING. Two questions, in order:

  1. IS THE LIMIT THE SELECTION, OR THE LAMBDAS?
     Learn, on the sweep seasons only, the empirically most common actual score
     for each (lambda_home, lambda_away) bucket. Apply that map to the holdout.
     If a lookup table beats the model's own mode, the lambdas carry usable
     signal the mode is throwing away and better selection is worth building.
     If it does not, the lambdas are the ceiling and no amount of re-reading the
     grid will help -- which would be the honest answer, and the end of it.

  2. DOES DISCRIMINATION EXIST AT ALL?
     Split holdout fixtures by how far the model's own top score sits from 1-1.
     If accuracy on the fixtures where the model picks something OTHER than 1-1
     is materially above the 1-1 base rate, there is a subset worth trusting
     even when the pooled number is flat.

Strictly leakage-free: the bucket map is fitted on pre-holdout seasons only, the
same boundary leagues.tune uses.
"""
import collections
import json
import sys

import numpy as np
import pandas as pd

from leagues import backtest, config, dataset, publish, tune
from leagues.model import LeagueModel

BASELINE = "1-1"
# Lambda buckets. Coarse enough that every bucket has a usable sample on ~4
# seasons, fine enough to separate a 1.1-goal side from a 2.2-goal one.
EDGES = [0.0, 0.9, 1.15, 1.4, 1.7, 2.1, 9.9]


def _bucket(x: float) -> int:
    return int(np.digitize(x, EDGES) - 1)


def walk(matches: pd.DataFrame, league: str, xi: float, xgw: float,
         min_train: int = 760, step_days: int = 7) -> pd.DataFrame:
    df = (matches.dropna(subset=["home_goals", "away_goals"])
                 .sort_values("date").reset_index(drop=True))
    rows = []
    for cutoff in pd.date_range(df.loc[min_train, "date"], df["date"].max(),
                                freq=f"{step_days}D"):
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
            g = p["grid"]
            h, a = np.unravel_index(np.argmax(g), g.shape)
            rows.append({
                "date": m["date"],
                "lh": p["lambda_home"], "la": p["lambda_away"],
                "mode": f"{int(h)}-{int(a)}",
                "actual": f"{int(m['home_goals'])}-{int(m['away_goals'])}",
                "p_mode": float(g[h, a]),
            })
    return pd.DataFrame(rows)


def bucket_map(train: pd.DataFrame, min_n: int = 25) -> dict:
    """(bucket_h, bucket_a) -> most common ACTUAL score in that bucket.

    A bucket thinner than min_n falls back to the overall most common score,
    because a mode taken from a handful of matches is noise wearing a table's
    clothes.
    """
    overall = collections.Counter(train["actual"]).most_common(1)[0][0]
    out, counts = {}, collections.defaultdict(collections.Counter)
    for _, r in train.iterrows():
        counts[(_bucket(r["lh"]), _bucket(r["la"]))][r["actual"]] += 1
    for k, c in counts.items():
        out[k] = c.most_common(1)[0][0] if sum(c.values()) >= min_n else overall
    return {"map": out, "fallback": overall}


def evaluate(league: str, res: pd.DataFrame) -> dict:
    boundary = tune.holdout_start(league)
    d = pd.to_datetime(res["date"])
    train, held = res[d < boundary], res[d >= boundary]
    if train.empty or held.empty:
        return {}

    bm = bucket_map(train)
    lookup = held.apply(
        lambda r: bm["map"].get((_bucket(r["lh"]), _bucket(r["la"])), bm["fallback"]),
        axis=1)

    n = len(held)
    mode_hit = (held["mode"] == held["actual"]).mean()
    look_hit = (lookup == held["actual"]).mean()
    base_hit = (held["actual"] == BASELINE).mean()

    # Question 2: is there a trustworthy subset?
    off11 = held[held["mode"] != BASELINE]
    on11 = held[held["mode"] == BASELINE]
    off_hit = (off11["mode"] == off11["actual"]).mean() if len(off11) else float("nan")
    on_hit = (on11["mode"] == on11["actual"]).mean() if len(on11) else float("nan")

    out = {
        "league": league, "n": n,
        "model_mode_pct": round(100 * mode_hit, 2),
        "empirical_lookup_pct": round(100 * look_hit, 2),
        "always_1_1_pct": round(100 * base_hit, 2),
        "lookup_lift_vs_baseline_pp": round(100 * (look_hit - base_hit), 2),
        "lookup_lift_vs_model_pp": round(100 * (look_hit - mode_hit), 2),
        "n_model_picks_non_1_1": int(len(off11)),
        "acc_when_model_avoids_1_1_pct": round(100 * off_hit, 2),
        "acc_when_model_says_1_1_pct": round(100 * on_hit, 2),
        "distinct_lookup_scores": int(len(set(lookup))),
    }
    print(f"\n=== {league} (holdout n={n}) ===")
    print(f"  model mode          {out['model_mode_pct']:.2f}%")
    print(f"  empirical lookup    {out['empirical_lookup_pct']:.2f}%   "
          f"(vs baseline {out['lookup_lift_vs_baseline_pp']:+.2f}pp, "
          f"vs model {out['lookup_lift_vs_model_pp']:+.2f}pp)")
    print(f"  always 1-1          {out['always_1_1_pct']:.2f}%")
    print(f"  when the model picks something OTHER than 1-1 "
          f"({out['n_model_picks_non_1_1']} fixtures): {out['acc_when_model_avoids_1_1_pct']:.2f}%")
    print(f"  when it picks 1-1: {out['acc_when_model_says_1_1_pct']:.2f}%")
    print(f"  lookup emits {out['distinct_lookup_scores']} distinct scores")
    return out


def main():
    leagues = sys.argv[1:] or list(config.LEAGUES)
    pol = publish._read_raw("release_policy.json").get("leagues", {})
    out = []
    for lg in leagues:
        p = pol.get(lg, {})
        print(f"\n--- {lg} ---")
        res = walk(dataset.build_matches(lg), lg,
                   p.get("xi", 0.003), p.get("xg_weight", 0.75))
        r = evaluate(lg, res)
        if r:
            out.append(r)
    path = "data-raw/leagues/correct_score_fix.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
