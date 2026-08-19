"""Do any of the published goal distributions predict EXACT SCORES better?

The literature and penaltyblog's own comparison rank these models on 1X2 RPS,
where they finish within 0.0002 of each other and Dixon-Coles wins. Nobody
publishes the number this project actually needs: TOP-1 EXACT SCORE accuracy.
A distribution can leave RPS untouched while reshaping the scoreline grid, so
that ranking does not answer our question.

Five distributions, all shipped in penaltyblog 1.11 (no new dependency):
Dixon-Coles, bivariate Poisson, negative binomial, zero-inflated Poisson, and
the Weibull-count-plus-Frank-copula model of Boshnakov, Kharrat and McHale
(IJF 2017) -- the one specifically built to fit scorelines rather than outcomes.

FAIRNESS. Every candidate is fitted on GOALS ONLY, so the honest comparison is
against penaltyblog's own Dixon-Coles, not against our production model, which
additionally blends 75% xG into its strengths. Production is scored too, for
context, but a candidate beating goals-only DC is the finding worth chasing --
the xG blend could then be layered on top.

Walk-forward across the held-out season only, refitting weekly on everything
before each cutoff, with the same exponential time decay production uses.
"""
import collections
import json
import sys
import time

import numpy as np
import pandas as pd
import penaltyblog as pb

from leagues import backtest, dataset, publish, tune
from leagues.model import LeagueModel
from leagues.weights import decay_weights, XI_PER_DAY

MODELS = {
    "dixon_coles": pb.models.DixonColesGoalModel,
    "bivariate_poisson": pb.models.BivariatePoissonGoalModel,
    "negative_binomial": pb.models.NegativeBinomialGoalModel,
    "zero_inflated_poisson": pb.models.ZeroInflatedPoissonGoalsModel,
    "weibull_copula": pb.models.WeibullCopulaGoalsModel,
}
BASELINE = "1-1"


def _grid(model, home, away):
    p = model.predict(home, away)
    g = np.asarray(p.grid if hasattr(p, "grid") else p, dtype=float)
    s = g.sum()
    return g / s if s > 0 else g


def run(league="PL", step_days=7):
    matches = (dataset.build_matches(league)
               .dropna(subset=["home_goals", "away_goals"])
               .sort_values("date").reset_index(drop=True))
    boundary = tune.holdout_start(league)
    params = publish.model_params(league)
    rows = []
    failures = collections.Counter()
    cutoffs = pd.date_range(boundary, matches["date"].max(), freq=f"{step_days}D")
    print(f"{league}: {len(cutoffs)} weekly cutoffs across the holdout")

    for n, cutoff in enumerate(cutoffs, 1):
        train = matches[matches["date"] < cutoff]
        test = matches[(matches["date"] >= cutoff) &
                       (matches["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        # .copy() on every array: penaltyblog's Cython kernels reject a read-only
        # buffer with "buffer source array is read-only", and a pandas .to_numpy()
        # view is read-only. Our own model.py already copies for this reason;
        # omitting it here silently failed EVERY fit on EVERY cutoff, and the run
        # still finished and printed a report -- with only the one model that did
        # not go through penaltyblog in it.
        w = decay_weights(train["date"], ref=cutoff, xi=XI_PER_DAY).to_numpy().copy()
        hg = train["home_goals"].astype(int).to_numpy().copy()
        ag = train["away_goals"].astype(int).to_numpy().copy()
        ht = train["home"].to_numpy().copy()
        at = train["away"].to_numpy().copy()

        fitted = {}
        for name, cls in MODELS.items():
            try:
                mod = cls(hg, ag, ht, at, weights=w)
                mod.fit()
                fitted[name] = mod
            except Exception as exc:
                failures[name] += 1
                if failures[name] <= 2:      # show the real message, not just the type
                    print(f"  {cutoff.date()} {name}: {type(exc).__name__}: {str(exc)[:120]}")
        try:                                   # our shipped xG-blended model
            prod = LeagueModel(**params).fit(train, ref=cutoff)
            priors = backtest._cutoff_priors(prod, league, cutoff,
                                             set(test["home"]) | set(test["away"]))
            if priors:
                prod = LeagueModel(**params).fit(train, ref=cutoff, priors=priors)
        except Exception:
            prod = None

        for _, m in test.iterrows():
            actual = f"{int(m['home_goals'])}-{int(m['away_goals'])}"
            row = {"date": m["date"], "actual": actual}
            for name, mod in fitted.items():
                try:
                    g = _grid(mod, m["home"], m["away"])
                    i, j = np.unravel_index(np.argmax(g), g.shape)
                    row[name] = f"{int(i)}-{int(j)}"
                    row[f"{name}_ll"] = -np.log(max(
                        g[int(m["home_goals"]), int(m["away_goals"])], 1e-12))
                except Exception:
                    row[name] = None
            if prod is not None:
                try:
                    p = prod.predict(m["home"], m["away"])
                    g = p["grid"]
                    i, j = np.unravel_index(np.argmax(g), g.shape)
                    row["production_xg"] = f"{int(i)}-{int(j)}"
                    row["production_xg_ll"] = -np.log(max(
                        g[int(m["home_goals"]), int(m["away_goals"])], 1e-12))
                except Exception:
                    row["production_xg"] = None
            rows.append(row)
        if n % 10 == 0:
            print(f"  ...{n}/{len(cutoffs)} cutoffs, {len(rows)} fixtures")
    for name, k in failures.items():
        if k:
            print(f"  WARNING: {name} failed to fit on {k}/{len(cutoffs)} cutoffs")
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, league: str) -> dict:
    n = len(res)
    base_hits = (res["actual"] == BASELINE).to_numpy()
    rng = np.random.default_rng(20260818)
    idx = rng.integers(0, n, size=(10000, n))

    out = {"league": league, "n": n,
           "always_1_1_pct": round(100 * base_hits.mean(), 2), "models": {}}
    print(f"\n=== {league}: exact-score top-1, holdout n={n} ===")
    print(f"  {'always 1-1':<24}{100*base_hits.mean():6.2f}%")
    names = [c for c in ["dixon_coles", "bivariate_poisson", "negative_binomial",
                         "zero_inflated_poisson", "weibull_copula", "production_xg"]
             if c in res.columns]
    for name in names:
        hits = (res[name] == res["actual"]).to_numpy()
        d = hits.astype(float) - base_hits.astype(float)
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        ll = res[f"{name}_ll"].mean() if f"{name}_ll" in res else float("nan")
        distinct = len(set(res[name].dropna()))
        top_share = collections.Counter(res[name].dropna()).most_common(1)[0]
        out["models"][name] = {
            "top1_pct": round(100 * hits.mean(), 2),
            "vs_baseline_pp": round(100 * d.mean(), 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "beats_baseline": bool(lo > 0),
            "grid_logloss": round(float(ll), 4),
            "distinct_scores": distinct,
            "most_predicted": f"{top_share[0]} on {round(100*top_share[1]/n,1)}%",
        }
        m = out["models"][name]
        flag = "  <-- BEATS BASELINE" if m["beats_baseline"] else ""
        print(f"  {name:<24}{m['top1_pct']:6.2f}%   vs 1-1 {m['vs_baseline_pp']:+5.2f}pp "
              f"[{m['ci95_pp'][0]:+.2f},{m['ci95_pp'][1]:+.2f}]  ll {m['grid_logloss']:.4f}  "
              f"{m['distinct_scores']} scores, {m['most_predicted']}{flag}")
    return out


def main():
    leagues = sys.argv[1:] or ["PL"]
    out = []
    for lg in leagues:
        t = time.time()
        res = run(lg)
        print(f"  walk took {(time.time()-t)/60:.1f} min")
        out.append(report(res, lg))
    path = "data-raw/leagues/score_distribution_bakeoff.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
