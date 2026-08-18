"""How much is there to win from BETTER LAMBDAS? An upper bound.

THE RESULT, so nobody re-runs this hoping for a different one. Feeding the
Premier League model the actual xG each side produced -- information no forecast
can have -- lifts correct score only from 11.84% to 16.84%. That is the entire
prize for perfecting the lambdas, and a real model captures a fraction of it.

But telling it the TRUE TOTAL GOALS as well jumps to 56.05%. So correct score is
a TOTAL-GOALS problem, not a split problem: given the total, the model already
picks the right split most of the time. The catch is that this engine has no edge
on totals -- over/under 2.5 was measured at 57.7% against a 58.8% base rate (see
model.goals_markets) -- while the market does. The only lever that would move
correct score in the Premier League is therefore the market''s total, which is
exactly the blend the owner has ruled out for PL.

Feeds the model lambdas it could never have -- the actual xG each side generated
in the match, and then the actual TOTAL goals -- and measures correct-score
accuracy. No real forecast can beat these, so whatever they score is the ceiling
on any lambda improvement.
"""
import numpy as np, pandas as pd
from leagues import dataset, publish, tune
from leagues.model import LeagueModel, scoreline_grid

for lg in ["PL", "LALIGA"]:
    m = dataset.build_matches(lg)
    mod = LeagueModel(**publish.model_params(lg)).fit(m)
    h = m[pd.to_datetime(m["date"]) >= tune.holdout_start(lg)].dropna(
        subset=["home_goals", "away_goals", "home_xg", "away_xg"])
    actual = h.apply(lambda r: f"{int(r.home_goals)}-{int(r.away_goals)}", axis=1)

    # ORACLE A: lambdas = the xG actually produced in the match
    def mode_from(lh, la):
        g = scoreline_grid(float(lh), float(la), mod.rho)
        i, j = np.unravel_index(np.argmax(g), g.shape)
        return f"{int(i)}-{int(j)}"
    xg_mode = h.apply(lambda r: mode_from(r.home_xg, r.away_xg), axis=1)

    # ORACLE B: also told the true TOTAL goals -- pick the likeliest score summing to it
    def mode_given_total(lh, la, tot):
        g = scoreline_grid(float(lh), float(la), mod.rho)
        best, bp = None, -1
        for i in range(g.shape[0]):
            j = tot - i
            if 0 <= j < g.shape[1] and g[i, j] > bp:
                best, bp = f"{i}-{j}", g[i, j]
        return best
    tot_mode = h.apply(lambda r: mode_given_total(r.home_xg, r.away_xg,
                                                  int(r.home_goals + r.away_goals)), axis=1)

    base = (actual == "1-1").mean()
    print(f"\n=== {lg}  (holdout n={len(h)}) ===")
    print(f"  always 1-1                          {100*base:5.2f}%")
    print(f"  shipped model (measured earlier)     {'11.84' if lg=='PL' else '18.42'}%")
    print(f"  ORACLE: lambdas = actual match xG   {100*(xg_mode==actual).mean():5.2f}%")
    print(f"  ORACLE: xG + true total goals       {100*(tot_mode==actual).mean():5.2f}%")
