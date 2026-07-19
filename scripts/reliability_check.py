"""Reliability: when the model says 65%, does it win 65% of the time?

This -- not the gap to the market -- is what decides whether a stated probability
can be trusted. A model can be miles behind the market and still perfectly
calibrated, or beat it and lie about its own confidence.
"""
import json, numpy as np, pandas as pd
from leagues import backtest, dataset

P = json.load(open('data-raw/leagues/backtest_report.json'))
rows = []
for lg in ("PL","LALIGA","BUNDESLIGA","LIGUE1"):
    p = P[lg]
    r = backtest.walk_forward(dataset.build_matches(lg), xi=p["xi"], xg_weight=p["xg_weight"])
    rows.append(r)
d = pd.concat(rows, ignore_index=True)
M = d[["p_home","p_draw","p_away"]].to_numpy()
y = d["y"].to_numpy() if "y" in d.columns else d["outcome"].to_numpy()

pick, p_pick = M.argmax(1), M.max(1)
hit = (pick == y)
print(f"n = {len(d)} matches, every one predicted using only earlier data\n")
print("  model says     n      actually won    gap")
for lo,hi in ((0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,0.75),(0.75,1.01)):
    m = (p_pick>=lo)&(p_pick<hi)
    if m.sum()<30: continue
    stated, actual, n = p_pick[m].mean(), hit[m].mean(), int(m.sum())
    se = np.sqrt(actual*(1-actual)/n)
    ok = "ok" if abs(stated-actual) < 1.96*se else "OFF"
    print(f"   {lo:.0%}-{hi if hi<=1 else 1:.0%}      {n:<6} "
          f"{stated:.1%} -> {actual:.1%}   {actual-stated:+.1%}  {ok}")
print(f"\noverall stated {p_pick.mean():.1%}  actual {hit.mean():.1%}  "
      f"({hit.mean()-p_pick.mean():+.1%})")
