"""Is there ANY subset where our model adds information the market lacks?

w=0 pooled does not rule out a niche. Testing per league, by how confident the
market is, and by season stage. Fitted on the earlier half, evaluated on the
later half throughout.

CAUTION, stated up front: this is many comparisons at once. A single positive w
in one bucket is what noise looks like when you slice enough ways. Only a weight
that is both meaningfully above zero AND improves the held-out RPS counts.
"""
import json, numpy as np, pandas as pd
from leagues import backtest, dataset

P = json.load(open('data-raw/leagues/backtest_report.json'))
rows = []
for lg in ("PL","LALIGA","BUNDESLIGA","LIGUE1"):
    p = P[lg]
    r = backtest.walk_forward(dataset.build_matches(lg), xi=p["xi"], xg_weight=p["xg_weight"])
    if "m_home" not in r.columns: continue
    r = r[r["m_home"].notna()].copy(); r["league"] = lg
    rows.append(r)
d = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)
M = d[["p_home","p_draw","p_away"]].to_numpy(); K = d[["m_home","m_draw","m_away"]].to_numpy()
y = d["y"].to_numpy() if "y" in d.columns else d["outcome"].to_numpy()
Y = np.zeros_like(M); Y[np.arange(len(y)),y] = 1

def rps(idx, p):
    c, cy = np.cumsum(p[idx],1), np.cumsum(Y[idx],1)
    return float(((c-cy)**2).sum(1).mean()/2)

ws = np.linspace(0,1,101)
def fit_eval(mask, label):
    idx = np.where(mask)[0]
    if len(idx) < 200: return
    cut = idx[len(idx)//2]
    tr, te = idx[idx < cut], idx[idx >= cut]
    if len(tr) < 100 or len(te) < 100: return
    w = min(ws, key=lambda w: rps(tr, w*M+(1-w)*K))
    base, bl = rps(te,K), rps(te, w*M+(1-w)*K)
    flag = "  <-- adds signal" if (w > 0.10 and bl < base - 1e-4) else ""
    print(f"  {label:<26} n={len(idx):<5} w={w:.2f}  market {base:.4f} -> blend {bl:.4f}"
          f"  {bl-base:+.4f}{flag}")

print("BY LEAGUE")
for lg in ("PL","LALIGA","BUNDESLIGA","LIGUE1"):
    fit_eval((d["league"]==lg).to_numpy(), lg)

print("\nBY HOW CONFIDENT THE MARKET IS (its favourite's probability)")
kmax = K.max(1)
for lo,hi,lab in ((0,0.40,"toss-up  <40%"),(0.40,0.50,"40-50%"),
                  (0.50,0.65,"50-65%"),(0.65,1.01,"clear fav >65%")):
    fit_eval((kmax>=lo)&(kmax<hi), lab)

print("\nBY SEASON STAGE (model leans on priors early, market may lag)")
mw = d.groupby([d["league"], d["date"].dt.year]).cumcount() if "date" in d else None
if mw is not None:
    for lo,hi,lab in ((0,60,"first ~6 weeks"),(60,240,"midseason"),(240,10**6,"run-in")):
        fit_eval(((mw>=lo)&(mw<hi)).to_numpy(), lab)
