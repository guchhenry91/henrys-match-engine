"""Two cheap candidates for closing the gap to the market.

1. CALIBRATION. If stated probabilities are systematically off, correcting them
   improves RPS using no new information at all -- the cheapest possible win.
2. BLEND. Fit w in (w*model + (1-w)*market). This is also a DIAGNOSTIC: if the
   best w is ~0, our model carries no information the market lacks, and no amount
   of tuning the same feature set will change that.

Both are fitted on the EARLIER half and evaluated on the LATER half. Fitting and
scoring on the same matches would manufacture an improvement that evaporates live,
which is the exact error this project keeps catching.
"""
import json, numpy as np, pandas as pd
from leagues import backtest, dataset

P = json.load(open('data-raw/leagues/backtest_report.json'))
rows = []
for lg in ("PL", "LALIGA", "BUNDESLIGA", "LIGUE1"):
    p = P[lg]
    r = backtest.walk_forward(dataset.build_matches(lg), xi=p["xi"], xg_weight=p["xg_weight"])
    if "m_home" not in r.columns: continue
    r = r[r["m_home"].notna()].copy(); r["league"] = lg
    rows.append(r)
d = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)

M = d[["p_home","p_draw","p_away"]].to_numpy()
K = d[["m_home","m_draw","m_away"]].to_numpy()
y = d["y"].to_numpy() if "y" in d.columns else d["outcome"].to_numpy()
Y = np.zeros_like(M); Y[np.arange(len(y)), y] = 1

def rps(p):
    c, cy = np.cumsum(p,1), np.cumsum(Y,1)
    return float(((c-cy)**2).sum(1).mean()/2)
def rps_i(p, i):
    c, cy = np.cumsum(p[i],1), np.cumsum(Y[i],1)
    return float(((c-cy)**2).sum(1).mean()/2)

half = len(d)//2
tr, te = np.arange(half), np.arange(half, len(d))
print(f"n={len(d)}  train={len(tr)} (earlier)  test={len(te)} (later)\n")
print(f"BASELINE on test:  model {rps_i(M,te):.4f}   market {rps_i(K,te):.4f}   "
      f"gap {rps_i(M,te)-rps_i(K,te):+.4f}\n")

# --- 1. calibration: sharpen/flatten with a single temperature on log-probs
def temper(p, t):
    q = np.clip(p,1e-9,1)**t
    return q/q.sum(1,keepdims=True)
ts = np.linspace(0.6,1.8,61)
best_t = min(ts, key=lambda t: rps_i(temper(M,t),tr))
print(f"1. CALIBRATION  best temperature on train = {best_t:.2f}  (1.00 = already calibrated)")
print(f"   test RPS  raw {rps_i(M,te):.4f} -> tempered {rps_i(temper(M,best_t),te):.4f}  "
      f"({rps_i(temper(M,best_t),te)-rps_i(M,te):+.4f})\n")

# --- 2. blend with the market
ws = np.linspace(0,1,101)
best_w = min(ws, key=lambda w: rps_i(w*M+(1-w)*K, tr))
B = best_w*M+(1-best_w)*K
print(f"2. BLEND        best weight on OUR model = {best_w:.2f}  (0.00 = model adds nothing)")
print(f"   test RPS  market {rps_i(K,te):.4f} -> blend {rps_i(B,te):.4f}  "
      f"({rps_i(B,te)-rps_i(K,te):+.4f})")
