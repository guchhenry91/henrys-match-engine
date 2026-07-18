"""Tune the empirical-Bayes prior strength against GRID LOG-LOSS.

Diagnosis: the shrinkage intended for thin-data promoted clubs lands on every
team. `eff` is a sum of decay weights, not a match count, so with a ~231-day
half-life a five-season club has eff ~= 38 and prior_strength=6 pulls it 13.5%
toward the league mean. Fitted attack sd came out 0.176 against ~0.30-0.35 for
real top-flight sides, and the modal scoreline was 1-1 in 87% of fixtures.

Some shrinkage is statistically correct, so rather than guess, sweep it and let a
proper scoring rule decide. Exact-hit rate cannot referee this (it is flat, even
non-monotonic, in model quality), so the criterion is mean log-loss over the full
scoreline grid, with attack-sd and modal-1-1 rate reported as diagnostics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid

ROOT = Path(__file__).resolve().parents[1]
GRID = [6.0, 3.0, 1.5, 0.75, 0.0]        # 6.0 = current behaviour, 0.0 = no shrinkage


def evaluate(df, ps, min_train=760, step_days=7):
    """Walk-forward, strictly causal. Returns grid log-loss + diagnostics."""
    lls, modal11, sds, n = [], 0, [], 0
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(prior_strength=ps).fit(train, ref=cutoff)
        except Exception:
            continue
        sds.append(float(np.std(list(model.attack.values()))))
        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            g = scoreline_grid(lh, la, model.rho)
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            p = g[hg, ag] if hg < g.shape[0] and ag < g.shape[1] else 1e-12
            lls.append(-np.log(max(float(p), 1e-12)))
            i, j = np.unravel_index(np.argmax(g), g.shape)
            modal11 += (i == 1 and j == 1)
            n += 1
    return {
        "prior_strength": ps,
        "n": n,
        "grid_logloss": round(float(np.mean(lls)), 4),
        "attack_sd": round(float(np.mean(sds)), 3),
        "modal_1_1_pct": round(100 * modal11 / max(n, 1), 1),
    }


def main():
    df = (dataset.build_matches("PL").dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    for ps in GRID:
        r = evaluate(df, ps)
        rows.append(r)
        print(f"  prior_strength={ps:<5} logloss={r['grid_logloss']:.4f}  "
              f"attack_sd={r['attack_sd']:.3f}  modal_1-1={r['modal_1_1_pct']:.1f}%  n={r['n']}",
              flush=True)
    best = min(rows, key=lambda r: r["grid_logloss"])
    print(f"\nBEST by grid log-loss: prior_strength={best['prior_strength']} "
          f"(logloss {best['grid_logloss']}, attack_sd {best['attack_sd']}, "
          f"modal 1-1 {best['modal_1_1_pct']}%)")
    out = {"grid": rows, "best": best}
    (ROOT / "data-raw" / "leagues" / "prior_strength_tuning.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
