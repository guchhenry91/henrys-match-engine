"""Phase 4a (complete version): de-vigged 1X2 AND Over/Under together.

ou_market_experiment.py already tested total-goals-from-O/U, split by the
MODEL's own attack/defence ratio. This extends it per the user's exact spec:
infer the market's expected total AND match balance, using both odds
markets, not the model's ratio for the split.

Method per held-out match:
  1. De-vig Avg>2.5/Avg<2.5 -> market total-goals lambda (same inversion as
     ou_market_experiment.py).
  2. De-vig the 1X2 odds -> market (p_home, p_draw, p_away).
  3. Solve for a split ratio r in (0,1) such that lambdas
     (r*total, (1-r)*total), run through the model's OWN fitted rho via
     scoreline_grid, produce (p_home, p_draw, p_away) closest (least-squares)
     to the market's devigged 1X2 -- i.e. the market's total AND its
     direction both come from the market, only rho (the low-score
     correlation shape) still comes from the model.
  4. Blend weight w against the model's own (lh, la) is fit on the EARLIER
     half of held-out matches, evaluated on a separate LATER half, exactly
     like every other market experiment in this repo.

LEAKAGE: odds_h/d/a and odds_over25/under25 are the historical CLOSING lines
already in leagues/history.py, used the same way scripts/market_gap_experiment.py
already uses odds_h/d/a for its own (already-shipped, negative) 1X2 blend --
this is a backtest against the best available historical archive, standard
practice for measuring whether market information helps at all. It is NOT
the same as using closing odds to grade an EARLIER prediction timestamp; if
this candidate ever moves toward production, the live pipeline would need
odds captured AT publish time, not the closing line -- noted here and in the
final report, not resolved by this script.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid, outcome_probs
from scripts.ou_market_experiment import devig_over_under, implied_total_lambda
from scripts.correct_score_benchmark import per_match_grid_probs, paired_ci

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def devig_1x2(oh, od, oa):
    raw = np.array([1 / oh, 1 / od, 1 / oa])
    out = raw / raw.sum()
    return float(out[0]), float(out[1]), float(out[2])


def solve_split(total: float, rho: float, target_1x2: tuple[float, float, float]) -> float:
    """Find r in (0.05, 0.95) minimizing squared error between the grid's
    1X2 probs at lambdas (r*total, (1-r)*total) and the market's 1X2."""
    target = np.array(target_1x2)

    def loss(r):
        lh, la = r * total, (1 - r) * total
        grid = scoreline_grid(max(lh, 0.05), max(la, 0.05), rho)
        probs = np.array(outcome_probs(grid))
        return float(((probs - target) ** 2).sum())

    res = minimize_scalar(loss, bounds=(0.05, 0.95), method="bounded")
    return float(res.x)


def run_league(league: str, xi=0.003, xg_weight=0.75, min_train=760, step_days=7):
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff)
        except Exception:
            continue
        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            row = {"hg": int(m["home_goals"]), "ag": int(m["away_goals"]),
                   "lh_model": lh, "la_model": la, "rho": model.rho}
            has_1x2 = pd.notna(m.get("odds_h")) and pd.notna(m.get("odds_d")) and pd.notna(m.get("odds_a"))
            has_ou = pd.notna(m.get("odds_over25")) and pd.notna(m.get("odds_under25"))
            if has_1x2 and has_ou:
                p1x2 = devig_1x2(m["odds_h"], m["odds_d"], m["odds_a"])
                p_over = devig_over_under(m["odds_over25"], m["odds_under25"])
                total = implied_total_lambda(p_over)
                r = solve_split(total, model.rho, p1x2)
                row["lh_market"] = r * total
                row["la_market"] = (1 - r) * total
            rows.append(row)
    return rows


def grid_for(r, lh_key, la_key):
    return scoreline_grid(r[lh_key], r[la_key], r["rho"])


def run():
    report = {}
    for league in LEAGUES:
        print(f"--- {league} ---")
        rows = run_league(league)
        rows_mkt = [r for r in rows if "lh_market" in r]
        n_total, n_mkt = len(rows), len(rows_mkt)
        print(f"  n={n_total}, with usable 1X2+O/U odds: {n_mkt} ({100*n_mkt/n_total:.0f}%)")
        if n_mkt < 200:
            report[league] = {"n_total": n_total, "n_with_odds": n_mkt,
                              "note": "insufficient combined-odds coverage"}
            continue

        half = n_mkt // 2
        tr, te = rows_mkt[:half], rows_mkt[half:]

        def blended(rows_, w):
            return [{"lh": w * r["lh_model"] + (1 - w) * r["lh_market"],
                     "la": w * r["la_model"] + (1 - w) * r["la_market"],
                     "rho": r["rho"], "hg": r["hg"], "ag": r["ag"]} for r in rows_]

        ws = np.linspace(0, 1, 21)
        train_losses = []
        for w in ws:
            b = blended(tr, w)
            p = per_match_grid_probs(b, lambda r: scoreline_grid(r["lh"], r["la"], r["rho"]))
            train_losses.append(float(-np.log(p).mean()))
        best_w = float(ws[int(np.argmin(train_losses))])

        model_only_test = [{"lh": r["lh_model"], "la": r["la_model"], "rho": r["rho"],
                            "hg": r["hg"], "ag": r["ag"]} for r in te]
        blend_test = blended(te, best_w)
        gfn = lambda r: scoreline_grid(r["lh"], r["la"], r["rho"])
        p_model = per_match_grid_probs(model_only_test, gfn)
        p_blend = per_match_grid_probs(blend_test, gfn)

        lo, hi = paired_ci(-np.log(p_model), -np.log(p_blend))
        report[league] = {
            "n_total": n_total, "n_with_odds": n_mkt,
            "train_n": len(tr), "test_n": len(te),
            "best_weight_on_model": round(best_w, 2),
            "test_grid_logloss": {
                "model_only": round(float(-np.log(p_model).mean()), 4),
                "combined_market_blend": round(float(-np.log(p_blend).mean()), 4),
            },
            "logloss_delta_model_minus_blend_95ci": [round(lo, 4), round(hi, 4)],
            "blend_significantly_better": bool(lo > 0),
        }
        print(f"  best_w={best_w:.2f}  model={report[league]['test_grid_logloss']['model_only']}  "
              f"blend={report[league]['test_grid_logloss']['combined_market_blend']}  "
              f"CI={report[league]['logloss_delta_model_minus_blend_95ci']}")
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "market_combined_experiment.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
