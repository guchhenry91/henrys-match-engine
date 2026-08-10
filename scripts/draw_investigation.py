"""Is the model systematically underpricing draws?

Zero draw picks across 1,372 full-season fixtures (Phase 1) is either (a)
correct -- home advantage genuinely keeps draw from ever being the largest
of the three probabilities in these leagues' current matchups -- or (b) a
sign the model's draw probability itself runs low versus reality. This
checks both: model vs. market vs. actual, and calibration by bucket.

Confirmed by direct code read (leagues/publish.py line 382, `pick =
max(probs, key=probs.get)`): the pick-selection code is a correct argmax
over the three raw model probabilities. No calibration layer is applied in
production -- leagues/model.py's Calibrator class exists but is not called
anywhere in leagues/ or scripts/, confirmed by repo-wide grep. So the
question is entirely about whether the RAW probabilities are honest, not
whether the selection logic mishandles them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues.model import LeagueModel
from leagues import dataset

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def devig_1x2(oh, od, oa):
    raw = np.array([1 / oh, 1 / od, 1 / oa])
    out = raw / raw.sum()
    return float(out[1])  # p_draw


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
            from leagues.model import scoreline_grid, outcome_probs
            grid = scoreline_grid(lh, la, model.rho)
            ph, pdw, pa = outcome_probs(grid)
            row = {"p_home": ph, "p_draw": pdw, "p_away": pa,
                   "actual_draw": int(m["home_goals"] == m["away_goals"]),
                   "model_picks_draw": int(pdw >= ph and pdw >= pa)}
            if pd.notna(m.get("odds_h")) and pd.notna(m.get("odds_d")) and pd.notna(m.get("odds_a")):
                mkt_pdraw = devig_1x2(m["odds_h"], m["odds_d"], m["odds_a"])
                raw = np.array([1/m["odds_h"], 1/m["odds_d"], 1/m["odds_a"]]); raw /= raw.sum()
                row["market_p_draw"] = mkt_pdraw
                row["market_picks_draw"] = int(raw[1] >= raw[0] and raw[1] >= raw[2])
            rows.append(row)
    return rows


def calibration_buckets(rows, prob_key, n_buckets=5):
    have = [r for r in rows if prob_key in r] if prob_key != "p_draw" else rows
    probs = np.array([r[prob_key] for r in have])
    actual = np.array([r["actual_draw"] for r in have])
    edges = np.quantile(probs, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -1, 2  # ensure all points fall in a bucket
    out = []
    for i in range(n_buckets):
        mask = (probs > edges[i]) & (probs <= edges[i + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "n": int(mask.sum()),
            "mean_predicted_p_draw": round(float(probs[mask].mean()), 4),
            "actual_draw_rate": round(float(actual[mask].mean()), 4),
        })
    return out


def run():
    report = {}
    for league in LEAGUES:
        print(f"--- {league} ---")
        rows = run_league(league)
        rows_mkt = [r for r in rows if "market_p_draw" in r]

        actual_draw_rate = float(np.mean([r["actual_draw"] for r in rows]))
        model_mean_pdraw = float(np.mean([r["p_draw"] for r in rows]))
        model_draw_picks = int(sum(r["model_picks_draw"] for r in rows))

        entry = {
            "n": len(rows),
            "actual_draw_rate_pct": round(100 * actual_draw_rate, 2),
            "model_mean_p_draw_pct": round(100 * model_mean_pdraw, 2),
            "model_n_draw_picks": model_draw_picks,
            "calibration_by_bucket_model": calibration_buckets(rows, "p_draw"),
        }

        if rows_mkt:
            market_mean_pdraw = float(np.mean([r["market_p_draw"] for r in rows_mkt]))
            market_draw_picks = int(sum(r["market_picks_draw"] for r in rows_mkt))
            entry.update({
                "n_with_market_odds": len(rows_mkt),
                "market_mean_p_draw_pct": round(100 * market_mean_pdraw, 2),
                "market_n_draw_picks": market_draw_picks,
                "market_draw_pick_pct": round(100 * market_draw_picks / len(rows_mkt), 2),
                "calibration_by_bucket_market": calibration_buckets(rows_mkt, "market_p_draw"),
            })

        report[league] = entry
        print(f"  actual_draw_rate={entry['actual_draw_rate_pct']}%  "
              f"model_mean_pdraw={entry['model_mean_p_draw_pct']}%  "
              f"market_mean_pdraw={entry.get('market_mean_p_draw_pct')}%  "
              f"model_draw_picks={model_draw_picks}  market_draw_picks={entry.get('market_n_draw_picks')}")
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "draw_investigation.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
