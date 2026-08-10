"""Phase 4 corrections for total-goals under-prediction, tested causally.

Mechanism audit (total_goals_mechanism_audit.json) ruled out the xG blend,
shrinkage, and home-advantage architecture as primary drivers (each moves the
in-sample level by only 0.02-0.09 goals, inconsistent in direction across
leagues) -- the gap is much larger in true walk-forward prediction than
in-sample, pointing to real-time forecasting conservatism (the
decay+shrinkage tradeoff paying for lower variance with a small bias) rather
than a fixed mechanism bug.

Candidates, all fit CAUSALLY (recent-window statistics computed only from
matches strictly before the test cutoff, same walk-forward loop as every
other benchmark here):
  total_scale     - single factor s from comparing actual vs model-predicted
                     total goals over the most recent RECENT_DAYS of the
                     training window; lh'=s*lh, la'=s*la (preserves H/A share)
  home_away_scale - separate factors sh, sa from the same recent window,
                     comparing actual home goals vs predicted lh and actual
                     away goals vs predicted la independently
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues.model import LeagueModel, scoreline_grid, outcome_probs
from leagues import dataset
from scripts.correct_score_benchmark import per_match_grid_probs, paired_ci

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]
RECENT_DAYS = 120


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

        # Recent-window recalibration factor: causal, uses only TRAIN matches
        # from the last RECENT_DAYS before cutoff.
        recent = train[train["date"] >= cutoff - pd.Timedelta(days=RECENT_DAYS)]
        s_total, sh, sa = 1.0, 1.0, 1.0
        if len(recent) >= 30:
            pred_h, pred_a, act_h, act_a = [], [], [], []
            for _, rr in recent.iterrows():
                try:
                    lh_r, la_r = model.lambdas(rr["home"], rr["away"])
                except KeyError:
                    continue
                pred_h.append(lh_r); pred_a.append(la_r)
                act_h.append(rr["home_goals"]); act_a.append(rr["away_goals"])
            if pred_h:
                pred_h, pred_a = np.array(pred_h), np.array(pred_a)
                act_h, act_a = np.array(act_h), np.array(act_a)
                pred_total, act_total = (pred_h + pred_a).mean(), (act_h + act_a).mean()
                if pred_total > 0:
                    s_total = float(np.clip(act_total / pred_total, 0.7, 1.4))
                if pred_h.mean() > 0:
                    sh = float(np.clip(act_h.mean() / pred_h.mean(), 0.7, 1.4))
                if pred_a.mean() > 0:
                    sa = float(np.clip(act_a.mean() / pred_a.mean(), 0.7, 1.4))

        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            rows.append({
                "hg": int(m["home_goals"]), "ag": int(m["away_goals"]),
                "lh": lh, "la": la, "rho": model.rho,
                "lh_total_scale": lh * s_total, "la_total_scale": la * s_total,
                "lh_hascale": lh * sh, "la_hascale": la * sa,
                "s_total": s_total, "sh": sh, "sa": sa,
            })
    return rows


def eval_candidate(rows, lh_key, la_key):
    def gfn(r):
        return scoreline_grid(r[lh_key], r[la_key], r["rho"])
    p = per_match_grid_probs(rows, gfn)
    hits1 = np.array([np.unravel_index(np.argmax(gfn(r)), gfn(r).shape) == (r["hg"], r["ag"]) for r in rows])

    def top3_hit(r):
        g = gfn(r)
        idx = np.argsort(-g.ravel())[:3]
        cells = {tuple(int(x) for x in np.unravel_index(k, g.shape)) for k in idx}
        return (r["hg"], r["ag"]) in cells
    hits3 = np.array([top3_hit(r) for r in rows])

    total_actual = np.array([r["hg"] + r["ag"] for r in rows])
    total_pred = np.array([r[lh_key] + r[la_key] for r in rows])
    tail_actual = float((total_actual >= 4).mean())
    tail_pred = np.mean([gfn(r)[np.add.outer(np.arange(gfn(r).shape[0]), np.arange(gfn(r).shape[1])) >= 4].sum() for r in rows])

    def outcome_idx(hg, ag):
        return 0 if hg > ag else (1 if hg == ag else 2)
    def rps_1x2(r):
        g = gfn(r)
        probs = np.array(outcome_probs(g))
        obs = np.zeros(3); obs[outcome_idx(r["hg"], r["ag"])] = 1.0
        cp, co = np.cumsum(probs), np.cumsum(obs)
        return float(((cp - co) ** 2)[:2].sum() / 2.0)

    return {
        "n": len(rows),
        "grid_logloss_nats": round(float(-np.log(p).mean()), 4),
        "top1_pct": round(100 * float(hits1.mean()), 2),
        "top3_pct": round(100 * float(hits3.mean()), 2),
        "mean_predicted_total": round(float(total_pred.mean()), 3),
        "mean_actual_total": round(float(total_actual.mean()), 3),
        "predicted_pct_4plus_goals": round(100 * tail_pred, 2),
        "actual_pct_4plus_goals": round(100 * tail_actual, 2),
        "rps_1x2": round(float(np.mean([rps_1x2(r) for r in rows])), 4),
    }, p


def run():
    report = {}
    for league in LEAGUES:
        print(f"--- {league} ---")
        rows = run_league(league)
        base_metrics, p_base = eval_candidate(rows, "lh", "la")
        total_metrics, p_total = eval_candidate(rows, "lh_total_scale", "la_total_scale")
        ha_metrics, p_ha = eval_candidate(rows, "lh_hascale", "la_hascale")

        lo_t, hi_t = paired_ci(-np.log(p_base), -np.log(p_total))
        lo_h, hi_h = paired_ci(-np.log(p_base), -np.log(p_ha))

        report[league] = {
            "existing": base_metrics,
            "total_scale_correction": {**total_metrics,
                "logloss_delta_vs_existing_95ci": [round(lo_t, 4), round(hi_t, 4)],
                "significantly_better": bool(lo_t > 0)},
            "home_away_scale_correction": {**ha_metrics,
                "logloss_delta_vs_existing_95ci": [round(lo_h, 4), round(hi_h, 4)],
                "significantly_better": bool(lo_h > 0)},
            "mean_scale_factors": {
                "s_total": round(float(np.mean([r["s_total"] for r in rows])), 3),
                "sh": round(float(np.mean([r["sh"] for r in rows])), 3),
                "sa": round(float(np.mean([r["sa"] for r in rows])), 3),
            },
        }
        print(f"  existing logloss={base_metrics['grid_logloss_nats']}  "
              f"total_scale={total_metrics['grid_logloss_nats']} CI={report[league]['total_scale_correction']['logloss_delta_vs_existing_95ci']}  "
              f"ha_scale={ha_metrics['grid_logloss_nats']} CI={report[league]['home_away_scale_correction']['logloss_delta_vs_existing_95ci']}")
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "total_goals_correction_experiment.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
