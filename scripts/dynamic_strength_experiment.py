"""Phase 4b (dynamic strengths): is the current recency-decay half-life optimal,
or would a MORE or LESS responsive xi reduce the total-goals under-prediction
found in Phase 2/3 (mean predicted goals consistently below actual, worst in
Bundesliga)?

Current default: xi=0.003/day, ~231-day half-life (leagues/weights.py), tuned
by 1X2 RPS in leagues/tune.py -- never specifically re-checked against grid
log-loss or total-goals calibration, which is what this tests.

Same causal discipline as every other candidate here: xi is selected on the
EARLIER half of each league's walk-forward matches, evaluated on a completely
separate LATER half. Selecting xi against the full backtest window (no split)
would be exactly the kind of in-sample tuning this project's own gate
(leagues/tune.py's docstring) warns against.
"""
import json
from pathlib import Path

import numpy as np

from leagues.model import scoreline_grid
from scripts.correct_score_benchmark import run_league, per_match_grid_probs, paired_ci

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]
XI_GRID = [0.0015, 0.002, 0.003, 0.005, 0.008, 0.012]  # 0.003 = current default


def logloss_for_rows(rows):
    p = per_match_grid_probs(rows, lambda r: scoreline_grid(r["lh"], r["la"], r["rho"]))
    return float(-np.log(p).mean())


def run():
    report = {}
    for league in LEAGUES:
        print(f"--- {league} ---")
        by_xi = {}
        for xi in XI_GRID:
            rows = run_league(league, xi=xi)
            by_xi[xi] = sorted(rows, key=lambda r: r["date"])
            print(f"  xi={xi}: n={len(rows)}")

        n = len(by_xi[XI_GRID[0]])
        half = n // 2
        # select xi on the earlier half
        train_losses = {xi: logloss_for_rows(rows[:half]) for xi, rows in by_xi.items()}
        best_xi = min(train_losses, key=train_losses.get)

        test_rows_default = by_xi[0.003][half:]
        test_rows_best = by_xi[best_xi][half:]
        p_default = per_match_grid_probs(test_rows_default, lambda r: scoreline_grid(r["lh"], r["la"], r["rho"]))
        p_best = per_match_grid_probs(test_rows_best, lambda r: scoreline_grid(r["lh"], r["la"], r["rho"]))

        total_pred_default = np.mean([r["lh"] + r["la"] for r in test_rows_default])
        total_pred_best = np.mean([r["lh"] + r["la"] for r in test_rows_best])
        total_actual = np.mean([r["hg"] + r["ag"] for r in test_rows_default])

        lo, hi = paired_ci(-np.log(p_default), -np.log(p_best))
        report[league] = {
            "train_losses_by_xi": {str(k): round(v, 4) for k, v in train_losses.items()},
            "selected_xi": best_xi,
            "test_n": len(test_rows_default),
            "test_grid_logloss": {
                "default_xi_0.003": round(float(-np.log(p_default).mean()), 4),
                f"selected_xi_{best_xi}": round(float(-np.log(p_best).mean()), 4),
            },
            "logloss_delta_default_minus_selected_95ci": [round(lo, 4), round(hi, 4)],
            "selected_significantly_better": bool(lo > 0),
            "mean_total_goals": {
                "actual": round(float(total_actual), 3),
                "default_xi_predicted": round(float(total_pred_default), 3),
                "selected_xi_predicted": round(float(total_pred_best), 3),
            },
        }
        print(f"  selected xi={best_xi} (default 0.003)  "
              f"test logloss default={report[league]['test_grid_logloss']['default_xi_0.003']}  "
              f"selected={report[league]['test_grid_logloss'][f'selected_xi_{best_xi}']}  "
              f"CI={report[league]['logloss_delta_default_minus_selected_95ci']}")
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "dynamic_strength_experiment.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
