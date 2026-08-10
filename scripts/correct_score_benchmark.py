"""Phase 2/3: leakage-free walk-forward benchmark for the correct-score engine.

Strictly causal (train on date < cutoff only, weekly steps, same discipline as
scripts/correct_score_backtest.py). Evaluates every candidate on the SAME
held-out matches per league, so comparisons are paired and fair. Primary
metric is full-grid log-loss; everything else is secondary, per Phase 3's
explicit instruction not to optimize on hit-rate alone.

Candidates evaluated here (others -- bivariate Poisson, negative binomial,
CMP -- are scored in their own scripts/*_experiment.py, same held-out
matches/methodology, results merged in the final report):
  existing        - production LeagueModel + Dixon-Coles tau (the baseline
                     to beat)
  no_rho          - same lambdas, tau correction removed (equivalent to a
                     simple independent-Poisson product on the SAME
                     strengths -- this is also Phase 3's "simple independent
                     Poisson" baseline; they are mathematically identical
                     under this codebase's grid construction, noted in the
                     report rather than implemented twice)
  always_1_1      - baseline
  most_common     - baseline (most common training-window score)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues.model import LeagueModel, scoreline_grid
from scripts.correct_score_common import fit_production_model  # noqa: F401 (reused shape reference)
from leagues import dataset

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def per_match_grid_probs(rows, grid_fn) -> np.ndarray:
    eps = 1e-12
    out = []
    for r in rows:
        grid = grid_fn(r)
        hg, ag = r["hg"], r["ag"]
        p = float(grid[hg, ag]) if hg < grid.shape[0] and ag < grid.shape[1] else 0.0
        out.append(max(p, eps))
    return np.array(out)


def top1_score(grid) -> tuple[int, int]:
    h, a = np.unravel_index(np.argmax(grid), grid.shape)
    return int(h), int(a)


def top3_scores(grid) -> set[tuple[int, int]]:
    flat = grid.ravel()
    idx = np.argsort(-flat)[:3]
    return {tuple(int(x) for x in np.unravel_index(k, grid.shape)) for k in idx}


def paired_ci(a: np.ndarray, b: np.ndarray, iters: int = 2000, seed: int = 7):
    """95% CI on paired mean(a - b) via bootstrap."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


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
        tc = (train["home_goals"].astype(int).astype(str) + "-"
              + train["away_goals"].astype(int).astype(str)).value_counts()
        common = tc.index[0] if len(tc) else "1-1"
        common_h, common_a = (int(x) for x in common.split("-"))

        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            rows.append({
                "date": m["date"], "league": league,
                "lh": lh, "la": la, "rho": model.rho,
                "hg": int(m["home_goals"]), "ag": int(m["away_goals"]),
                "common_h": common_h, "common_a": common_a,
            })
    return rows


def grid_existing(r):
    return scoreline_grid(r["lh"], r["la"], r["rho"])


def grid_no_rho(r):
    return scoreline_grid(r["lh"], r["la"], 0.0)


def grid_always_11(r):
    g = np.zeros((11, 11)); g[1, 1] = 1.0
    return g


def grid_most_common(r):
    g = np.zeros((11, 11))
    h, a = min(r["common_h"], 10), min(r["common_a"], 10)
    g[h, a] = 1.0
    return g


CANDIDATES = {
    "existing": grid_existing,
    "no_rho_indep_poisson": grid_no_rho,
    "always_1_1": grid_always_11,
    "most_common": grid_most_common,
}


def score_candidates(rows: list[dict]) -> dict:
    out = {}
    p_by_cand = {}
    for name, fn in CANDIDATES.items():
        p = per_match_grid_probs(rows, fn)
        p_by_cand[name] = p
        logloss = float(-np.log(p).mean())

        hits1 = np.array([top1_score(fn(r)) == (r["hg"], r["ag"]) for r in rows])
        hits3 = np.array([(r["hg"], r["ag"]) in top3_scores(fn(r)) for r in rows])
        pred_11 = np.array([top1_score(fn(r)) == (1, 1) for r in rows])
        actual_11 = np.array([(r["hg"], r["ag"]) == (1, 1) for r in rows])

        out[name] = {
            "n": len(rows),
            "grid_logloss_nats": round(logloss, 4),
            "top1_pct": round(100 * float(hits1.mean()), 2),
            "top3_pct": round(100 * float(hits3.mean()), 2),
            "pred_1_1_pct": round(100 * float(pred_11.mean()), 2),
            "actual_1_1_pct": round(100 * float(actual_11.mean()), 2),
        }

    base = p_by_cand["existing"]
    for name, p in p_by_cand.items():
        if name == "existing":
            continue
        # existing_loss - candidate_loss: positive means candidate is BETTER
        lo, hi = paired_ci(-np.log(base), -np.log(p))
        out[name]["logloss_delta_vs_existing_95ci"] = [round(lo, 4), round(hi, 4)]
        out[name]["existing_beats_candidate_significantly"] = bool(hi < 0)
        out[name]["candidate_beats_existing_significantly"] = bool(lo > 0)

    # Per-scoreline calibration for a handful of common scores, existing model only
    calib = {}
    for score in ["0-0", "1-0", "0-1", "1-1", "2-1", "1-2"]:
        h, a = (int(x) for x in score.split("-"))
        preds = np.array([scoreline_grid(r["lh"], r["la"], r["rho"])[h, a] for r in rows])
        actual = np.array([(r["hg"] == h and r["ag"] == a) for r in rows], dtype=float)
        calib[score] = {
            "mean_predicted_pct": round(100 * float(preds.mean()), 2),
            "actual_pct": round(100 * float(actual.mean()), 2),
        }
    out["_calibration_existing"] = calib

    # Total-goals calibration + tail (4+ goals)
    total_actual = np.array([r["hg"] + r["ag"] for r in rows])
    total_pred_mean = np.array([r["lh"] + r["la"] for r in rows])
    tail_actual = float((total_actual >= 4).mean())
    grids = [scoreline_grid(r["lh"], r["la"], r["rho"]) for r in rows]
    rr, cc = np.indices(grids[0].shape)
    tail_pred = float(np.mean([g[(rr + cc) >= 4].sum() for g in grids]))
    out["_totals"] = {
        "mean_actual_total_goals": round(float(total_actual.mean()), 3),
        "mean_predicted_total_goals": round(float(total_pred_mean.mean()), 3),
        "actual_pct_4plus_goals": round(100 * tail_actual, 2),
        "predicted_pct_4plus_goals": round(100 * tail_pred, 2),
    }

    # RPS for 1X2 (existing model), for reference against the 1X2 gate already
    # tracked elsewhere (leagues/tune.py) -- included here so this report is
    # self-contained.
    def outcome_idx(hg, ag):
        return 0 if hg > ag else (1 if hg == ag else 2)
    def rps_1x2(r):
        g = scoreline_grid(r["lh"], r["la"], r["rho"])
        ph = float(np.tril(g, -1).sum()); pdw = float(np.trace(g)); pa = float(np.triu(g, 1).sum())
        probs = np.array([ph, pdw, pa])
        obs = np.zeros(3); obs[outcome_idx(r["hg"], r["ag"])] = 1.0
        cp, co = np.cumsum(probs), np.cumsum(obs)
        return float(((cp - co) ** 2)[:2].sum() / 2.0)
    out["_rps_1x2_existing"] = round(float(np.mean([rps_1x2(r) for r in rows])), 4)

    return out


def run():
    report = {}
    all_rows = []
    for league in LEAGUES:
        print(f"--- {league} ---")
        rows = run_league(league)
        all_rows.extend(rows)
        report[league] = score_candidates(rows)
        print(f"  n={len(rows)}  existing logloss={report[league]['existing']['grid_logloss_nats']}  "
              f"pred_1-1={report[league]['existing']['pred_1_1_pct']}%  "
              f"actual_1-1={report[league]['existing']['actual_1_1_pct']}%")

    print("--- COMBINED ---")
    report["COMBINED"] = score_candidates(all_rows)
    print(f"  n={len(all_rows)}  existing logloss={report['COMBINED']['existing']['grid_logloss_nats']}")
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "correct_score_benchmark_phase2.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
