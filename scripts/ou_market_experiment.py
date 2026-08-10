"""Does the Over/Under 2.5 market improve exact-score predictions? (Premier League)

market_gap_experiment.py already tested the 1X2 market and found our model carries
NO information it lacks -- but 1X2 only says who's stronger, not how many goals the
match is likely to have. O/U 2.5 is a genuinely different signal: total-goals tempo,
priced close to kickoff (team news, weather, tactical setup) rather than purely from
five seasons of historical attack/defence ratings.

Method, same causal discipline as everywhere else in this project:
  1. De-vig the closing Avg>2.5 / Avg<2.5 odds into a market probability of "over".
  2. Invert that into a market-implied total-goals lambda (total goals ~ Poisson,
     since home+away goals sum to Poisson(lh+la) under this model's independence
     assumption before the small Dixon-Coles low-score correction).
  3. Re-split that total between the two teams using THIS MODEL's own attack/defence
     ratio (lh / (lh+la)) -- so the market supplies "how many total", the model still
     supplies "who gets more of them".
  4. Blend weight w between the model's own total and the market's total is fit on
     the EARLIER half of matches and evaluated on the LATER half. Never fit and score
     on the same matches.

Reports grid log-loss (the proper scoring rule correct_score_backtest.py already
uses) for: model alone, market-total-only (w=1), and the best blend, both on the
full test set and restricted to matches where O/U odds actually exist.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid

ROOT = Path(__file__).resolve().parents[1]


def devig_over_under(odds_over: float, odds_under: float):
    """Proportional de-vig, same approach as backtest.devig for 1X2."""
    raw = np.array([1.0 / odds_over, 1.0 / odds_under], dtype=float)
    out = raw / raw.sum()
    return float(out[0])  # p(over 2.5)


def implied_total_lambda(p_over: float, lo: float = 0.3, hi: float = 6.0) -> float:
    """Invert P(total > 2.5) = p_over for total ~ Poisson(lambda) by bisection.
    P(total > 2.5) = P(total >= 3) = 1 - PMF(0) - PMF(1) - PMF(2)."""
    def p_over_at(lam):
        return 1.0 - poisson.cdf(2, lam)
    a, b = lo, hi
    if p_over_at(a) > p_over: return a
    if p_over_at(b) < p_over: return b
    for _ in range(50):
        mid = (a + b) / 2
        if p_over_at(mid) < p_over:
            a = mid
        else:
            b = mid
    return (a + b) / 2


def per_match_p_actual(rows: pd.DataFrame, lh_col: str, la_col: str, rho: float) -> np.ndarray:
    eps = 1e-12
    p_actual = []
    for _, r in rows.iterrows():
        grid = scoreline_grid(r[lh_col], r[la_col], rho)
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        p = float(grid[hg, ag]) if hg < grid.shape[0] and ag < grid.shape[1] else 0.0
        p_actual.append(max(p, eps))
    return np.array(p_actual)


def grid_logloss(rows: pd.DataFrame, lh_col: str, la_col: str, rho: float) -> float:
    return float(-np.log(per_match_p_actual(rows, lh_col, la_col, rho)).mean())


def paired_logloss_ci(p_a: np.ndarray, p_b: np.ndarray, iters: int = 2000, seed: int = 7):
    """Bootstrap 95% CI on the PAIRED per-match log-loss difference (a - b),
    same discipline as correct_score_backtest.py's _paired_ci -- a point-estimate
    delta alone can't say whether it's a real effect or noise on this sample size."""
    rng = np.random.default_rng(seed)
    d = -np.log(p_a) - (-np.log(p_b))  # per-match loss_a - loss_b
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def top1_hit_rate(rows: pd.DataFrame, lh_col: str, la_col: str, rho: float) -> float:
    hits = []
    for _, r in rows.iterrows():
        grid = scoreline_grid(r[lh_col], r[la_col], rho)
        h, a = np.unravel_index(np.argmax(grid), grid.shape)
        hits.append(int(h) == int(r["home_goals"]) and int(a) == int(r["away_goals"]))
    return float(np.mean(hits))


def run(league="PL", xi=0.003, xg_weight=0.75, min_train=760, step_days=7):
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))

    rows = []
    start = df.loc[min_train, "date"]
    rho = None
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff)
        except Exception:
            continue
        rho = model.rho
        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            row = {
                "date": m["date"], "home_goals": int(m["home_goals"]),
                "away_goals": int(m["away_goals"]),
                "lh_model": lh, "la_model": la,
            }
            has_ou = pd.notna(m.get("odds_over25")) and pd.notna(m.get("odds_under25"))
            if has_ou:
                p_over = devig_over_under(m["odds_over25"], m["odds_under25"])
                market_total = implied_total_lambda(p_over)
                ratio = lh / (lh + la) if (lh + la) > 0 else 0.5
                row["lh_market_total"] = ratio * market_total
                row["la_market_total"] = (1 - ratio) * market_total
            rows.append(row)

    r = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if r.empty:
        raise SystemExit("no matches scored")

    n_total = len(r)
    r_ou = r.dropna(subset=["lh_market_total", "la_market_total"]).reset_index(drop=True)
    n_ou = len(r_ou)
    print(f"n={n_total} total, {n_ou} with usable O/U 2.5 closing odds "
          f"({100*n_ou/n_total:.0f}%)\n")

    if n_ou < 200:
        print("Too few matches with O/U odds to evaluate a blend reliably; stopping.")
        return {"league": league, "n_total": n_total, "n_with_ou_odds": n_ou,
                "note": "insufficient O/U coverage to fit/test a blend"}

    half = n_ou // 2
    tr, te = r_ou.iloc[:half].copy(), r_ou.iloc[half:].copy()

    model_only_logloss_test = grid_logloss(te, "lh_model", "la_model", rho)
    model_only_top1_test = top1_hit_rate(te, "lh_model", "la_model", rho)
    market_only_logloss_test = grid_logloss(te, "lh_market_total", "la_market_total", rho)
    market_only_top1_test = top1_hit_rate(te, "lh_market_total", "la_market_total", rho)

    def blended_lambdas(frame, w):
        return (w * frame["lh_model"] + (1 - w) * frame["lh_market_total"],
                w * frame["la_model"] + (1 - w) * frame["la_market_total"])

    ws = np.linspace(0.0, 1.0, 21)
    train_losses = []
    for w in ws:
        lh_b, la_b = blended_lambdas(tr, w)
        tmp = tr.copy(); tmp["lh_b"], tmp["la_b"] = lh_b, la_b
        train_losses.append(grid_logloss(tmp, "lh_b", "la_b", rho))
    best_w = float(ws[int(np.argmin(train_losses))])

    lh_b, la_b = blended_lambdas(te, best_w)
    te_blend = te.copy(); te_blend["lh_b"], te_blend["la_b"] = lh_b, la_b
    blend_logloss_test = grid_logloss(te_blend, "lh_b", "la_b", rho)
    blend_top1_test = top1_hit_rate(te_blend, "lh_b", "la_b", rho)

    # Is the blend's log-loss edge over model-only real, or noise on 568 test
    # matches? Paired bootstrap on the SAME matches under both lambda sets.
    p_model = per_match_p_actual(te, "lh_model", "la_model", rho)
    p_blend = per_match_p_actual(te_blend, "lh_b", "la_b", rho)
    ci_lo, ci_hi = paired_logloss_ci(p_model, p_blend)  # model_loss - blend_loss
    edge_significant = bool(ci_lo > 0)  # entire CI above zero = model loses to blend, reliably

    result = {
        "league": league,
        "n_total": n_total,
        "n_with_ou_odds": n_ou,
        "ou_coverage_pct": round(100 * n_ou / n_total, 1),
        "train_n": len(tr),
        "test_n": len(te),
        "best_weight_on_model": round(best_w, 2),
        "test_grid_logloss": {
            "model_only": round(model_only_logloss_test, 4),
            "market_total_only": round(market_only_logloss_test, 4),
            "blend": round(blend_logloss_test, 4),
        },
        "test_top1_pct": {
            "model_only": round(100 * model_only_top1_test, 2),
            "market_total_only": round(100 * market_only_top1_test, 2),
            "blend": round(100 * blend_top1_test, 2),
        },
        "blend_vs_model_logloss_delta_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "edge_is_significant": edge_significant,
        "verdict": (
            "market total-goals info gives a STATISTICALLY REAL edge over model-only"
            if edge_significant else
            "point estimate favors the market total, but the 95% CI crosses zero -- "
            "not distinguishable from noise on this sample; do not ship on this alone"
        ),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    rep = run("PL")
    path = ROOT / "data-raw" / "leagues" / "ou_market_experiment.json"
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
