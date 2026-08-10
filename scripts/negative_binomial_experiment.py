"""Do overdispersed Negative Binomial marginals beat Dixon-Coles on scoreline log-loss?

Companion to bivariate_poisson_experiment.py -- see that file's docstring for the
shared context (root-cause finding on 1-1, why log-loss over the full grid is the
metric, why exact-hit rate is saturated). This script tests a different
hypothesis: maybe goals are simply overdispersed relative to Poisson (variance >
mean), which a plain Poisson marginal can't express regardless of any
home/away correlation structure.

Model: replace each Poisson(lh)/Poisson(la) marginal with NegativeBinomial(mean=lh,
dispersion=alpha) / NegativeBinomial(mean=la, dispersion=alpha), NB2
parameterization: Var = mu + alpha*mu^2 (alpha -> 0 recovers Poisson exactly).
lh, la come from the SAME production LeagueModel (attack/defence/xG never
re-fit here) -- only a dispersion layer is added on top.

alpha is fit ONE shared value per league-window (not separate home/away) via 1-D
MLE, weighted log-likelihood of observed goal counts against NB(mean=lh_i or la_i,
alpha) on the TRAINING window, same decay weights the production model itself
uses. A single shared alpha is the simplest, cleanest read of "an extra dispersion
parameter" from the brief, and keeps this a fair one-parameter-vs-DC's-one-
parameter (rho) comparison; a home/away-split alpha was considered and rejected
for that reason -- see the report for the option not tried.

PRIMARY combination is the plain independent product of the two NB marginals (no
Dixon-Coles-style low-score tau on top) -- the brief is explicit that the primary
version should be a clean, uncorrected comparison so any effect is attributable to
the marginal shape alone, not a second correlation patch layered on top.

Walk-forward, strictly causal: min_train=760, step_days=7, xi=0.003,
xg_weight=0.75, judged on full-grid log-loss with a paired bootstrap 95% CI
against production's scoreline_grid (Poisson marginals + dc_tau).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid, MAX_GOALS
from leagues.weights import decay_weights

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]
EPS = 1e-12
ALPHA_BOUNDS = (1e-6, 5.0)  # alpha -> 0 is Poisson; wide enough for real overdispersion


def batch_lambdas(model: LeagueModel, df: pd.DataFrame):
    """Vectorized equivalent of LeagueModel.lambdas() over a whole frame
    (same clip bounds; see bivariate_poisson_experiment.py's twin helper)."""
    home_att = df["home"].map(model.attack)
    away_def = df["away"].map(model.defence)
    away_att = df["away"].map(model.attack)
    home_def = df["home"].map(model.defence)
    ok = (home_att.notna() & away_def.notna() & away_att.notna() & home_def.notna()).to_numpy()
    lh = np.exp((home_att + away_def + model.home_adv).to_numpy()[ok])
    la = np.exp((away_att + home_def).to_numpy()[ok])
    return ok, np.clip(lh, 0.05, 6.0), np.clip(la, 0.05, 6.0)


def nb_logpmf(k, mu, alpha):
    """NB2 parameterization: mean=mu, Var=mu + alpha*mu^2. r=1/alpha,
    p=r/(r+mu). alpha -> 0 recovers Poisson(mu) in the limit."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    alpha = max(float(alpha), 1e-9)
    r = 1.0 / alpha
    p = r / (r + mu)
    return nbinom.logpmf(k, r, p)


def fit_alpha(lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray,
              w: np.ndarray) -> float:
    """1-D MLE for a single shared dispersion parameter across both marginals,
    weighted by the production model's own decay weights."""
    def neg_ll(alpha):
        ll_h = nb_logpmf(hg, lh, alpha)
        ll_a = nb_logpmf(ag, la, alpha)
        return -float(np.sum(w * (ll_h + ll_a)))

    res = minimize_scalar(neg_ll, bounds=ALPHA_BOUNDS, method="bounded",
                           options={"xatol": 1e-6})
    return float(res.x)


def nb_grid(lh: float, la: float, alpha: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Independent product of two NB marginals -- the PRIMARY, uncorrected
    comparison (see module docstring)."""
    xs = np.arange(max_goals + 1)
    hp = np.exp(nb_logpmf(xs, lh, alpha))
    ap = np.exp(nb_logpmf(xs, la, alpha))
    grid = np.outer(hp, ap)
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    return grid / total if total > 0 else grid


def paired_logloss_ci(loss_a: np.ndarray, loss_b: np.ndarray, iters: int = 3000, seed: int = 7):
    """Bootstrap 95% CI on the paired per-match log-loss diff (a - b);
    positive => b has lower loss (b better)."""
    rng = np.random.default_rng(seed)
    d = loss_a - loss_b
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run(league="PL", xi=0.003, xg_weight=0.75, min_train=760, step_days=7):
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    if len(df) <= min_train:
        raise SystemExit(f"{league}: not enough matches ({len(df)}) for min_train={min_train}")

    rows = []
    alpha_by_window = []
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

        ok, lh_tr, la_tr = batch_lambdas(model, train)
        if ok.sum() < 20:
            continue
        w_tr = decay_weights(train["date"], ref=cutoff, xi=xi).to_numpy()[ok]
        hg_tr = train["home_goals"].astype(int).to_numpy()[ok]
        ag_tr = train["away_goals"].astype(int).to_numpy()[ok]
        alpha = fit_alpha(lh_tr, la_tr, hg_tr, ag_tr, w_tr)
        alpha_by_window.append(alpha)

        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            grid_prod = scoreline_grid(lh, la, model.rho)
            grid_cand = nb_grid(lh, la, alpha)

            p_prod = float(grid_prod[hg, ag]) if hg < grid_prod.shape[0] and ag < grid_prod.shape[1] else 0.0
            p_cand = float(grid_cand[hg, ag]) if hg < grid_cand.shape[0] and ag < grid_cand.shape[1] else 0.0

            def top_n_hit(grid, n):
                flat = grid.ravel()
                idx = np.argsort(-flat)[:n]
                scores = {tuple(np.unravel_index(k, grid.shape)) for k in idx}
                return (hg, ag) in scores

            rows.append({
                "league": league, "actual": f"{hg}-{ag}",
                "p_actual_prod": max(p_prod, EPS), "p_actual_cand": max(p_cand, EPS),
                "top1_prod": top_n_hit(grid_prod, 1), "top1_cand": top_n_hit(grid_cand, 1),
                "top3_prod": top_n_hit(grid_prod, 3), "top3_cand": top_n_hit(grid_cand, 3),
                "p11_prod": float(grid_prod[1, 1]) if grid_prod.shape[0] > 1 else 0.0,
                "p11_cand": float(grid_cand[1, 1]) if grid_cand.shape[0] > 1 else 0.0,
                "is_11": (hg == 1 and ag == 1),
            })

    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit(f"{league}: nothing scored")

    loss_prod = -np.log(r["p_actual_prod"].to_numpy())
    loss_cand = -np.log(r["p_actual_cand"].to_numpy())
    delta = loss_prod - loss_cand  # positive => candidate better
    ci_lo, ci_hi = paired_logloss_ci(loss_prod, loss_cand)

    out = {
        "league": league,
        "n": int(len(r)),
        "alpha_fitted": {
            "mean": round(float(np.mean(alpha_by_window)), 5),
            "median": round(float(np.median(alpha_by_window)), 5),
            "min": round(float(np.min(alpha_by_window)), 5),
            "max": round(float(np.max(alpha_by_window)), 5),
            "n_windows": len(alpha_by_window),
            "n_windows_near_poisson_floor": int(sum(1 for v in alpha_by_window if v < 1e-4)),
        },
        "grid_logloss_nats": {
            "production_dc": round(float(loss_prod.mean()), 4),
            "negative_binomial": round(float(loss_cand.mean()), 4),
        },
        "logloss_delta_prod_minus_cand": round(float(delta.mean()), 4),
        "logloss_delta_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "candidate_significantly_better": bool(ci_lo > 0),
        "candidate_significantly_worse": bool(ci_hi < 0),
        "top1_pct": {
            "production_dc": round(100 * r["top1_prod"].mean(), 2),
            "negative_binomial": round(100 * r["top1_cand"].mean(), 2),
        },
        "top3_pct": {
            "production_dc": round(100 * r["top3_prod"].mean(), 2),
            "negative_binomial": round(100 * r["top3_cand"].mean(), 2),
        },
        "p_1_1": {
            "actual_frequency_pct": round(100 * r["is_11"].mean(), 2),
            "mean_predicted_production_pct": round(100 * r["p11_prod"].mean(), 2),
            "mean_predicted_nb_pct": round(100 * r["p11_cand"].mean(), 2),
        },
    }
    return out, r


def main():
    per_league = {}
    all_rows = []
    for league in LEAGUES:
        print(f"--- {league} ---", flush=True)
        rep, r = run(league)
        per_league[league] = rep
        all_rows.append(r)
        print(json.dumps(rep, indent=2))

    pooled = pd.concat(all_rows, ignore_index=True)
    loss_prod = -np.log(pooled["p_actual_prod"].to_numpy())
    loss_cand = -np.log(pooled["p_actual_cand"].to_numpy())
    delta = loss_prod - loss_cand
    ci_lo, ci_hi = paired_logloss_ci(loss_prod, loss_cand)

    combined = {
        "n": int(len(pooled)),
        "grid_logloss_nats": {
            "production_dc": round(float(loss_prod.mean()), 4),
            "negative_binomial": round(float(loss_cand.mean()), 4),
        },
        "logloss_delta_prod_minus_cand": round(float(delta.mean()), 4),
        "logloss_delta_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "candidate_significantly_better": bool(ci_lo > 0),
        "candidate_significantly_worse": bool(ci_hi < 0),
        "top1_pct": {
            "production_dc": round(100 * pooled["top1_prod"].mean(), 2),
            "negative_binomial": round(100 * pooled["top1_cand"].mean(), 2),
        },
        "top3_pct": {
            "production_dc": round(100 * pooled["top3_prod"].mean(), 2),
            "negative_binomial": round(100 * pooled["top3_cand"].mean(), 2),
        },
        "p_1_1": {
            "actual_frequency_pct": round(100 * pooled["is_11"].mean(), 2),
            "mean_predicted_production_pct": round(100 * pooled["p11_prod"].mean(), 2),
            "mean_predicted_nb_pct": round(100 * pooled["p11_cand"].mean(), 2),
        },
        "verdict": (
            "negative binomial marginals give a STATISTICALLY SIGNIFICANT log-loss "
            "edge over production Dixon-Coles, pooled across leagues"
            if ci_lo > 0 else
            "negative binomial marginals are STATISTICALLY SIGNIFICANTLY WORSE than "
            "production, pooled"
            if ci_hi < 0 else
            "point estimate is inconclusive either way -- the 95% CI crosses zero; "
            "not distinguishable from noise on this sample, do not ship on this alone"
        ),
    }
    print("\n=== COMBINED (pooled across all 4 leagues) ===")
    print(json.dumps(combined, indent=2))

    result = {"per_league": per_league, "combined": combined}
    path = ROOT / "data-raw" / "leagues" / "negative_binomial_experiment.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
