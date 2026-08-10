"""Does a genuine BIVARIATE Poisson correlation layer beat Dixon-Coles' tau patch?

Context (see leagues/model.py's dc_tau/scoreline_grid): production builds the
scoreline grid as an outer product of two independent Poisson marginals, with a
4-cell ad-hoc multiplicative correction (dc_tau) that only touches (0,0),(0,1),
(1,0),(1,1). Root-cause analysis already found 1-1 being the most likely single
scoreline is a genuine thin-margin property of low-scoring discrete distributions,
not a bug -- but the question here is whether a DIFFERENT, principled correlation
structure across the WHOLE grid (not just 4 cells) redistributes probability mass
in a way that measurably improves exact-score log-loss.

Model: Karlis & Ntzoufras shared-shock / trivariate reduction. X = W1 + W3,
Y = W2 + W3, W1 ~ Poisson(l1), W2 ~ Poisson(l2), W3 ~ Poisson(l3), l3 is the
shared component that induces POSITIVE correlation between home and away goals.

lambda1/lambda2 vs lh/la -- a deliberate reading of the brief: the task said to
fit lambda3 "given lambda1=lh, lambda2=la", but literally setting lambda1=lh,
lambda2=la would shift the marginal means to lh+lambda3 / la+lambda3, which
violates the stated goal of this whole experiment -- holding the attack/defence/
xG strength estimation constant so we're testing ONLY the joint-distribution
shape. The standard, mean-preserving way to bolt a shared-shock term onto
externally-given marginal means is l1 = lh - lambda3, l2 = la - lambda3 (so
E[X] = l1 + lambda3 = lh, E[Y] = la, exactly as production's marginals). That is
what's implemented here; noted so the choice is auditable, not silently assumed.

lambda3 is fit by 1-D MLE (scipy.optimize.minimize_scalar, bounded) maximizing
weighted log-likelihood of the ACTUAL (home_goals, away_goals) pairs on the
TRAINING window only, using that window's own fitted lh_i/la_i per historical
match (from the same production LeagueModel -- attack/defence/xG is never
re-fit here). l1_i/l2_i are clipped to >=1e-6 per match rather than constraining
the global optimization to a feasible range for every training match
simultaneously: with the model's own 0.05 lambda floor, a global feasibility
bound would collapse lambda3 to near-zero every window (dominated by whichever
match had the single weakest team), which would make the correlation layer
structurally unable to do anything. Per-match clipping lets lambda3 reflect the
BULK of the training data's correlation while degrading gracefully (silently
back toward the independent-Poisson case) for the handful of extreme mismatches
where lh_i or la_i < lambda3. This is a documented pragmatic choice, not a hidden
one.

Walk-forward, strictly causal, same discipline as correct_score_backtest.py and
weibull_experiment.py: min_train=760, step_days=7, xi=0.003, xg_weight=0.75.
Judged on full-grid log-loss with a PAIRED bootstrap 95% CI against the existing
production Dixon-Coles grid (scoreline_grid with dc_tau), because exact-hit rate
is saturated at this sample size (correct_score_backtest.py's own finding).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid, MAX_GOALS

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]
EPS = 1e-12
LAMBDA3_UPPER_BOUND = 2.5  # generous prior cap; per-match clipping does the real work


def batch_lambdas(model: LeagueModel, df: pd.DataFrame):
    """Vectorized equivalent of LeagueModel.lambdas() over a whole frame.

    Mirrors model.lambdas() exactly (same clip bounds); avoids a Python-level
    loop + per-row KeyError catch across thousands of historical rows per
    walk-forward window. Returns (ok_mask, lh, la) restricted to rows where
    both teams are in the fitted model (should be ~all rows, since the model
    was fit on this same df -- kept as a guard, not expected to trigger).
    """
    home_att = df["home"].map(model.attack)
    away_def = df["away"].map(model.defence)
    away_att = df["away"].map(model.attack)
    home_def = df["home"].map(model.defence)
    ok = (home_att.notna() & away_def.notna() & away_att.notna() & home_def.notna()).to_numpy()
    lh = np.exp((home_att + away_def + model.home_adv).to_numpy()[ok])
    la = np.exp((away_att + home_def).to_numpy()[ok])
    return ok, np.clip(lh, 0.05, 6.0), np.clip(la, 0.05, 6.0)


def bipois_logpmf(x, y, l1, l2, l3, max_k: int = 10):
    """Log P(X=x, Y=y) under the Karlis-Ntzoufras shared-shock bivariate Poisson.

    P(x,y) = exp(-(l1+l2+l3)) * sum_{k=0}^{min(x,y)} l1^(x-k)/(x-k)! *
             l2^(y-k)/(y-k)! * l3^k/k!

    Vectorized over match arrays x, y, l1, l2 (all broadcastable to the same
    shape); l3 may be scalar (a single window-level fitted parameter) or an
    array. Computed in log-space with a logsumexp over k for numerical
    stability (l3 can be tiny, driving some terms to extreme exponents).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    l1 = np.broadcast_to(np.atleast_1d(np.asarray(l1, dtype=float)), x.shape)
    l2 = np.broadcast_to(np.atleast_1d(np.asarray(l2, dtype=float)), x.shape)
    l3 = np.broadcast_to(np.atleast_1d(np.asarray(l3, dtype=float)), x.shape)
    l1 = np.clip(l1, 1e-12, None)
    l2 = np.clip(l2, 1e-12, None)
    l3 = np.clip(l3, 1e-12, None)

    ks = np.arange(max_k + 1, dtype=float)                      # (K,)
    minxy = np.minimum(x, y)                                     # (N,)
    xk = x[:, None] - ks[None, :]                                # (N,K)
    yk = y[:, None] - ks[None, :]
    valid = (ks[None, :] <= minxy[:, None]) & (xk >= 0) & (yk >= 0)

    log_l1 = np.log(l1)[:, None]
    log_l2 = np.log(l2)[:, None]
    log_l3 = np.log(l3)[:, None]
    logterm = np.where(
        valid,
        xk * log_l1 - gammaln(xk + 1) + yk * log_l2 - gammaln(yk + 1)
        + ks[None, :] * log_l3 - gammaln(ks[None, :] + 1),
        -np.inf,
    )
    m = np.max(logterm, axis=1)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(logterm - m_safe[:, None]), axis=1)
    logsum = m_safe + np.log(np.clip(s, 1e-300, None))
    return -(l1 + l2 + l3) + logsum


def fit_lambda3(lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray,
                 w: np.ndarray) -> float:
    """1-D MLE for the shared-shock parameter, weighted by the same decay
    weights the production model itself trains with."""
    def neg_ll(lam3):
        l1 = np.clip(lh - lam3, 1e-6, None)
        l2 = np.clip(la - lam3, 1e-6, None)
        logp = bipois_logpmf(hg, ag, l1, l2, lam3)
        return -float(np.sum(w * logp))

    res = minimize_scalar(neg_ll, bounds=(0.0, LAMBDA3_UPPER_BOUND), method="bounded",
                           options={"xatol": 1e-4})
    return float(res.x)


def bipois_grid(lh: float, la: float, lambda3: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    l1 = max(lh - lambda3, 1e-6)
    l2 = max(la - lambda3, 1e-6)
    xs = np.arange(max_goals + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    logp = bipois_logpmf(X.ravel(), Y.ravel(), l1, l2, lambda3)
    grid = np.exp(logp).reshape(X.shape)
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    return grid / total if total > 0 else grid


def paired_logloss_ci(loss_a: np.ndarray, loss_b: np.ndarray, iters: int = 3000, seed: int = 7):
    """Bootstrap 95% CI on the PAIRED per-match log-loss difference (a - b).
    Positive => b has lower loss (b better). Same discipline as
    ou_market_experiment.py's paired_logloss_ci -- a point estimate alone
    cannot say whether an edge is real or noise at this sample size."""
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
    lambda3_by_window = []
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
        from leagues.weights import decay_weights
        w_tr = decay_weights(train["date"], ref=cutoff, xi=xi).to_numpy()[ok]
        hg_tr = train["home_goals"].astype(int).to_numpy()[ok]
        ag_tr = train["away_goals"].astype(int).to_numpy()[ok]
        lambda3 = fit_lambda3(lh_tr, la_tr, hg_tr, ag_tr, w_tr)
        lambda3_by_window.append(lambda3)

        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            grid_prod = scoreline_grid(lh, la, model.rho)
            grid_cand = bipois_grid(lh, la, lambda3)

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
        "lambda3_fitted": {
            "mean": round(float(np.mean(lambda3_by_window)), 4),
            "median": round(float(np.median(lambda3_by_window)), 4),
            "min": round(float(np.min(lambda3_by_window)), 4),
            "max": round(float(np.max(lambda3_by_window)), 4),
            "n_windows": len(lambda3_by_window),
            "n_windows_at_upper_bound": int(sum(1 for v in lambda3_by_window
                                                 if v > LAMBDA3_UPPER_BOUND - 1e-3)),
        },
        "grid_logloss_nats": {
            "production_dc": round(float(loss_prod.mean()), 4),
            "bivariate_poisson": round(float(loss_cand.mean()), 4),
        },
        "logloss_delta_prod_minus_cand": round(float(delta.mean()), 4),
        "logloss_delta_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "candidate_significantly_better": bool(ci_lo > 0),
        "candidate_significantly_worse": bool(ci_hi < 0),
        "top1_pct": {
            "production_dc": round(100 * r["top1_prod"].mean(), 2),
            "bivariate_poisson": round(100 * r["top1_cand"].mean(), 2),
        },
        "top3_pct": {
            "production_dc": round(100 * r["top3_prod"].mean(), 2),
            "bivariate_poisson": round(100 * r["top3_cand"].mean(), 2),
        },
        "p_1_1": {
            "actual_frequency_pct": round(100 * r["is_11"].mean(), 2),
            "mean_predicted_production_pct": round(100 * r["p11_prod"].mean(), 2),
            "mean_predicted_bivariate_pct": round(100 * r["p11_cand"].mean(), 2),
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
            "bivariate_poisson": round(float(loss_cand.mean()), 4),
        },
        "logloss_delta_prod_minus_cand": round(float(delta.mean()), 4),
        "logloss_delta_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "candidate_significantly_better": bool(ci_lo > 0),
        "candidate_significantly_worse": bool(ci_hi < 0),
        "top1_pct": {
            "production_dc": round(100 * pooled["top1_prod"].mean(), 2),
            "bivariate_poisson": round(100 * pooled["top1_cand"].mean(), 2),
        },
        "top3_pct": {
            "production_dc": round(100 * pooled["top3_prod"].mean(), 2),
            "bivariate_poisson": round(100 * pooled["top3_cand"].mean(), 2),
        },
        "p_1_1": {
            "actual_frequency_pct": round(100 * pooled["is_11"].mean(), 2),
            "mean_predicted_production_pct": round(100 * pooled["p11_prod"].mean(), 2),
            "mean_predicted_bivariate_pct": round(100 * pooled["p11_cand"].mean(), 2),
        },
        "verdict": (
            "bivariate Poisson gives a STATISTICALLY SIGNIFICANT log-loss edge over "
            "production Dixon-Coles, pooled across leagues"
            if ci_lo > 0 else
            "bivariate Poisson is STATISTICALLY SIGNIFICANTLY WORSE than production, pooled"
            if ci_hi < 0 else
            "point estimate is inconclusive either way -- the 95% CI crosses zero; "
            "not distinguishable from noise on this sample, do not ship on this alone"
        ),
    }
    print("\n=== COMBINED (pooled across all 4 leagues) ===")
    print(json.dumps(combined, indent=2))

    result = {"per_league": per_league, "combined": combined}
    path = ROOT / "data-raw" / "leagues" / "bivariate_poisson_experiment.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
