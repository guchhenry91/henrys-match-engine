"""Does WeibullCopulaGoalsModel beat Dixon-Coles on SCORELINE quality?

Research said this is the one model pointing the right way on both known
properties of football goals: Weibull counts handle conditional UNDERdispersion
(shape > 1) and the copula captures NEGATIVE goal dependence with kappa
unrestricted in sign. Dixon-Coles only patches dependence crudely in four
low-score cells; bivariate Poisson can only express POSITIVE dependence; negative
binomial assumes overdispersion. So Weibull is the only credible challenger.

It is NOT a drop-in for our pipeline -- it exposes `defense_` (American) where DC
exposes `defence_`, and `shape`/`kappa` instead of `rho`, so our grid
reconstruction does not apply. We therefore compare each model on ITS OWN grid,
plus our full pipeline, so model-vs-model is separated from what xG blending buys:

  A  full pipeline : DC + xG-blended strengths + our DC-tau grid   (what ships)
  B  DC raw        : penaltyblog DixonColes .predict() grid        (control)
  C  Weibull raw   : penaltyblog WeibullCopula .predict() grid     (challenger)

Judged on grid log-loss (a local proper scoring rule) with a PAIRED bootstrap CI,
because exact-hit rate is saturated and cannot referee this.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import penaltyblog as pb

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid
from leagues.weights import decay_weights

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def _ll(grid, hg, ag):
    g = np.asarray(grid)
    p = g[hg, ag] if hg < g.shape[0] and ag < g.shape[1] else EPS
    return -np.log(max(float(p), EPS))


def run(league="PL", min_train=760, step_days=7):
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue

        w = decay_weights(train["date"], ref=cutoff).to_numpy().copy()
        hg_tr = train["home_goals"].astype(int).to_numpy().copy()
        ag_tr = train["away_goals"].astype(int).to_numpy().copy()
        h_tr = train["home"].to_numpy().copy()
        a_tr = train["away"].to_numpy().copy()

        try:
            full = LeagueModel().fit(train, ref=cutoff)                      # A
            dc = pb.models.DixonColesGoalModel(hg_tr, ag_tr, h_tr, a_tr, weights=w)
            dc.fit()                                                          # B
            wb = pb.models.WeibullCopulaGoalsModel(hg_tr, ag_tr, h_tr, a_tr, weights=w)
            wb.fit()                                                          # C
        except Exception as exc:
            print(f"  skip {cutoff.date()}: {exc}", flush=True)
            continue

        for _, m in test.iterrows():
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            try:
                lh, la = full.lambdas(m["home"], m["away"])
                gA = scoreline_grid(lh, la, full.rho)
                gB = dc.predict(m["home"], m["away"]).grid
                gC = wb.predict(m["home"], m["away"]).grid
            except Exception:
                continue
            rows.append({"A_full": _ll(gA, hg, ag), "B_dc": _ll(gB, hg, ag),
                         "C_weibull": _ll(gC, hg, ag)})

    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("nothing scored")

    def paired_ci(a, b, iters=4000, seed=7):
        rng = np.random.default_rng(seed)
        d = (r[a] - r[b]).to_numpy()          # positive => b better
        boot = d[rng.integers(0, len(d), size=(iters, len(d)))].mean(axis=1)
        return round(float(d.mean()), 5), [round(float(np.percentile(boot, 2.5)), 5),
                                           round(float(np.percentile(boot, 97.5)), 5)]

    out = {"league": league, "n": int(len(r)),
           "logloss": {k: round(float(r[k].mean()), 4) for k in r.columns}}
    for pair in (("B_dc", "C_weibull"), ("A_full", "C_weibull"), ("A_full", "B_dc")):
        mean, ci = paired_ci(*pair)
        out[f"{pair[0]}_minus_{pair[1]}"] = {
            "mean": mean, "ci95": ci, "significant": bool(ci[0] > 0 or ci[1] < 0)}
    return out


def main():
    rep = run("PL")
    print(json.dumps(rep, indent=2))
    (ROOT / "data-raw" / "leagues" / "weibull_experiment.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


# RESULT (PL, n=1106, walk-forward, 2026-07-19) -- NEGATIVE. Do not re-run.
#
#   A  full pipeline (DC + xG blend + tau)  2.9633   <- best, ships
#   B  Dixon-Coles raw                      2.9941
#   C  Weibull-copula raw                   2.9985
#
#   B vs C: -0.00443  CI [-0.01401, +0.00471]  NOT significant
#   A vs B: -0.03078  CI [-0.04667, -0.01683]  SIGNIFICANT
#   A vs C: -0.03521  CI [-0.05598, -0.01663]  SIGNIFICANT
#
# Weibull-copula does NOT beat Dixon-Coles on scoreline quality despite the strong
# theoretical case (it is the only penaltyblog model pointing the right way on both
# conditional underdispersion and NEGATIVE goal dependence). The difference is
# indistinguishable from zero and directionally slightly worse, and it failed to
# converge on two cutoffs ("Iteration limit reached") -- a robustness mark against
# it for an unattended weekly job. Keeping Dixon-Coles.
#
# The valuable finding is A vs B: our xG blending + tau + tuned shrinkage beats raw
# Dixon-Coles by 0.031 nats, SIGNIFICANTLY. That is the first statistically solid
# result of the whole correct-score investigation -- the pipeline earns its keep.
#
# Ensembling A with C was considered and rejected: A beats C significantly, so
# averaging would most likely drag A toward C. Ensembles pay when components are
# comparable with decorrelated errors; that is not the case here.
