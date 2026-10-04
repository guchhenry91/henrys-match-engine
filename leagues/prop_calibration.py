"""Recalibrate the soccer player-prop probabilities from their own graded record.

WHY. The soccer props are UNDERCONFIDENT live, the opposite of the NFL: picks the
model rated 60-70% on 2+ shot attempts hit 92% (12 of 13), shots-on-target
60-70% hit 83%, anytime scorers rated ~32% hit 45% (2026-09-27). A board that
understates its picks hides good ones and misprices the edge.

HOW. Per market, a logistic map of the model's number:
    logit(p') = a + b * logit(p)
fitted by maximum likelihood on the graded picks, PENALISED TOWARD NO CHANGE
(a=0, b=1) with the weight of PRIOR_PICKS pseudo-picks -- so thirty graded picks
nudge it and three hundred move it. Below MIN_PICKS a market is left as is. The
raw number is kept on every card as p_model; nothing is hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

MIN_PICKS = 40
PRIOR_PICKS = 60.0
MARKETS = ("goal", "shots", "sot")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(p, outcome) -> tuple:
    """(a, b) for one market, penalised toward (0, 1)."""
    x, y = _logit(p), np.asarray(outcome, float)

    def loss(ab):
        a, b = ab
        q = 1.0 / (1.0 + np.exp(-(a + b * x)))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        ll = np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
        # A Gaussian prior worth PRIOR_PICKS picks of evidence for "no change".
        return -ll + 0.5 * PRIOR_PICKS * 0.25 * (a ** 2 + (b - 1.0) ** 2)

    res = minimize(loss, x0=[0.0, 1.0], method="L-BFGS-B",
                   bounds=[(-3.0, 3.0), (0.3, 3.0)])
    return float(res.x[0]), float(res.x[1])


def apply_p(p: float, ab: tuple) -> float:
    a, b = ab
    return float(1.0 / (1.0 + np.exp(-(a + b * _logit(p)))))


def fit_all(settled: list) -> dict:
    out = {}
    for market in MARKETS:
        rows = [s for s in settled if s.get("market") == market
                and s.get("graded") in ("correct", "wrong") and s.get("p_pick") is not None]
        if len(rows) < MIN_PICKS:
            continue
        p = [float(s.get("p_model", s["p_pick"])) for s in rows]
        y = [1.0 if s["graded"] == "correct" else 0.0 for s in rows]
        out[market] = fit(p, y) + (len(rows),)
    return out


def apply(upcoming: list, settled: list) -> dict:
    """Recalibrate each upcoming pick's p_pick in place; returns the fits used."""
    fits = fit_all(settled)
    for pick in upcoming:
        f = fits.get(pick.get("market"))
        if not f or pick.get("p_pick") is None:
            continue
        if pick.get("provisional") is False:
            continue     # FROZEN: it shows the probability it was locked at, never a refit
        raw = float(pick.get("p_model", pick["p_pick"]))
        pick["p_model"] = round(raw, 4)
        pick["p_pick"] = round(apply_p(raw, f[:2]), 4)
        pick["calibrated_on"] = f[2]
    return {m: {"a": round(f[0], 3), "b": round(f[1], 3), "n": f[2]} for m, f in fits.items()}
