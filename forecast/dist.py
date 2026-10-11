"""Count distributions for stat forecasts: negative binomial with a fitted dispersion.

A player's points / rebounds / assists / threes are over-dispersed counts: the
variance is larger than the mean (a Poisson understates the tails, so it overstates
how sure an over or under is). Var = mu + alpha * mu^2, with alpha fitted per stat on
out-of-sample residuals -- and allowed to depend on the size of the mean, because a
3-point shooter's threes and a centre's rebounds do not spread alike.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def nb_params(mu, alpha):
    """scipy's (n, p) for a negative binomial with mean mu and Var = mu + alpha mu^2."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    alpha = np.maximum(np.asarray(alpha, dtype=float), 1e-6)
    n = 1.0 / alpha
    p = n / (n + mu)
    return n, p


def p_over(mu, alpha, line):
    """P(X > line) for half-point lines (P(X >= floor(line) + 1))."""
    n, p = nb_params(mu, alpha)
    k = np.floor(np.asarray(line, dtype=float))
    return stats.nbinom.sf(k, n, p)


def p_exact(mu, alpha, k):
    n, p = nb_params(mu, alpha)
    return stats.nbinom.pmf(k, n, p)


class Dispersion:
    """alpha as a step function of the predicted mean, fitted by moments.

    In each band of predicted mean: alpha = (mean((y-mu)^2) - mean(mu)) / mean(mu^2),
    floored at a small positive number (a band that looks under-dispersed is treated
    as Poisson, never as tighter than Poisson)."""

    def __init__(self, bands: int = 8):
        self.bands = bands
        self.edges = None
        self.alphas = None

    def fit(self, mu, y):
        mu, y = np.asarray(mu, float), np.asarray(y, float)
        qs = np.quantile(mu, np.linspace(0, 1, self.bands + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        self.edges = np.unique(qs)
        idx = np.clip(np.searchsorted(self.edges, mu, side="right") - 1, 0, len(self.edges) - 2)
        alphas = []
        for b in range(len(self.edges) - 1):
            m, t = mu[idx == b], y[idx == b]
            if len(m) < 200:
                alphas.append(np.nan)
                continue
            a = (np.mean((t - m) ** 2) - np.mean(m)) / max(np.mean(m ** 2), 1e-9)
            alphas.append(max(a, 1e-3))
        alphas = np.array(alphas, float)
        if np.isnan(alphas).all():
            alphas[:] = 0.1
        alphas[np.isnan(alphas)] = np.nanmedian(alphas)
        self.alphas = alphas
        return self

    def __call__(self, mu):
        mu = np.asarray(mu, float)
        idx = np.clip(np.searchsorted(self.edges, mu, side="right") - 1, 0, len(self.alphas) - 1)
        return self.alphas[idx]

    def to_dict(self):
        return {"edges": [None if not np.isfinite(e) else float(e) for e in self.edges],
                "alphas": [float(a) for a in self.alphas]}

    @classmethod
    def from_dict(cls, d):
        obj = cls(bands=len(d["alphas"]))
        obj.edges = np.array([-np.inf if e is None and i == 0 else np.inf if e is None else e
                              for i, e in enumerate(d["edges"])], float)
        obj.alphas = np.array(d["alphas"], float)
        return obj


def calibration(p, y, bins: int = 10) -> dict:
    """{ece, n, table:[(stated, landed, n)]} with equal-count bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p)
    table, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        s, l = float(p[chunk].mean()), float(y[chunk].mean())
        table.append((round(s, 4), round(l, 4), int(len(chunk))))
        ece += len(chunk) / len(p) * abs(s - l)
    return {"ece": round(ece, 4), "n": int(len(p)), "table": table}
