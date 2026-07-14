"""The match model.

penaltyblog's DixonColesGoalModel requires INTEGER goals, so xG cannot be the
response. Instead we:
  1. fit Dixon-Coles on actual goals  -> rho, home advantage, attack/defence
  2. compute xG-based attack/defence separately (log ratio to league average)
  3. blend the two strength sets (default 75% xG / 25% goals)
  4. build the scoreline grid ourselves with the Dixon-Coles tau correction
This keeps penaltyblog's rigorous MLE while gaining xG's lower-variance signal.

penaltyblog's get_params() (verified empirically on real PL data) returns a
flat dict with keys "attack_{team}", "defence_{team}", "home_advantage" and
"rho". Sign convention: a STRONG defence has a MORE NEGATIVE value (e.g.
Arsenal -1.24 vs Burnley -0.63), which correctly lowers the opponent's
expected goals when added into the away lambda's log-rate.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.stats import poisson
from leagues.weights import XI_PER_DAY, decay_weights

XG_WEIGHT = 0.75
MAX_GOALS = 10
PRIOR_STRENGTH = 6.0


def dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles low-score correction; negative rho lifts 0-0 and 1-1."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Normalized correct-score matrix [home, away]."""
    hp = poisson.pmf(np.arange(max_goals + 1), lh)
    ap = poisson.pmf(np.arange(max_goals + 1), la)
    grid = np.outer(hp, ap)
    for h in range(min(2, max_goals + 1)):
        for a in range(min(2, max_goals + 1)):
            grid[h, a] *= dc_tau(h, a, lh, la, rho)
    grid = np.clip(grid, 0.0, None)
    return grid / grid.sum()


def outcome_probs(grid: np.ndarray) -> tuple[float, float, float]:
    return (float(np.tril(grid, -1).sum()), float(np.trace(grid)),
            float(np.triu(grid, 1).sum()))


@dataclass
class LeagueModel:
    xi: float = XI_PER_DAY
    xg_weight: float = XG_WEIGHT
    rho: float = 0.0
    home_adv: float = 0.0
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)

    def fit(self, matches, ref=None, priors=None):
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if df.empty:
            raise ValueError("no played matches to fit on")
        ref = ref or pd.to_datetime(df["date"]).max()
        w = decay_weights(df["date"], ref=ref, xi=self.xi).to_numpy().copy()

        clf = pb.models.DixonColesGoalModel(
            df["home_goals"].astype(int).to_numpy().copy(),
            df["away_goals"].astype(int).to_numpy().copy(),
            df["home"].to_numpy().copy(), df["away"].to_numpy().copy(), weights=w)
        clf.fit()
        goal_att, goal_def = self._parse_params(clf)

        xg_att, xg_def = self._xg_strengths(df, ref)
        teams = sorted(set(goal_att) | set(goal_def))
        self.attack, self.defence = {}, {}
        for t in teams:
            ga, gd = goal_att.get(t, 0.0), goal_def.get(t, 0.0)
            xa, xd = xg_att.get(t), xg_def.get(t)
            if xa is None or xd is None:
                self.attack[t], self.defence[t] = ga, gd
            else:
                self.attack[t] = self.xg_weight * xa + (1 - self.xg_weight) * ga
                self.defence[t] = self.xg_weight * xd + (1 - self.xg_weight) * gd
        if priors:
            self._apply_priors(df, w, priors)
        return self

    def _parse_params(self, clf):
        """Extract attack/defence dicts + set self.rho/self.home_adv from
        penaltyblog's get_params().

        Verified empirically (Task 9, Step 3) on real PL data: get_params()
        returns a flat dict keyed "attack_{team}", "defence_{team}", plus
        scalar "home_advantage" and "rho" keys.
        """
        params = clf.get_params()
        self.rho = float(params["rho"])
        self.home_adv = float(params["home_advantage"])
        attack, defence = {}, {}
        for key, value in params.items():
            if key.startswith("attack_"):
                attack[key[len("attack_"):]] = float(value)
            elif key.startswith("defence_"):
                defence[key[len("defence_"):]] = float(value)
        return attack, defence

    def _xg_strengths(self, df, ref):
        if "home_xg" not in df or df["home_xg"].isna().all():
            return {}, {}
        d = df.dropna(subset=["home_xg", "away_xg"])
        if d.empty:
            return {}, {}
        wd = decay_weights(d["date"], ref=ref, xi=self.xi).to_numpy()
        rows = pd.concat([
            pd.DataFrame({"team": d["home"].values, "xgf": d["home_xg"].values,
                          "xga": d["away_xg"].values, "w": wd}),
            pd.DataFrame({"team": d["away"].values, "xgf": d["away_xg"].values,
                          "xga": d["home_xg"].values, "w": wd}),
        ])
        avg = np.average(rows["xgf"], weights=rows["w"])
        att, dfn = {}, {}
        for team, g in rows.groupby("team"):
            f = np.average(g["xgf"], weights=g["w"])
            a = np.average(g["xga"], weights=g["w"])
            att[team] = float(np.log(max(f, 0.05) / avg))
            dfn[team] = float(np.log(max(a, 0.05) / avg))
        return att, dfn

    def _apply_priors(self, df, w, priors):
        """Shrink strengths toward a prior for teams with little history."""
        for team in list(self.attack):
            p = priors.get(team)
            if p is None:
                continue
            mask = ((df["home"] == team) | (df["away"] == team)).to_numpy()
            eff = float(w[mask].sum())
            k = PRIOR_STRENGTH / (PRIOR_STRENGTH + eff)
            self.attack[team] = (1 - k) * self.attack[team] + k * p
            self.defence[team] = (1 - k) * self.defence[team] - k * p

    def lambdas(self, home: str, away: str):
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(f"team {t!r} not in fitted model")
        lh = np.exp(self.attack[home] + self.defence[away] + self.home_adv)
        la = np.exp(self.attack[away] + self.defence[home])
        return float(np.clip(lh, 0.05, 6.0)), float(np.clip(la, 0.05, 6.0))

    def predict(self, home: str, away: str) -> dict:
        lh, la = self.lambdas(home, away)
        grid = scoreline_grid(lh, la, self.rho)
        ph, pdw, pa = outcome_probs(grid)
        h, a = np.unravel_index(np.argmax(grid), grid.shape)
        return {"p_home": ph, "p_draw": pdw, "p_away": pa,
                "lambda_home": lh, "lambda_away": la,
                "score": f"{int(h)}-{int(a)}", "grid": grid}
