"""Season Monte Carlo: sample every remaining fixture, tally the table, repeat.

PL tie-breakers, in order: points -> goal difference -> goals for. (Head-to-head
only separates teams still level on all three, which is rare enough that we leave
it to the alphabetical fallback rather than pretend to a precision the model
does not have.)
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from leagues import config
from leagues.model import scoreline_grid

N_SIMS = 10000


def rank_teams(table: pd.DataFrame) -> list[str]:
    """Order teams by points, then goal difference, then goals for."""
    t = table.sort_values(["points", "gd", "gf", "team"],
                          ascending=[False, False, False, True])
    return t["team"].tolist()


def final_table(played: pd.DataFrame, remaining: pd.DataFrame, sample) -> pd.DataFrame:
    """One simulated season. `sample(home, away) -> (home_goals, away_goals)`.

    Real results in `played` are locked in; only `remaining` is sampled.
    """
    pts, gf, ga = defaultdict(int), defaultdict(int), defaultdict(int)

    def record(h, a, hg, ag):
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif ag > hg:
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1

    for _, m in played.iterrows():
        record(m["home"], m["away"], int(m["home_goals"]), int(m["away_goals"]))
    for _, m in remaining.iterrows():
        hg, ag = sample(m["home"], m["away"])
        record(m["home"], m["away"], int(hg), int(ag))

    teams = sorted(set(gf) | set(ga) | set(pts))
    return pd.DataFrame([{"team": t, "points": pts[t], "gf": gf[t], "ga": ga[t],
                          "gd": gf[t] - ga[t]} for t in teams])


def _sampler(model, rng):
    """Draw a scoreline from the model's Dixon-Coles grid (cached per fixture)."""
    cache = {}

    def sample(home, away):
        key = (home, away)
        if key not in cache:
            lh, la = model.lambdas(home, away)
            grid = scoreline_grid(lh, la, model.rho)
            cache[key] = (grid.ravel(), grid.shape)
        flat, shape = cache[key]
        idx = rng.choice(len(flat), p=flat)
        return np.unravel_index(idx, shape)

    return sample


def simulate_season(model, played: pd.DataFrame, remaining: pd.DataFrame,
                    league: str = "PL", n: int = N_SIMS, seed: int = 7) -> pd.DataFrame:
    """Run n seasons; return the projected table with title/top-4/relegation %."""
    lg = config.get(league)
    rng = np.random.default_rng(seed)
    sample = _sampler(model, rng)

    finishes = defaultdict(lambda: defaultdict(int))
    points_sum = defaultdict(float)

    for _ in range(n):
        table = final_table(played, remaining, sample)
        order = rank_teams(table)
        pts = dict(zip(table["team"], table["points"]))
        for pos, team in enumerate(order, start=1):
            finishes[team][pos] += 1
            points_sum[team] += pts[team]

    rows = []
    for team, counts in finishes.items():
        total = sum(counts.values())
        top4 = sum(c for p, c in counts.items() if p <= lg.europe_spots)
        rel = sum(c for p, c in counts.items()
                  if p > lg.n_teams - lg.relegation_spots)
        rows.append({
            "team": team,
            "proj_points": round(points_sum[team] / total, 1),
            "title_pct": round(100 * counts.get(1, 0) / total, 1),
            "top4_pct": round(100 * top4 / total, 1),
            "relegation_pct": round(100 * rel / total, 1),
        })
    return pd.DataFrame(rows).sort_values("proj_points", ascending=False)
