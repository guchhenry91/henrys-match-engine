"""Walk-forward validation against de-vigged bookmaker closing odds.

The closing line is the benchmark. Getting CLOSE to it is success; consistently
BEATING it on accuracy would be extraordinary and is far more likely to indicate
lookahead leakage than genuine skill.
"""
import numpy as np
import pandas as pd

from leagues import second_tier
from leagues.model import LeagueModel, promoted_priors


def outcome_index(hg: int, ag: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if hg > ag:
        return 0
    return 1 if hg == ag else 2


def devig(odds_h: float, odds_d: float, odds_a: float):
    """Proportional de-vig: raw implied probabilities normalized to sum to 1."""
    raw = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], dtype=float)
    out = raw / raw.sum()
    return float(out[0]), float(out[1]), float(out[2])


def _onehot(outcomes: np.ndarray, k: int = 3) -> np.ndarray:
    obs = np.zeros((len(outcomes), k), dtype=float)
    obs[np.arange(len(outcomes)), outcomes] = 1.0
    return obs


def rps_per_match(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """RPS for each match separately. The paired bootstrap needs the per-match
    series, not the mean: pairing two models on the SAME fixtures is what removes
    fixture difficulty from the comparison."""
    probs = np.asarray(probs, dtype=float)
    obs = _onehot(np.asarray(outcomes, dtype=int))
    cp, co = np.cumsum(probs, axis=1), np.cumsum(obs, axis=1)
    return ((cp - co) ** 2)[:, :2].sum(axis=1) / 2.0


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(rps_per_match(probs, outcomes).mean())


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    obs = _onehot(np.asarray(outcomes, dtype=int))
    return float(((probs - obs) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(outcomes)).mean())


def _cutoff_priors(base, league: str, cutoff, teams) -> dict:
    """Priors for teams in the test window with no history yet, built EXACTLY the
    way publish.py builds them: the club's own second-tier season first, the
    weakest-side fallback for whatever that feed cannot resolve.

    The season is derived from the cutoff (`feeder_season`), never from today, so
    a 2022 cutoff is seeded from the 2021-22 second tier and not from a table that
    had not been played yet. That is what keeps this strictly causal.
    """
    no_history = [t for t in sorted(teams) if t not in base.attack]
    if not no_history:
        return {}
    priors = {}
    try:
        priors = second_tier.second_tier_priors(
            base, league, no_history, season=second_tier.feeder_season(cutoff))
    except Exception as exc:
        print(f"  {cutoff.date()}: second-tier feed unavailable ({exc}); "
              f"weakest-side fallback for {len(no_history)} club(s)")
    still_missing = [t for t in no_history if t not in priors]
    if still_missing:
        priors.update(promoted_priors(base, still_missing))
    return priors


def walk_forward(matches: pd.DataFrame, league: str, xi: float = 0.003,
                 xg_weight: float = 0.75, min_train: int = 760,
                 step_days: int = 7) -> pd.DataFrame:
    """Refit weekly on everything BEFORE the cutoff, predict the next 7 days.

    STRICTLY causal: training data is always `date < cutoff`, and a promoted
    club's prior comes from the second-tier season that had already finished by
    that cutoff.

    THIS RUNS THE MODEL THAT SHIPS. It used to run a different one: promoted clubs
    got no prior here, so `model.predict` raised KeyError and the fixture was
    silently dropped from the sample. That quietly excluded every promoted club's
    matches -- the hardest and least predictable fixtures in the league -- and
    measured a model publish.py never uses. Expect the reported RPS to be WORSE
    than the old number; the old number was flattering, not better.

    `league` is required for exactly that reason: there is no configuration of
    this function that scores a model production does not run.
    """
    df = (matches.dropna(subset=["home_goals", "away_goals"])
                 .sort_values("date").reset_index(drop=True))
    if len(df) <= min_train:
        raise ValueError(f"need > {min_train} matches, got {len(df)}")

    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) &
                  (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff)
            # Refit with priors ONLY when the window actually contains a club with
            # no history -- true for the opening weeks of a season and nowhere
            # else, so the second fit costs little across the whole walk.
            priors = _cutoff_priors(model, league, cutoff,
                                    set(test["home"]) | set(test["away"]))
            if priors:
                model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(
                    train, ref=cutoff, priors=priors)
        except Exception as exc:
            print(f"  skip {cutoff.date()}: fit failed ({exc})")
            continue
        for _, m in test.iterrows():
            # No KeyError guard. Every team in this window was either fitted or
            # seeded above, so a KeyError here is a real bug and must surface
            # rather than silently shrink the evaluation sample.
            p = model.predict(m["home"], m["away"])
            row = {"date": m["date"], "home": m["home"], "away": m["away"],
                   "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                   "outcome": outcome_index(int(m["home_goals"]), int(m["away_goals"])),
                   # True when either side was seeded from a prior rather than
                   # fitted, so score() can report how much of the sample is the
                   # promoted-club fixtures the old backtest threw away.
                   "seeded": bool(priors and (m["home"] in priors or m["away"] in priors))}
            if pd.notna(m.get("odds_h")):
                mh, md, ma = devig(m["odds_h"], m["odds_d"], m["odds_a"])
                row.update({"m_home": mh, "m_draw": md, "m_away": ma})
            rows.append(row)
    return pd.DataFrame(rows)


def score(results: pd.DataFrame) -> dict:
    """Model vs market on the SAME matches (market subset only, for fairness)."""
    p = results[["p_home", "p_draw", "p_away"]].to_numpy()
    y = results["outcome"].to_numpy()
    out = {"n": int(len(results)), "accuracy": accuracy(p, y),
           "rps": rps(p, y), "brier": brier(p, y)}
    if "seeded" in results.columns:
        # Fixtures involving a club seeded from a prior. These are the ones the
        # pre-M-01 backtest dropped entirely, so this number says how much of the
        # sample is newly measured rather than newly invented.
        out["n_seeded"] = int(results["seeded"].sum())
    if "m_home" in results.columns:
        mk = results.dropna(subset=["m_home"])
        if len(mk):
            mp = mk[["m_home", "m_draw", "m_away"]].to_numpy()
            my = mk["outcome"].to_numpy()
            # score the MODEL on the same subset so the comparison is apples-to-apples
            sp = mk[["p_home", "p_draw", "p_away"]].to_numpy()
            out.update({
                "market_n": int(len(mk)),
                "model_rps_on_market_subset": rps(sp, my),
                "model_accuracy_on_market_subset": accuracy(sp, my),
                "market_accuracy": accuracy(mp, my),
                "market_rps": rps(mp, my),
                "market_brier": brier(mp, my),
            })
    return out
