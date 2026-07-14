"""Walk-forward validation against de-vigged bookmaker closing odds.

The closing line is the benchmark. Getting CLOSE to it is success; consistently
BEATING it on accuracy would be extraordinary and is far more likely to indicate
lookahead leakage than genuine skill.
"""
import numpy as np
import pandas as pd

from leagues.model import LeagueModel


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


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    obs = _onehot(np.asarray(outcomes, dtype=int))
    cp, co = np.cumsum(probs, axis=1), np.cumsum(obs, axis=1)
    return float((((cp - co) ** 2)[:, :2].sum(axis=1) / 2.0).mean())


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    obs = _onehot(np.asarray(outcomes, dtype=int))
    return float(((probs - obs) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(outcomes)).mean())


def walk_forward(matches: pd.DataFrame, xi: float = 0.003, xg_weight: float = 0.75,
                 min_train: int = 760, step_days: int = 7) -> pd.DataFrame:
    """Refit weekly on everything BEFORE the cutoff, predict the next 7 days.

    STRICTLY causal: training data is always `date < cutoff`.
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
        except Exception as exc:
            print(f"  skip {cutoff.date()}: fit failed ({exc})")
            continue
        for _, m in test.iterrows():
            try:
                p = model.predict(m["home"], m["away"])
            except KeyError:
                continue   # promoted team with no history yet — skip, never guess
            row = {"date": m["date"], "home": m["home"], "away": m["away"],
                   "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                   "outcome": outcome_index(int(m["home_goals"]), int(m["away_goals"]))}
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
