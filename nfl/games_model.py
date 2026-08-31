"""Team-winner model: Elo ratings with margin of victory and home field.

WHY ELO AND NOT THE SOCCER ENGINE. Dixon-Coles models two low-count Poisson goal
processes with a correlation correction for 0-0 and 1-1. NFL scores are 20-30
points assembled from touchdowns, field goals and two-point conversions; nothing
about that shape survives the translation. Elo makes no distributional claim at
all -- it tracks who has been beating whom and by how much -- which is the honest
level of assumption for a 17-game season.

IT IS CAUSAL BY CONSTRUCTION. A rating only ever reflects games already played,
so the probability attached to a game cannot contain that game. There is no
`.shift(1)` to get wrong here, which is a real advantage over the props side.

PARAMETERS ARE FITTED ON TRAINING SEASONS ONLY. K, the home-field edge and the
between-season regression are chosen by grid search on the burn-in window and
then frozen before any scored season is touched.
"""
import numpy as np
import pandas as pd

START = 1500.0
SCALE = 400.0        # a 400-point gap is the classic 10:1 odds


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / SCALE))


def mov_multiplier(margin: float, rating_diff: float) -> float:
    """Weight a result by how emphatic it was, damped for one-sided fixtures.

    A three-point win is mostly a coin landing; a 30-point win is information. But
    a strong favourite blowing out a weak team says less than the same margin the
    other way, so the multiplier shrinks as the rating gap grows -- otherwise good
    teams inflate forever by beating bad ones. This is the standard correction and
    it exists to stop exactly that runaway.
    """
    return float(np.log(abs(margin) + 1.0) * (2.2 / (rating_diff * 0.001 + 2.2)))


def run_elo(games: pd.DataFrame, k: float, home_edge: float,
            regression: float) -> pd.DataFrame:
    """Walk the schedule once, recording each game's PRE-GAME probability."""
    ratings = {}
    season_seen = None
    rows = []
    # Chronological, by whatever column the sport orders games with. The NFL
    # numbers them by week and date; the NBA has only a date. NFL frames still
    # carry both, so their iteration order is unchanged.
    order = [c for c in ("season", "week", "gameday", "game_date") if c in games.columns]
    for _, game in games.sort_values(order).iterrows():
        home, away = game["home_team"], game["away_team"]
        if game["season"] != season_seen:
            # Between seasons, pull every team back toward the mean. Rosters turn
            # over, coaches change, and a team's rating in September is a weaker
            # claim than the same number in December.
            for team in ratings:
                ratings[team] = START + (ratings[team] - START) * (1 - regression)
            season_seen = game["season"]
        rating_home = ratings.setdefault(home, START)
        rating_away = ratings.setdefault(away, START)

        # Neutral-site games get no home edge -- the schedule says which they are.
        # Two feeds say it two ways: the NFL schedule carries a `location` string,
        # the NBA one a boolean, set where BOTH team rows read "@" and neither
        # side owns the court. Either is honoured; granting home advantage on a
        # neutral floor would be inventing an effect.
        neutral = (bool(game.get("neutral", False))
                   or str(game.get("location", "Home")).lower() == "neutral")
        edge = 0.0 if neutral else home_edge
        prob_home = expected(rating_home + edge, rating_away)
        # `week` is the NFL's ordering key and simply absent for the NBA.
        rows.append({"season": game["season"], "week": game.get("week"),
                     "home_team": home, "away_team": away,
                     "prob_home": prob_home, "winner": game["winner"],
                     "played": bool(game["played"]),
                     "rating_home": rating_home, "rating_away": rating_away})

        if not game["played"] or pd.isna(game["winner"]):
            continue
        # A tie is half a win to each side, which is what it is. Collapsing it to a
        # loss would quietly punish whichever team the model happened to favour.
        actual = {"home": 1.0, "away": 0.0, "tie": 0.5}[game["winner"]]
        margin = float(game["home_score"] - game["away_score"])
        diff = (rating_home + edge - rating_away) * (1 if actual >= 0.5 else -1)
        change = k * mov_multiplier(margin, diff) * (actual - prob_home)
        ratings[home] = rating_home + change
        ratings[away] = rating_away - change
    return pd.DataFrame(rows)


def fit_parameters(games: pd.DataFrame, grid=None) -> dict:
    """Grid search K, home edge and regression by log-loss. TRAINING GAMES ONLY."""
    grid = grid or {
        # Ranges chosen to BRACKET the optimum, not to sit on it. A first pass
        # returned home_edge=25 at the very edge of the grid, which usually means
        # the real answer is outside it -- and it was: home advantage in the NFL
        # has fallen a long way from the ~65 Elo points of the 2000s.
        "k": (8.0, 12.0, 16.0, 20.0, 24.0, 28.0),
        "home_edge": (0.0, 10.0, 20.0, 25.0, 32.0, 40.0, 55.0, 70.0),
        "regression": (0.10, 0.20, 0.33, 0.45, 0.60),
    }
    best, best_loss = None, None
    for k in grid["k"]:
        for edge in grid["home_edge"]:
            for regression in grid["regression"]:
                out = run_elo(games, k, edge, regression)
                out = out[out["played"] & out["winner"].notna()]
                if out.empty:
                    continue
                actual = out["winner"].map({"home": 1.0, "away": 0.0, "tie": 0.5})
                prob = out["prob_home"].clip(1e-6, 1 - 1e-6)
                loss = float(-np.mean(actual * np.log(prob)
                                      + (1 - actual) * np.log(1 - prob)))
                if best_loss is None or loss < best_loss:
                    best, best_loss = {"k": k, "home_edge": edge,
                                       "regression": regression}, loss
    return {**(best or {"k": 20.0, "home_edge": 55.0, "regression": 0.33}),
            "log_loss": round(best_loss, 6) if best_loss else None}
