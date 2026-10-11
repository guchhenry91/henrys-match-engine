"""The rebuilt NFL prop model: forecast the player's game, never read a line.

docs/REBUILD_PLAN.md. The old NFL props (nfl/features.augment_lines + PropModel)
were trained on lines set from each player's own median with the line as an input,
and deliberately ignored the game's spread and total. This model forecasts the stat:

  yards  -- a gradient-boosted Poisson mean on ROLE (targets, carries, attempts,
            snap share, target and air-yards share), GAME SCRIPT (the team's implied
            points from the spread and total), the opponent's defence, home, roof,
            wind and temperature; the spread of outcomes is EMPIRICAL
            (forecast.dist.EmpiricalRatio) -- yards are skewed and zero-heavy;
  TD     -- a gradient-boosted classifier for "scores at least once".

Every feature is ENTERING (built from earlier games), except what is known before
kickoff: the spread, total, venue and weather. The bookmaker's prop line is never
an input. Data: nflverse (stats_player_week, snap_counts, players, games), read
through nfl.data's cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast import dist
from nfl import data

MARKETS = {
    "passing_yards": {"stat": "passing_yards", "positions": ("QB",)},
    "rushing_yards": {"stat": "rushing_yards", "positions": ("RB", "QB", "WR")},
    "receiving_yards": {"stat": "receiving_yards", "positions": ("WR", "TE", "RB")},
    "anytime_touchdown": {"stat": "touchdowns", "positions": ("RB", "WR", "TE", "QB")},
}
POS = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
ROLL = ("targets", "receptions", "receiving_yards", "carries", "rushing_yards", "attempts",
        "completions", "passing_yards", "target_share", "air_yards_share",
        "receiving_air_yards", "passing_air_yards", "offense_pct", "touchdowns",
        "rushing_tds", "receiving_tds")
TRAIN_SEASONS = 5
RAW = ["player_id", "player_display_name", "position", "season", "week", "season_type",
       "game_id", "team", "opponent_team", "completions", "attempts", "passing_yards",
       "passing_tds", "passing_air_yards", "carries", "rushing_yards", "rushing_tds",
       "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
       "target_share", "air_yards_share", "special_teams_tds", "def_tds", "fumble_recovery_tds"]


def load(seasons) -> pd.DataFrame:
    """Player-weeks (regular season and playoffs) with snap share and game context."""
    frames = []
    for s in seasons:
        try:
            raw = data._read_csv(data.PLAYER_STATS_URL.format(season=s), f"stats_player_week_{s}.csv",
                                 max_age_hours=6 if s >= data.config.CURRENT_SEASON else None)
        except Exception:
            if s < data.config.CURRENT_SEASON:
                raise
            print(f"  no {s} player feed yet; completed seasons only")
            continue
        if "team" not in raw and "recent_team" in raw:
            raw = raw.rename(columns={"recent_team": "team"})
        frames.append(raw.reindex(columns=RAW))
    d = pd.concat(frames, ignore_index=True)
    for c in RAW[9:]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["touchdowns"] = data.anytime_touchdown(d).astype(float)
    d["offense_pct"] = _snaps(seasons, d)
    return _context(d)


def _snaps(seasons, d: pd.DataFrame) -> pd.Series:
    try:
        ids = data._read_csv(data.PLAYERS_URL, "players.csv", max_age_hours=72)
    except Exception:
        return pd.Series(np.nan, index=d.index)
    pfr = {r.pfr_id: r.gsis_id for r in ids[["pfr_id", "gsis_id"]].dropna().itertuples()}
    frames = []
    for s in seasons:
        try:
            sn = data._read_csv(data.SNAPS_URL.format(season=s), f"snap_counts_{s}.csv",
                                max_age_hours=6 if s >= data.config.CURRENT_SEASON else None)
        except Exception:
            continue
        if sn.empty or "pfr_player_id" not in sn:
            continue
        sn = sn.assign(player_id=sn["pfr_player_id"].map(pfr))
        frames.append(sn[["game_id", "player_id", "offense_pct"]].dropna(subset=["player_id"]))
    if not frames:
        return pd.Series(np.nan, index=d.index)
    sn = pd.concat(frames).drop_duplicates(["game_id", "player_id"])
    merged = d[["game_id", "player_id"]].merge(sn, on=["game_id", "player_id"], how="left")
    return pd.to_numeric(merged["offense_pct"], errors="coerce").set_axis(d.index)


def games_table() -> pd.DataFrame:
    g = data._read_csv(data.GAMES_URL, "games.csv", max_age_hours=3)
    g = g.copy()
    for c in ("spread_line", "total_line", "wind", "temp", "home_score", "away_score"):
        g[c] = pd.to_numeric(g.get(c), errors="coerce")
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    g["dome"] = g.get("roof", pd.Series("", index=g.index)).astype(str).isin(["dome", "closed"]).astype(float)
    return g


def _context(d: pd.DataFrame) -> pd.DataFrame:
    """Join the game: date, home, the team's implied points (from spread and total --
    a positive spread_line favours the home side), venue and weather."""
    g = games_table()[["game_id", "gameday", "home_team", "away_team", "spread_line",
                       "total_line", "dome", "wind", "temp"]]
    d = d.merge(g, on="game_id", how="left")
    d["home"] = (d["team"] == d["home_team"]).astype(float)
    sign = np.where(d["home"] == 1.0, 1.0, -1.0)
    d["spread_team"] = sign * d["spread_line"]
    d["implied_team"] = (d["total_line"] + d["spread_team"]) / 2.0
    d["implied_opp"] = d["total_line"] - d["implied_team"]
    d["wind"] = d["wind"].where(d["dome"] == 0, 0.0)
    return d


def build(d: pd.DataFrame) -> pd.DataFrame:
    """ENTERING features per player-week. Upcoming games may be appended with
    is_next=True and NaN stats; they get features, not targets."""
    d = d.copy()
    if "is_next" not in d:
        d["is_next"] = False
    d["is_next"] = d["is_next"].fillna(False).astype(bool)
    d["pos"] = d["position"].map(POS).fillna(4).astype(float)
    d = d.sort_values(["player_id", "gameday", "game_id"]).reset_index(drop=True)
    g = d.groupby("player_id", sort=False)
    d["games_before"] = g.cumcount()
    d["rest_days"] = (d["gameday"] - g["gameday"].shift(1)).dt.days.clip(upper=21).fillna(21)
    d["new_season"] = (d["season"] != g["season"].shift(1)).astype(float)
    for c in ROLL:
        s = pd.to_numeric(d[c], errors="coerce")
        for span in (3, 8):
            d[f"{c}_e{span}"] = s.groupby(d["player_id"]).transform(
                lambda x, k=span: x.shift(1).ewm(span=k, min_periods=1).mean())
    for num, den in (("receiving_yards", "targets"), ("rushing_yards", "carries"),
                     ("passing_yards", "attempts")):
        d[f"{num}_per_{den}"] = d[f"{num}_e8"] / d[f"{den}_e8"].clip(lower=0.5)
    # The team's volume and the opponent's defence, both entering.
    tg = d[~d["is_next"]].groupby(["game_id", "gameday", "team", "opponent_team"], as_index=False)[
        ["attempts", "carries", "passing_yards", "rushing_yards"]].sum()
    tg = tg.sort_values(["team", "gameday"])
    for c in ("attempts", "carries"):
        tg[f"team_{c}_in"] = tg.groupby("team")[c].transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    allowed = tg.rename(columns={"team": "x", "opponent_team": "def_team"}).sort_values(["def_team", "gameday"])
    for c in ("passing_yards", "rushing_yards"):
        allowed[f"def_{c}_in"] = allowed.groupby("def_team")[c].transform(
            lambda x: x.shift(1).ewm(span=8, min_periods=1).mean())
    d = d.merge(tg[["game_id", "team", "team_attempts_in", "team_carries_in"]], on=["game_id", "team"], how="left")
    d = d.merge(allowed[["game_id", "def_team", "def_passing_yards_in", "def_rushing_yards_in"]]
                .rename(columns={"def_team": "opponent_team"}), on=["game_id", "opponent_team"], how="left")
    # Upcoming games have no team-game row yet: carry each team's / defence's latest.
    for col, key in (("team_attempts_in", "team"), ("team_carries_in", "team"),
                     ("def_passing_yards_in", "opponent_team"), ("def_rushing_yards_in", "opponent_team")):
        latest = tg if key == "team" else allowed.rename(columns={"def_team": "opponent_team"})
        src = col.replace("_in", "")
        if key == "team":
            last = tg.groupby("team")[src.replace("team_", "")].apply(lambda x: x.ewm(span=6).mean().iloc[-1])
        else:
            base = src.replace("def_", "")
            last = latest.groupby("opponent_team")[base].apply(lambda x: x.ewm(span=8).mean().iloc[-1])
        miss = d["is_next"] & d[col].isna()
        d.loc[miss, col] = d.loc[miss, key].map(last)
    return d.sort_values(["gameday", "game_id", "player_id"]).reset_index(drop=True)


def features(market: str) -> list:
    base = ["pos", "home", "games_before", "rest_days", "new_season", "spread_team",
            "implied_team", "implied_opp", "total_line", "dome", "wind", "temp",
            "team_attempts_in", "team_carries_in", "def_passing_yards_in", "def_rushing_yards_in",
            "receiving_yards_per_targets", "rushing_yards_per_carries", "passing_yards_per_attempts"]
    return base + [f"{c}_e{k}" for c in ROLL for k in (3, 8)]


def eligible(frame: pd.DataFrame, market: str) -> pd.Series:
    """Players with a role in this market: the right position and recent volume."""
    pos = frame["position"].isin(MARKETS[market]["positions"])
    if market == "passing_yards":
        role = frame["attempts_e3"] >= 15
    elif market == "rushing_yards":
        role = frame["carries_e3"] >= 3
    elif market == "receiving_yards":
        role = frame["targets_e3"] >= 2
    else:
        role = (frame["carries_e3"] + frame["targets_e3"]) >= 3
    return pos & role & (frame["games_before"] >= 2)


class MarketModel:
    def __init__(self, market: str, seed: int = 0):
        self.market = market
        self.stat = MARKETS[market]["stat"]
        self.cols = features(market)
        self.binary = market == "anytime_touchdown"
        self.seed = seed
        self.model, self.spread = None, None

    def _gbm(self):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                  l2_regularization=1.0, random_state=self.seed)
        return (HistGradientBoostingClassifier(**kw) if self.binary
                else HistGradientBoostingRegressor(loss="poisson", **kw))

    def _y(self, frame):
        y = frame[self.stat].astype(float)
        return (y > 0).astype(int) if self.binary else y.clip(lower=0)

    def fit(self, train: pd.DataFrame):
        train = train[eligible(train, self.market) & ~train["is_next"]]
        x, y = train[self.cols], self._y(train)
        if not self.binary:
            # The spread of outcomes from OUT-OF-FOLD forecasts (two folds by season).
            seasons = sorted(train["season"].unique())
            half = set(seasons[len(seasons) // 2:])
            oof = np.full(len(train), np.nan)
            for hold in (half, set(seasons) - half):
                mask = train["season"].isin(hold).to_numpy()
                if mask.all() or (~mask).all():
                    continue
                oof[mask] = self._gbm().fit(x[~mask], y[~mask]).predict(x[mask])
            ok = ~np.isnan(oof)
            self.spread = dist.EmpiricalRatio().fit(oof[ok], train[self.stat].to_numpy(float)[ok])
        self.model = self._gbm().fit(x, y)
        return self

    def mean(self, rows):
        return np.clip(self.model.predict(rows[self.cols]), 0.5, None)

    def p_over(self, rows, lines):
        if self.binary:
            return self.model.predict_proba(rows[self.cols])[:, 1]
        return self.spread.p_over(self.mean(rows), lines)
