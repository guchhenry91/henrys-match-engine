"""The rebuilt NBA prop model: forecast each player's stat, never read a line.

docs/REBUILD_PLAN.md. The old model (nba/features.augment_lines + PropModel) was
trained on lines it set from each player's own median, with the line as an input,
and read bookmakers' lines backwards. This one forecasts the stat itself:

  mean   = a gradient-boosted Poisson regression on what drives the stat --
           minutes (recent and long-run), per-minute rates, usage, the team's and
           opponent's pace, what the opponent allows, rest, home;
  spread = a negative binomial dispersion fitted on out-of-sample residuals.

P(over ANY line) is read off that distribution (forecast.dist.p_over). Every
feature is ENTERING: built only from games before the one being forecast.

Source: data-raw/nba/history_full/ (every box-score column, scripts/
export_nba_full_history.py) plus this season's box scores (nba.current).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecast import dist

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data-raw" / "nba" / "history_full"
STATS = ("PTS", "REB", "AST", "FG3M")
MARKET_STAT = {"points": "PTS", "rebounds": "REB", "assists": "AST", "threes": "FG3M"}
RATE_COLS = ("PTS", "REB", "AST", "FG3M", "FGA", "FTA", "FG3A", "TOV", "OREB", "DREB")
TRAIN_SEASONS = 5


def load_history(seasons=None) -> pd.DataFrame:
    files = sorted(HISTORY.glob("player_*.csv.gz"))
    frames = []
    for f in files:
        season = int(f.stem.split("_")[1].split(".")[0])
        if seasons is not None and season not in seasons:
            continue
        d = pd.read_csv(f)
        d["season"] = season
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prep(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows.copy()
    if "is_next" not in d:
        d["is_next"] = False
    d["is_next"] = d["is_next"].fillna(False).astype(bool)
    d["MIN"] = pd.to_numeric(d["MIN"], errors="coerce")
    # Played games, plus the upcoming game being forecast (its stats are unknown).
    d = d[(d["MIN"].fillna(0) > 0) | d["is_next"]].copy()
    d["game_date"] = pd.to_datetime(d["GAME_DATE"].astype(str).str[:10])
    m = d["MATCHUP"].astype(str)
    d["home"] = m.str.contains("vs.", regex=False).astype(float)
    d["opp"] = m.str.split(r"\s+(?:@|vs\.)\s+", regex=True).str[-1]
    d["team"] = d["TEAM_ABBREVIATION"]
    d["GAME_ID"] = d["GAME_ID"].astype(str)
    # A column a feed did not carry stays MISSING, never zero: zero shot attempts
    # would read as a player who stopped shooting. The rolling means skip gaps.
    for c in RATE_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    return d.sort_values(["game_date", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


def _team_games(d: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: possessions, and what the team ALLOWED (opponent's
    output), so pace and defence can be forecast like anything else."""
    t = d.groupby(["GAME_ID", "game_date", "team", "opp"], as_index=False)[
        ["FGA", "FTA", "OREB", "TOV", "PTS", "REB", "AST", "FG3M", "MIN"]].sum()
    t["poss"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    opp = t[["GAME_ID", "team", "poss", "PTS", "REB", "AST", "FG3M"]].rename(columns={
        "team": "opp", "poss": "opp_poss", "PTS": "allow_PTS", "REB": "allow_REB",
        "AST": "allow_AST", "FG3M": "allow_FG3M"})
    t = t.merge(opp, on=["GAME_ID", "opp"], how="left")
    t = t.sort_values(["team", "game_date", "GAME_ID"])
    g = t.groupby("team", sort=False)
    t["pace_in"] = g["poss"].transform(lambda s: s.shift(1).ewm(span=20, min_periods=1).mean())
    for s in STATS:
        per = t[f"allow_{s}"] / t["opp_poss"].clip(lower=1)
        t[f"allow_{s}_in"] = per.groupby(t["team"]).transform(
            lambda x: x.shift(1).ewm(span=30, min_periods=1).mean())
    return t[["GAME_ID", "team", "pace_in"] + [f"allow_{s}_in" for s in STATS]]


def _vacated(d: pd.DataFrame, out_players: dict | None) -> pd.DataFrame:
    """Per team-game: how much of the team's usual usage is MISSING.

    The rotation is every player who played for the team in any of its last three
    games averaging 15+ minutes. In a played game, a rotation player with no row is
    absent; for an UPCOMING game nobody's row exists yet, so only the players the
    news has ruled OUT (`out_players`: {team: {PLAYER_ID}}) count as absent. A player
    who has left the team drops out of its rotation after three games.

    vac_usage = possessions-used per game (ewm) of the absent; vac_share = that over
    the whole rotation's. The GBM learns who inherits it (a guard's usage goes to the
    other guards and the stars, not the backup centre)."""
    out_players = out_players or {}
    used = d["FGA"] + 0.44 * d["FTA"] + d["TOV"]
    d = d.assign(_u=used.groupby(d["PLAYER_ID"]).transform(lambda s: s.ewm(span=10, min_periods=1).mean()),
                 _m=d["MIN"].groupby(d["PLAYER_ID"]).transform(lambda s: s.ewm(span=10, min_periods=1).mean()))
    rows = {}
    order = d.sort_values(["team", "game_date", "GAME_ID"])
    for (team, _date, gid), grp in order.groupby(["team", "game_date", "GAME_ID"], sort=False):
        rows.setdefault(team, []).append((gid, grp))
    out = []
    for team, games in rows.items():
        recent = []                              # the last three PLAYED games' rosters
        for gid, grp in games:
            rot = {}
            for roster in recent:
                rot.update(roster)
            rot = {p: v for p, v in rot.items() if v[1] >= 15}
            nxt = bool(grp["is_next"].all())
            if nxt:
                absent = set(rot) & set(out_players.get(team, ()))
            else:
                absent = set(rot) - set(grp["PLAYER_ID"])
            vac = sum(rot[p][0] for p in absent if np.isfinite(rot[p][0]))
            tot = sum(v[0] for v in rot.values() if np.isfinite(v[0]))
            out.append((gid, team, vac, vac / tot if tot > 0 else 0.0))
            if not nxt:
                recent.append({int(p): (u, m) for p, u, m in zip(grp["PLAYER_ID"], grp["_u"], grp["_m"])})
                recent = recent[-3:]
    return pd.DataFrame(out, columns=["GAME_ID", "team", "vac_usage", "vac_share"])


ELO_K, ELO_HOME, ELO_REGRESS = 10.0, 55.0, 0.35      # data-raw/nba/elo_params.json


def _elo(d: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: the PRE-game Elo gap (team minus opponent, home edge included).
    The expected margin is what decides blowouts -- and a blowout is what sits a
    starter for the fourth quarter. Historical point spreads are not on hand; the
    ratings the board already uses for team winners stand in for them. Upcoming games
    get today's ratings (played games only ever update them)."""
    g = (d.groupby(["GAME_ID", "game_date", "season", "team", "opp"], as_index=False)
         .agg(home=("home", "max"), pts=("PTS", "sum"), nxt=("is_next", "max")))
    homes = g[g["home"] >= 0.5].merge(
        g[["GAME_ID", "team", "pts"]].rename(columns={"team": "opp", "pts": "opp_pts"}),
        on=["GAME_ID", "opp"], how="left").sort_values(["game_date", "GAME_ID"])
    rating, season_seen, rows = {}, None, []
    for r in homes.itertuples():
        if r.season != season_seen:
            rating = {t: 1500 + (v - 1500) * (1 - ELO_REGRESS) for t, v in rating.items()}
            season_seen = r.season
        rh, ra = rating.setdefault(r.team, 1500.0), rating.setdefault(r.opp, 1500.0)
        diff = rh + ELO_HOME - ra
        rows += [(r.GAME_ID, r.team, diff), (r.GAME_ID, r.opp, -diff)]
        if not r.nxt and pd.notna(r.pts) and pd.notna(r.opp_pts) and r.pts != r.opp_pts:
            exp = 1 / (1 + 10 ** (-diff / 400))
            delta = ELO_K * ((1.0 if r.pts > r.opp_pts else 0.0) - exp)
            rating[r.team], rating[r.opp] = rh + delta, ra - delta
    out = pd.DataFrame(rows, columns=["GAME_ID", "team", "elo_diff"])
    out["elo_abs"] = out["elo_diff"].abs()
    return out.drop_duplicates(["GAME_ID", "team"])


def build(rows: pd.DataFrame, out_players: dict | None = None) -> pd.DataFrame:
    """Player-game rows with ENTERING features. Upcoming games may be appended as
    rows with NaN stats and is_next=True -- they get features, not targets.
    `out_players` ({team: {PLAYER_ID}}) says who is ruled out of upcoming games."""
    d = _prep(rows)
    d = d.sort_values(["PLAYER_ID", "game_date", "GAME_ID"]).reset_index(drop=True)
    g = d.groupby("PLAYER_ID", sort=False)
    prev_date = g["game_date"].shift(1)
    d["rest_days"] = (d["game_date"] - prev_date).dt.days.clip(upper=10).fillna(10)
    d["b2b"] = (d["rest_days"] <= 1).astype(float)
    d["games_before"] = g.cumcount()
    same = d["season"] == g["season"].shift(1)
    d["season_games_before"] = d.groupby(["PLAYER_ID", "season"]).cumcount()
    d["new_season"] = (~same).astype(float)
    mins = g["MIN"]
    d["min_last"] = mins.shift(1)
    for span in (3, 10, 40):
        d[f"min_ewm{span}"] = mins.transform(lambda s, k=span: s.shift(1).ewm(span=k, min_periods=1).mean())
    d["min_sd10"] = mins.transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    for c in RATE_COLS:
        mins_where = d["MIN"].where(d[c].notna())      # minutes only where the stat exists
        for span in (10, 40):
            num = d[c].groupby(d["PLAYER_ID"]).transform(
                lambda s, k=span: s.shift(1).ewm(span=k, min_periods=1).mean())
            den = mins_where.groupby(d["PLAYER_ID"]).transform(
                lambda s, k=span: s.shift(1).ewm(span=k, min_periods=1).mean())
            d[f"{c}_pm{span}"] = num / den.clip(lower=1)
    d["usage_pm10"] = d["FGA_pm10"] + 0.44 * d["FTA_pm10"] + d["TOV_pm10"]
    for s in STATS:
        d[f"{s}_last"] = g[s].shift(1)
    team = _team_games(d)
    d = d.merge(team, on=["GAME_ID", "team"], how="left")
    d = d.merge(_vacated(d, out_players), on=["GAME_ID", "team"], how="left")
    d = d.merge(_elo(d), on=["GAME_ID", "team"], how="left")
    opp = team.rename(columns={"team": "opp", "pace_in": "opp_pace_in"})
    opp = opp[["GAME_ID", "opp", "opp_pace_in"] + [f"allow_{s}_in" for s in STATS]].rename(
        columns={f"allow_{s}_in": f"opp_allow_{s}_in" for s in STATS})
    d = d.merge(opp, on=["GAME_ID", "opp"], how="left")
    d = d.drop(columns=[f"allow_{s}_in" for s in STATS])
    d["day_of_season"] = (d["game_date"] - d.groupby("season")["game_date"].transform("min")).dt.days
    return d.sort_values(["game_date", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


# ELO GAP: built and A/B-gated 2026-10-12 (no change in MAE or calibration in any of the
# four stats -- pace and defence already carry team strength), so it is OFF by default.
def features(stat: str, vacated: bool = True, elo: bool = False) -> list:
    base = (["vac_usage", "vac_share"] if vacated else []) + (["elo_diff", "elo_abs"] if elo else []) + ["home", "rest_days", "b2b", "games_before", "season_games_before", "new_season",
            "min_last", "min_ewm3", "min_ewm10", "min_ewm40", "min_sd10", "usage_pm10",
            "pace_in", "opp_pace_in", f"opp_allow_{stat}_in", "day_of_season", f"{stat}_last"]
    rates = [f"{c}_pm{k}" for c in RATE_COLS for k in (10, 40)]
    return base + rates


class StatModel:
    """Mean forecast (Poisson GBM) + NB dispersion for one stat."""

    def __init__(self, stat: str, seed: int = 0, vacated: bool = True, elo: bool = False):
        self.stat = stat
        self.cols = features(stat, vacated, elo)
        self.seed = seed
        self.model = None
        self.disp = None

    def _gbm(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, min_samples_leaf=200,
                                             l2_regularization=1.0, random_state=self.seed)

    def fit(self, train: pd.DataFrame):
        train = train[(train["games_before"] >= 3) & ~train["is_next"] & train[self.stat].notna()]
        x, y = train[self.cols], train[self.stat].astype(float)
        # Dispersion from OUT-OF-FOLD means (2 folds by season), never from the rows
        # a model was fitted on -- in-sample residuals are too small and would make
        # every tail look tighter than it is.
        seasons = sorted(train["season"].unique())
        half = set(seasons[len(seasons) // 2:])
        oof = np.full(len(train), np.nan)
        for hold in (half, set(seasons) - half):
            mask = train["season"].isin(hold).to_numpy()
            if mask.all() or (~mask).all():
                continue
            m = self._gbm().fit(x[~mask], y[~mask])
            oof[mask] = m.predict(x[mask])
        ok = ~np.isnan(oof)
        self.disp = dist.Dispersion().fit(oof[ok], y.to_numpy()[ok])
        self.model = self._gbm().fit(x, y)
        return self

    def mean(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(rows[self.cols]), 0.05, None)

    def p_over(self, rows: pd.DataFrame, lines) -> np.ndarray:
        mu = self.mean(rows)
        return dist.p_over(mu, self.disp(mu), lines)
