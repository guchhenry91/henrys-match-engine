"""Per-market model inputs, built strictly from what was known BEFORE each game.

Same construction as nba/features.py -- every window is ENTERING (shift(1)), the
line is built from the player's own entering median, and the model is trained
across a SPREAD of lines around it (augment_lines) so it can be asked about a
bookmaker's number. Column names match nfl/model.py so its PropModel is reused.

The "player" is whoever the market is about: a starting batter, a starting
pitcher, or -- for the team-total market -- the team itself.
"""
import numpy as np
import pandas as pd

from mlb import config


def _shifted(frame, column, fn):
    return (frame.groupby("PLAYER_ID")[column]
            .apply(lambda s: fn(s.shift(1)))
            .reset_index(level=0, drop=True))


def _opponent_allowed(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """What the opponent has conceded of this stat per game, entering the game."""
    per_game = (frame.groupby(["season", "opponent", "GAME_ID", "game_date"])[stat]
                .sum().reset_index().sort_values(["game_date", "GAME_ID"]))
    per_game["allowed"] = (per_game.groupby(["season", "opponent"])[stat]
                           .apply(lambda s: s.shift(1).expanding().mean())
                           .reset_index(level=[0, 1], drop=True))
    return per_game[["season", "opponent", "GAME_ID", "allowed"]]


PARK_SEASONS = 3        # seasons of history a park factor is measured over
PARK_SHRINK_GAMES = 81  # pull toward neutral with a season's worth of games


def park_factors(rows: pd.DataFrame, stat: str) -> pd.DataFrame:
    """How much each home park inflates or suppresses `stat`, from PRIOR seasons only.

    PHASE 3: what books are slow on. Coors Field adds runs and homers, Petco and
    Oracle take them away. For each team, the factor is its home games' per-game
    total of `stat` (both sides) over its road games' total, pooled across the
    PARK_SEASONS seasons BEFORE the one being priced and shrunk toward 1.0 by
    PARK_SHRINK_GAMES -- so a new or rebuilt park starts neutral rather than at a
    small-sample extreme. Retrosheet carries no park key on these rows; the home
    team stands in for its park (neutral-site games are a handful a season).

    Returns home_team, season, park_pf.
    """
    r = rows[["GAME_ID", "season", "TEAM", "opponent", "is_home", stat]].copy()
    r["home_team"] = np.where(r["is_home"] >= 0.5, r["TEAM"], r["opponent"])
    r["away_team"] = np.where(r["is_home"] >= 0.5, r["opponent"], r["TEAM"])
    g = (r.groupby(["GAME_ID", "season", "home_team", "away_team"])[stat].sum()
         .reset_index())
    home = g.groupby(["home_team", "season"])[stat].agg(["sum", "count"]).reset_index()
    home.columns = ["team", "season", "h_sum", "h_n"]
    road = g.groupby(["away_team", "season"])[stat].agg(["sum", "count"]).reset_index()
    road.columns = ["team", "season", "r_sum", "r_n"]
    per = home.merge(road, on=["team", "season"], how="outer").fillna(0.0)
    out = []
    for season in sorted(r["season"].unique()):
        prior = per[(per["season"] < season) & (per["season"] >= season - PARK_SEASONS)]
        agg = prior.groupby("team")[["h_sum", "h_n", "r_sum", "r_n"]].sum()
        for team, a in agg.iterrows():
            if a["h_n"] <= 0 or a["r_n"] <= 0 or a["r_sum"] <= 0:
                continue
            raw = (a["h_sum"] / a["h_n"]) / (a["r_sum"] / a["r_n"])
            w = a["h_n"] / (a["h_n"] + PARK_SHRINK_GAMES)
            out.append((team, season, 1.0 + (raw - 1.0) * w))
    return pd.DataFrame(out, columns=["home_team", "season", "park_pf"])


def build(rows: pd.DataFrame, market: str) -> pd.DataFrame:
    spec = config.MARKETS[market]
    stat, opp, kind = spec["stat"], spec["opp"], spec["kind"]
    frame = rows[rows[stat].notna() & rows[opp].notna()].copy()
    # The park this game is played in, and how it treats this stat (park_factors).
    frame["home_team"] = np.where(frame["is_home"] >= 0.5, frame["TEAM"], frame["opponent"])
    frame = frame.merge(park_factors(frame, stat), on=["home_team", "season"], how="left")
    frame["park_pf"] = frame["park_pf"].fillna(1.0)
    frame = frame.sort_values(["PLAYER_ID", "game_date", "GAME_ID"]).reset_index(drop=True)

    frame["games_before"] = frame.groupby("PLAYER_ID").cumcount()
    frame["hist_rate"] = _shifted(frame, stat, lambda s: s.expanding().mean())
    frame["form5"] = _shifted(frame, stat, lambda s: s.rolling(config.FORM_GAMES, min_periods=1).mean())
    frame["form10"] = _shifted(frame, stat, lambda s: s.rolling(config.FORM_LONG, min_periods=1).mean())
    frame["min5"] = _shifted(frame, opp, lambda s: s.rolling(config.FORM_GAMES, min_periods=1).mean())
    frame["_eff"] = (frame[stat] / frame[opp].replace(0, np.nan)).fillna(0.0)
    frame["eff5"] = _shifted(frame, "_eff", lambda s: s.rolling(config.FORM_GAMES, min_periods=1).mean())
    team_total = frame.groupby(["GAME_ID", "TEAM"])[stat].transform("sum")
    frame["_share"] = (frame[stat] / team_total.replace(0, np.nan)).fillna(0.0)
    frame["share5"] = _shifted(frame, "_share", lambda s: s.rolling(config.FORM_GAMES, min_periods=1).mean()).fillna(0.0)
    rest = frame.groupby("PLAYER_ID")["game_date"].diff().dt.days
    frame["rest_days"] = rest.fillna(7.0).clip(0, 14)

    allowed = _opponent_allowed(frame, stat)
    frame = frame.merge(allowed, on=["season", "opponent", "GAME_ID"], how="left")
    frame["opp_allowed"] = frame["allowed"]
    frame["opp5"] = frame["opp_allowed"].fillna(frame["opp_allowed"].mean())

    # The line: integer median + 0.5, so every line is on the half and no result
    # can land on it. Floored and capped at what a book actually quotes.
    median = _shifted(frame, stat, lambda s: s.expanding().median())
    frame["base_line"] = np.floor(median + 0.5) - 0.5
    frame["line"] = frame["base_line"].clip(lower=config.MIN_LINE[market],
                                            upper=config.MAX_LINE[market])
    frame["outcome"] = (frame[stat] > frame["line"]).astype(float)

    frame = frame[frame["games_before"] >= config.MIN_GAMES[kind]]
    frame = frame[frame["min5"] >= config.MIN_OPPORTUNITY[kind]]
    frame = frame.dropna(subset=["hist_rate", "form5", "form10", "line"])
    return frame.sort_values(["game_date", "GAME_ID"]).reset_index(drop=True)


def augment_lines(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    """Each game asked against a SPREAD of lines; the outcome re-settled at each.
    A step outside what a book quotes (MIN_LINE..MAX_LINE) is dropped, never
    clipped, so the floor cannot manufacture duplicate rows."""
    if frame.empty:
        return frame
    stat = config.MARKETS[market]["stat"]
    parts = []
    for step in config.LINE_STEPS[market]:
        part = frame.copy()
        part["line"] = (part["base_line"] + step).clip(lower=config.MIN_LINE[market]) \
            if step == 0 else part["base_line"] + step
        part = part[(part["line"] >= config.MIN_LINE[market])
                    & (part["line"] <= config.MAX_LINE[market])]
        part["outcome"] = (part[stat] > part["line"]).astype(float)
        part["line_step"] = step
        parts.append(part)
    # ONE ROW PER GAME PER LINE. Where a median sits below the floor (most
    # batters' RBI median is 0), the floored step-0 line and the step-1 line are
    # the SAME 0.5 -- without this every such game was counted twice.
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["PLAYER_ID", "GAME_ID", "line"]).reset_index(drop=True)


def at_line(frame: pd.DataFrame, line) -> pd.DataFrame:
    out = frame.copy()
    out["line"] = line
    return out
