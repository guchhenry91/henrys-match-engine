"""Per-market model inputs, built strictly from what was known BEFORE each game.

EVERY WINDOW IS ENTERING. `shift(1)` before every expanding or rolling statistic,
so a row never sees its own result. That is the whole difference between a
backtest and a description of the past, and it is easy to lose: a rolling mean
computed without the shift reports a model that knows the answer and scores about
eight points better for it.

THE LINE IS THE PLAYER'S OWN ENTERING MEDIAN, quoted on the half. Same decision
the NFL engine made and for the same measured reason: these distributions are
right-skewed, so a line at the MEAN is beaten well under half the time, and a
model looks clever simply by saying "under" against a baseline flattered the same
way. The median splits his own history near 50/50, which is what a book is trying
to do. Half-points mean no result can land exactly on a line, so there is never a
push to settle.

FEATURE NAMES MATCH nfl/model.py DELIBERATELY. That model is a logistic fit
blended with an empirical baseline, and none of it is football-specific; building
the same column names lets this engine reuse it instead of keeping a second copy
of an ensemble, a cross-fitted calibrator and an isotonic/Platt switch that would
then drift apart.
"""
import numpy as np
import pandas as pd

from nba import config


def _prior_mean(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (frame.groupby("PLAYER_ID")[column]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True))


def _prior_expanding_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame.groupby("PLAYER_ID")[column]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True))


def _prior_median(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame.groupby("PLAYER_ID")[column]
            .apply(lambda s: s.shift(1).expanding().median())
            .reset_index(level=0, drop=True))


def _opponent_allowed(frame: pd.DataFrame, stat: str) -> pd.Series:
    """How much of this stat the opponent has been giving up, entering the game.

    Built from team-game totals rather than per-player rows so it measures the
    DEFENCE rather than whichever players happened to face it.
    """
    per_game = (frame.groupby(["season", "opponent", "GAME_ID"])[stat]
                .sum().reset_index()
                .sort_values("GAME_ID"))
    per_game["allowed"] = (per_game.groupby(["season", "opponent"])[stat]
                           .apply(lambda s: s.shift(1).expanding().mean())
                           .reset_index(level=[0, 1], drop=True))
    return per_game[["season", "opponent", "GAME_ID", "allowed"]]


def build(player_games: pd.DataFrame, market: str) -> pd.DataFrame:
    """One row per player-game that is eligible for `market`, with its outcome."""
    stat = config.MARKETS[market]
    frame = player_games.copy()
    frame = frame[frame["MIN"].notna() & frame[stat].notna()]
    frame = frame.sort_values(["PLAYER_ID", "game_date"]).reset_index(drop=True)

    frame["games_before"] = frame.groupby("PLAYER_ID").cumcount()
    frame["hist_rate"] = _prior_expanding_mean(frame, stat)
    frame["form5"] = _prior_mean(frame, stat, config.FORM_GAMES)
    frame["form10"] = _prior_mean(frame, stat, config.FORM_LONG)
    frame["min5"] = _prior_mean(frame, "MIN", config.FORM_GAMES)

    # EFFICIENCY, SEPARATED FROM MINUTES. Twelve points in 20 minutes and twelve
    # in 38 are not the same bet, and a five-game average of the total cannot tell
    # them apart -- it blends the rate with the opportunity.
    per_minute = frame[stat] / frame["MIN"].replace(0, np.nan)
    frame["_eff"] = per_minute.fillna(0.0)
    frame["eff5"] = _prior_mean(frame, "_eff", config.FORM_GAMES)

    # Share of his own team's production, entering.
    team_total = frame.groupby(["GAME_ID", "TEAM_ABBREVIATION"])[stat].transform("sum")
    frame["_share"] = frame[stat] / team_total.replace(0, np.nan)
    frame["share5"] = _prior_mean(frame, "_share", config.FORM_GAMES).fillna(0.0)

    rest = frame.groupby("PLAYER_ID")["game_date"].diff().dt.days
    # A first game has no rest history. Filled with a week rather than zero, which
    # would read as a back-to-back and is the opposite of the truth.
    frame["rest_days"] = rest.fillna(7.0).clip(0, 14)

    allowed = _opponent_allowed(frame, stat)
    frame = frame.merge(allowed, on=["season", "opponent", "GAME_ID"], how="left")
    frame["opp_allowed"] = frame["allowed"]
    frame["opp5"] = frame["opp_allowed"].fillna(frame["opp_allowed"].mean())

    # THE LINE, on the half point, floored by what a book would actually quote.
    #
    # THE MEDIAN IS ROUNDED TO A WHOLE NUMBER FIRST, and that is a fix rather than
    # a detail. It used to be rounded to the nearest HALF (`(median*2).round()/2`),
    # so a median of 12.5 plus the +0.5 offset produced a line of 13.0 -- an
    # INTEGER. 9,625 points lines (4.1%) came out whole, and on a whole line a
    # result can land exactly on it: 528 rows scored exactly their line and every
    # one was graded a LOSS, because `outcome` is `stat > line`. The module
    # docstring's claim that "there is never a push to settle" was false, and
    # test_lines_are_always_on_the_half_point passed only because its synthetic
    # medians happened to be whole.
    #
    # An integer median plus a half-point offset is always a half-point line, so
    # the tie cannot arise. The bias was small -- 0.23% of rows, all in the same
    # direction -- but it was a silent thumb on the scale against every over.
    median = _prior_median(frame, stat)
    frame["line"] = np.floor(median + 0.5) + config.LINE_OFFSET[market]
    frame["line"] = frame["line"].clip(lower=config.MIN_LINE[market])
    frame["outcome"] = (frame[stat] > frame["line"]).astype(float)

    # A ROLE, not an appearance. See config.MIN_MINUTES.
    frame = frame[frame["games_before"] >= config.MIN_GAMES_FOR_PROP]
    frame = frame[frame["min5"] >= config.MIN_MINUTES]
    frame = frame.dropna(subset=["hist_rate", "form5", "form10", "line"])
    return frame.sort_values("game_date").reset_index(drop=True)
