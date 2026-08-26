"""Strictly pre-game features. Nothing here may see the game it describes.

THE ONLY BUG THAT MATTERS IN THIS FILE IS LEAKAGE. A feature that accidentally
includes the target game makes every backtest number beautiful and every live
prediction worthless, and it does not announce itself -- the report just looks
better than it should. So every rolling window here is `.shift(1)` FIRST and
aggregated second, and the tests assert that a player's own game cannot reach his
own features.

The "line" for a yards market is the player's OWN entering average, not a
sportsbook number. That is a real limitation and is stated on the board: we are
asking "will he beat his own recent standard?", which is a well-posed question
the data can answer, not "will he beat the book?", which needs a price we do not
have. When a live line exists it should replace this.
"""
import numpy as np
import pandas as pd

from nfl import config

# What each market settles on, and the opportunity stat that drives it.
MARKET_STAT = dict(config.MARKETS)
OPPORTUNITY = {
    "anytime_touchdown": "touches",
    "receiving_yards": "targets",
    "rushing_yards": "carries",
    "passing_yards": "attempts",
}
# Only these positions are credible in each market. Without this a punter with one
# freak carry acquires a rushing-yards projection, and a lineman who caught a
# tipped pass becomes a receiving prop.
ELIGIBLE = {
    "anytime_touchdown": {"RB", "WR", "TE", "FB", "QB"},
    "receiving_yards": {"WR", "TE", "RB", "FB"},
    "rushing_yards": {"RB", "QB", "FB", "WR"},
    "passing_yards": {"QB"},
}


def _prior_mean(frame, column, window=None):
    """Mean of `column` over a player's PREVIOUS games. Never the current one."""
    grouped = frame.groupby("player_id", sort=False)[column]
    shifted = grouped.shift(1)                    # drop the current game first
    if window is None:
        return shifted.groupby(frame["player_id"], sort=False).expanding().mean() \
                      .reset_index(level=0, drop=True)
    return shifted.groupby(frame["player_id"], sort=False) \
                  .rolling(window, min_periods=1).mean() \
                  .reset_index(level=0, drop=True)


def _prior_median(frame, column):
    """Median of a player's PREVIOUS games. Never the current one.

    The MEDIAN, not the mean, sets the line. Yardage is heavily right-skewed -- one
    80-yard game drags a receiver's average above the score he posts in a typical
    week -- so a line at the mean is beaten only about 36% of the time. That is a
    market a model can look clever in by simply always saying "under", and a
    baseline that flatters itself for the same reason. At the median the question
    is genuinely 50/50 and the model has to actually know something. It is also
    what a sportsbook does, so the line a reader sees resembles one they could bet.
    """
    shifted = frame.groupby("player_id", sort=False)[column].shift(1)
    return (shifted.groupby(frame["player_id"], sort=False)
                   .expanding().median().reset_index(level=0, drop=True))


def _prior_count(frame):
    """How many games a player has already played. Never counts the current one."""
    return frame.groupby("player_id", sort=False).cumcount()


def _opponent_allowance(frame, stat):
    """What this opponent has allowed in `stat` over its previous games.

    Aggregated per defence per week first, then rolled -- rolling the raw player
    rows would weight a week by how many players happened to record a stat in it.
    """
    per_week = (frame.groupby(["opponent_team", "season", "week"], as_index=False)[stat]
                     .sum().sort_values(["opponent_team", "season", "week"]))
    shifted = per_week.groupby("opponent_team", sort=False)[stat].shift(1)
    per_week["opp_allowed"] = (shifted.groupby(per_week["opponent_team"], sort=False)
                               .rolling(config.OPPONENT_GAMES, min_periods=1).mean()
                               .reset_index(level=0, drop=True))
    return frame.merge(per_week[["opponent_team", "season", "week", "opp_allowed"]],
                       on=["opponent_team", "season", "week"], how="left")


def last_five(frame, stat):
    """The player's last five values of `stat`, oldest first, per row.

    This is what the board shows. It is built from the SAME shifted history the
    model uses, so the five numbers a reader sees are literally the five the
    projection was computed from -- a board that displays one form window while
    the model used another is explaining itself with numbers it never saw.
    """
    out = []
    for _, group in frame.groupby("player_id", sort=False):
        values = group[stat].tolist()
        for i in range(len(values)):
            out.append(values[max(0, i - config.FORM_GAMES):i])
    return out


def attach_game_context(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Home/away and days of rest, from the schedule.

    Missing from the first build, and it showed: passing yards could not beat a
    five-game-form baseline without knowing whether the quarterback was at home or
    how long since he last played. Both are known the moment the fixture is, so
    neither can leak.

    The betting line is deliberately NOT used. spread_line and total_line are in
    the same file and would predict passing volume well -- a high total means a
    shootout -- but that makes the model a reader of the market rather than a
    check on it, which is the opposite of the point.
    """
    home = games[["season", "week", "home_team", "gameday"]].rename(
        columns={"home_team": "team"})
    home["is_home"] = 1.0
    away = games[["season", "week", "away_team", "gameday"]].rename(
        columns={"away_team": "team"})
    away["is_home"] = 0.0
    schedule = pd.concat([home, away], ignore_index=True)

    out = frame.merge(schedule, on=["season", "week", "team"], how="left")
    out["is_home"] = out["is_home"].fillna(0.5)      # unknown -> neutral, never a guess

    out = out.sort_values(["player_id", "season", "week"])
    previous = out.groupby("player_id", sort=False)["gameday"].shift(1)
    rest = (out["gameday"] - previous).dt.days
    # A normal week is 7 days. Clipped because an offseason gap is not "rest", it
    # is a different season, and a 200-day number would dominate a scaled feature.
    out["rest_days"] = rest.clip(3, 21).fillna(7.0)
    return out.reset_index(drop=True)


def opportunity_share(player_weeks: pd.DataFrame, opportunity: str) -> pd.Series:
    """A player's share of his own team's volume in that game.

    The signal both weak markets were missing. Ten carries means one thing on a
    team that ran forty times and something else entirely on a team that ran
    twelve -- the raw count cannot tell a lead back from a committee, and the
    committee is where rushing projections go wrong. Share separates them.

    Computed on the raw game, then lagged like everything else: the share used for
    a game is always the share he had BEFORE it.
    """
    totals = (player_weeks.groupby(["team", "season", "week"])[opportunity]
              .transform("sum"))
    return (player_weeks[opportunity] / totals.replace(0, np.nan)).fillna(0.0)


def team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's prior scoring and conceding rate, per game, strictly lagged.

    GAME SCRIPT is what drives passing volume, and it is the thing a quarterback's
    five-game form cannot see. A team with a leaky defence spends the second half
    behind and throwing; a team that leads runs the clock out. Points scored and
    allowed going INTO the game are the cheapest honest proxy for that, and both
    are known before kickoff.

    Deliberately not the betting spread or total, which live in the same file and
    would predict this better. Reading the market is not the same as modelling the
    game, and a market-derived feature makes the board a mirror rather than a check.
    """
    home = games[["season", "week", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "allowed"})
    away = games[["season", "week", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "allowed"})
    rows = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "season", "week"])
    for column in ("scored", "allowed"):
        shifted = rows.groupby("team", sort=False)[column].shift(1)
        rows[f"team_{column}5"] = (shifted.groupby(rows["team"], sort=False)
                                   .rolling(config.FORM_GAMES, min_periods=1).mean()
                                   .reset_index(level=0, drop=True))
    return rows[["season", "week", "team", "team_scored5", "team_allowed5"]]


def build(player_weeks: pd.DataFrame, market: str, games: pd.DataFrame = None) -> pd.DataFrame:
    """Per-player-game rows with pre-game features and the settled outcome."""
    stat = MARKET_STAT[market]
    opportunity = OPPORTUNITY[market]
    frame = player_weeks[player_weeks["position"].isin(ELIGIBLE[market])].copy()
    frame = frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Share must be computed against the FULL squad, before the position filter --
    # a running back's share of his team's carries is meaningless if the other
    # backs have already been filtered out of the denominator.
    share_source = player_weeks.copy()
    share_source["_share"] = opportunity_share(share_source, opportunity)
    frame = frame.merge(
        share_source[["player_id", "season", "week", "_share"]],
        on=["player_id", "season", "week"], how="left")
    frame["_share"] = frame["_share"].fillna(0.0)
    frame = frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    frame["games_before"] = _prior_count(frame)
    frame["share5"] = _prior_mean(frame, "_share", config.FORM_GAMES)
    frame["hist_rate"] = _prior_mean(frame, stat)
    frame["form5"] = _prior_mean(frame, stat, config.FORM_GAMES)
    frame["form10"] = _prior_mean(frame, stat, config.LONG_FORM_GAMES)
    frame["opp5"] = _prior_mean(frame, opportunity, config.FORM_GAMES)

    # EFFICIENCY, separated from volume. Yards are opportunity times efficiency,
    # and a five-game yardage average silently blends the two -- a quarterback
    # throwing 40 times for 6.0 a go and one throwing 28 times for 8.6 look
    # identical in it, and they are not the same bet. Yards per attempt, per
    # carry, per target isolates the half that form cannot see.
    per_unit = frame[stat] / frame[opportunity].replace(0, np.nan)
    frame["_eff"] = per_unit.fillna(0.0)
    frame["eff5"] = _prior_mean(frame, "_eff", config.FORM_GAMES)
    frame["last_five"] = last_five(frame, stat)

    frame = _opponent_allowance(frame, stat)

    # THE LINE. For a binary market there is none -- the outcome is the event. For
    # a yards market it is the player's own entering MEDIAN, so "over" means "a
    # better day than his typical one". Quoted on the half yard the way a book
    # does, which also stops a line landing exactly on a whole-yard result and
    # leaving the outcome ambiguous.
    if market == "anytime_touchdown":
        frame["line"] = np.nan
        frame["outcome"] = (frame[stat] > 0).astype(float)
    else:
        frame["median"] = _prior_median(frame, stat)
        frame["line"] = (frame["median"] * 2).round() / 2 + 0.5
        frame["outcome"] = (frame[stat] > frame["line"]).astype(float)

    # A ROLE, not just an appearance. See config.MIN_OPPORTUNITY.
    frame["opp_median"] = _prior_median(frame, opportunity)
    frame = frame[frame["games_before"] >= config.MIN_GAMES_FOR_PROP].copy()
    frame = frame[frame["opp_median"] >= config.MIN_OPPORTUNITY[market]]
    if market != "anytime_touchdown":
        frame = frame[frame["line"].notna()
                      & (frame["line"] >= config.MIN_LINE[market])]
    if games is not None:
        frame = attach_game_context(frame, games)
        form = team_form(games)
        frame = frame.merge(form, on=["season", "week", "team"], how="left")
        opponent = form.rename(columns={"team": "opponent_team",
                                        "team_scored5": "opp_scored5",
                                        "team_allowed5": "opp_allowed5"})
        frame = frame.merge(opponent, on=["season", "week", "opponent_team"], how="left")
        for column in ("team_scored5", "team_allowed5", "opp_scored5", "opp_allowed5"):
            frame[column] = frame[column].fillna(frame[column].mean())
    else:
        frame["is_home"] = 0.5
        frame["rest_days"] = 7.0
        for column in ("team_scored5", "team_allowed5", "opp_scored5", "opp_allowed5"):
            frame[column] = 0.0
    frame["market"] = market
    return frame.reset_index(drop=True)
