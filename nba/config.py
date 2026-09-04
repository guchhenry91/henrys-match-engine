"""NBA engine configuration.

A third sport, in its own package for the same reason the NFL got one: nothing
about a Dixon-Coles model over two low-count goal processes survives translation
to a sport where both sides score 110 points. What IS shared is the discipline --
walk-forward validation on seasons the model never saw, a release gate that
WITHHOLDS rather than caveats, and evidence published beside the picks.

THE SOURCE IS stats.nba.com's leaguegamelog, one request per season per side.
That was chosen over the flat-file alternative after checking both: hoopR's CSV
releases are free and fast but stop at season 2023, three years short, so they
cannot see the 2025-26 season at all. The official endpoint returns an entire
season -- 26,651 player rows, 2,460 team rows -- in a single request, covers the
current season, and reaches back two decades. Fifteen seasons therefore cost
thirty requests, cached to disk.
"""

# FIFTEEN SEASONS, ending with the one that just finished. NBA seasons are named
# by their end year here (2026 == 2025-26, which ran October 2025 to June 2026).
SEASONS = tuple(range(2012, 2027))
CURRENT_SEASON = 2026

# 2011-12 was the LOCKOUT season: 66 games a team, not 82, so its 1,980 team rows
# are correct rather than a truncated download. Noted because a row-count sanity
# check that assumes 2,460 would flag the one season that is legitimately short.
SHORT_SEASONS = {2012: "lockout, 66 games a team"}

# The seasons the gate never scores. The model needs history before its first
# graded prediction, and grading a season it partly trained on measures nothing.
BURN_IN_SEASONS = 4

# HOW MANY SEASONS OF HISTORY A PROP MODEL MAY TRAIN ON.
#
# THIS IS WHY POINTS USED TO BE WITHHELD, and the reason is drift, not a broken
# market. Training on ALL prior seasons anchors the model to a decade-long
# average scoring environment, and the NBA's has moved: the points over-rate rose
# from 0.461 in 2012 to 0.555 by 2019 as the line -- a player's own expanding
# CAREER median -- lagged further and further behind what he was actually
# scoring. Measured on the walk-forward, the model predicted 39.9% overs in 2024
# when 53.2% landed, a 13-point miscalibration, and lost to the baseline in 2023,
# 2024 and 2025 for exactly that reason.
#
# The tell was that 2026 recovered on its own (bias -0.5pp): by then the training
# set finally contained the high-scoring seasons. A market that fixes itself once
# history catches up was never a bad market, it was a late one.
#
# FIVE, not three, and the distinction matters. Windows of 3 and 5 BOTH clear the
# gate on all four markets, and 3 scores better on every one. Five is chosen
# because when two options both pass, taking the one with more data is the choice
# that is not reaching for the scoreboard -- picking the window that maximises
# performance on the very seasons the gate scores is how a selector overfits a
# gate, which is the same failure `nfl.model` avoids by averaging its candidate
# feature sets instead of selecting among them.
#
# Effect, measured across all four markets (failing seasons, all-history -> 5):
#   points   3 -> 0     rebounds 0 -> 0
#   assists  0 -> 0     threes   0 -> 0
# Assists' worst season also improves (+0.0080 -> +0.0114); rebounds and threes
# are unchanged. Nothing was traded away to fix points.
#
# NOT applied to team_winner: `games_backtest` fits an Elo, which is already
# recency-weighted by construction through its k-factor and per-season regression
# to the mean, and it beats home court in every one of the eleven scored seasons.
TRAIN_SEASONS = 5

# The four markets, mapped to the column that settles them. Points, rebounds and
# assists are the three every book quotes; threes made is the fourth because it is
# the one genuinely different shape -- a low-count count, not a continuous total.
MARKETS = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threes": "FG3M",
}

# A LINE WORTH QUOTING. Same reasoning as the NFL's MIN_LINE: nobody offers "over
# 0.5 rebounds", and a floor that exists in the board but not the backtest would
# measure a different product from the one on screen. These are the thresholds a
# book effectively applies by only pricing players with a role.
MIN_LINE = {
    "points": 7.5,
    "rebounds": 3.5,
    "assists": 2.5,
    "threes": 0.5,
}

# WHERE THE LINE SITS RELATIVE TO THE PLAYER'S OWN ROUNDED MEDIAN.
#
# MEASURED, not chosen. The aim is a line that splits his own history near 50/50,
# so neither side is free -- the same reason the NFL engine quotes the median
# rather than the mean. But basketball props are DISCRETE COUNTS, and a half-point
# line cannot always straddle a small integer: for a player whose median is 4
# rebounds, 4.5 needs five and 3.5 needs four, and nothing in between exists.
#
# Base rates over 30,561 rows of 2024-25 and 2025-26, by offset:
#
#     market     -0.5    0.0    +0.5
#     points     0.540  0.501  0.494
#     rebounds   0.487  0.410  0.402
#     assists    0.434  0.367  0.361
#     threes     0.565  0.364  0.354
#
# So the offset differs by market because the distributions do. Points is nearly
# continuous at these volumes and lands on 0.494 at +0.5; the three count markets
# are pushed well under 50% by the same offset, because adding half a point to an
# integer median means "over" requires median PLUS ONE.
#
# ASSISTS (0.434) AND THREES (0.565) ARE THE BEST AVAILABLE, NOT 0.50. That is
# discreteness, not a defect, and it is stated rather than tuned away: forcing
# them closer would mean quoting lines no book offers.
LINE_OFFSET = {
    "points": +0.5,
    "rebounds": -0.5,
    "assists": -0.5,
    "threes": -0.5,
}

# A ROLE, NOT AN APPEARANCE. Minutes are the opportunity that every one of these
# markets runs through, so a player has to be playing real ones before his line
# means anything. Without it the board fills with benchwarmers whose entering
# median is zero and who clear a floored line only when they happen to get twenty
# minutes in a blowout.
MIN_MINUTES = 20.0

# Games a player must already have this season before he is projected at all.
MIN_GAMES_FOR_PROP = 8

# Release gate. Deliberately the same shape as the NFL's: beat the baseline, stay
# calibrated, and do it in EVERY scored season rather than on average -- a market
# that works in twelve seasons and fails in three is not a market.
MIN_PREDICTIONS_TOTAL = 2000
MIN_PREDICTIONS_PER_SEASON = 500
MAX_ECE = 0.04

# Form windows.
FORM_GAMES = 5
FORM_LONG = 10
