"""Seasons, markets and the constants the NFL engine is tuned by."""

# Six full regular seasons, as asked. 2019 exists upstream but 2020 was played
# without crowds in most stadiums, which is where the home-field estimate starts
# behaving normally again -- taking 2020-2025 keeps the window contiguous and
# recent rather than reaching back for one more year of thinner relevance.
# EIGHT seasons loaded, FOUR scored. 2018-2021 are burn-in and are never scored;
# the released numbers still come from 2022-2025 exactly as before. Widening the
# window is not a change to the evaluation, it is more evidence for the model to
# learn from before facing it -- passing yards trains on ~500 quarterback-games a
# season, and doubling that is the cheapest honest gain available.
SEASONS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
CURRENT_SEASON = 2026

# The first two seasons establish player roles and opponent history and are never
# scored: a player's "historical rate" has to come from somewhere, and scoring a
# fold whose features were built from nothing measures the warm-up, not the model.
BURN_IN_SEASONS = 4

# REGULAR SEASON ONLY. Preseason is played by men who will not be on the roster in
# week 1 and its snap counts are meaningless; playoff samples are tiny and the
# field is self-selected. Both would corrupt the rates.
SEASON_TYPE = "REG"

# The markets this engine publishes, and the stat each settles on.
MARKETS = {
    "anytime_touchdown": "touchdowns",
    "receiving_yards": "receiving_yards",
    "rushing_yards": "rushing_yards",
    "passing_yards": "passing_yards",
}

# A market must clear ALL of these on the held-out seasons or it is withheld.
# Deliberately the same shape as the soccer side's release policy: beat the
# baseline, stay calibrated, and do it in EVERY season rather than on average --
# a market that works in two seasons and fails in the third is not a market, it
# is a coin that landed twice.
MIN_PREDICTIONS_TOTAL = 1000
MIN_PREDICTIONS_PER_SEASON = 400
MAX_ECE = 0.04

# Form windows. Five is what the board displays and what the user reads as "last
# five", so the model uses the same number -- a model whose form window disagrees
# with the one on screen is explaining itself with numbers it did not use.
FORM_GAMES = 5
LONG_FORM_GAMES = 10
OPPONENT_GAMES = 8

# Minimum games before a player is publishable at all. Below this his "historical
# rate" is one or two performances and the projection is noise wearing a number.
MIN_GAMES_FOR_PROP = 6

# A player needs a real ROLE in a market before he can be quoted in it, measured
# on his own prior median opportunity. Without this, a wide receiver whose median
# is zero carries acquires a rushing line of 0.5 yards that he clears only when
# he happens to get a handoff -- 22,993 such rows dragged the rushing market's
# base rate to 26% and would have handed the model free accuracy for saying
# "under" to players who were never going to run. These are the thresholds a
# sportsbook effectively applies by only quoting players with a job.
MIN_OPPORTUNITY = {
    "anytime_touchdown": 3.0,     # touches a game
    "receiving_yards": 2.0,       # targets a game
    "rushing_yards": 4.0,         # carries a game
    "passing_yards": 15.0,        # pass attempts a game
}

# A yards market needs a line worth quoting. Without this the board's top picks
# were fringe tight ends at "over 0.5 receiving yards" -- trivially likely, utterly
# uninformative, and not a bet any sportsbook offers. The backtest applies the
# same floor, because a gate that validates lines the board will never publish is
# measuring a different product from the one on screen.
MIN_LINE = {
    "receiving_yards": 15.0,
    "rushing_yards": 20.0,
    "passing_yards": 150.0,
}

# A player must have appeared in the most recent completed season to be projected.
# Attribution follows a player's last appearance, so without this the week 1 board
# filled up with men who retired years ago -- Alfred Blue, C.J. Anderson and Colt
# McCoy were all being projected onto 2026 fixtures from their final seasons.
ACTIVE_WITHIN_SEASONS = 1


# HOW FAR DOWN THE DEPTH CHART A MARKET STAYS CREDIBLE.
#
# `pos_rank` is a player's rank WITHIN his position slot, so 1 is the starter at
# that spot. These caps close the hole the board used to state and not fix: "a
# backup quarterback carries a low line and can top a market he may not play in."
#
# Measured on the 2026 week 1 board before the gate existed. Six of nineteen
# passing picks were BACKUP quarterbacks -- Marcus Mariota (WAS QB2) was the
# second-highest passing pick on the whole board at 62%, and Cleveland published
# its QB2 and QB3 while its starter appeared nowhere. Their lines sit at 150-180
# yards precisely because they have only ever played in relief, which makes "over"
# look easy right up until they take no snap at all.
#
# PASSING IS 1 BECAUSE THE MARKET IS WINNER-TAKE-ALL. One quarterback takes
# essentially every drop-back, so a QB2 is not a smaller share of the market, he
# is usually none of it. The other three markets genuinely share: a WR3 and an RB2
# play real snaps, so they stay, and only deep reserves are cut. Rank 4+ removed
# Mack Hollins (WR4) at 74.7%, then the highest receiving pick on the board.
#
# This REMOVES a player rather than flagging him, for the same reason a player
# reported OUT is removed: his last five games look exactly as good as anyone's
# right up until he is inactive.
MAX_DEPTH_RANK = {
    "anytime_touchdown": 3,
    "receiving_yards": 3,
    "rushing_yards": 3,
    "passing_yards": 1,
}

# The depth chart may only overrule the board if it describes the same league the
# board does -- the lesson of rosters.corroborates, where a file with 43-71
# plausible names a team passed every count-based check while omitting Patrick
# Mahomes. Below this share of the board's own players, the chart is ignored
# entirely and the gate is reported as not applied.
MIN_DEPTH_COVERAGE = 0.80
