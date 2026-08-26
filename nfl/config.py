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
