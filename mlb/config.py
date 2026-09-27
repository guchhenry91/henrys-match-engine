"""MLB engine configuration.

A fourth sport, in its own package for the same reason as the NFL and NBA: the
soccer model does not translate, but the DISCIPLINE does -- walk-forward on
seasons the model never saw, a release gate that withholds rather than caveats,
and evidence published beside the picks.

DATA: Retrosheet's game-level CSVs (batting, pitching, gameinfo, teamstats), one
zip per season, free. Player-by-game lines for every regular-season game.
"""

# NINETEEN seasons load, FIFTEEN are scored: every scored season is predicted by
# a model trained only on the seasons before it, so the first four are history
# the gate never grades. Seasons are named by calendar year.
SEASONS = tuple(range(2007, 2026))
BURN_IN_SEASONS = 4
SCORED_SEASONS = SEASONS[BURN_IN_SEASONS:]          # 2011-2025, fifteen seasons

# 2020 was 60 games a team -- correct, not a truncated download.
SHORT_SEASONS = {2020: "pandemic season, 60 games a team"}

# How many seasons of history a prop model trains on. Same reasoning as the NBA
# (nba/config.py): an unbounded window anchors calibration to an old run
# environment, and MLB's has moved (the 2019 ball, the 2023 rule changes).
TRAIN_SEASONS = 5

# THE MARKETS. `stat` is the column that settles it; `opp` the opportunity it runs
# through (plate appearances for a batter, batters faced for a pitcher).
#   hits  -- batter hits                 (books: 0.5 / 1.5)
#   hrr   -- hits + runs + RBIs          (books: 1.5 / 2.5)
#   hr    -- home runs                   (books: 0.5)
#   rbi   -- runs batted in              (books: 0.5)
#   strikeouts -- STARTING pitcher Ks    (books: ~3.5 - 8.5)
#   team_runs  -- a team's runs, the team-total over/under (books: ~3.5 - 5.5)
MARKETS = {
    "hits": {"stat": "H", "opp": "PA", "kind": "batter"},
    "hrr": {"stat": "HRR", "opp": "PA", "kind": "batter"},
    "hr": {"stat": "HR", "opp": "PA", "kind": "batter"},
    "rbi": {"stat": "RBI", "opp": "PA", "kind": "batter"},
    "strikeouts": {"stat": "K", "opp": "BFP", "kind": "pitcher"},
    "team_runs": {"stat": "RUNS", "opp": "PA", "kind": "team"},
}

# The lowest line a book quotes, and the spread of lines around each player's own
# median that the model is trained AND gated on -- so it can be asked about a
# bookmaker's line and mean it (the lesson of the NFL and NBA engines).
MIN_LINE = {"hits": 0.5, "hrr": 0.5, "hr": 0.5, "rbi": 0.5,
            "strikeouts": 2.5, "team_runs": 2.5}
MAX_LINE = {"hits": 2.5, "hrr": 4.5, "hr": 0.5, "rbi": 0.5,
            "strikeouts": 10.5, "team_runs": 7.5}
LINE_STEPS = {
    "hits": (0, 1),
    "hrr": (-1, 0, 1, 2),
    "hr": (0,),
    # RBI is a 0.5 market at the books, like home runs. "Over 1.5" appeared in
    # only 497 of 571,902 games (almost no batter's median is 1+) and the model
    # could not learn it -- 28.7% predicted, 9.9% landed -- so it is not offered.
    "rbi": (0,),
    "strikeouts": (-2, -1, 0, 1, 2),
    "team_runs": (-2, -1, 0, 1, 2),
}

# ONE MODEL PER LINE STEP for these markets. Hits+runs+RBIs is lumpy -- a hit
# often brings a run or an RBI with it -- and a single model shared across the
# whole line spread under-reacted to the line: 56.8% predicted at the median
# where 59.3% landed, 29.4% at +2 where 24.4% landed. A model per step learns
# each question's own base rate; the gate is unchanged and still tests every step.
PER_STEP_MODELS = {"hrr"}

# A ROLE, NOT AN APPEARANCE. A batter must be a regular (a starter averaging 3+
# plate appearances); a pitcher must be a starter with a few starts behind him.
MIN_GAMES = {"batter": 20, "pitcher": 5, "team": 10}
MIN_OPPORTUNITY = {"batter": 3.0, "pitcher": 15.0, "team": 0.0}

FORM_GAMES = 5
FORM_LONG = 10

# Release gate: beat the baseline in EVERY scored season, stay calibrated overall
# and at every line step (same shape as the NFL and NBA gates).
MIN_PREDICTIONS_TOTAL = 2000
MIN_PREDICTIONS_PER_SEASON = 300
MAX_ECE = 0.04

# Elo for the team-winner market. MLB outcomes are close to coin flips, so the
# grid is centred on small K. Parameters are fitted ONCE on the burn-in seasons
# and then held, which keeps every scored season out of the fit.
# WIDENED 2026-09-27: the first fit landed on three grid edges at once (k=2,
# home 40, regression 0.2), which usually means the optimum lies outside.
ELO_GRID = {"k": (0.5, 1.0, 1.5, 2.0, 3.0),
            "home_edge": (30.0, 40.0, 50.0, 60.0, 75.0),
            "regression": (0.0, 0.05, 0.1, 0.2, 0.33)}
