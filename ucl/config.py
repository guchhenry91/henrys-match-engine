"""The 2026/27 field, the draw pots, and the constants the UCL engine runs on."""

API_LEAGUE_ID = 2

# Sixteen seasons are available from the API (2011-2026), 3,407 finished matches.
# All of them are LOADED; the earliest are burn-in and are never scored, because a
# club's strength on the first scored matchday has to come from somewhere.
SEASONS = tuple(range(2011, 2027))
CURRENT_SEASON = 2026
BURN_IN_SEASONS = 5          # 2011-2015 establish strengths, never graded

# THE FORMAT CHANGED IN 2024. Before that it was eight groups of four and a
# knockout; since then a 36-team single league phase, which is why the fixture
# count jumps from ~215 a season to ~280. Team strength transfers across that
# change -- Real Madrid did not become a different club -- but anything that
# projects a TABLE must only use the Swiss seasons, and the backtest reports the
# two eras separately so a reader can see whether they behave alike.
SWISS_FROM = 2024

# The draw made on 2026-08-27. Pot order IS UEFA's coefficient seeding, so it is
# an external strength estimate available for every club including the debutants.
POTS = {
    1: ["Paris Saint Germain", "Bayern Munich", "Real Madrid", "Liverpool",
        "Inter", "Manchester City", "Arsenal", "Barcelona", "Atletico Madrid"],
    2: ["Borussia Dortmund", "AS Roma", "Sporting CP", "Aston Villa", "FC Porto",
        "Manchester United", "Club Brugge KV", "Real Betis", "PSV Eindhoven"],
    3: ["Feyenoord", "Lille", "Bodo/Glimt", "Napoli", "RB Leipzig", "Villarreal",
        "Fenerbahce", "Shakhtar Donetsk", "Galatasaray"],
    4: ["Slavia Praha", "Slovan Bratislava", "VfB Stuttgart", "AEK Athens", "LASK",
        "Como", "Lens", "Viking", "Sabah"],
}

# Our spelling -> the API's. Every one of these was a club the probe reported as
# having ZERO history, which would have seeded a real European side from nothing:
# Betis, Fenerbahce and AEK have all played plenty. A name mismatch and a genuine
# debutant look identical in the data and must not be treated alike.
NAME_TO_API = {
    "Real Betis": "Betis",
    "Fenerbahce": "Fenerbahçe",     # the API spells it with the cedilla
    "AEK Athens": "AEK Athens FC",
    "LASK": "Lask Linz",
    "Sabah": "Sabah FA",
    "Slovan Bratislava": "Slovan Bratislava",
    "Paris Saint Germain": "Paris Saint Germain",
    "Sporting CP": "Sporting CP",
    "Club Brugge KV": "Club Brugge KV",
}

# Below this many European matches a club is mostly its pot rather than its own
# record. Not a cliff: the shrinkage is continuous and this only sets the scale.
PRIOR_STRENGTH = 25.0

# Under this many matches the number on the card is mostly a seed, and the board
# says so. Measured, not guessed: seven of the 36 drawn clubs sit here -- Viking
# has two European matches in sixteen seasons, Lens and LASK six, Stuttgart eight.
THIN_HISTORY = 20

# Clubs with NO Champions League history at all in the loaded window. Real Betis
# and Como are not name mismatches -- Betis has been a Europa League side and Como
# is a debutant -- so they cannot be fitted and are seeded from the weakest sides
# in the competition, flagged on the card. "Viking" is also distinct from
# "Vikingur Reykjavik" and "Vikingur Gota" and must never be merged with them.
SEEDED_NOTE = ("no Champions League history in the loaded window; seeded at the "
               "strength of the competition's weakest sides")

# Time decay, per day. Matches the leagues engine's XI_PER_DAY: a club's form from
# 2013 is not evidence about it now, but it is not nothing either.
XI_PER_DAY = 0.003
