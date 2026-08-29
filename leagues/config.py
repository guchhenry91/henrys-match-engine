"""Per-league configuration. One entry per competition; the engine is generic."""
from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    key: str              # our canonical key, used in paths/URLs
    name: str             # display name
    fd_code: str          # football-data.co.uk division code
    fixture_slug: str     # fixturedownload.com slug for 2026-27
    understat: str        # soccerdata/Understat league id
    fbref: str            # soccerdata/FBref league id
    n_teams: int
    relegation_spots: int
    europe_spots: int     # top-N qualifying for the Champions League
    # second-tier football-data.co.uk code, source of promoted-club priors
    fd_code2: str = ""
    # how clubs level on points are separated: "gd" = goal difference then goals
    # for; "h2h" = head-to-head first (La Liga's actual rule).
    tiebreak: str = "gd"
    # 5 completed seasons used to fit, as football-data.co.uk season codes
    history_seasons: tuple = ("2122", "2223", "2324", "2425", "2526")


LEAGUES = {
    "PL": League("PL", "Premier League", "E0", "epl-2026",
                 "ENG-Premier League", "ENG-Premier League", 20, 3, 4, "E1"),
    "LALIGA": League("LALIGA", "La Liga", "SP1", "la-liga-2026",
                     "ESP-La Liga", "ESP-La Liga", 20, 3, 4, "SP2", "h2h"),
    "BUNDESLIGA": League("BUNDESLIGA", "Bundesliga", "D1", "bundesliga-2026",
                         "GER-Bundesliga", "GER-Bundesliga", 18, 2, 4, "D2"),
    "LIGUE1": League("LIGUE1", "Ligue 1", "F1", "ligue-1-2026",
                     "FRA-Ligue 1", "FRA-Ligue 1", 18, 2, 4, "F2"),
}


def get(key: str) -> League:
    if key not in LEAGUES:
        raise KeyError(f"unknown league {key!r}; known: {sorted(LEAGUES)}")
    return LEAGUES[key]

# HOW LONG BEFORE KICKOFF A PICK FREEZES.
#
# Lives here, in the one module that imports nothing heavier than dataclasses, so
# scripts/lock_picks.py can read it without dragging in penaltyblog, scipy and
# sklearn. That locker exists to be fast; a two-minute dependency install to learn
# one float would defeat the point of it.
#
# WIDENED TO 2 HOURS ON 2026-08-26, overturning a 60-minute setting on evidence.
# The old reasoning was that 60 minutes gave "four chances" and traded "a rare
# void for a systematic loss" of late team news. Measured over 109 scheduled runs,
# both halves were wrong:
#
#   * There are not four chances. A run starts a MEDIAN 8 minutes after its cron
#     slot (90th percentile 14) and the publish job needs about 10 more to reach
#     the lock, so a nominal 18:45 run froze a pick at roughly 19:05 -- after a
#     19:00 kickoff. Two or three usable slots, and 29% of scheduled runs fail.
#   * The void was not rare. In a fortnight it took four La Liga fixtures, eight
#     player picks and thirty-seven parlays out of the record. On 2026-08-26
#     GitHub fired nothing at all between 17:30 and 19:09 for a 19:00 kickoff.
#   * What it protected is small: the old note's own example is a confirmed XI
#     moving Arsenal from 77.4% to 77.1%.
#
# scripts/lock_picks.py freezes from the already-published board in seconds, so it
# CAN run every few minutes. The open question was whether GitHub actually runs
# it that often, and the answer decides whether this can go back to an hour.
#
# MEASURED 2026-08-29 (scripts/measure_lock_reliability.py, written to
# data-raw/lock_reliability.json). It does not:
#
#     cron slots due over 27.4 hours   97
#     runs that actually fired          3   (3.1%)
#     longest gap with no locker     16.1 hours
#
# Not one of those three failed -- GitHub simply never started the other 94.
# High-frequency schedules are dropped under load, and one of the three fired at
# 05:12 UTC, a time the cron does not even cover. The workflow's own comment
# claimed the design was "many cheap chances rather than one expensive one"; there
# are not many chances, so this SO FAR remains a hope rather than a mechanism.
#
# So the window STAYS AT 2.0, and narrowing it to 1.0 would have been actively
# harmful: with a 16.1-hour gap, a one-hour window means a pick can reach kickoff
# having never had a locker run at all -- unfrozen, then frozen late, then
# tainted, then VOID. That is the exact failure the widening fixed. Re-run the
# measurement before revisiting; the floor for this constant is
# `worst_gap_hours`, not a judgement call.
LOCK_WINDOW_HOURS = 2.0
