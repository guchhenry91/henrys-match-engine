"""nflverse fetch + cache. The only place that talks to the network.

WHY NFLVERSE AND NOT THE API-NFL KEY. nflverse publishes the same weekly box
scores as flat CSVs, free, with no per-day request cap. The API-Football account
on this project hit its daily limit twice in one week, and a six-season backfill
is thousands of player-games -- exactly the shape of job that causes that. The
API key stays for live in-season data where freshness actually matters.

CACHED ON DISK, because a backtest gets run many times while it is being written
and re-downloading six seasons each time is both slow and rude to a free host.
"""
import io
from pathlib import Path

import time

import pandas as pd

from nfl import config

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data-raw" / "nfl" / "_cache"

# The CURRENT release path. nflverse retired `player_stats/player_stats_{season}`
# after 2024 -- it still serves 2020-2024 and 404s for 2025, which is the kind of
# break that looks like "no recent data" rather than "wrong URL". `stats_player`
# carries all six seasons. The only schema difference is `recent_team` -> `team`,
# normalised below so nothing downstream has to know which era it came from.
PLAYER_STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                    "stats_player/stats_player_week_{season}.csv")
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
# Who is ON each team right now, which the box scores cannot say. Carries gsis_id,
# the SAME key as player_stats' player_id, so clubs are reconciled by identity
# rather than by matching names.
ROSTER_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "rosters/roster_{season}.csv")
ROSTER_COLUMNS = ["season", "team", "position", "status", "full_name", "gsis_id"]

DEPTH_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "depth_charts/depth_charts_{season}.csv")
DEPTH_COLUMNS = ["dt", "team", "player_name", "gsis_id", "pos_abb", "pos_rank"]

# Columns we actually use. Named explicitly so an upstream schema change fails
# loudly here rather than silently producing a column of NaN three layers down.
PLAYER_COLUMNS = [
    "player_id", "player_display_name", "position", "team",
    "season", "week", "season_type", "opponent_team",
    "passing_yards", "rushing_yards", "receiving_yards",
    "receptions", "carries", "targets", "attempts", "completions",
    "passing_tds", "rushing_tds", "receiving_tds",
]
GAME_COLUMNS = ["game_id", "season", "game_type", "week", "gameday", "gametime",
                "home_team", "away_team",
                "home_score", "away_score", "result", "spread_line", "total_line",
                "location"]


def _read_csv(url: str, cache_name: str, refresh: bool = False,
              max_age_hours: float | None = None) -> pd.DataFrame:
    """Read a cached nflverse CSV, refetching when it is older than max_age_hours.

    THE CACHE USED TO BE FOREVER, and in CI the download directory is itself
    cached between runs, so a file fetched once in pre-season was still being
    served weeks later. On 2026-08-30 the roster cache was 68 hours old with the
    board eleven days from kickoff.

    `max_age_hours` is None for anything that CANNOT change -- a completed
    season's box scores -- and a few hours for anything that moves: rosters,
    depth charts, the schedule, the current season's player weeks.

    A failed refetch falls back to the cached copy with a LOUD warning rather
    than raising. Serving slightly old data beats taking the board down, but it
    must never be silent, because a quiet fallback is indistinguishable from
    fresh data and that is how staleness hides.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name
    if path.exists() and not refresh and max_age_hours is not None:
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        if age_h > max_age_hours:
            refresh = True
            print(f"  {cache_name}: cache {age_h:.1f}h old (limit "
                  f"{max_age_hours:g}h) -- refetching")
    if path.exists() and not refresh:
        return pd.read_csv(path, low_memory=False)
    try:
        frame = pd.read_csv(url, low_memory=False)
    except Exception as exc:
        if not path.exists():
            raise
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        print(f"WARNING: could not refetch {cache_name} ({exc}); serving a cached "
              f"copy {age_h:.1f}h old -- anything newer than that is MISSING")
        return pd.read_csv(path, low_memory=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)            # atomic: a killed download never leaves a half file
    return frame


def anytime_touchdown(frame: pd.DataFrame) -> pd.Series:
    """Did he score? A FLAG, never a count.

    A player who ran one in and caught one has still hit the market exactly once,
    so a count would settle a single bet twice over.

    Passing touchdowns are excluded on purpose: the quarterback who threw it did
    not score it. Settling a QB's anytime market on his own passing TDs would make
    every starter look like a scorer and quietly turn the market into something
    else entirely.
    """
    return ((frame["rushing_tds"] + frame["receiving_tds"]) > 0).astype(int)


def player_weeks(seasons=None, refresh: bool = False) -> pd.DataFrame:
    """One row per player per game, regular season only, oldest first."""
    seasons = tuple(seasons or config.SEASONS)
    frames = []
    for season in seasons:
        # A COMPLETED season never changes, so it caches forever. The CURRENT
        # one gains a row every week and is what grading reads, so it gets a few
        # hours at most.
        raw = _read_csv(PLAYER_STATS_URL.format(season=season),
                        f"stats_player_week_{season}.csv", refresh,
                        max_age_hours=(6 if int(season) >= int(config.CURRENT_SEASON)
                                       else None))
        if "team" not in raw.columns and "recent_team" in raw.columns:
            raw = raw.rename(columns={"recent_team": "team"})
        missing = [c for c in PLAYER_COLUMNS if c not in raw.columns]
        if missing:
            raise RuntimeError(
                f"nflverse stats_player_week_{season} is missing {missing}; the upstream "
                f"schema changed and the features built from it would be silently wrong")
        frames.append(raw[PLAYER_COLUMNS])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["season_type"] == config.SEASON_TYPE].copy()

    for column in ("passing_yards", "rushing_yards", "receiving_yards", "receptions",
                   "carries", "targets", "attempts", "completions",
                   "passing_tds", "rushing_tds", "receiving_tds"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    # An "anytime touchdown" is any of the three, and a player who both ran one in
    # and caught one has still only hit the market once -- so this is a FLAG, never
    # a count. Passing touchdowns are excluded: the quarterback throwing it did not
    # score it, and settling a QB's anytime market on his own passing TDs would
    # make every starting QB look like a scorer.
    out["touchdowns"] = anytime_touchdown(out)

    # Opportunity for the touchdown market. A player scores from a carry or a
    # catch, so his chance tracks how often he gets the ball -- not his yardage.
    # Receptions rather than targets: an incomplete pass to him was never a
    # scoring chance.
    out["touches"] = out["carries"] + out["receptions"]

    out = out.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    return out


def rosters(season=None, refresh: bool = False) -> pd.DataFrame:
    """Current rosters for one season. Empty frame if unavailable."""
    season = season or config.CURRENT_SEASON
    try:
        # Rosters move constantly in camp and after cuts. Six hours, because the
        # cache was 68 hours old eleven days before kickoff.
        raw = _read_csv(ROSTER_URL.format(season=season),
                        f"roster_{season}.csv", refresh, max_age_hours=6)
    except Exception as exc:
        print(f"WARNING: no roster file for {season} ({exc}); clubs will fall back "
              f"to each player's last appearance")
        return pd.DataFrame(columns=ROSTER_COLUMNS)
    missing = [c for c in ROSTER_COLUMNS if c not in raw.columns]
    if missing:
        print(f"WARNING: roster_{season} is missing {missing}; not trusting it")
        return pd.DataFrame(columns=ROSTER_COLUMNS)
    out = raw[ROSTER_COLUMNS].copy()
    out["gsis_id"] = out["gsis_id"].astype(str)
    return out


def _kickoff_utc(frame: pd.DataFrame) -> pd.Series:
    """The real kickoff instant, in UTC.

    `gameday` alone is a DATE. Everything downstream treated it as the kickoff,
    so every game on the board carried a midnight timestamp -- and a pick frozen
    against midnight is frozen ~20 hours before a 20:20 kickoff, long before any
    inactives report, or is marked late and voided. `gametime` has been in the
    feed all along.

    nflverse quotes `gametime` in US EASTERN, not UTC and not local-to-stadium, so
    it is localised through the America/New_York zone rather than by subtracting a
    fixed offset: the season spans the DST change in early November, and a fixed
    -4 would put every late-season kickoff an hour wrong in the direction that
    makes a lock late.
    """
    when = pd.to_datetime(
        frame["gameday"].dt.strftime("%Y-%m-%d") + " " + frame["gametime"].fillna(""),
        errors="coerce")
    # A game with no time falls back to its date, and is flagged by being midnight
    # UTC-of-Eastern rather than silently pretending to a kickoff it does not know.
    when = when.fillna(frame["gameday"])
    return (when.dt.tz_localize("America/New_York", ambiguous=True,
                                nonexistent="shift_forward")
                .dt.tz_convert("UTC"))


def depth_charts(season=None, refresh: bool = False) -> pd.DataFrame:
    """THE LATEST depth-chart snapshot, one row per player, with his rank.

    WHY THIS MATTERS MORE THAN IT LOOKS. Two long-standing holes close here.

    First, WHO IS ACTUALLY ON THE TEAM. The season roster file is a full-season
    record: on 2026-08-30, four days after the cut to 53, it still listed 90
    active players a team, and so did the weekly file. It cannot say who was cut.
    The depth chart is republished continuously (that morning at 12:30 UTC) and
    only contains players a team is actually carrying.

    Second, WHO STARTS. `pos_rank` is 1 for a starter. The board's own stated hole
    was that "a backup quarterback carries a low line and can top a market he may
    not play in" -- his line is low precisely because he has only played in
    relief, which makes "over" look easy right up until he takes no snap.

    Only the most recent snapshot is returned. The file holds every snapshot of
    the season (485k rows in 2026), and an older one would reinstate players who
    have since been cut -- the exact staleness this is here to remove.
    """
    season = season or config.CURRENT_SEASON
    try:
        raw = _read_csv(DEPTH_URL.format(season=season),
                        f"depth_charts_{season}.csv", refresh, max_age_hours=6)
    except Exception as exc:
        print(f"WARNING: no depth chart for {season} ({exc}); the board falls back "
              f"to roster-only filtering and cannot flag starters")
        return pd.DataFrame(columns=DEPTH_COLUMNS)
    missing = [c for c in DEPTH_COLUMNS if c not in raw.columns]
    if missing:
        print(f"WARNING: depth chart is missing {missing}; ignoring it rather than "
              f"guessing at its shape")
        return pd.DataFrame(columns=DEPTH_COLUMNS)
    out = raw[DEPTH_COLUMNS].copy()
    out["dt"] = pd.to_datetime(out["dt"], errors="coerce", utc=True)
    out = out.dropna(subset=["dt", "gsis_id"])
    if out.empty:
        return out
    out = out[out["dt"] == out["dt"].max()].copy()
    out["gsis_id"] = out["gsis_id"].astype(str)
    out["pos_rank"] = pd.to_numeric(out["pos_rank"], errors="coerce")
    # One row per player: a man can appear at more than one position (a returner
    # listed at WR and KR), and his BEST rank is the one that decides whether he
    # is a starter somewhere.
    out = out.sort_values("pos_rank").drop_duplicates(subset=["gsis_id"], keep="first")
    return out.reset_index(drop=True)


def games(seasons=None, refresh: bool = False) -> pd.DataFrame:
    """One row per game, regular season, with the result filled in where played."""
    seasons = tuple(seasons or config.SEASONS)
    # The schedule carries kickoff times and results, both of which move.
    raw = _read_csv(GAMES_URL, "games.csv", refresh, max_age_hours=3)
    missing = [c for c in GAME_COLUMNS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"nflverse games.csv is missing {missing}")
    out = raw[GAME_COLUMNS]
    out = out[(out["season"].isin(seasons)) & (out["game_type"] == config.SEASON_TYPE)].copy()
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    out["kickoff"] = _kickoff_utc(out)
    for column in ("home_score", "away_score", "result", "spread_line", "total_line"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["played"] = out["home_score"].notna() & out["away_score"].notna()
    # A tie is a real NFL outcome (there were three between 2020 and 2025), so the
    # winner is deliberately three-valued. Collapsing it to a binary would score a
    # tie as a loss for whichever side the model named and quietly overstate error.
    out["winner"] = pd.NA
    played = out["played"]
    out.loc[played & (out["home_score"] > out["away_score"]), "winner"] = "home"
    out.loc[played & (out["home_score"] < out["away_score"]), "winner"] = "away"
    out.loc[played & (out["home_score"] == out["away_score"]), "winner"] = "tie"
    return out.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
