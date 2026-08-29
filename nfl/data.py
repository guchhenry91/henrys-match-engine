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


def _read_csv(url: str, cache_name: str, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name
    if path.exists() and not refresh:
        return pd.read_csv(path, low_memory=False)
    frame = pd.read_csv(url, low_memory=False)
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
        raw = _read_csv(PLAYER_STATS_URL.format(season=season),
                        f"stats_player_week_{season}.csv", refresh)
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
        raw = _read_csv(ROSTER_URL.format(season=season),
                        f"roster_{season}.csv", refresh)
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


def games(seasons=None, refresh: bool = False) -> pd.DataFrame:
    """One row per game, regular season, with the result filled in where played."""
    seasons = tuple(seasons or config.SEASONS)
    raw = _read_csv(GAMES_URL, "games.csv", refresh)
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
