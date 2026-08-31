"""Season game logs from stats.nba.com, cached to disk.

ONE REQUEST PER SEASON PER SIDE. `leaguegamelog` returns every player-game (or
team-game) row for a whole season at once, so fifteen seasons of both cost thirty
requests rather than one per game. That is what makes a fifteen-season backtest
practical without a paid feed.

THE HEADERS ARE NOT DECORATION. stats.nba.com refuses a bare request; it wants a
browser User-Agent and its own `x-nba-stats-*` pair. Without them the endpoint
hangs or returns an empty body rather than an error, which reads exactly like "no
data for that season" -- a failure mode worth naming, because acting on it would
mean silently backtesting on fewer seasons than intended.

A COMPLETED SEASON NEVER CHANGES, so it caches forever. Only the current one has
a max age, for the same reason the NFL loader grew one: a cache with no expiry
served pre-season roster data for 68 hours while the board was eleven days from
kickoff.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from nba import config

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data-raw" / "nba" / "_cache"

BASE = "https://stats.nba.com/stats/leaguegamelog"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Politeness between requests. The endpoint is free and undocumented; hammering it
# is how an IP stops being served, and thirty requests do not need to be fast.
PAUSE_SECONDS = 1.2
CURRENT_MAX_AGE_HOURS = 6.0


def season_label(season: int) -> str:
    """2026 -> '2025-26'. Seasons are named here by the year they END."""
    return f"{season - 1}-{str(season)[-2:]}"


def _fetch(season: int, side: str, attempts: int = 3) -> dict:
    params = {
        "Counter": "0", "DateFrom": "", "DateTo": "", "Direction": "DESC",
        "LeagueID": "00", "PlayerOrTeam": side, "Season": season_label(season),
        "SeasonType": "Regular Season", "Sorter": "DATE",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:                      # noqa: BLE001 - reported below
            last = exc
            if attempt < attempts:
                time.sleep(PAUSE_SECONDS * attempt * 2)
    raise RuntimeError(f"stats.nba.com refused {side} log for "
                       f"{season_label(season)} after {attempts} tries: {last}")


def _cached(season: int, side: str, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{'player' if side == 'P' else 'team'}_{season}.csv"
    max_age = CURRENT_MAX_AGE_HOURS if season >= config.CURRENT_SEASON else None
    if path.exists() and not refresh and max_age is not None:
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        if age_h > max_age:
            refresh = True
    if path.exists() and not refresh:
        return pd.read_csv(path, low_memory=False)
    try:
        payload = _fetch(season, side)
    except Exception as exc:
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            print(f"WARNING: could not refresh {path.name} ({exc}); serving a "
                  f"cached copy {age_h:.1f}h old -- anything newer is MISSING")
            return pd.read_csv(path, low_memory=False)
        raise
    block = payload["resultSets"][0]
    frame = pd.DataFrame(block["rowSet"], columns=block["headers"])
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)                # atomic: a killed download leaves no half file
    time.sleep(PAUSE_SECONDS)
    return frame


def _normalise(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    out = frame.copy()
    out["season"] = season
    out["game_date"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    # MATCHUP is "ATL @ MIA" (away) or "ATL vs. MIA" (home) -- the only place the
    # venue is recorded, and venue is a real effect in every one of these markets.
    matchup = out["MATCHUP"].astype(str)
    out["is_home"] = (~matchup.str.contains("@")).astype(float)
    out["opponent"] = matchup.str.split(r"\s+(?:@|vs\.)\s+", regex=True).str[-1]
    out["won"] = (out["WL"].astype(str).str.upper() == "W").astype(float)
    for column in ("MIN", "PTS", "REB", "AST", "FG3M"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def player_games(seasons=None, refresh: bool = False) -> pd.DataFrame:
    """One row per player per game, oldest first."""
    seasons = tuple(seasons or config.SEASONS)
    frames = [_normalise(_cached(s, "P", refresh), s) for s in seasons]
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["game_date", "PLAYER_ID"])
    return out.sort_values(["game_date", "PLAYER_ID"]).reset_index(drop=True)


def team_games(seasons=None, refresh: bool = False) -> pd.DataFrame:
    """One row per team per game, oldest first."""
    seasons = tuple(seasons or config.SEASONS)
    frames = [_normalise(_cached(s, "T", refresh), s) for s in seasons]
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["game_date", "TEAM_ID"])
    return out.sort_values(["game_date", "TEAM_ID"]).reset_index(drop=True)


def games(seasons=None, refresh: bool = False) -> pd.DataFrame:
    """One row per GAME -- home team, away team, both scores, winner.

    Built by pairing the two team rows that share a GAME_ID. A game whose pair is
    incomplete is dropped rather than half-guessed: a one-sided row cannot say who
    won, and inventing the other side is how a fabricated result enters a record.
    """
    team = team_games(seasons, refresh)
    rows = []
    for game_id, pair in team.groupby("GAME_ID"):
        if len(pair) != 2:
            continue
        home = pair[pair["is_home"] == 1.0]
        away = pair[pair["is_home"] == 0.0]

        # NEUTRAL-VENUE GAMES HAVE NO HOME SIDE, and the feed says so honestly:
        # both rows read "@" ("DAL @ DET" and "DET @ DAL"). The NBA plays a handful
        # abroad each year -- five in 2025-26, none in the seasons before it.
        #
        # Dropping them was the first behaviour and it was wrong twice over: it
        # silently lost five real games from the season the board most needs, and
        # it lost them from the CURRENT one specifically, so the loss would have
        # grown every year rather than staying a historical curiosity. They are
        # kept, with a nominal home taken from the order the matchup names them --
        # arbitrary but consistent -- and flagged so no model grants home
        # advantage on a court neither side owns.
        neutral = len(home) != 1 or len(away) != 1
        if neutral:
            if len(pair) != 2:
                continue
            first, second = pair.iloc[0], pair.iloc[1]
        else:
            first, second = home.iloc[0], away.iloc[0]
        rows.append({
            "game_id": game_id,
            "season": int(first["season"]),
            "game_date": first["game_date"],
            "home_team": first["TEAM_ABBREVIATION"],
            "away_team": second["TEAM_ABBREVIATION"],
            "home_score": float(first["PTS"]),
            "away_score": float(second["PTS"]),
            "neutral": bool(neutral),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["played"] = out["home_score"].notna() & out["away_score"].notna()
    # Basketball has no draws, so this is genuinely two-valued -- unlike the NFL,
    # where a tie is a real outcome and collapsing it would misgrade three games.
    out["winner"] = None
    out.loc[out["home_score"] > out["away_score"], "winner"] = "home"
    out.loc[out["home_score"] < out["away_score"], "winner"] = "away"
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)
