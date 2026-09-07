"""5 seasons of results + closing odds from football-data.co.uk, cached on disk.

WHY THE CACHE EXISTS, AND WHY IT IS COMMITTED. This module had none: every
publish downloaded 5 seasons x 5 leagues = 25 CSVs, and at a run every half hour
that is roughly 1,200 requests a day at a small free site for files that cannot
change. On 2026-09-05 football-data.co.uk began answering 503 to GitHub's
runners, and the publish failed for two days -- predictions frozen while results
stayed current, because the fast lock path needs no history.

THE FAILURE WAS SELF-SUSTAINING, which is why an actions/cache entry would not
have fixed it: that cache only saves when a job SUCCEEDS, and no job was
succeeding. Cold run -> 25 requests -> 503 -> failure -> nothing cached -> cold
run. The files are therefore committed to the repo, so CI reads them from the
checkout and never asks the network for a finished season at all.

Caching this is also just the correct way to treat the source. Re-downloading an
immutable 175KB file a thousand times a day is the behaviour that got the IP
range refused, and it would have been wrong even if it had never been noticed.
"""
import io
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leagues import config
from leagues.names import canonical

URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
CACHE = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "_fdcache"


def current_fd_season(now=None) -> str:
    """football-data's code for the season in progress, e.g. "2627".

    A European season starting in August is named by both years, so the code
    rolls over at midyear rather than at New Year.
    """
    now = now or datetime.now(timezone.utc)
    start = now.year if now.month >= 7 else now.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _cache_path(season: str, div: str) -> Path:
    return CACHE / f"{season}_{div}.csv"


def fetch_csv(season: str, div: str, now=None) -> str:
    """The raw CSV text, from disk where possible.

    A FINISHED season is cached permanently -- it cannot change, so re-fetching
    it is pure cost to us and to the source. The season IN PROGRESS is refetched
    every time, because results land in it.

    A failed fetch falls back to the cached copy with a LOUD warning rather than
    silently, and only raises when there is no cached copy to fall back to. That
    is the same rule nfl/data.py follows: a stale file that says it is stale
    beats no board at all, and a stale file that says nothing is how a fixture
    list quietly goes a week out of date.
    """
    path = _cache_path(season, div)
    live = season == current_fd_season(now)
    if path.exists() and not live:
        return path.read_text(encoding="latin-1")

    url = URL.format(season=season, div=div)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("latin-1")
    except Exception as exc:
        if path.exists():
            print(f"  WARNING football-data {season}/{div} unreachable "
                  f"({type(exc).__name__}: {exc}); using the cached copy")
            return path.read_text(encoding="latin-1")
        raise
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text(text, encoding="latin-1")
    tmp.replace(path)                    # atomic: never leave a half-written cache
    return text
# Average closing odds; fall back to Bet365 closing, then Bet365 pre-match.
ODDS_SETS = [("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"),
             ("B365H", "B365D", "B365A")]
# Over/Under 2.5 goals closing odds, same fallback discipline as ODDS_SETS.
# Unused by the match model today -- captured so a total-goals market signal
# (distinct from the 1X2 market already tested and found to carry no edge) can
# be evaluated without a second network dependency; see scripts/ou_market_experiment.py.
OU_ODDS_SETS = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("Max>2.5", "Max<2.5")]


def parse_history(buf, league: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(buf, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "home": [canonical(t, league) for t in df["HomeTeam"]],
        "away": [canonical(t, league) for t in df["AwayTeam"]],
        "home_goals": df["FTHG"].astype(int).values,
        "away_goals": df["FTAG"].astype(int).values,
    })
    for h, d, a in ODDS_SETS:
        if h in df.columns:
            out["odds_h"] = pd.to_numeric(df[h], errors="coerce").values
            out["odds_d"] = pd.to_numeric(df[d], errors="coerce").values
            out["odds_a"] = pd.to_numeric(df[a], errors="coerce").values
            break
    else:
        out["odds_h"] = out["odds_d"] = out["odds_a"] = pd.NA
    for over, under in OU_ODDS_SETS:
        if over in df.columns:
            out["odds_over25"] = pd.to_numeric(df[over], errors="coerce").values
            out["odds_under25"] = pd.to_numeric(df[under], errors="coerce").values
            break
    else:
        out["odds_over25"] = out["odds_under25"] = pd.NA
    # Always Bet365 specifically (never the Avg-across-bookmakers fallback
    # odds_h/d/a above prefers) -- lets a bookmaker-stability check compare
    # a single named book against the multi-book consensus on the SAME
    # matches; see scripts/market_model_ab_report.py.
    if "B365CH" in df.columns:
        out["odds_b365_h"] = pd.to_numeric(df["B365CH"], errors="coerce").values
        out["odds_b365_d"] = pd.to_numeric(df["B365CD"], errors="coerce").values
        out["odds_b365_a"] = pd.to_numeric(df["B365CA"], errors="coerce").values
    elif "B365H" in df.columns:
        out["odds_b365_h"] = pd.to_numeric(df["B365H"], errors="coerce").values
        out["odds_b365_d"] = pd.to_numeric(df["B365D"], errors="coerce").values
        out["odds_b365_a"] = pd.to_numeric(df["B365A"], errors="coerce").values
    else:
        out["odds_b365_h"] = out["odds_b365_d"] = out["odds_b365_a"] = pd.NA
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_history(league: str) -> pd.DataFrame:
    """All configured history seasons for a league, concatenated."""
    lg = config.get(league)
    frames = []
    for season in lg.history_seasons:
        buf = io.StringIO(fetch_csv(season, lg.fd_code))
        frames.append(parse_history(buf, league, season))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
