"""Per-match team xG from Understat (via soccerdata, disk-cached)."""
import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical

# BUG FIX: plain 4-digit ints like 2021 are ambiguous to soccerdata's
# SeasonCode.parse() for MULTI_YEAR leagues — "2021" hits a hardcoded
# special case and resolves to season "20-21" instead of "21-22", while
# 2022/2023/2024/2025 resolve to "22-23"/"23-24"/"24-25"/"25-26". That
# silently dropped the 21-22 season and pulled in an unwanted 20-21 season,
# costing ~20% xG coverage vs. history.py's 5 seasons. Using the same
# 4-char season code strings as config.League.history_seasons ("2122" etc.)
# parses unambiguously and keeps both sources season-aligned.


def fetch_team_match_xg(league: str) -> pd.DataFrame:
    """Return one row per team-match: date, team, home flag, xg, xga."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=list(lg.history_seasons))
    raw = us.read_team_match_stats().reset_index()

    required = ["date", "home_team", "away_team", "home_xg", "away_xg"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(
            f"Understat schema changed; missing {missing}. Got: {list(raw.columns)}"
        )

    raw = raw.assign(date=pd.to_datetime(raw["date"]))

    home_rows = pd.DataFrame({
        "date": raw["date"],
        "team": [canonical(t, league) for t in raw["home_team"]],
        "xg": pd.to_numeric(raw["home_xg"], errors="coerce"),
        "xga": pd.to_numeric(raw["away_xg"], errors="coerce"),
        "is_home": True,
    })
    away_rows = pd.DataFrame({
        "date": raw["date"],
        "team": [canonical(t, league) for t in raw["away_team"]],
        "xg": pd.to_numeric(raw["away_xg"], errors="coerce"),
        "xga": pd.to_numeric(raw["home_xg"], errors="coerce"),
        "is_home": False,
    })

    out = pd.concat([home_rows, away_rows], ignore_index=True)
    out = out.sort_values(["date", "team"]).reset_index(drop=True)
    return out.dropna(subset=["xg"]).reset_index(drop=True)
