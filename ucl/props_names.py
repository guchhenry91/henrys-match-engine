"""Champions League club names (API-Football) -> Understat team names.

Understat covers the big five leagues only, so these are the clubs whose players the
soccer props model (leagues/forecast.py) can price. Every other club -- Benfica, PSV,
Celtic, Slavia, Sabah... -- has no Understat data and is absent on purpose.
Clubs spelled identically in both feeds need no entry; UNDERSTAT_TEAMS is the check.
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = ROOT / "data-raw" / "leagues" / "understat_matches"

DIFFERENT = {
    "Bayern München": "Bayern Munich",
    "Newcastle": "Newcastle United",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Stade Brestois 29": "Brest",
    "VfL Wolfsburg": "Wolfsburg",
    "AS Roma": "Roma",
}


@lru_cache(maxsize=None)
def understat_teams() -> dict:
    """{understat team: league} over every season on disk (latest league wins)."""
    out = {}
    for f in sorted(glob.glob(str(MATCHES_DIR / "*_20*.csv.gz"))):
        league = Path(f).name.rsplit("_", 1)[0]
        for t in pd.read_csv(f, usecols=["team"])["team"].unique():
            out[str(t)] = league
    return out


def to_understat(club: str) -> str | None:
    """The Understat name of a UCL club, or None for a club outside the big five."""
    name = DIFFERENT.get(club, club)
    return name if name in understat_teams() else None
