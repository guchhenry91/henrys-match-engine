"""ClubElo ratings — the cross-league strength prior (free, no key, HTTP only)."""
import io
import urllib.request
from datetime import date

import pandas as pd

from leagues.names import UnknownTeam, canonical

API = "http://api.clubelo.com/{d}"   # NOTE: http only — clubelo does not serve https


def fetch_elo_snapshot(on: date | None = None) -> pd.DataFrame:
    """Every club's Elo on a date: columns Rank, Club, Country, Level, Elo, From, To."""
    d = (on or date.today()).isoformat()
    with urllib.request.urlopen(API.format(d=d), timeout=30) as resp:
        buf = io.StringIO(resp.read().decode("utf-8"))
    return pd.read_csv(buf)


def elo_for_league(league: str, teams: list[str], on: date | None = None) -> dict[str, float]:
    """Map our canonical team names -> ClubElo rating. Teams ClubElo doesn't know
    get the league's median (a safe neutral prior rather than a wrong number)."""
    snap = fetch_elo_snapshot(on)
    found: dict[str, float] = {}
    for _, row in snap.iterrows():
        try:
            name = canonical(str(row["Club"]), league)
        except UnknownTeam:
            continue
        if name in teams and name not in found:
            found[name] = float(row["Elo"])
    if found:
        median = pd.Series(list(found.values())).median()
    else:
        median = 1500.0
    return {t: found.get(t, float(median)) for t in teams}
