"""5 seasons of results + closing odds from football-data.co.uk."""
import io
import urllib.request

import pandas as pd

from leagues import config
from leagues.names import canonical

URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
# Average closing odds; fall back to Bet365 closing, then Bet365 pre-match.
ODDS_SETS = [("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"),
             ("B365H", "B365D", "B365A")]


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
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_history(league: str) -> pd.DataFrame:
    """All configured history seasons for a league, concatenated."""
    lg = config.get(league)
    frames = []
    for season in lg.history_seasons:
        url = URL.format(season=season, div=lg.fd_code)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            buf = io.StringIO(resp.read().decode("latin-1"))
        frames.append(parse_history(buf, league, season))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
