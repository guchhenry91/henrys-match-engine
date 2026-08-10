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
        url = URL.format(season=season, div=lg.fd_code)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            buf = io.StringIO(resp.read().decode("latin-1"))
        frames.append(parse_history(buf, league, season))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
