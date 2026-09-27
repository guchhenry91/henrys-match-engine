"""Retrosheet game-level CSVs -> the frames the MLB engine reads.

Each season is one zip (batting, pitching, gameinfo, teamstats). A finished
season never changes, so each is trimmed to the columns the engine uses and
written once to data-raw/mlb/history/ (committed, like the NBA's history), and
read from there before the network is ever touched.

Regular season only (`gametype == "regular"`), official box lines only
(`stattype == "value"`).
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from mlb import config

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data-raw" / "mlb" / "_cache"
HISTORY = ROOT / "data-raw" / "mlb" / "history"
URL = "https://www.retrosheet.org/downloads/{y}/{y}csvs.zip"

BAT = ["gid", "id", "team", "b_lp", "b_seq", "b_pa", "b_h", "b_r", "b_rbi", "b_hr",
       "b_k", "date", "vishome", "opp"]
PIT = ["gid", "id", "team", "p_seq", "p_bfp", "p_k", "p_ipouts", "p_gs", "date",
       "vishome", "opp"]
GAME = ["gid", "visteam", "hometeam", "site", "date", "number", "vruns", "hruns",
        "wteam", "lteam"]
TEAM = ["gid", "team", "b_pa", "b_r", "b_h", "b_k", "date", "vishome", "opp"]


def _zip(season: int) -> zipfile.ZipFile:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{season}csvs.zip"
    if not path.exists():
        with urllib.request.urlopen(URL.format(y=season), timeout=120) as r:
            path.write_bytes(r.read())
        time.sleep(1)
    return zipfile.ZipFile(path)


def _member(z: zipfile.ZipFile, name: str) -> str:
    return next(n for n in z.namelist() if n.endswith(f"{name}.csv"))


def _extract(season: int) -> None:
    """Trim one season's zip into the four committed history files."""
    z = _zip(season)
    HISTORY.mkdir(parents=True, exist_ok=True)
    for name, cols in (("batting", BAT), ("pitching", PIT), ("gameinfo", GAME),
                       ("teamstats", TEAM)):
        frame = pd.read_csv(z.open(_member(z, name)), low_memory=False)
        if "gametype" in frame.columns:
            frame = frame[frame["gametype"].astype(str).str.lower() == "regular"]
        if "stattype" in frame.columns:
            frame = frame[frame["stattype"].astype(str).str.lower() == "value"]
        frame[[c for c in cols if c in frame.columns]].to_csv(
            HISTORY / f"{name}_{season}.csv.gz", index=False, compression="gzip")


def _read(name: str, season: int) -> pd.DataFrame:
    path = HISTORY / f"{name}_{season}.csv.gz"
    if not path.exists():
        _extract(season)
    frame = pd.read_csv(path, low_memory=False)
    frame["season"] = season
    frame["game_date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d",
                                        errors="coerce")
    return frame


def batter_games(seasons=None) -> pd.DataFrame:
    """One row per STARTING batter per game (lineup slot's first occupant)."""
    frames = [_read("batting", s) for s in (seasons or config.SEASONS)]
    b = pd.concat(frames, ignore_index=True)
    b = b[b["b_seq"] == 1]
    out = pd.DataFrame({
        "PLAYER_ID": b["id"], "TEAM": b["team"], "GAME_ID": b["gid"],
        "game_date": b["game_date"], "season": b["season"], "opponent": b["opp"],
        "is_home": (b["vishome"].astype(str) == "h").astype(float),
        "PA": b["b_pa"], "H": b["b_h"], "R": b["b_r"], "RBI": b["b_rbi"],
        "HR": b["b_hr"], "lineup": b["b_lp"],
    })
    out["HRR"] = out["H"] + out["R"] + out["RBI"]
    return out.dropna(subset=["game_date"]).sort_values(["game_date", "GAME_ID"]).reset_index(drop=True)


def pitcher_games(seasons=None) -> pd.DataFrame:
    """One row per STARTING pitcher per game."""
    frames = [_read("pitching", s) for s in (seasons or config.SEASONS)]
    p = pd.concat(frames, ignore_index=True)
    p = p[p["p_gs"] == 1]
    out = pd.DataFrame({
        "PLAYER_ID": p["id"], "TEAM": p["team"], "GAME_ID": p["gid"],
        "game_date": p["game_date"], "season": p["season"], "opponent": p["opp"],
        "is_home": (p["vishome"].astype(str) == "h").astype(float),
        "BFP": p["p_bfp"], "K": p["p_k"], "IPOUTS": p["p_ipouts"],
    })
    return out.dropna(subset=["game_date"]).sort_values(["game_date", "GAME_ID"]).reset_index(drop=True)


def team_games(seasons=None) -> pd.DataFrame:
    """One row per team per game: the team-total market's rows. The team IS the
    'player', so the prop machinery applies unchanged."""
    frames = [_read("teamstats", s) for s in (seasons or config.SEASONS)]
    t = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "PLAYER_ID": t["team"], "TEAM": t["team"], "GAME_ID": t["gid"],
        "game_date": t["game_date"], "season": t["season"], "opponent": t["opp"],
        "is_home": (t["vishome"].astype(str) == "h").astype(float),
        "PA": t["b_pa"], "RUNS": t["b_r"],
    })
    return out.dropna(subset=["game_date"]).sort_values(["game_date", "GAME_ID"]).reset_index(drop=True)


def games(seasons=None) -> pd.DataFrame:
    """One row per game in the shape nfl.games_model's Elo expects."""
    frames = [_read("gameinfo", s) for s in (seasons or config.SEASONS)]
    g = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "game_id": g["gid"], "season": g["season"], "game_date": g["game_date"],
        "home_team": g["hometeam"], "away_team": g["visteam"],
        "home_score": pd.to_numeric(g["hruns"], errors="coerce"),
        "away_score": pd.to_numeric(g["vruns"], errors="coerce"),
        "neutral": False,
    })
    out["played"] = out["home_score"].notna() & out["away_score"].notna()
    out["winner"] = None
    out.loc[out["home_score"] > out["away_score"], "winner"] = "home"
    out.loc[out["home_score"] < out["away_score"], "winner"] = "away"
    return out.dropna(subset=["game_date"]).sort_values(["game_date", "game_id"]).reset_index(drop=True)
