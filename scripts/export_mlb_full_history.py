"""Export Retrosheet seasons with everything the MLB rebuild needs (docs/REBUILD_PLAN.md).

The original export kept a handful of columns and regular-season games only. The
rebuild needs: every batting/pitching column, POSTSEASON games (the October picks
that failed were never in any test), batter and pitcher handedness, real pitch
counts (workload), and platoon splits. All from Retrosheet's free season zips.

Writes, per season, into --out (default data-raw/mlb/history_full):
  batting_{y}.csv.gz    every column, all game types but the All-Star game
  pitching_{y}.csv.gz   ditto, plus `pitches` (summed from play-by-play)
  gameinfo_{y}.csv.gz   venue, day/night, DH, weather, game type, scores
  players_{y}.csv.gz    bats / throws per player
  platoon_{y}.csv.gz    per game, per batter, per pitcher hand: PA, H, HR, K, BB
                        (and the pitcher-side mirror: per pitcher, per batter hand)
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.retrosheet.org/downloads/{y}/{y}csvs.zip"
GAME_COLS = ["gid", "visteam", "hometeam", "site", "date", "number", "starttime", "daynight",
             "innings", "usedh", "temp", "winddir", "windspeed", "precip", "sky",
             "gametype", "vruns", "hruns", "wteam", "lteam"]
DROP = {"box", "pbp", "stattype"}


def _member(z, name):
    return next(n for n in z.namelist() if n.endswith(f"{name}.csv"))


def _read(z, name, **kw):
    frame = pd.read_csv(z.open(_member(z, name)), low_memory=False, **kw)
    if "stattype" in frame.columns:
        frame = frame[frame["stattype"].astype(str).str.lower() == "value"]
    if "gametype" in frame.columns:
        frame = frame[frame["gametype"].astype(str).str.lower() != "allstar"]
    return frame


def export(season: int, out: Path, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / f"{season}csvs.zip"
    if not zpath.exists():
        with urllib.request.urlopen(URL.format(y=season), timeout=300) as r:
            zpath.write_bytes(r.read())
    z = zipfile.ZipFile(zpath)
    out.mkdir(parents=True, exist_ok=True)
    bat = _read(z, "batting")
    pit = _read(z, "pitching")
    gi = _read(z, "gameinfo")
    players = pd.read_csv(z.open(_member(z, "allplayers")), low_memory=False)
    plays = _read(z, "plays", usecols=lambda c: c in {
        "gid", "batter", "pitcher", "bathand", "pithand", "nump", "pa", "single", "double",
        "triple", "hr", "k", "walk", "gametype"})
    plays = plays[plays["pa"] == 1].copy()
    plays["h"] = plays[["single", "double", "triple", "hr"]].sum(axis=1)
    pitches = plays.groupby(["gid", "pitcher"], as_index=False)["nump"].sum().rename(
        columns={"pitcher": "id", "nump": "pitches"})
    pit = pit.merge(pitches, on=["gid", "id"], how="left")
    agg = {"pa": "sum", "h": "sum", "hr": "sum", "k": "sum", "walk": "sum"}
    bside = plays.groupby(["gid", "batter", "pithand"], as_index=False).agg(agg).rename(
        columns={"batter": "id", "pithand": "vs_hand"})
    bside["role"] = "bat"
    pside = plays.groupby(["gid", "pitcher", "bathand"], as_index=False).agg(agg).rename(
        columns={"pitcher": "id", "bathand": "vs_hand"})
    pside["role"] = "pit"
    platoon = pd.concat([bside, pside], ignore_index=True)
    files = {
        "batting": bat[[c for c in bat.columns if c not in DROP]],
        "pitching": pit[[c for c in pit.columns if c not in DROP]],
        "gameinfo": gi[[c for c in GAME_COLS if c in gi.columns]],
        "players": players[["id", "last", "first", "bat", "throw", "team"]].drop_duplicates("id"),
        "platoon": platoon,
    }
    for name, frame in files.items():
        frame.to_csv(out / f"{name}_{season}.csv.gz", index=False, compression="gzip")
    zpath.unlink()                       # 10 MB a season; the outputs are what we keep
    return {"batting": len(bat), "pitching": len(pit), "games": len(gi),
            "postseason_games": int((gi["gametype"].astype(str).str.lower() != "regular").sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2007)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--out", default=str(ROOT / "data-raw" / "mlb" / "history_full"))
    ap.add_argument("--cache", default=str(ROOT / "data-raw" / "mlb" / "_cache"))
    args = ap.parse_args()
    out, failed = Path(args.out), []
    for season in range(args.first, args.last + 1):
        if (out / f"platoon_{season}.csv.gz").exists():
            print(f"{season}: already exported")
            continue
        try:
            print(f"{season}: {export(season, out, Path(args.cache))}", flush=True)
        except Exception as exc:                       # noqa: BLE001 - reported, retried by rerun
            print(f"{season}: FAILED {type(exc).__name__}: {exc}", flush=True)
            failed.append(season)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
