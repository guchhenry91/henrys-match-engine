"""Every player's minutes, goals, shots and xG in every match (soccer rebuild).

docs/REBUILD_PLAN.md: the anytime-scorer and shots rebuild needs each player's
MINUTES in every match (starts, subs, zero-shot appearances) and every shot with
penalties marked -- Understat's per-season totals cannot say who played when.
Understat's own JSON endpoints serve this directly: getLeagueData/<league>/<year>
lists the matches, getMatchData/<id> returns both rosters and every shot.
(soccerdata's Understat reader needs a TLS helper library that cloud sandboxes
cannot download, so this talks to the endpoints itself.)

Writes, per league and season (Understat's year = the year the season STARTS):
  data-raw/leagues/history_full/understat/<LEAGUE>_<year>_players.csv.gz
  data-raw/leagues/history_full/understat/<LEAGUE>_<year>_shots.csv.gz
Resumable: a finished league-season file is skipped.
"""
import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {"PL": "EPL", "LALIGA": "La_liga", "BUNDESLIGA": "Bundesliga",
           "SERIEA": "Serie_A", "LIGUE1": "Ligue_1"}
HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
           "Accept-Encoding": "gzip"}
PAUSE = 0.35


def get(url: str, tries: int = 4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                return json.loads(body.decode("utf-8"))
        except Exception as exc:                       # noqa: BLE001 - retried
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"    retry {url.rsplit('/', 1)[-1]}: {type(exc).__name__}", flush=True)


def league_season(code: str, year: int, out: Path) -> str:
    pfile = out / f"{code}_{year}_players.csv.gz"
    sfile = out / f"{code}_{year}_shots.csv.gz"
    if pfile.exists() and sfile.exists():
        return "already exported"
    league = get(f"https://understat.com/getLeagueData/{LEAGUES[code]}/{year}")
    matches = [m for m in league.get("dates") or [] if m.get("isResult")]
    players, shots = [], []
    for i, m in enumerate(matches):
        data = get(f"https://understat.com/getMatchData/{m['id']}")
        meta = {"match_id": m["id"], "datetime": m.get("datetime"),
                "home": m["h"]["title"], "away": m["a"]["title"],
                "home_goals": m["goals"]["h"], "away_goals": m["goals"]["a"],
                "home_xg": m["xG"]["h"], "away_xg": m["xG"]["a"]}
        for side in ("h", "a"):
            for row in ((data.get("rosters") or {}).get(side) or {}).values():
                players.append({**meta, **row})
            for row in (data.get("shots") or {}).get(side) or []:
                shots.append({**meta, **row})
        time.sleep(PAUSE)
        if (i + 1) % 100 == 0:
            print(f"    {code} {year}: {i + 1}/{len(matches)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shots).to_csv(sfile, index=False, compression="gzip")
    pd.DataFrame(players).to_csv(pfile, index=False, compression="gzip")
    return f"{len(matches)} matches, {len(players)} player rows, {len(shots)} shots"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2017)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--out", default=str(ROOT / "data-raw" / "leagues" / "history_full" / "understat"))
    args = ap.parse_args()
    failed = []
    for year in range(args.last, args.first - 1, -1):          # newest first
        for code in LEAGUES:
            try:
                print(f"{code} {year}: {league_season(code, year, Path(args.out))}", flush=True)
            except Exception as exc:                            # noqa: BLE001 - rerun resumes
                print(f"{code} {year}: FAILED {type(exc).__name__}: {exc}", flush=True)
                failed.append((code, year))
    print("FAILED:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
