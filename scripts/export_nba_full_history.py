"""Export the completed NBA seasons WITH EVERY BOX-SCORE COLUMN (model rebuild).

docs/REBUILD_PLAN.md: the rebuilt NBA forecasts need usage and pace -- shot
attempts, free-throw attempts, turnovers, offensive rebounds -- which the original
export (scripts/export_nba_history.py) trimmed away. stats.nba.com ignores cloud
job containers but answers the Worker's stats proxy (/api/nba/leaguegamelog), so the
history is fetched through it: one request per season, ~9s each.

Writes data-raw/nba/history_full/player_{season}.csv.gz (season = the year it ENDS).
Team totals (pace, team shot volume) are sums of these player rows, so no separate
team export is needed. A finished season never changes: run once, commit.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from nba import config, current, data

OUT = Path(data.HISTORY).parent / "history_full"
KEEP = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID",
        "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]


def fetch(season: int, season_type: str = "Regular Season", opener=urllib.request.urlopen) -> pd.DataFrame:
    url = current.PROXY_URL + "/api/nba/leaguegamelog?" + urllib.parse.urlencode(
        {"season": data.season_label(season), "season_type": season_type})
    req = urllib.request.Request(url, headers={"User-Agent": "henrys-match-engine"})
    with opener(req, timeout=120) as r:
        payload = json.loads(r.read().decode("utf-8"))
    if "resultSets" not in payload:
        raise RuntimeError(payload.get("error") or "no resultSets")
    block = payload["resultSets"][0]
    frame = pd.DataFrame(block["rowSet"], columns=block["headers"])
    return frame[[c for c in KEEP if c in frame.columns]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None, help="comma list of season END years")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else list(config.SEASONS)
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for season in seasons:
        out = OUT / f"player_{season}.csv.gz"
        if out.exists():
            print(f"{out.name}: already exported")
            continue
        for attempt in range(3):
            try:
                frame = fetch(season)
                break
            except Exception as exc:                  # noqa: BLE001 - retried, reported
                print(f"  {season}: {type(exc).__name__}: {exc} (attempt {attempt + 1})")
                time.sleep(10 * (attempt + 1))
        else:
            failed.append(season)
            continue
        frame.to_csv(out, index=False, compression="gzip")
        print(f"{out.name}: {len(frame)} rows, {frame['GAME_ID'].nunique()} games")
        time.sleep(3)
    if failed:
        print(f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
