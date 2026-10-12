"""The NBA's official pre-game injury reports, one per game day, as a table.

For NBA teammates-out (nba/forecast._vacated). The feature was first trained on
"absent from the box score" -- known only after a game, late scratches included --
while the live board could only know the injury report. This file gives training the
SAME information the board has: who the official report listed OUT before tip-off.

Source: the NBA's report archive, ak-static.cms.nba.com/referee/injury/
Injury-Report_<YYYY-MM-DD>_<HH>PM.pdf (from 2026 the name carries minutes, _05_30PM).
The 5:30pm ET report is used -- the one in force when the live board freezes picks
for evening games. Writes data-raw/nba/injury_reports/<season>.csv.gz with
date, player (as the report spells it, "Last, First"), status.

  python -m scripts.export_nba_injury_reports --first 2022 --last 2026
(seasons named by the year they END, as nba/config does). Resumable per season.
"""
import argparse
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nba import injury_report

ROOT = Path(__file__).resolve().parents[1]
OUT = injury_report.DIR
SLOTS = ("05PM", "05_30PM")


def fetch(day: date, opener=urllib.request.urlopen) -> bytes | None:
    for t in SLOTS:
        pdf = injury_report.fetch(day.isoformat(), t, opener)
        if pdf:
            return pdf
    return None


parse = injury_report.parse


def season_days(season: int, today: date):
    start, end = date(season - 1, 10, 1), min(date(season, 6, 25), today)
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2022)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    today = date.today()
    for season in range(args.first, args.last + 1):
        path = out / f"{season}.csv.gz"
        done = set()
        have = pd.DataFrame()
        if path.exists():
            have = pd.read_csv(path)
            done = set(have["date"])
            if season < args.last and len(have):
                print(f"{season}: already exported ({len(have)} rows)")
                continue
        rows, days = [], 0
        for d in season_days(season, today):
            if d.isoformat() in done:
                continue
            pdf = fetch(d)
            time.sleep(0.25)
            if pdf is None:
                continue
            days += 1
            rows += [{"date": d.isoformat(), "player": n, "status": s} for n, s in parse(pdf)]
            if days % 25 == 0:
                print(f"  {season}: {days} report days, {len(rows)} rows", flush=True)
        frame = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if len(have) else pd.DataFrame(rows)
        frame.to_csv(path, index=False, compression="gzip")
        print(f"{season}: {days} new report days, {len(frame)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
