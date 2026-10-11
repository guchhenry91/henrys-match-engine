"""Keep this season's per-match Understat rows current, and list upcoming fixtures.

For the rebuilt soccer props (leagues/forecast.py). Each league's current season
file data-raw/leagues/understat_matches/<LEAGUE>_<year>.csv.gz gains the matches
played since the last run (one getMatchData request per NEW match only -- about ten
a league a week), and data-raw/leagues/understat_matches/upcoming.json lists every
unplayed fixture with Understat's match id, so the board can ask the model about it.

Never fatal: a league that fails keeps its previous file.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leagues import forecast as fc
from scripts.export_understat_matches import LEAGUES, get

UPCOMING = fc.MATCHES_DIR / "upcoming.json"


def season_year(now: datetime) -> int:
    """Understat's year is the year a season STARTS (July onwards)."""
    return now.year if now.month >= 7 else now.year - 1


def sync_league(code: str, year: int, base: Path) -> tuple[int, list]:
    league = get(f"https://understat.com/getLeagueData/{LEAGUES[code]}/{year}")
    dates = league.get("dates") or []
    path = base / f"{code}_{year}.csv.gz"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(have["match_id"].astype(str)) if len(have) else set()
    new_players, new_shots = [], []
    for m in dates:
        if not m.get("isResult") or str(m["id"]) in done:
            continue
        data = get(f"https://understat.com/getMatchData/{m['id']}")
        meta = {"match_id": m["id"], "datetime": m.get("datetime"),
                "home": m["h"]["title"], "away": m["a"]["title"],
                "home_goals": m["goals"]["h"], "away_goals": m["goals"]["a"],
                "home_xg": m["xG"]["h"], "away_xg": m["xG"]["a"]}
        for side in ("h", "a"):
            for row in ((data.get("rosters") or {}).get(side) or {}).values():
                new_players.append({**meta, **row})
            for row in (data.get("shots") or {}).get(side) or []:
                new_shots.append({**meta, **row})
    added = 0
    if new_players:
        shots = pd.DataFrame(new_shots)
        if shots.empty:
            shots = pd.DataFrame(columns=["match_id", "player_id", "result", "situation", "xG"])
        rows = fc.trim(pd.DataFrame(new_players), shots, code, year)
        added = rows["match_id"].nunique()
        out = pd.concat([have, rows], ignore_index=True) if len(have) else rows
        base.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)
    upcoming = [{"league": code, "season": year, "match_id": int(m["id"]),
                 "date": m.get("datetime"), "home": m["h"]["title"], "away": m["a"]["title"]}
                for m in dates if not m.get("isResult")]
    return added, upcoming


def main() -> int:
    now = datetime.now(timezone.utc)
    year = season_year(now)
    upcoming, failed = [], []
    for code in LEAGUES:
        try:
            added, up = sync_league(code, year, fc.MATCHES_DIR)
            upcoming += up
            print(f"{code} {year}: +{added} matches, {len(up)} upcoming")
        except Exception as exc:                       # noqa: BLE001 - keep the old file
            failed.append(code)
            print(f"{code} {year}: FAILED {type(exc).__name__}: {exc}")
    if upcoming or not UPCOMING.exists():
        UPCOMING.write_text(json.dumps({"updated": now.isoformat(timespec="seconds"),
                                        "failed": failed, "matches": upcoming}, indent=1) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
