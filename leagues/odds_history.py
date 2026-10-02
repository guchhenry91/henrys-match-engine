"""A timestamped archive of soccer prices, one snapshot per change.

WHY (improvement report, point #2). Our soccer prices were only ever kept at two
moments -- the board's current price and the price frozen with a pick -- so there
was no honest record of how a price moved between first publication and kickoff,
and no way to say whether a pick beat the closing line. football-data.co.uk's
fixtures.csv (already fetched every run, free) carries bet365, the Betfair
Exchange and the market average for match result, over/under 2.5 and the Asian
handicap. Each run appends a snapshot for every upcoming fixture in our leagues --
but only when a price actually changed, so the file grows with the market's moves,
not with the run count. Fixtures more than KEEP_DAYS past kickoff are dropped.

data-raw/leagues/odds_history.json:
  {"fixtures": {"PL|2026-10-10|Arsenal|Leeds": {"kickoff": iso,
                 "snapshots": [{"at": iso, "b365": {...}, "exchange": {...}, "avg": {...}}]}}}
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from leagues.mispricing import LEAGUE_OF, _kickoff
from leagues.names import UnknownTeam, canonical

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data-raw" / "leagues" / "odds_history.json"
KEEP_DAYS = 14
COLS = {"b365": {"home": "B365H", "draw": "B365D", "away": "B365A", "over_2.5": "B365>2.5",
                 "under_2.5": "B365<2.5", "ah_home": "B365AHH", "ah_away": "B365AHA"},
        "exchange": {"home": "BFEH", "draw": "BFED", "away": "BFEA", "over_2.5": "BFE>2.5",
                     "under_2.5": "BFE<2.5", "ah_home": "BFEAHH", "ah_away": "BFEAHA"},
        "avg": {"home": "AvgH", "draw": "AvgD", "away": "AvgA", "over_2.5": "Avg>2.5",
                "under_2.5": "Avg<2.5"}}


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 3) if x > 1.0 else None


def rows(frame: pd.DataFrame):
    """(key, kickoff, snapshot-without-time) for every fixture in our leagues."""
    for _, r in frame.iterrows():
        league = LEAGUE_OF.get(r.get("Div"))
        if not league:
            continue
        try:
            home, away = canonical(r["HomeTeam"], league), canonical(r["AwayTeam"], league)
        except UnknownTeam:
            continue
        ko = _kickoff(r.get("Date"), r.get("Time"))
        if ko is None:
            continue
        snap = {group: {k: _num(r.get(c)) for k, c in cols.items()} for group, cols in COLS.items()}
        try:
            snap["ah_line"] = float(r.get("AHh"))
        except (TypeError, ValueError):
            snap["ah_line"] = None
        if not any(v for g in COLS for v in snap[g].values()):
            continue
        yield f"{league}|{ko.date().isoformat()}|{home}|{away}", ko, snap


def record(frame: pd.DataFrame, path: Path = PATH, now=None) -> int:
    """Append a snapshot per fixture whose prices changed. Returns how many."""
    now = now or datetime.now(timezone.utc)
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        store = {}
    fixtures = store.setdefault("fixtures", {})
    added = 0
    for key, ko, snap in rows(frame):
        if ko <= now:
            continue                        # in play or finished: the close is the last snapshot
        entry = fixtures.setdefault(key, {"kickoff": ko.isoformat(), "snapshots": []})
        last = entry["snapshots"][-1] if entry["snapshots"] else None
        if last and {k: last.get(k) for k in snap} == snap:
            continue
        entry["kickoff"] = ko.isoformat()
        entry["snapshots"].append({"at": now.isoformat(timespec="seconds"), **snap})
        added += 1
    cutoff = now - timedelta(days=KEEP_DAYS)
    for key in [k for k, e in fixtures.items()
                if datetime.fromisoformat(e["kickoff"]) < cutoff]:
        del fixtures[key]
    store["_note"] = ("bet365 / Betfair Exchange / market-average prices per fixture, one snapshot "
                      "per change (leagues/odds_history.py). Source: football-data.co.uk fixtures.csv.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return added
