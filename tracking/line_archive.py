"""Every bookmaker line the engine pays for, kept (the model rebuild's test data).

docs/REBUILD_PLAN.md: the old prop models were judged on lines they set themselves,
and no history of REAL bookmaker lines existed to test against. Every
/events/{id}/odds and /odds reply from The Odds API passes through
oddsapi.client.Client.get, which hands it here, so every sport is archived by one
hook and nothing extra is ever requested.

One CSV per sport per UTC day: data-raw/odds_api/lines/<sport>/<YYYY-MM-DD>.csv.gz,
one row per (book, market, player, side, line) quote with the time it was seen, so
the first and last quotes before a start are both there (open and close).

Size: main markets are kept from every book in the reply (best price, consensus);
alternate-line ladders only from the first book that quotes them, which is enough
to test the forecast at off-main lines without storing every book's ladder.
"""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data-raw" / "odds_api" / "lines"
COLUMNS = ["seen_at", "event_id", "commence", "home", "away", "book", "market",
           "player", "side", "line", "price"]


def rows(payload, seen_at: str) -> list:
    """Flatten one reply (an event dict, or a list of them) into archive rows."""
    events = payload if isinstance(payload, list) else [payload]
    out = []
    for ev in events:
        if not isinstance(ev, dict) or not ev.get("bookmakers"):
            continue
        alt_from = {}                     # alternate market -> the one book kept
        for book in ev.get("bookmakers") or []:
            bkey = book.get("key")
            for m in book.get("markets") or []:
                mkey = str(m.get("key") or "")
                if "alternate" in mkey:
                    if alt_from.setdefault(mkey, bkey) != bkey:
                        continue
                for o in m.get("outcomes") or []:
                    if o.get("price") is None:
                        continue
                    out.append([seen_at, ev.get("id"), ev.get("commence_time"),
                                ev.get("home_team"), ev.get("away_team"), bkey, mkey,
                                o.get("description") or "", str(o.get("name") or ""),
                                "" if o.get("point") is None else o.get("point"),
                                o.get("price")])
    return out


def record(sport: str, payload, seen_at: str, base: Path | None = None) -> int:
    """Append one reply's quotes to today's file for `sport`. Returns rows added."""
    new = rows(payload, seen_at)
    if not new:
        return 0
    folder = (base or DIR) / str(sport).lower()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{seen_at[:10]}.csv.gz"
    old = []
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            old = list(reader)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    writer.writerows(old)
    writer.writerows(new)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    tmp.replace(path)
    return len(new)


def load(sport: str, base: Path | None = None):
    """Every archived row for `sport` as a DataFrame (pandas imported lazily)."""
    import pandas as pd
    folder = (base or DIR) / str(sport).lower()
    frames = [pd.read_csv(p) for p in sorted(folder.glob("*.csv.gz"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
