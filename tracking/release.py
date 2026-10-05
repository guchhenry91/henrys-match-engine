"""Official or tracked? What the app actually recommended, frozen with each pick.

WHY (improvement report, points #4 and #8: explicit release states, a shadow stage
before production). Every pick is graded, but only some were ever RECOMMENDED -- a
value verdict at bet365 in a market whose own record had earned a bet (the Bet
list, Phase 4). Without recording which, the record of "what the app told me to
bet" is mixed in with everything it merely tracks, and could only be rebuilt with
hindsight about market status. So, at freeze:

  release = "official"  value at bet365 AND the market was Promising/Proven then
            "tracked"   everything else -- the shadow record that earns a market
                        its place

The market status is read from the board as last PUBLISHED (before this run
overwrites it), i.e. the status a reader saw when the pick was made. Stamped only
on recently frozen picks with no release yet; never back-dated.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BETTABLE = {"promising", "proven"}
WINDOW_HOURS = 3
BOARDS = {"nfl": ["data/nfl/board.json"], "nba": ["data/nba/board.json"],
          "mlb": ["data/mlb/board.json"],
          "soccer": ["data/leagues/best.json", "data/leagues/player_picks.json"]}


def status_map(sport: str) -> dict:
    """{market: status} from the boards as last published."""
    out = {}
    for rel in BOARDS[sport]:
        try:
            perf = json.loads((ROOT / rel).read_text(encoding="utf-8")).get("performance") or {}
        except Exception:
            continue
        for market, s in perf.items():
            # Stakes follow the VALUE picks' own record (performance.bet_status);
            # a board published before that existed earns nothing.
            if isinstance(s, dict) and s.get("status"):
                out[market] = s.get("bet_status") or "testing"
    return out


def mark(entries, sport: str, market_of, statuses: dict | None = None, now=None) -> int:
    """Set entry["release"] on recently frozen picks. Returns how many were official."""
    # The RUN's time, not the wall clock: a run (or a test) labels the picks it froze
    # itself. Accepts a pandas Timestamp; a naive time is UTC.
    now = now if now is not None else datetime.now(timezone.utc)
    if hasattr(now, "to_pydatetime"):
        now = now.to_pydatetime()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    statuses = status_map(sport) if statuses is None else statuses
    official = 0
    for e in entries:
        if not isinstance(e, dict) or "release" in e or not e.get("locked_at"):
            continue
        try:
            locked = datetime.fromisoformat(str(e["locked_at"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if locked.tzinfo is None:
            locked = locked.replace(tzinfo=timezone.utc)
        if now - locked > timedelta(hours=WINDOW_HOURS):
            continue
        market = market_of(e)
        is_value = bool((e.get("value") or {}).get("value"))
        e["release"] = "official" if is_value and statuses.get(market) in BETTABLE else "tracked"
        e["release_status_seen"] = statuses.get(market)
        official += e["release"] == "official"
    return official
