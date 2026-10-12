"""THE CLOSING PRICE: one more look at each locked prop just before it starts.

Why (owner approved the credits, 2026-10-12): profit needs about a thousand bets
before luck washes out; whether the price we took beats the market's FINAL price
shows up in a few hundred. Until now the "close" on a prop was the lock-time quote
-- the price taken, not a close -- so tracking/performance.clv had nothing to score.

What it does, on the frequent lock job: for NBA, NFL and MLB player props that are
LOCKED, not yet closed, and start within CLOSE_MINUTES, fetch that game's lines once
(only the markets those picks need, Pinnacle first) and stamp on each entry:
  closing     {line, p (de-vigged chance of OUR side), price, book, at, moved,
              line_before}; "moved" compares the closing book's line with that SAME
              book's line at the last check before the lock ("unknown" without one)
  close_p / close_price   only when the closing line is the line we took -- a
              different line is a different bet, so its price says nothing about
              ours. Scored at the bet365 price in tracking/performance.clv.
Never rewrites a stamped entry. Never fatal: a failed fetch leaves the pick as it was.

COST: one /events/{id}/odds call per game with a locked prop, priced at the number of
markets asked (three named books are one region): NBA ~4 a game, NFL <= 4, MLB <= 4.
About 50-150 credits a day in season, inside the per-run and per-day caps.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSE_MINUTES = 20            # the lock job runs every 10 minutes: each game gets one look


def _utc(v) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _nfl_store_key(game_id: str) -> str | None:
    parts = str(game_id).split("_")                  # 2026_04_PIT_CLE: away PIT, home CLE
    return f"{parts[3]}|{parts[2]}" if len(parts) == 4 else None


def sports() -> dict:
    """Per sport: where its locked props live and how to reach its lines."""
    from nba import book_lines as nba_bl
    from mlb import book_lines as mlb_bl
    from nfl import book_lines as nfl_bl
    return {
        "nba": {"log": ROOT / "data-raw" / "nba" / "picks_log.json", "section": None,
                "store": ROOT / "data-raw" / "nba" / "odds_api.json", "bl": nba_bl,
                "start": "tipoff", "store_key": lambda e: str(e.get("game_id"))},
        "mlb": {"log": ROOT / "data-raw" / "mlb" / "picks_log.json", "section": None,
                "store": ROOT / "data-raw" / "mlb" / "odds_api.json", "bl": mlb_bl,
                "start": "tipoff", "store_key": lambda e: str(e.get("game_id"))},
        "nfl": {"log": ROOT / "data-raw" / "nfl" / "picks_log.json", "section": "props",
                "store": ROOT / "data-raw" / "nfl" / "odds_api_props.json", "bl": nfl_bl,
                "start": "kickoff", "store_key": lambda e: _nfl_store_key(e.get("game_id"))},
    }


def due(entries: dict, start_key: str, markets: set, now: datetime) -> dict:
    """{store key fn input: ...} -> the entries to close now, grouped by key later."""
    out = {}
    for k, e in entries.items():
        if str(k).startswith("_") or not isinstance(e, dict):
            continue
        if e.get("kind", "prop") != "prop" or "closing" in e or e.get("graded"):
            continue
        if e.get("market") not in markets or e.get("line") is None or e.get(start_key) is None:
            continue
        try:
            mins = (_utc(e[start_key]) - now).total_seconds() / 60.0
        except Exception:
            continue
        if 0 < mins <= CLOSE_MINUTES:
            out[k] = e
    return out


def stamp(entry: dict, quote: dict | None, now: datetime, before: dict | None = None) -> None:
    """Write the closing quote onto one locked prop.

    `before` is the SAME source's quote from the last check before the lock (the odds
    store). The line MOVED only if that book's own line moved: comparing the close
    with the line we took from another book (bet365 56.5 vs Pinnacle 58.5) records a
    move when nothing moved (third review). CLOSING-LINE VALUE needs the closing line
    to be OUR line -- the fair closing chance of exactly the bet we took -- and is
    scored at the bet365 price (tracking/performance.clv)."""
    at = now.isoformat(timespec="seconds")
    if not quote or quote.get("line") is None:
        entry["closing"] = {"at": at, "line": None}           # looked; not quoted at the close
        return
    side = entry.get("side") or "over"
    line, ours = float(quote["line"]), float(entry["line"])
    p, price = quote.get(side), quote.get(f"odd_{side}")
    same_book = bool(before and before.get("line") is not None
                     and str(before.get("book") or "").lower() == str(quote.get("book") or "").lower())
    if not same_book:
        moved = "unknown"                                    # no same-book line before the lock
    else:
        was = float(before["line"])
        moved = ("same" if line == was else
                 "ours" if (line > was) == (side == "over") else "against")
    entry["closing"] = {"line": line, "p": p, "price": price, "book": quote.get("book"),
                        "at": at, "moved": moved, "line_before": before.get("line") if before else None}
    if line == ours and p is not None and price:
        entry["close_p"] = round(float(p), 4)
        entry["close_price"] = float(price)


def run(client, now=None, only=None) -> dict:
    from nfl import odds as odds_mod
    from oddsapi import client as oc
    now = now or datetime.now(timezone.utc)
    summary = {}
    for sport, cfg in sports().items():
        if only and sport not in only:
            continue
        bl = cfg["bl"]
        wanted = {ours: theirs for theirs, ours in bl.MARKETS.items()}
        log = _read(cfg["log"])
        entries = log.get(cfg["section"], {}) if cfg["section"] else log
        todo = due(entries, cfg["start"], set(wanted), now)
        if not todo:
            continue
        store = (_read(cfg["store"]).get("games")) or {}
        by_game = {}
        for k, e in todo.items():
            by_game.setdefault(cfg["store_key"](e), []).append(e)
        stamped = calls = 0
        for gkey, es in by_game.items():
            event_id = (store.get(gkey) or {}).get("event_id")
            if not event_id:
                for e in es:
                    e["closing"] = {"at": now.isoformat(timespec="seconds"), "line": None,
                                    "why": "no bookmaker event on record"}
                continue
            markets = sorted({wanted[e["market"]] for e in es})
            try:
                payload = client.get(
                    f"sports/{bl.SPORT_KEY}/events/{event_id}/odds", sport=sport,
                    purpose=f"{sport} closing check {gkey}",
                    est=oc.cost(len(markets), bookmakers=len(bl.BOOK_ORDER)),
                    bookmakers=",".join(bl.BOOK_ORDER), markets=",".join(markets),
                    oddsFormat="decimal")
            except oc.BudgetExceeded as exc:
                print(f"  budget stop: {exc}")
                break
            except RuntimeError as exc:
                print(f"  {sport} {gkey}: closing check failed ({exc})")
                continue
            calls += 1
            props = bl.parse_event(payload) or {}
            before_props = (store.get(gkey) or {}).get("props") or {}
            for e in es:
                stamp(e, odds_mod.match_player(props.get(e["market"]) or {}, e["player"]), now,
                      odds_mod.match_player(before_props.get(e["market"]) or {}, e["player"]))
                stamped += 1
        if stamped or any("closing" in e for e in todo.values()):
            cfg["log"].write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary[sport] = {"games": calls, "picks": stamped}
    return summary


def main() -> int:
    from oddsapi import client as oc
    if not oc.enabled():
        print("The Odds API is switched off (ODDS_API_ENABLED); no closing check")
        return 0
    client = oc.Client()
    try:
        out = run(client)
    except (oc.Disabled, oc.BudgetExceeded) as exc:
        print(f"closing check stopped: {exc}")
        out = {}
    print(f"closing prices: {out or 'nothing starting in the next %d minutes' % CLOSE_MINUTES}")
    print(client.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
