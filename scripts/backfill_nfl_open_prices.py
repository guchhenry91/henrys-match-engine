"""One-off: price and OPEN quotes for NFL picks frozen before they were recorded.

Until 2026-10-01 an NFL team-winner pick froze the book's fair probability but not
its price, and no first-seen quote was kept for winners or props -- so the record
could not score winners at a price or measure closing-line value at all. Every
version of the odds files is in git, so both are recoverable from what the app
actually held at the time, never estimated:

  price at lock -- the newest odds.json version at or before the pick froze
  open          -- the oldest version that quotes it (winner, or that player)

Only entries with no "open" yet are touched; run it again and nothing changes.
    python -m scripts.backfill_nfl_open_prices
"""
import json
import subprocess
import sys
from pathlib import Path

from leagues.picks import save_log
from nfl.odds import match_player
from tracking.performance import side_quote

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data-raw" / "nfl" / "picks_log.json"
B365 = "data-raw/nfl/odds.json"
ODDS_API = "data-raw/nfl/odds_api_props.json"


def versions(path):
    """[(time, payload)] oldest first. The time is the file's own "updated" stamp
    (when the sync ran), else the commit time."""
    log = subprocess.run(["git", "log", "--format=%H %cI", "--", path], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    out = []
    for line in filter(None, log):
        sha, when = line.split(" ", 1)
        try:
            blob = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout
            data = json.loads(blob)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        out.append((str(data.get("updated") or when), data))
    return sorted(out, key=lambda v: v[0])


def _iso_le(a, b):
    from datetime import datetime
    pa = datetime.fromisoformat(str(a).replace("Z", "+00:00"))
    pb = datetime.fromisoformat(str(b).replace("Z", "+00:00"))
    return pa <= pb


def winner(entry, b365):
    key = f"{entry.get('home')}|{entry.get('away')}"
    side = ("home" if entry.get("pick") == entry.get("home")
            else "away" if entry.get("pick") == entry.get("away") else None)
    if side is None or not entry.get("locked_at"):
        return False
    seen = [(t, d["games"][key]) for t, d in b365 if key in (d.get("games") or {})]
    at_lock = [(t, g) for t, g in seen if _iso_le(t, entry["locked_at"])]
    if not at_lock:
        return False
    (t_open, g_open), (t_close, g_close) = seen[0], at_lock[-1]

    def price(g):
        return g.get(f"odd_{side}") or (round(1.0 / g[f"raw_{side}"], 3)
                                        if g.get(f"raw_{side}") else None)
    if price(g_close) is None:
        return False
    entry["book_price"] = price(g_close)
    entry["book"] = entry.get("book") or g_close.get("book")
    entry["close_p"] = g_close.get(side)
    entry["close_at"] = t_close
    entry["open"] = {"p": g_open.get(side), "price": price(g_open),
                     "book": g_open.get("book"), "at": t_open}
    entry["price_from"] = "odds.json history (backfilled)"
    return True


def prop(entry, b365, odds_api):
    # nflverse ids read season_week_AWAY_HOME; the odds files key HOME|AWAY. (A
    # prop's own "home" field is a flag or his team, and "away" his opponent.)
    parts = str(entry.get("game_id") or "").split("_")
    if len(parts) != 4:
        return False
    key = f"{parts[3]}|{parts[2]}"
    market, name = entry.get("market"), entry.get("player")
    if entry.get("book_p") is None or not entry.get("locked_at"):
        return False
    if str(entry.get("line_source")).lower() == "bet365":
        blocks = [(t, ((d.get("props") or {}).get(key) or {}).get(market) or {})
                  for t, d in b365]
    else:
        blocks = [(t, ((((d.get("games") or {}).get(key) or {}).get("props")) or {})
                   .get(market) or {}) for t, d in odds_api]
    seen = [(t, q) for t, b in blocks if (q := match_player(b, name)) is not None]
    at_lock = [(t, q) for t, q in seen if _iso_le(t, entry["locked_at"])]
    if not at_lock:
        return False
    t_open, q_open = seen[0]
    p, price = side_quote(q_open, entry.get("side"))
    entry["open"] = {"line": q_open.get("line"), "p": p, "price": price,
                     "book": q_open.get("book"), "at": t_open}
    entry["close_at"] = at_lock[-1][0]
    return True


def repair_open_moneylines(b365) -> int:
    """odds.json games whose kept "open" lacks a time (written in the hour before
    decimal prices were kept) get their true first-seen quote from history."""
    path = ROOT / B365
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for key, line in (data.get("games") or {}).items():
        if (line.get("open") or {}).get("at"):
            continue
        seen = [(t, d["games"][key]) for t, d in b365 if key in (d.get("games") or {})]
        if not seen:
            continue
        t, g = seen[0]
        line["open"] = {"home": g.get("home"), "away": g.get("away"), "book": g.get("book"),
                        "odd_home": g.get("odd_home") or round(1.0 / g["raw_home"], 3),
                        "odd_away": g.get("odd_away") or round(1.0 / g["raw_away"], 3),
                        "at": t}
        n += 1
    if n:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return n


def main() -> int:
    log = json.loads(LOG.read_text(encoding="utf-8"))
    b365, odds_api = versions(B365), versions(ODDS_API)
    print(f"odds.json versions: {len(b365)}; odds_api_props.json versions: {len(odds_api)}")
    w = sum(winner(e, b365) for k, e in (log.get("games") or {}).items()
            if not k.startswith("_") and isinstance(e, dict) and "open" not in e)
    p = sum(prop(e, b365, odds_api) for k, e in (log.get("props") or {}).items()
            if not k.startswith("_") and isinstance(e, dict) and "open" not in e)
    print(f"backfilled {w} winner pick(s) and {p} prop(s); "
          f"repaired {repair_open_moneylines(b365)} open moneyline(s) in odds.json")
    save_log(log, LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
