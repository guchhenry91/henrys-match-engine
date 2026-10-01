"""Phase 3: bet before the book moves -- picks lifted by FRESH team news.

When a starter is ruled out, his teammates inherit his volume (nfl/vacancy.py,
the NBA boost in nfl/model.py). Books do re-price them, but not instantly: the
window between the news and the move is where a bettor at one book can still get
the old line. So the board needs to know not only WHO is out but SINCE WHEN.

Neither feed timestamps a status change, so this module keeps its own record: the
first run that saw each player ruled out. A pick whose lift comes from a teammate
ruled out within FRESH_HOURS is tagged a "news edge" -- check bet365 now, before
the line catches up. After that it is ordinary news the price already carries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

FRESH_HOURS = 12.0
MIN_VACATED = 0.05      # a teammate out who carried less than this is not news


def _utc(x) -> datetime:
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def remember(path, out: dict, now=None) -> dict:
    """Record when each player was FIRST seen ruled out; forget anyone no longer out.

    `out` is {player name: {"team": code, ...}} for players ruled out right now.
    Returns {name: {"team", "since"}} and writes it back to `path`.
    """
    now = _utc(now or datetime.now(timezone.utc))
    path = Path(path)
    try:
        held = json.loads(path.read_text(encoding="utf-8")).get("players") or {}
    except Exception:
        held = {}
    state = {}
    for name, info in (out or {}).items():
        team = (info or {}).get("team")
        prev = held.get(name)
        since = prev["since"] if prev and prev.get("team") == team else now.isoformat(timespec="seconds")
        state[name] = {"team": team, "since": since}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_note": ("When each player was first seen ruled OUT (tracking/news_edge.py). "
                  "Written by the board run; a player leaves when no longer out."),
        "players": dict(sorted(state.items()))}, indent=1) + "\n", encoding="utf-8")
    return state


def annotate(picks, state: dict, now=None) -> int:
    """Tag picks whose team lost a player within FRESH_HOURS. Returns how many."""
    now = _utc(now or datetime.now(timezone.utc))
    by_team = {}
    for name, s in (state or {}).items():
        try:
            hours = (now - _utc(s["since"])).total_seconds() / 3600.0
        except Exception:
            continue
        by_team.setdefault(s.get("team"), []).append((hours, name))
    n = 0
    for p in picks:
        p["news_edge"] = None
        if (p.get("vacated") or 0) < MIN_VACATED or p.get("side") == "under":
            continue                      # the lift favours the over; no lift, no edge
        fresh = sorted(x for x in by_team.get(p.get("team"), []) if x[0] <= FRESH_HOURS)
        if not fresh:
            continue
        p["news_edge"] = {"out": [name for _, name in fresh[:3]],
                          "hours_ago": round(fresh[0][0], 1)}
        n += 1
    return n
