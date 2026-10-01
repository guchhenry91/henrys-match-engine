"""bet365's prices for PLAYED fixtures, stamped onto the frozen match picks.

football-data.co.uk's season file carries, per played match, bet365's pre-match
1X2 (B365H/D/A -- the same snapshot fixtures.csv showed before kickoff, so the
same number the board offered) and bet365's CLOSING 1X2 (B365CH/CD/CA). With
both, a graded pick can be scored at a price and for closing-line value
(tracking/performance.py) even when no price was frozen at lock -- which was
every soccer pick before 2026-10-01.

A price frozen at lock is never overwritten; only missing fields are filled.
"""
from __future__ import annotations

import io

import pandas as pd

from leagues import config, history
from leagues.names import UnknownTeam, canonical

SIDES = ("home", "draw", "away")


def devig(h: float, d: float, a: float):
    """Proportional de-vig, as leagues.backtest.devig -- repeated here so this
    module needs pandas only (backtest imports the whole model stack)."""
    raw = (1.0 / h, 1.0 / d, 1.0 / a)
    total = sum(raw)
    return tuple(x / total for x in raw)


def _triple(r, cols):
    try:
        vals = tuple(float(r[c]) for c in cols)
    except (KeyError, TypeError, ValueError):
        return None
    return vals if all(v > 1.0 for v in vals) else None


def b365_rows(league: str, text: str | None = None) -> dict:
    """{(home, away): {"pre": (h, d, a) | None, "close": (h, d, a) | None}}."""
    lg = config.get(league)
    if text is None:
        text = history.fetch_csv(history.current_fd_season(), lg.fd_code)
    df = pd.read_csv(io.StringIO(text), encoding="latin-1")
    out = {}
    for _, r in df.iterrows():
        try:
            key = (canonical(r["HomeTeam"], league), canonical(r["AwayTeam"], league))
        except (UnknownTeam, KeyError, TypeError):
            continue
        out[key] = {"pre": _triple(r, ("B365H", "B365D", "B365A")),
                    "close": _triple(r, ("B365CH", "B365CD", "B365CA"))}
    return out


def stamp(log: dict, played: pd.DataFrame, rows: dict, season_tag: str) -> int:
    """Fill price, open and close on graded picks. Returns how many changed."""
    by_id = {int(m["match_id"]): m for _, m in played.iterrows()}
    n = 0
    for key, entry in log.items():
        if (not str(key).startswith(f"{season_tag}:") or not isinstance(entry, dict)
                or entry.get("graded") is None or "close_p" in entry):
            continue
        m = by_id.get(int(str(key).split(":", 1)[1]))
        q = rows.get((m["home"], m["away"])) if m is not None else None
        if not q:
            continue
        pick = entry.get("pick")
        i = 0 if pick == m["home"] else 2 if pick == m["away"] else 1
        pre, close = q["pre"], q["close"]
        if pre and entry.get("odds") is None:
            entry["odds"] = pre[i]
            entry["price_from"] = "bet365 pre-match, football-data (after the fact)"
        if pre and "open" not in entry:
            entry["open"] = {"p": round(devig(*pre)[i], 4), "price": pre[i],
                             "book": "bet365"}
        if close:
            entry["close_p"] = round(devig(*close)[i], 4)
            entry["close_price"] = close[i]
        n += 1
    return n
