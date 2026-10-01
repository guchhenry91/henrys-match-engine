"""bet365 NFL yardage lines that beat Pinnacle's fair price -- no model needed.

The same idea as leagues/mispricing.py. Pinnacle is the sharpest bookmaker; its
de-vigged price is the reference for the true chance. bet365's yardage lines come
from API-NFL (data-raw/nfl/odds.json), Pinnacle's from The Odds API
(data-raw/nfl/odds_api_props.json), both already fetched -- no extra credits.

SAME LINE: EV = Pinnacle's fair chance of the side x bet365's price - 1.
NEARBY LINE: bet365 often hangs a line a yard or few away from Pinnacle. The fair
chance is moved across the gap with a normal approximation of the stat around
Pinnacle's line (SPREAD per market, typical game-to-game spread for a starter),
and flagged as "line-adjusted" -- an estimate, held to a higher bar. Lines further
apart than MAX_GAP, and tiny lines (a quarterback's 1.5 rushing yards), are
skipped: lumpy stats are not normal there.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from nfl.odds import match_player

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data-raw" / "nfl" / "mispriced_log.json"
MIN_EV = 0.03
MIN_EV_ADJUSTED = 0.05
SPREAD = {"passing_yards": 65.0, "rushing_yards": 25.0, "receiving_yards": 25.0}
MAX_GAP = {"passing_yards": 8.0, "rushing_yards": 4.0, "receiving_yards": 4.0}
MIN_LINE = {"passing_yards": 150.0, "rushing_yards": 15.0, "receiving_yards": 15.0}


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p):
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def _fair_over(q):
    o, u = q.get("odd_over"), q.get("odd_under")
    if not o or not u:
        return None
    io, iu = 1.0 / float(o), 1.0 / float(u)
    return io / (io + iu)


def scan(bet365: dict, odds_api_games: dict) -> list:
    out = []
    for game, markets in (bet365 or {}).items():
        sharp_game = (odds_api_games.get(game) or {}).get("props") or {}
        for market, players in (markets or {}).items():
            if market not in SPREAD:
                continue
            for name, b in (players or {}).items():
                p = match_player(sharp_game.get(market) or {}, name)
                if not p or str(p.get("book", "")).lower() != "pinnacle":
                    continue
                fo = _fair_over(p)
                bl, pl = float(b.get("line") or 0), float(p.get("line") or 0)
                if fo is None or min(bl, pl) < MIN_LINE[market]:
                    continue
                gap = bl - pl
                if abs(gap) > MAX_GAP[market]:
                    continue
                if gap == 0:
                    over, basis = fo, "same line"
                else:
                    mu = pl + SPREAD[market] * _phi_inv(fo)         # Pinnacle's implied centre
                    over, basis = 1.0 - _phi((bl - mu) / SPREAD[market]), "line-adjusted"
                bar = MIN_EV if basis == "same line" else MIN_EV_ADJUSTED
                for side, prob, price in (("over", over, b.get("odd_over")), ("under", 1 - over, b.get("odd_under"))):
                    if not price:
                        continue
                    ev = prob * float(price) - 1.0
                    if ev >= bar:
                        out.append({"game": game, "market": market, "player": b.get("_name") or name,
                                    "side": side, "line": bl, "b365": float(price),
                                    "pinnacle_line": pl, "fair_p": round(prob, 4),
                                    "ev": round(ev, 4), "basis": basis})
    return out


def key(e) -> str:
    return f"{e['game']}|{e['market']}|{e['player']}|{e['side']}|{e['line']}"


def update(found: list, games: dict, stats_for, now=None) -> dict:
    """Log flags at first sight; grade finished ones.

    games: "HOME|AWAY" -> {"kickoff", "season", "week"} for the slate being priced.
    stats_for(season, week, name, teams) -> the player's stat row or None.
    """
    now = now or datetime.now(timezone.utc)
    try:
        log = json.loads(LOG.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    stamp = now.isoformat(timespec="seconds")
    for e in found:
        g = games.get(e["game"])
        if not g or datetime.fromisoformat(str(g["kickoff"]).replace("Z", "+00:00")) <= now:
            continue
        k = key(e)
        if k not in log:
            log[k] = {**e, **g, "first_seen": stamp, "book_price": e["b365"], "book": "bet365"}
        log[k]["last_seen"], log[k]["ev_latest"] = stamp, e["ev"]
    for e in log.values():
        if e.get("graded"):
            continue
        ko = datetime.fromisoformat(str(e["kickoff"]).replace("Z", "+00:00"))
        if (now - ko).total_seconds() < 4 * 3600:
            continue
        row = stats_for(int(e["season"]), int(e["week"]), e["player"], e["game"].split("|"))
        if row is None:
            continue
        actual = float(row[e["market"]])
        e["actual"] = actual
        hit = actual > e["line"] if e["side"] == "over" else actual < e["line"]
        e["graded"] = "correct" if hit else "wrong"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    upcoming = sorted((e for e in log.values() if not e.get("graded")
                       and datetime.fromisoformat(str(e["kickoff"]).replace("Z", "+00:00")) > now),
                      key=lambda e: -e.get("ev_latest", e["ev"]))
    settled = sorted((e for e in log.values() if e.get("graded")), key=lambda e: e["kickoff"], reverse=True)
    return {"updated": stamp, "min_ev": MIN_EV, "source": "Pinnacle (The Odds API) vs bet365 (API-NFL)",
            "upcoming": upcoming, "settled": settled}
