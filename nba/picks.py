"""Freeze NBA picks before tip-off and grade them from the box score.

The same discipline as every other board in this repo: a pick enters the record
only if it was frozen BEFORE the game started, it is graded against exactly what
was frozen (the line included), and nothing is re-picked afterwards.

WHEN A PICK FREEZES. On the last scheduled NBA run before tip-off
(nba.book_lines.is_last_run_before) -- the same run that takes the second, "lock"
bookmaker check, so the frozen line is the freshest one the budget allows -- or
on any run inside FREEZE_HOURS of tip-off. A run that is skipped by GitHub means
the pick is never frozen and never counted, which is an honest gap rather than a
late lock.

GRADING
  * Team winner: the final score in the NBA's schedule feed.
  * Props: the player's line in that game's box score (nba.current). Over wins
    on stat > line; a whole-number line landing exactly is a PUSH (void), as a
    book settles it. A player with no box-score row in a game whose box score we
    DO hold did not play: VOID, as books void a prop for a player who never takes
    the floor. A game whose box score we do not hold yet stays PENDING -- never
    graded against missing data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from oddsapi import props as shared

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data-raw" / "nba" / "picks_log.json"
FREEZE_HOURS = 2.0


def load_log(path=LOG) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_log(log: dict, path=LOG) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(Path(path))


def game_key(g: dict) -> str:
    return f"{g['game_id']}:winner"


def prop_key(p: dict) -> str:
    key = f"{p['game_id']}:{p['market']}:{p['player_id']}"
    return key + ":under" if p.get("side") == "under" else key


def should_freeze(tipoff, now, last_run_before) -> bool:
    """Freeze on the last scheduled run before tip-off, or within FREEZE_HOURS."""
    hours = (shared.utc(tipoff) - shared.utc(now)).total_seconds() / 3600.0
    return hours > 0 and (hours <= FREEZE_HOURS or last_run_before(tipoff, now))


def freeze(payload: dict, log: dict, now, last_run_before) -> int:
    """Freeze every published pick that is due. Idempotent: a frozen key is never
    rewritten, so a later run cannot move the line or the probability."""
    frozen = 0
    stamp = shared.utc(now).isoformat(timespec="seconds")
    for g in payload.get("games") or []:
        if g.get("pick") and game_key(g) not in log and \
                should_freeze(g["tipoff"], now, last_run_before):
            log[game_key(g)] = {
                "kind": "winner", "game_id": g["game_id"], "tipoff": g["tipoff"],
                "home": g["home"], "away": g["away"], "pick": g["pick"],
                "p_pick": g["p_pick"], "book_p_pick": g.get("book_p_pick"),
                "book": g.get("book"), "locked_at": stamp}
            frozen += 1
    for market, block in (payload.get("props") or {}).items():
        for p in block.get("picks") or []:
            if prop_key(p) in log or not should_freeze(p["tipoff"], now, last_run_before):
                continue
            log[prop_key(p)] = {
                "kind": "prop", "game_id": p["game_id"], "tipoff": p["tipoff"],
                "market": market, "player": p["player"], "player_id": p["player_id"],
                "team": p["team"], "opponent": p["opponent"], "line": p["line"],
                "line_source": p["line_source"], "book": p.get("book"),
                "probability": p["probability"], "book_p": p.get("book_p"),
                "book_price": p.get("book_price"), "edge": p.get("edge"),
                "availability": p.get("availability"), "side": p.get("side", "over"),
                "p_model": p.get("p_model"), "locked_at": stamp}
            frozen += 1
    return frozen


STAT = {"points": "PTS", "rebounds": "REB", "assists": "AST", "threes": "FG3M"}


def grade(log: dict, finals: dict, box: dict, covered: set) -> None:
    """Settle every frozen pick the data can settle, in place.

    finals:  game_id -> "home" / "away" (the winner), finished games only
    box:     (game_id, player_id) -> {stat: value}
    covered: game ids whose box score we hold"""
    for entry in log.values():
        if entry.get("graded"):
            continue
        gid = str(entry["game_id"])
        if entry["kind"] == "winner":
            winner = finals.get(gid)
            if winner is None:
                continue
            team = entry["home"] if winner == "home" else entry["away"]
            entry["result"] = team
            entry["graded"] = "correct" if entry["pick"] == team else "wrong"
            continue
        if gid not in covered:
            continue
        row = box.get((gid, int(entry["player_id"])))
        if row is None:
            entry["graded"], entry["actual"] = "void", None
            entry["void_reason"] = "did not play"
            continue
        actual = float(row[STAT[entry["market"]]])
        entry["actual"] = actual
        line = float(entry["line"])
        hit = actual < line if entry.get("side") == "under" else actual > line
        entry["graded"] = "void" if actual == line else "correct" if hit else "wrong"


def record(log: dict) -> dict:
    """{team_winner: {...}, props: {...}, props_by_market: {...}}."""
    def tally(entries):
        out = {"correct": 0, "wrong": 0, "void": 0, "pending": 0}
        for e in entries:
            out[e.get("graded") or "pending"] += 1
        settled = out["correct"] + out["wrong"]
        out["settled"] = settled
        out["hit_rate"] = round(out["correct"] / settled, 4) if settled else None
        return out
    games = [e for e in log.values() if e["kind"] == "winner"]
    props = [e for e in log.values() if e["kind"] == "prop"]
    return {"team_winner": tally(games), "props": tally(props),
            "props_by_market": {m: tally([e for e in props if e["market"] == m])
                                for m in STAT}}


def settled(log: dict, limit: int = 200) -> list:
    rows = [dict(e, key=k) for k, e in log.items() if e.get("graded")]
    rows.sort(key=lambda e: e["tipoff"], reverse=True)
    return rows[:limit]


def now_utc():
    return datetime.now(timezone.utc)
