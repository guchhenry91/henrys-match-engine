"""Freeze MLB picks before first pitch and grade them from StatsAPI box scores.

Same rules as nba/picks.py: frozen on the last scheduled run before the game (or
within FREEZE_HOURS), graded against exactly what was frozen, never re-picked.
  * Team winner: the final score.
  * Props: the player's line in the box score. Over wins on stat > line; a whole
    line landing exactly is a push (void). A player not in the starting lineup of
    a game whose box score we hold did not start: VOID (books void a prop for a
    player who does not play). A box score we do not hold yet: PENDING.
  * Team totals: the team's runs.
"""
from __future__ import annotations

from nba import picks as base

LOG = base.ROOT / "data-raw" / "mlb" / "picks_log.json"
STAT = {"hits": "H", "hrr": "HRR", "hr": "HR", "rbi": "RBI", "strikeouts": "K",
        "team_runs": "RUNS"}


def load_log():
    return base.load_log(LOG)


def save_log(log):
    base.save_log(log, LOG)


freeze = base.freeze
game_key = base.game_key
prop_key = base.prop_key


def grade(log: dict, finals: dict, box: dict, covered: set) -> None:
    """finals: game_id -> "home"/"away"; box: (game_id, player_id) -> {stat: v};
    covered: game ids whose box score we hold."""
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
        row = box.get((gid, str(entry["player_id"])))
        if row is None:
            entry["graded"], entry["actual"] = "void", None
            entry["void_reason"] = "did not start"
            continue
        actual = float(row[STAT[entry["market"]]])
        entry["actual"] = actual
        line = float(entry["line"])
        hit = actual < line if entry.get("side") == "under" else actual > line
        entry["graded"] = "void" if actual == line else "correct" if hit else "wrong"


def record(log: dict) -> dict:
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
            "props_by_market": {m: tally([e for e in props if e["market"] == m]) for m in STAT}}


settled = base.settled
