"""Model-built accumulators (parlays) from the published cross-league boards.

A parlay stacks legs the model ALREADY publishes and grades -- match-winner Best
Picks and player props -- into one all-or-nothing bet. Two rules keep the combined
number honest rather than decorative:

  1. ONE LEG PER MATCH. Two legs from the same fixture (say "Dortmund win" and
     "Guirassy to score") are correlated, so multiplying their probabilities is a
     lie. Every leg in a parlay comes from a DIFFERENT match, which keeps them
     independent and makes the combined probability the true product of the legs.

  2. THE SAME FREEZING DISCIPLINE AS EVERY OTHER PICK. A parlay's legs are frozen
     at lock time (see picks.py) and graded from the result -- never re-picked in
     hindsight. The parlay is CORRECT only if every leg is correct, WRONG the
     moment any leg is wrong, and VOID if it was first locked after its earliest
     leg had already kicked off.

The legs themselves are the model's real published selections; this module only
COMBINES and FREEZES them, it never invents a pick. So a parlay is exactly as
accurate as the picks underneath it -- no more, no less, which is the honest
ceiling.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from leagues import picks

# The prop markets, in the order a mixed "props" parlay prefers them, with the
# short tag the UI shows. Match wins use the "w" tag.
PROP_TAG = {"shots": "a", "sot": "o", "goal": "g"}   # a=attempts, o=on target, g=goal
TAG_NAME = {"w": "Win", "g": "Goal", "a": "Attempts", "o": "On target"}
LEAGUES_ORDER = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def _leg_id(lk: str, mid, mkt: str, player: str | None) -> str:
    """Stable identity for a leg, used to freeze it and to look up its result.
    A match-winner leg is one-per-match (`mkt='w'`, no player); a prop leg is
    keyed by its market and player so it grades against the right line."""
    return f"{lk}#{int(mid)}#{mkt}" + (f"#{player}" if player else "")


def _match_leg(u: dict) -> dict:
    return {"id": _leg_id(u["league_key"], u["id"], "w", None),
            "kind": "match", "tag": "w",
            "selection": f"{u['pick']} to win",
            "league_key": u["league_key"], "league": u["league"],
            "match": f"{u['home']} v {u['away']}", "date": u["date"],
            "mid": int(u["id"]), "p": float(u["p_pick"] or 0)}


def _prop_leg(u: dict) -> dict:
    verb = "to score" if u["market"] == "goal" else u["line"]
    sel = (f"{u['player']} {verb}" if u["market"] == "goal"
           else f"{u['player']} {u['line']}")
    return {"id": _leg_id(u["league_key"], u["id"], u["market"], u["player"]),
            "kind": "prop", "tag": PROP_TAG.get(u["market"], "a"),
            "selection": sel, "player": u["player"], "market": u["market"],
            "league_key": u["league_key"], "league": u["league"],
            "match": f"{u['home']} v {u['away']}", "date": u["date"],
            "mid": int(u["id"]), "p": float(u["p_pick"] or 0)}


def _independent(pool: list[dict], n: int, per_league: bool = False,
                 used_matches: set | None = None) -> list[dict]:
    """Greedily take the highest-probability legs with DISTINCT matches (and,
    when per_league, distinct leagues too). Never two legs from one fixture."""
    used_matches = set() if used_matches is None else set(used_matches)
    used_leagues: set = set()
    out: list[dict] = []
    for leg in sorted(pool, key=lambda x: -x["p"]):
        key = (leg["league_key"], leg["mid"])
        if key in used_matches:
            continue
        if per_league and leg["league_key"] in used_leagues:
            continue
        out.append(leg)
        used_matches.add(key)
        used_leagues.add(leg["league_key"])
        if len(out) == n:
            break
    return out


def _combined(legs: list[dict]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg["p"]
    return p


def _one_per_league(team_legs: list[dict], prop_legs: list[dict]) -> list[dict]:
    """Best available leg (match OR prop) from each league, distinct matches --
    the featured 'one from every league' four-fold."""
    best_by_lg: dict[str, dict] = {}
    for leg in team_legs + prop_legs:
        cur = best_by_lg.get(leg["league_key"])
        if cur is None or leg["p"] > cur["p"]:
            best_by_lg[leg["league_key"]] = leg
    out, used = [], set()
    for lk in LEAGUES_ORDER:
        leg = best_by_lg.get(lk)
        if leg and (leg["league_key"], leg["mid"]) not in used:
            out.append(leg)
            used.add((leg["league_key"], leg["mid"]))
    return out


def _tier(name: str, note: str, legs: list[dict], feat: bool = False) -> dict | None:
    """One parlay. Returns None if it could not be filled with enough legs."""
    if len(legs) < 2:
        return None
    return {"tier": name, "note": note, "feat": feat,
            "combined": round(_combined(legs), 4),
            "legs": [{"selection": l["selection"], "match": l["match"],
                      "league": l["league"], "league_key": l["league_key"],
                      "tag": l["tag"], "p": round(l["p"], 4), "id": l["id"],
                      "date": l["date"]} for l in legs]}


def _build_sections(team_legs: list[dict], prop_legs: list[dict]) -> list[dict]:
    sections = []

    # ---- All four leagues -------------------------------------------------
    all_parlays = []
    banker = _independent(team_legs, 2, per_league=True)
    all_parlays.append(_tier("Banker", "Two safest calls, two leagues", banker))
    four = _one_per_league(team_legs, prop_legs)
    all_parlays.append(_tier("Balanced four-fold", "One leg from every league", four, feat=True))
    # Props treble preferring three DIFFERENT markets across different matches.
    treble, used, seen_mkt = [], set(), set()
    for leg in sorted(prop_legs, key=lambda x: -x["p"]):
        key = (leg["league_key"], leg["mid"])
        if key in used:
            continue
        treble.append(leg); used.add(key); seen_mkt.add(leg["tag"])
        if len(treble) == 3:
            break
    all_parlays.append(_tier("Props treble", "Attempt · on target · goal player markets", treble))
    sections.append({"title": "All four leagues",
                     "sub": "Legs spread across the Premier League, La Liga, Bundesliga and Ligue 1",
                     "parlays": [p for p in all_parlays if p]})

    # ---- Premier League only ---------------------------------------------
    pl_team = [l for l in team_legs if l["league_key"] == "PL"]
    pl_prop = [l for l in prop_legs if l["league_key"] == "PL"]
    pl_parlays = []
    pl_bank = _independent(pl_team + pl_prop, 2)
    pl_parlays.append(_tier("PL Banker", "Two safest Premier League calls", pl_bank))
    pl_acca = _independent(pl_team + pl_prop, 3)
    pl_parlays.append(_tier("PL Acca", "The model's Premier League treble", pl_acca, feat=True))
    pl_goals = _independent([l for l in pl_prop if l["market"] == "goal"], 2)
    pl_parlays.append(_tier("PL Goals double", "Two Premier League scorers", pl_goals))
    sections.append({"title": "Premier League only",
                     "sub": "Only Premier League games and Premier League player props",
                     "parlays": [p for p in pl_parlays if p]})
    return sections


# --------------------------------------------------------------------------
# Freezing + grading
# --------------------------------------------------------------------------
def _parlay_key(legs: list[dict]) -> str:
    return "|".join(sorted(l["id"] for l in legs))


def lock_parlay(log: dict, legs: list[dict], now=None) -> dict:
    """Freeze a parlay once, before its EARLIEST leg kicks off -- the last moment
    the whole acca could have been placed. Frozen legs and their lock-time
    probabilities are stored so the combined number cannot be re-chosen later."""
    key = _parlay_key(legs)
    if key in log:
        return log[key]
    now = picks._utc(now if now is not None else pd.Timestamp.now("UTC"))
    earliest = min(picks._utc(l["date"]) for l in legs)
    late_by = (now - earliest).total_seconds() / 3600.0
    log[key] = {
        "legs": [{"id": l["id"], "selection": l["selection"],
                  "p": round(l["p"], 4)} for l in legs],
        "combined": round(_combined(legs), 4),
        "locked_at": now.isoformat(),
        "earliest_kickoff": earliest.isoformat(),
        "tainted": bool(late_by > picks.LATE_LOCK_HOURS),
    }
    return log[key]


def grade_parlay(entry: dict, outcomes: dict) -> str:
    """All-or-nothing. `outcomes` maps leg id -> 'correct'/'wrong'/'void'.
    A parlay is void if it locked late or any leg voided; wrong the moment a leg
    is wrong; correct only when EVERY leg is correct; else still pending."""
    if entry.get("tainted"):
        return "void"
    grades = [outcomes.get(l["id"]) for l in entry["legs"]]
    if any(g == "void" for g in grades):
        return "void"
    if any(g == "wrong" for g in grades):
        return "wrong"
    if all(g == "correct" for g in grades):
        return "correct"
    return "pending"


def _outcome_lookup(best: dict, pp: dict) -> dict:
    """Map every settled leg id -> its grade, from the two boards' settled lists."""
    out = {}
    for s in best.get("settled", []):
        out[_leg_id(s["league_key"], s["id"], "w", None)] = s.get("graded")
    for s in pp.get("settled", []):
        out[_leg_id(s["league_key"], s["id"], s["market"], s["player"])] = s.get("graded")
    return out


def build_parlays(best: dict, pp: dict, log_path: str | Path, now=None) -> dict:
    """Assemble, freeze and grade the model's parlays. `best`/`pp` are the freshly
    built cross-league board dicts (build_best_picks / build_player_picks)."""
    team_legs = [_match_leg(u) for u in best.get("upcoming", []) if u.get("p_pick")]
    prop_legs = [_prop_leg(u) for u in pp.get("upcoming", []) if u.get("p_pick")]
    sections = _build_sections(team_legs, prop_legs)

    log = picks.load_log(log_path)
    now = picks._utc(now if now is not None else pd.Timestamp.now("UTC"))
    LOCK_H = 48.0    # lock a parlay once its earliest leg is inside this window

    # Freeze any upcoming parlay whose earliest leg is within the lock window.
    for sec in sections:
        for pa in sec["parlays"]:
            legs = pa["legs"]
            earliest = min(picks._utc(l["date"]) for l in legs)
            hours_out = (earliest - now).total_seconds() / 3600.0
            pa["provisional"] = True
            if hours_out <= LOCK_H:
                entry = lock_parlay(log, legs, now=now)
                pa["provisional"] = False
                pa["locked_at"] = entry["locked_at"]
    picks.save_log(log, log_path)

    # Grade every FROZEN parlay against the settled boards.
    outcomes = _outcome_lookup(best, pp)
    settled = []
    for key, entry in log.items():
        g = grade_parlay(entry, outcomes)
        if g in ("correct", "wrong", "void"):
            settled.append({"combined": entry["combined"], "graded": g,
                            "legs": entry["legs"], "kickoff": entry["earliest_kickoff"]})
    rec = {"correct": 0, "wrong": 0, "void": 0, "total": 0, "pending": 0}
    for s in settled:
        if s["graded"] == "void":
            rec["void"] += 1
        else:
            rec["total"] += 1
            rec[s["graded"]] += 1
    # Parlays still open (locked, not yet fully settled).
    for key, entry in log.items():
        if grade_parlay(entry, outcomes) == "pending":
            rec["pending"] += 1

    return {"updated": now.isoformat(), "method": "one leg per match; combined = product of legs",
            "sections": sections, "record": rec,
            "settled": sorted(settled, key=lambda s: s["kickoff"], reverse=True)}
