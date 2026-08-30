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
LEAGUES_ORDER = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1", "SERIEA"]

# A parlay's legs must all fall inside this window, measured from the EARLIEST
# one. Four days covers a Friday-to-Monday round or a Tuesday-to-Thursday midweek
# -- the shape an accumulator actually takes.
#
# TWO REASONS, and the second is the serious one.
#
# PLACEABILITY: an acca spanning nine days is not a bet anyone places. The board
# built a four-fold running from 19 to 29 August, across two matchweeks, that
# could not settle for eleven days.
#
# STALENESS: a parlay freezes when its EARLIEST leg is 48h out, because that is
# the last moment the whole thing could have been placed -- correct, but it means
# every later leg freezes at the same instant. That four-fold committed "Dortmund
# to win" ELEVEN DAYS before kickoff, and a props treble committed Mbappe's shot
# line eight days out. The rest of this app locks a pick 45 minutes before kickoff
# precisely so team news reaches it; those legs were frozen before a single
# lineup, injury or transfer could be known, with the window still open until
# 1 September. Capping the span bounds how stale the freshest-frozen leg can be.
MAX_LEG_SPAN_HOURS = 96.0


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
    # NAME IT AFTER WHAT IT IS. "_one_per_league" returns one leg per league that
    # HAS a qualifying leg in the window, so it is not always four -- and once the
    # span cap started dropping out-of-round legs, a three-leg parlay was still
    # being labelled "Balanced four-fold". A label that contradicts the legs
    # printed underneath it is the kind of small dishonesty that makes a reader
    # doubt the rest of the board.
    FOLD = {2: "double", 3: "treble", 4: "four-fold", 5: "five-fold", 6: "six-fold"}
    all_parlays.append(_tier(f"Balanced {FOLD.get(len(four), f'{len(four)}-fold')}",
                             "One leg from every league with a pick this round",
                             four, feat=True))
    # Props treble: three DIFFERENT markets across different matches, so it
    # actually earns its "attempt / on target / goal" billing instead of stacking
    # two of the same market. Prefer distinct markets first; only if fewer than
    # three markets are available do we fill with the next-best distinct-match leg.
    treble, used, seen_mkt = [], set(), set()
    for leg in sorted(prop_legs, key=lambda x: -x["p"]):
        key = (leg["league_key"], leg["mid"])
        if key in used or leg["tag"] in seen_mkt:
            continue
        treble.append(leg); used.add(key); seen_mkt.add(leg["tag"])
        if len(treble) == 3:
            break
    if len(treble) < 3:
        for leg in sorted(prop_legs, key=lambda x: -x["p"]):
            key = (leg["league_key"], leg["mid"])
            if key in used:
                continue
            treble.append(leg); used.add(key)
            if len(treble) == 3:
                break
    all_parlays.append(_tier("Props treble", "Three player-prop legs, distinct markets where the board allows", treble))
    sections.append({"title": "All four leagues",
                     "sub": "Legs spread across the Premier League, La Liga, Bundesliga and Ligue 1",
                     "parlays": [p for p in all_parlays if p]})

    # ---- Shots & on target only -------------------------------------------
    # Shot-VOLUME markets pooled on their own. These are the two best-calibrated
    # markets the board has (scripts/props_pick_calibration.py: shots and sot both
    # land within a few points of their stated rate on the leagues with a
    # judgeable sample, while the goalscorer market has too few distinct players
    # to judge), so an accumulator built only from them rests on firmer ground
    # than one that mixes in scorer legs.
    #
    # ONE LEG PER MATCH still applies and it bites hard here: Haaland and Semenyo
    # are the same fixture, as are Mbappe and Vinicius, and Bundesliga props are
    # ungradeable and already excluded upstream -- so nine published picks collapse
    # to three usable legs today. That shrinks as the fixture window widens; it is
    # the honest count, not a cap.
    shot_legs = [l for l in prop_legs if l["market"] in ("shots", "sot")]
    so_parlays = []
    pair = _independent(shot_legs, 2)
    so_parlays.append(_tier("Shots double",
                            "The two safest shot-volume calls", pair))
    full = _independent(shot_legs, 99)
    # Only worth showing when it actually adds legs over the double -- otherwise
    # it is the same bet under a second name.
    if len(full) > len(pair):
        # Deliberately NOT forced to alternate markets. Within one fixture a
        # player's "2+ attempts" and "1+ on target" are the same bet twice, so
        # only the stronger of the two can be used; taking the best available leg
        # per match is what maximises the combined number. Today that resolves to
        # on-target every time, which is a result rather than a restriction.
        so_parlays.append(_tier(
            "Shots accumulator",
            "Best shot-volume leg from every available match, drawn from both "
            "2+ attempts and 1+ on target", full, feat=True))
    so_parlays = [p for p in so_parlays if p]
    if so_parlays:
        sections.append({
            "title": "Shots & on target",
            "sub": "2+ attempts and 1+ on target only -- no scorer or match-winner legs",
            "parlays": so_parlays})

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


def _within_window(team_legs: list[dict], prop_legs: list[dict], now) -> tuple:
    """Keep only legs inside MAX_LEG_SPAN_HOURS of the earliest UPCOMING fixture.

    Anchored on the earliest leg still ahead of `now`, so the board follows the
    next round rather than trailing a fixture that has already kicked off. Returns
    both lists filtered against the SAME anchor, so a parlay mixing a match winner
    and a prop cannot straddle two rounds.
    """
    all_legs = team_legs + prop_legs
    ahead = [picks._utc(l["date"]) for l in all_legs if picks._utc(l["date"]) > now]
    if not ahead:
        return team_legs, prop_legs
    cutoff = min(ahead) + pd.Timedelta(hours=MAX_LEG_SPAN_HOURS)
    keep = lambda ls: [l for l in ls if picks._utc(l["date"]) <= cutoff]
    return keep(team_legs), keep(prop_legs)


def build_parlays(best: dict, pp: dict, log_path: str | Path, now=None) -> dict:
    """Assemble, freeze and grade the model's parlays. `best`/`pp` are the freshly
    built cross-league board dicts (build_best_picks / build_player_picks)."""
    now = picks._utc(now if now is not None else pd.Timestamp.now("UTC"))
    team_legs = [_match_leg(u) for u in best.get("upcoming", []) if u.get("p_pick")]
    # ONLY gradeable prop legs. A pick that can never settle never appears in
    # player_picks.json's `settled` list, so its leg-id would never resolve in
    # _outcome_lookup and the parlay would sit "pending" forever, silently
    # dropped from the record. Match-winner legs are always fine -- results are
    # gradeable everywhere -- so this filter is on props only.
    #
    # This USED to mean "no Bundesliga props", since its shot events crash
    # upstream and gradeable was tied to that one feed. It no longer does:
    # API-Football settles Bundesliga fixtures, so its goal and shots picks are
    # gradeable and eligible here. Its shots-on-target market is still withheld
    # at source, for a different reason -- the rates behind it cannot be measured
    # (see build_player_picks) -- so no SOT leg arises there anyway.
    prop_legs = [_prop_leg(u) for u in pp.get("upcoming", [])
                 if u.get("p_pick") and u.get("gradeable") is not False]
    team_legs, prop_legs = _within_window(team_legs, prop_legs, now)
    sections = _build_sections(team_legs, prop_legs)

    log = picks.load_log(log_path)
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
