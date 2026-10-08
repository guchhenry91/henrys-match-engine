"""Who is on each NBA team NOW, from a published roster (data-raw/nba/rosters.json).

Box scores say where a player last PLAYED; before opening night that is last
season for every team that has not played a preseason game yet, so a summer trade
(Durant to Houston, Trae Young to Washington) stays invisible for weeks. A roster
says where he IS.

Sources, both free and keyless, checked 2026-10-08 against the 328 players in the
preseason box scores then held:

  * Sleeper (api.sleeper.app/v1/players/nba) -- the whole league in ONE request,
    30 teams, 327 of 328 agreeing, none contradicting. Primary.
  * ESPN team rosters (site.api.espn.com, one request a team) -- 325 of 328, none
    contradicting. Fallback when Sleeper does not answer.

Players are joined by NAME (neither feed carries the NBA's PLAYER_ID), with the
same rule as everywhere else: a name two players share -- in the roster or in our
own history -- is dropped, never resolved by guessing.

CORROBORATION. A roster may only be used if it agrees with the box scores we
already hold: of the players both know, at least MIN_AGREEMENT must be on the
same team. The NFL learned this the expensive way (nfl/rosters.py: a plausible
feed that recognised 49% of real players deleted 177 of them). A refused roster
is reported and the previous one kept.

Refreshed at most every MAX_AGE_HOURS (Sleeper asks for at most one players call a
day). Pure stdlib.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nfl.odds import norm_name

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data-raw" / "nba" / "rosters.json"
SLEEPER_URL = "https://api.sleeper.app/v1/players/nba"
ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_ROSTER_URL = ESPN_TEAMS_URL + "/{id}/roster"
# ESPN's short codes -> the NBA tricodes the rest of the engine uses.
ESPN_CODES = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA",
              "WSH": "WAS", "PHO": "PHX", "BRK": "BKN"}
TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU",
         "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL",
         "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}
MAX_AGE_HOURS = 20.0
MIN_TEAMS = 30               # every team, or it is not a roster of the league
MIN_PER_TEAM = 13            # an NBA team carries 15+ (18-24 in camp)
MIN_AGREEMENT = 0.90         # of the players the box scores and the roster both know
MIN_OVERLAP = 50             # fewer shared players than this cannot corroborate


def _json(url: str, opener, timeout: float = 45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 henrys-match-engine",
                                               "Accept": "application/json"})
    with opener(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _one_team_each(pairs) -> dict:
    """{norm name: team} from (name, team) pairs; a name on two teams is dropped."""
    seen, dupes = {}, set()
    for name, team in pairs:
        key = norm_name(name)
        if not key or team not in TEAMS:
            continue
        if key in seen and seen[key] != team:
            dupes.add(key)
        seen[key] = team
    return {k: v for k, v in seen.items() if k not in dupes}


def parse_sleeper(payload: dict) -> dict:
    pairs = []
    for p in (payload or {}).values():
        if not isinstance(p, dict) or not p.get("active") or not p.get("team"):
            continue
        name = p.get("full_name") or f"{p.get('first_name') or ''} {p.get('last_name') or ''}"
        pairs.append((name, str(p["team"]).upper()))
    return _one_team_each(pairs)


def fetch_sleeper(opener=urllib.request.urlopen) -> dict:
    return parse_sleeper(_json(SLEEPER_URL, opener))


def fetch_espn(opener=urllib.request.urlopen) -> dict:
    teams = _json(ESPN_TEAMS_URL, opener)["sports"][0]["leagues"][0]["teams"]
    pairs = []
    for t in teams:
        t = t.get("team") or {}
        code = ESPN_CODES.get(t.get("abbreviation"), t.get("abbreviation"))
        roster = _json(ESPN_ROSTER_URL.format(id=t["id"]), opener)
        pairs += [(a.get("fullName") or a.get("displayName"), code)
                  for a in roster.get("athletes") or []]
    return _one_team_each(pairs)


def check(players: dict, box_teams: dict) -> tuple:
    """(ok, reason). `box_teams` is {norm name: team} from this season's box scores."""
    per_team = {}
    for team in players.values():
        per_team[team] = per_team.get(team, 0) + 1
    thin = sorted(t for t in TEAMS if per_team.get(t, 0) < MIN_PER_TEAM)
    if len(per_team) < MIN_TEAMS or thin:
        return False, f"{len(per_team)} teams, thin or missing: {thin}"
    shared = [k for k in box_teams if k in players]
    if len(shared) < MIN_OVERLAP:
        # Before any box score this season there is nothing to corroborate against;
        # a complete 30-team roster is then accepted on its shape alone.
        return (True, f"{len(shared)} players to corroborate against; shape checks only") \
            if not box_teams else (False, f"only {len(shared)} players shared with the box scores")
    agree = sum(players[k] == box_teams[k] for k in shared) / len(shared)
    if agree < MIN_AGREEMENT:
        return False, f"agrees on {agree:.0%} of {len(shared)} shared players"
    return True, f"agrees on {agree:.1%} of {len(shared)} shared players"


def load() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _utc(text):
    t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def sync(box_teams: dict, now=None, opener=urllib.request.urlopen, fetchers=None) -> dict:
    """Refresh the store when stale. Returns {"status", "source", "reason"}."""
    now = now or datetime.now(timezone.utc)
    store = load()
    if store.get("fetched_at") and now - _utc(store["fetched_at"]) < timedelta(hours=MAX_AGE_HOURS):
        return {"status": "fresh", "source": store.get("source"), "reason": store.get("check")}
    fetchers = fetchers or (("sleeper", lambda: fetch_sleeper(opener)),
                            ("espn", lambda: fetch_espn(opener)))
    reasons = []
    for source, fetch in fetchers:
        try:
            players = fetch()
        except Exception as exc:                      # noqa: BLE001 - reported
            reasons.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        ok, why = check(players, box_teams)
        if not ok:
            reasons.append(f"{source} refused: {why}")
            continue
        out = {"source": source, "fetched_at": now.isoformat(timespec="seconds"),
               "check": why, "players": dict(sorted(players.items())),
               "_note": "Current NBA rosters, norm name -> team. Written by "
                        "nba/rosters.py -- never by hand."}
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(STORE)
        return {"status": "updated", "source": source, "reason": why,
                "players": len(players), "failed": reasons}
    return {"status": "kept previous" if store else "none", "source": store.get("source"),
            "reason": "; ".join(reasons)}


def teams_for(names: dict, store=None) -> tuple:
    """({PLAYER_ID: team}, fetched date 'YYYY-MM-DD' or None) for players the roster
    names. `names` is {norm name: PLAYER_ID} (already free of shared names)."""
    store = store if store is not None else load()
    players = store.get("players") or {}
    out = {int(pid): players[key] for key, pid in names.items() if key in players}
    day = str(store.get("fetched_at") or "")[:10] or None
    return out, day
