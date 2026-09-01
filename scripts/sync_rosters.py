"""Snapshot current first-team squads for every configured league, from API-Football.

ONE SOURCE, ON PURPOSE. This used to fall back to ESPN's public feeds whenever
API-Football failed for a league. That fallback is gone, and removing it made the
sync MORE honest rather than less robust:

  * IT HAD STOPPED WORKING. ESPN began answering 403 to every request -- for all
    five leagues, on a plain urllib call -- so the "fallback" could not have
    rescued anything. A fallback that cannot run is not redundancy, it is a false
    sense of it, and it hid the real state: rosters simply stopped refreshing.
  * IT MIXED TWO SCHEMAS. An ESPN snapshot and an API-Football one carry different
    player ids, so a league rescued by the fallback silently changed identity
    space mid-file. `_league_verified_at` was deliberately NOT stamped for those
    leagues to keep them due for a real refresh, which is a lot of machinery to
    maintain around a source we do not trust enough to stamp.

This snapshot is an audit input, not a source of performance statistics:
Understat remains the rate source.

WITHOUT A KEY IT NOW FAILS LOUDLY. Previously a missing API_FOOTBALL_KEY silently
took the ESPN path, so a run with no key looked like a successful refresh. There
is no second path any more, so there is nothing to fail over TO, and pretending
otherwise is how a stale file passes for a fresh one.

Run during an open transfer window and before each publish:
    python -m scripts.sync_rosters
"""
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time

from leagues.names import canonical, UnknownTeam
from leagues import config
from leagues.api_football import Client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "leagues" / "rosters.json"
# DERIVED FROM THE ENGINE'S OWN LEAGUE LIST, not restated. Adding Serie A found
# the four-league list written out in eleven places; this is one fewer.
LEAGUES = {key: key for key in config.LEAGUES}

# API-Football's league ids. Serie A is 135, UNVERIFIED against this account but
# FAIL-SAFE: a wrong id returns no teams, which raises and leaves that league's
# previous snapshot untouched rather than replacing it with someone else's squad.
API_LEAGUES = {"PL": 39, "LALIGA": 140, "BUNDESLIGA": 78, "LIGUE1": 61,
               "SERIEA": 135}
API_SEASON = 2026


def _stamp(payload, league):
    return (payload.get("_league_verified_at") or {}).get(
        league, payload.get("_verified_at"))


def _fresh(payload, league, now, hours=36):
    try:
        checked = datetime.fromisoformat(str(_stamp(payload, league)).replace("Z", "+00:00"))
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
        return (now - checked.astimezone(timezone.utc)).total_seconds() < hours * 3600
    except (TypeError, ValueError):
        return False


def fetch_api_league(client, key):
    teams = client.get("teams", league=API_LEAGUES[key], season=API_SEASON)
    # Resolve every team's canonical name FIRST, before spending any
    # players/squads calls -- one unmapped name used to abort mid-league after
    # already burning quota on the teams fetched before it, and only reported
    # that one name, so fixing the alias file took one CI round-trip per name
    # instead of one round-trip for the whole league.
    unmapped = []
    resolved = []
    for item in teams:
        team = item["team"]
        try:
            resolved.append((team, canonical(team["name"], key)))
        except UnknownTeam:
            unmapped.append(team["name"])
    if unmapped:
        raise UnknownTeam(
            f"{len(unmapped)} team(s) not mapped for league {key}: "
            f"{sorted(unmapped)}. Add them to leagues/names.py ALIASES.")
    result = {}
    for team, club in resolved:
        squad_rows = client.get("players/squads", team=team["id"])
        players = (squad_rows[0].get("players") or []) if squad_rows else []
        result[club] = {
            "source": f"api-football:team:{team['id']}",
            # API-Football returns names HTML-escaped ("N. O&apos;Reilly"), which
            # never matches Understat's "Nico O'Reilly" and silently drops the
            # player as departed. Decode at ingest so the stored snapshot holds
            # the real name.
            "players": sorted([{
                "id": str(player["id"]),
                "name": html.unescape(player["name"] or ""),
                "position": player.get("position") or "",
            } for player in players], key=lambda player: player["name"]),
        }
    return result


def main():
    previous = None
    if OUT.exists():
        try:
            previous = json.loads(OUT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    now = datetime.now(timezone.utc)
    if not os.environ.get("API_FOOTBALL_KEY"):
        # NO SECOND PATH TO FAIL OVER TO. This used to fall through to ESPN, so a
        # run without a key looked like a successful refresh and the snapshot
        # quietly aged. Refusing is the honest outcome: the caller keeps its last
        # verified file and knows why it did not move.
        print("API_FOOTBALL_KEY is not set; rosters NOT refreshed. There is no "
              "fallback source any more -- the previous snapshot is retained "
              "unchanged, and publish will warn that it is stale.")
        return 1

    payload = previous or {}
    payload.setdefault("_league_verified_at", {})
    # EVERY CONFIGURED LEAGUE, not a hardcoded rotation.
    #
    # This alternated two hardcoded pairs -- (PL, BUNDESLIGA) and
    # (LALIGA, LIGUE1) -- so a fifth league was not merely late, it was NEVER
    # FETCHED FROM THE API AT ALL. Serie A silently fell through to the ESPN
    # fallback, and when ESPN started answering 403 it had no roster evidence
    # of any kind. A rotation that names leagues by hand cannot survive a new
    # one being added, so it now comes from config.LEAGUES.
    #
    # The pairing existed to ration a 100-a-day free plan: ~21 calls a league
    # (one teams call plus one squad call per club) made 42 the sensible daily
    # spend. The account actually allows 7,500 -- the client now reads that
    # from the response headers rather than anyone assuming it -- so all five
    # leagues cost about 105, or 1.4% of a day's allowance.
    #
    # `_fresh` remains the real guard: a league refreshed inside 36 hours is
    # skipped, so repeated runs on the same day cost nothing.
    due = [key for key in LEAGUES if not _fresh(payload, key, now)]
    # Manual override (FORCE_ROSTER_LEAGUES=PL,BUNDESLIGA) for catching a pair
    # up outside its normal 48h slot -- e.g. right after a key change, when a
    # league has been sitting on the ESPN fallback and shouldn't wait out the
    # rotation. Never used by the scheduled runs, only workflow_dispatch.
    forced = [k.strip().upper() for k in
              os.environ.get("FORCE_ROSTER_LEAGUES", "").split(",") if k.strip()]
    for key in forced:
        if key in LEAGUES and key not in due:
            due.append(key)
    if not due:
        print("API-Football rosters are fresh; no quota used")
        return 0
    # Per-league, not one try/except around the whole loop: one club's name
    # unmapped in ONE league used to raise mid-loop and discard every league
    # already fetched successfully in the same run (observed: PL fetched fine,
    # Bundesliga then raised on an unmapped club, and the handler threw away
    # the good PL fetch too). A league that fails now simply KEEPS ITS
    # PREVIOUS DATA -- untouched, unstamped, and therefore due again next run.
    #
    # There is deliberately nothing to fall back to. The old ESPN rescue could
    # not have run (403 on every request) and mixed two id schemas when it
    # did, so its absence costs nothing real and removes a branch that made a
    # dead source look like redundancy.
    # Budget sized from what is actually due -- one teams call plus one squad
    # call per club, with headroom -- instead of a constant that silently
    # truncated a league mid-fetch once the list grew.
    budget = sum(config.get(key).n_teams + 2 for key in due) + 5
    client = Client(limit=budget)
    print(f"refreshing {len(due)} league(s): {due} (budget {budget} requests)")
    failed = []
    for key in due:
        try:
            payload[key] = fetch_api_league(client, key)
            payload["_league_verified_at"][key] = now.isoformat()
            print(f"{key}: {len(payload[key])} clubs refreshed from API-Football")
        except Exception as exc:
            print(f"WARNING: roster refresh FAILED for {key} ({exc}); keeping "
                  f"its last verified snapshot, which stays due next run")
            failed.append(key)
    # NOTHING REFRESHED MEANS NOTHING WRITTEN. Rewriting the file to record that
    # every league failed would move `_source` and the modification time while the
    # squads themselves are untouched -- a file that looks freshly written but
    # holds nothing new is exactly the confusion this whole path is trying to
    # avoid. The previous snapshot is left byte-identical.
    if failed and len(failed) == len(due):
        print("no league refreshed; the previous snapshot is retained unchanged")
        return 1

    if payload.get("_league_verified_at"):
        payload["_verified_at"] = max(payload["_league_verified_at"].values())
    payload["_source"] = ("API-Football current squad feeds" if not failed
                          else "API-Football current squad feeds "
                               "(NOT refreshed this run: " + ", ".join(failed)
                               + " -- their previous snapshot is retained)")
    payload["_provisional"] = (
        "Squads rotate through a quota-aware 48-hour refresh; team news and "
        "confirmed lineups are checked separately near kickoff.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT}; {client.report()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
