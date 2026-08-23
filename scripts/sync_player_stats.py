"""Per-match player stats from API-Football, for fixtures Understat has not filed.

WHY THIS EXISTS. Understat is the only per-match player feed the engine had, and
on 2026-08-23 it had published NOTHING for the 2026-27 season -- newest row
2026-05-24 -- while 26 fixtures had been played. Because the frame was full of
LAST season it did not look absent, so every player pick was graded against a
fixture the feed had never seen, and every one of them lost. The board read 0
correct / 18 wrong. None of it was real.

leagues.publish now refuses to grade a side the feed cannot speak about, which
makes those picks pending instead of false. Pending is honest but it is not a
record. This closes the gap from the other side: API-Football already supplies
lineups and results here, and its fixtures/players endpoint carries exactly the
three numbers these markets settle on -- goals, shot attempts, shots on target.

IT ONLY FETCHES WHAT IT MUST. One request per fixture, and only for fixtures
that have a frozen pick, have been played, are absent from Understat, and are
not already cached. On a normal matchday that is a handful of calls; once
Understat catches up it is none.

NAMES ARE RESOLVED HERE, ONCE. The two feeds spell players differently, so the
join happens at write time against the pick names we actually need, using the
same guarded matcher the roster rescue uses (players.resolve_squad_name),
constrained to ONE CLUB IN ONE FIXTURE. The raw API squad is stored alongside so
a questionable join can be audited later rather than taken on trust. A player
who cannot be matched confidently is left out and stays pending -- an ungraded
pick is a gap, a wrongly-joined one is a lie about a different footballer.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leagues import fixtures, players
from leagues import picks as picks_mod
from leagues.api_football import Client
from leagues.names import canonical, UnknownTeam
from scripts.sync_rosters import API_LEAGUES

ROOT = Path(__file__).resolve().parents[1]
PICKS_DIR = ROOT / "data-raw" / "leagues"
CACHE = PICKS_DIR / "player_stats.json"
LEAGUES = ("PL", "LALIGA", "LIGUE1", "BUNDESLIGA")
# One call per fixture. Keep well inside the free tier, which the lineup ladder
# and the result sync also draw on.
RUN_BUDGET = 25


def _fixture_key(date_iso, home, away):
    return f"{date_iso}|{home}|{away}"


def _understat_cover(league):
    """(team, date) pairs Understat already covers -- never spend quota on these."""
    try:
        a = players.match_player_stats(league)
    except Exception as exc:
        print(f"  {league}: Understat unreadable ({exc}); treating as uncovered")
        return set()
    if a.empty:
        return set()
    return {(r["team"], pd.to_datetime(r["date"]).date()) for _, r in a.iterrows()}


def wanted(league):
    """Frozen picks on played fixtures: {(date, home, away): {team: {players}}}."""
    log_path = PICKS_DIR / league.lower() / "player_picks_log.json"
    if not log_path.exists():
        return {}
    log = picks_mod.load_log(log_path)
    try:
        fx = fixtures.fetch_fixtures(league)
    except Exception as exc:
        print(f"  {league}: no fixtures ({exc})")
        return {}
    by_id = {int(r["match_id"]): r for _, r in fx.iterrows()}
    out = {}
    for key, entry in log.items():
        parts = str(key).split(":")
        if len(parts) < 4 or not parts[1].isdigit():
            continue
        row = by_id.get(int(parts[1]))
        if row is None or not bool(row["played"]):
            continue
        date = pd.Timestamp(row["date"]).date()
        slot = out.setdefault((date, row["home"], row["away"]), {})
        slot.setdefault(entry.get("team"), set()).add(entry.get("player"))
    return out


def _extract(side, league, squads):
    """One team's block from fixtures/players -> (team, raw names, matched stats)."""
    try:
        team = canonical(side["team"]["name"], league)
    except UnknownTeam:
        return None, [], {}
    names, stats_by_name = [], {}
    for item in side.get("players") or []:
        nm = (item.get("player") or {}).get("name")
        if not nm:
            continue
        names.append(nm)
        st = (item.get("statistics") or [{}])[0] or {}
        shots = st.get("shots") or {}
        goals = st.get("goals") or {}
        mins = (st.get("games") or {}).get("minutes")
        stats_by_name[nm] = {
            "goals": int(goals.get("total") or 0),
            "shots": int(shots.get("total") or 0),
            "sot": int(shots.get("on") or 0),
            "minutes": None if mins is None else int(mins),
        }
    matched = {}
    for ours in squads.get(team, ()):
        theirs = players.resolve_squad_name(ours, names)
        if theirs is None:
            print(f"  UNMATCHED {ours} ({team}) -- left pending on purpose")
            continue
        matched[ours] = {"team": team, "api_name": theirs, **stats_by_name[theirs]}
    return team, names, matched


def main(now=None):
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("API_FOOTBALL_KEY is not set; skipping player-stat sync")
        return 0
    now = now or datetime.now(timezone.utc)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    todo = []
    for league in LEAGUES:
        want = wanted(league)
        if not want:
            continue
        have_us = _understat_cover(league)
        cached = cache.get(league, {})
        for (date, home, away), squads in sorted(want.items()):
            if _fixture_key(date.isoformat(), home, away) in cached:
                continue
            # If Understat already covers every side we hold a pick for, the
            # better feed has it and this one should not spend a request.
            if all((t, date) in have_us for t in squads if t):
                continue
            todo.append((league, date, home, away, squads))

    if not todo:
        print("every played fixture with a pick is already covered; no quota used")
        return 0
    print(f"{len(todo)} fixture(s) need player stats")

    client = Client(limit=RUN_BUDGET)
    listings, added = {}, 0
    for league, date, home, away, squads in todo:
        iso = date.isoformat()
        try:
            if (league, iso) not in listings:
                listings[(league, iso)] = client.get(
                    "fixtures", date=iso, timezone="UTC",
                    league=API_LEAGUES.get(league), season=date.year)
            fid = None
            for c in listings[(league, iso)]:
                try:
                    h = canonical(c["teams"]["home"]["name"], league)
                    a = canonical(c["teams"]["away"]["name"], league)
                except UnknownTeam:
                    continue
                if (h, a) == (home, away):
                    fid = c["fixture"]["id"]
                    break
            if fid is None:
                print(f"  no API fixture for {home} v {away} on {iso}")
                continue
            rows = client.get("fixtures/players", fixture=fid)
        except RuntimeError as exc:
            print(f"player-stat sync stopped early ({exc}); {added} added so far")
            break

        squad_raw, matched = {}, {}
        for side in rows:
            team, names, got = _extract(side, league, squads)
            if team is None:
                continue
            squad_raw[team] = names
            matched.update(got)

        if not squad_raw:
            print(f"  {home} v {away} ({iso}): feed returned no squads; skipped")
            continue

        cache.setdefault(league, {})[_fixture_key(iso, home, away)] = {
            "date": iso, "home": home, "away": away, "api_fixture_id": fid,
            "fetched_at": now.isoformat(),
            # Both squads verbatim, so a suspicious join can be re-checked against
            # what the feed actually said rather than argued about.
            "api_squads": squad_raw,
            "players": matched,
        }
        added += 1
        print(f"  {home} v {away} ({iso}): {len(matched)} pick player(s) matched")

    if added:
        cache["_note"] = (
            "Per-match player goals/shots/SOT from API-Football, used ONLY where "
            "Understat has not published the fixture. Written by "
            "scripts/sync_player_stats.py -- never by hand. `players` is keyed by "
            "OUR player name; `api_squads` keeps the feed's own spelling so a join "
            "can be audited. Understat wins wherever it has the match.")
        tmp = CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(CACHE)
        print(f"wrote {added} fixture(s) to {CACHE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
