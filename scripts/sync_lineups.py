"""Fetch confirmed XIs shortly before kickoff without wasting the daily quota.

POLLED ON A LADDER: 60, 45, 30 and 15 minutes before kickoff, one attempt per
rung. A single attempt in a 40-minute window was too brittle -- clubs publish
late and unevenly, so one poll that happened to land before a slow club released
its XI got nothing and never looked again, and the props for that fixture went
out on appearance probabilities instead of the actual eleven.

Quota is protected by stopping early rather than by refusing to look: a fixture
whose BOTH XIs are already confirmed is skipped on every later rung, so the extra
polls are only ever spent on the fixtures still missing one. Most resolve at 60
or 45 and never reach the lower rungs.

NOTE ON ORDERING. publish.py locks a pick at LOCK_WINDOW_HOURS (60 minutes), so a
pick can freeze before its XI arrives. That is deliberate -- see
publish._player_pick_publishable: requiring a confirmed XI to lock was tried and
acted as a kill switch. A confirmed XI arriving after the lock still updates the
PROPS through props.match_props; it just cannot move the frozen match pick.
"""
import json
import os
from datetime import datetime, timezone

from leagues.api_football import Client
from leagues.names import canonical, UnknownTeam
from leagues.team_news import NEWS_PATH, upcoming_fixtures
from scripts.sync_rosters import API_LEAGUES

# Minutes before kickoff at which to poll for a confirmed XI, highest first.
RUNGS = (60, 45, 30, 15)


def main(now=None):
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("API_FOOTBALL_KEY is not set; skipping confirmed lineups")
        return 0
    now = now or datetime.now(timezone.utc)
    imminent = []
    for fixture in upcoming_fixtures(now=now):
        kickoff = datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
        minutes = (kickoff - now).total_seconds() / 60
        # SMALLEST rung at or above the time remaining, so consecutive runs consume
        # different rungs: 58->60, 43->45, 28->30, 12->15. Iterating RUNGS as
        # written (descending) matched 60 for everything, spending one rung and
        # silently skipping the other three.
        rung = next((r for r in sorted(RUNGS) if r >= minutes), None)
        if rung is not None and minutes > 0:
            imminent.append((fixture, kickoff, rung))
    if not imminent:
        print(f"no fixtures inside the {RUNGS[0]}-minute lineup window; no quota used")
        return 0

    news = json.loads(NEWS_PATH.read_text(encoding="utf-8")) if NEWS_PATH.exists() else {}
    client = Client(limit=60)   # four rungs need more headroom than one poll did
    by_date = {}
    changed = False
    confirmed = 0
    for fixture, kickoff, rung in imminent:
        league = fixture["league_key"]
        section = news.setdefault(league, {})
        if all((section.get(team) or {}).get("lineup_confirmed") is True
               for team in (fixture["home"], fixture["away"])):
            continue
        date = kickoff.date().isoformat()
        if date not in by_date:
            by_date[date] = client.get("fixtures", date=date, timezone="UTC")
        match = None
        for candidate in by_date[date]:
            if candidate.get("league", {}).get("id") != API_LEAGUES[league]:
                continue
            try:
                home = canonical(candidate["teams"]["home"]["name"], league)
                away = canonical(candidate["teams"]["away"]["name"], league)
            except UnknownTeam:
                continue
            if (home, away) == (fixture["home"], fixture["away"]):
                match = candidate
                break
        if not match:
            print(f"WARNING: API-Football fixture not matched: {league} "
                  f"{fixture['home']} v {fixture['away']}")
            continue
        fixture_id = match["fixture"]["id"]
        # ONE ATTEMPT PER RUNG, not one per fixture. The marker records which rungs
        # have been spent on THIS fixture, so a 15-minute cadence cannot burn the
        # same rung twice while the ladder still gets its four looks.
        done = set()
        for team in (fixture["home"], fixture["away"]):
            e = section.get(team) or {}
            if e.get("lineup_api_attempted_fixture") == fixture_id:
                done |= set(e.get("lineup_api_rungs") or [])
        if rung in done:
            continue
        try:
            lineups = client.get("fixtures/lineups", fixture=fixture_id)
        except RuntimeError as exc:
            # Out of per-run budget. Stop cleanly and keep what was gathered --
            # this step must never fail the publish it precedes.
            print(f"lineup poll stopped early ({exc}); {confirmed} confirmed so far")
            break
        for team in (fixture["home"], fixture["away"]):
            section.setdefault(team, {}).update({
                "lineup_api_attempted_fixture": fixture_id,
                "lineup_api_attempted_at": now.isoformat(),
                "lineup_api_rungs": sorted(done | {rung}, reverse=True),
            })
        changed = True
        if len(lineups) != 2:
            print(f"lineup not published yet at the {rung}-minute rung: "
                  f"{fixture['home']} v {fixture['away']}")
            continue
        for row in lineups:
            try:
                team = canonical(row["team"]["name"], league)
            except UnknownTeam:
                continue
            starters = [item["player"]["name"] for item in row.get("startXI", [])]
            bench = [item["player"]["name"] for item in row.get("substitutes", [])]
            if len(starters) != 11:
                continue
            entry = section.setdefault(team, {})
            entry.update({"starters": starters, "bench": bench,
                          "lineup_confirmed": True,
                          "lineup_checked_at": now.isoformat(),
                          "lineup_source": "API-Football",
                          "lineup_fixture_id": fixture_id})
            confirmed += 1
    if changed:
        tmp = NEWS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(news, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(NEWS_PATH)
    print(f"confirmed {confirmed} team lineups; {client.used} API-Football requests used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
