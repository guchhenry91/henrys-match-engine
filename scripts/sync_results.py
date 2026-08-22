"""Pull FINAL scores from API-Football so grading does not need a laptop.

WHY THIS EXISTS. fixturedownload is the fixture feed and, in the 2026-27 season,
has published almost no results: 1 of 380 a week in, hours or days after full
time. Every grade so far has come from results_override.json, which is written by
the local scheduled task -- so the record only advanced while the owner's machine
was on, and several finished matches sat ungraded whenever it was not.

API-Football already provides confirmed lineups here (scripts/sync_lineups.py)
and its key is a GitHub Actions secret, so the same client can close the loop
entirely in the cloud.

IT WRITES THE SAME OVERRIDE FILE, deliberately. leagues.fixtures already prefers
the real feed and falls back to an override only where the feed's score is
missing, reporting any disagreement loudly. Reusing that path means this adds a
source, not a second mechanism with its own rules.

ONLY GENUINELY FINISHED MATCHES. A live or half-time score written as final would
grade a pick against a scoreline that had not happened yet, into an append-only
record. FINISHED holds the only statuses that mean full time.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from leagues.api_football import Client
from leagues.names import canonical, UnknownTeam
from scripts.sync_rosters import API_LEAGUES

ROOT = Path(__file__).resolve().parents[1]
OVERRIDE = ROOT / "data-raw" / "leagues" / "results_override.json"
OUT = ROOT / "data" / "leagues"
FILE_FOR = {"PL": "pl", "LALIGA": "laliga", "BUNDESLIGA": "bundesliga", "LIGUE1": "ligue1"}

# Full time, after extra time, and after penalties. Everything else -- NS, 1H,
# HT, 2H, LIVE, PST, CANC, SUSP -- is not a result.
FINISHED = {"FT", "AET", "PEN"}
# Leave a match alone until it has had time to finish: 90 minutes plus stoppage,
# half time and any delay.
MIN_MINUTES_AFTER_KICKOFF = 130


def ungraded(now):
    """Fixtures that kicked off long enough ago to be over and still have no score."""
    out = []
    for league, fn in FILE_FOR.items():
        p = OUT / f"{fn}.json"
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        for m in payload.get("matches", []) + payload.get("season", []):
            if m.get("result") is not None:
                continue
            try:
                ko = datetime.fromisoformat(str(m["date"]).replace("Z", "+00:00"))
            except Exception:
                continue
            if (now - ko) >= timedelta(minutes=MIN_MINUTES_AFTER_KICKOFF):
                out.append((league, m["home"], m["away"], ko))
    # de-duplicate: a fixture appears in both `matches` and `season`
    return sorted(set(out), key=lambda x: x[3])


def main(now=None):
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("API_FOOTBALL_KEY is not set; skipping result sync")
        return 0
    now = now or datetime.now(timezone.utc)
    todo = ungraded(now)
    if not todo:
        print("no finished-but-ungraded fixtures; no quota used")
        return 0

    data = json.loads(OVERRIDE.read_text(encoding="utf-8")) if OVERRIDE.exists() else {}
    client = Client(limit=40)
    by_date, added = {}, 0
    for league, home, away, ko in todo:
        if (data.get(league) or {}).get(f"{home}|{away}"):
            continue                                   # already recorded
        date = ko.date().isoformat()
        try:
            if date not in by_date:
                by_date[date] = client.get("fixtures", date=date, timezone="UTC")
        except RuntimeError as exc:
            print(f"result sync stopped early ({exc}); {added} added so far")
            break
        for c in by_date[date]:
            if c.get("league", {}).get("id") != API_LEAGUES.get(league):
                continue
            try:
                h = canonical(c["teams"]["home"]["name"], league)
                a = canonical(c["teams"]["away"]["name"], league)
            except UnknownTeam:
                continue
            if (h, a) != (home, away):
                continue
            status = (c.get("fixture", {}).get("status") or {}).get("short")
            if status not in FINISHED:
                print(f"  not finished ({status}): {home} v {away}")
                break
            g = c.get("goals") or {}
            if not isinstance(g.get("home"), int) or not isinstance(g.get("away"), int):
                break
            data.setdefault(league, {})[f"{home}|{away}"] = {
                "home_goals": g["home"], "away_goals": g["away"], "status": "FT",
                # PROVENANCE, and it matters. Hand-written entries in this file
                # carry two independent sources because an agent reading match
                # reports can hallucinate a scoreline. This entry was not read,
                # it was fetched: a structured feed reporting integer goals under
                # a finished status cannot misread itself the way prose can, and
                # it can only ever cite one source -- itself. `auto` marks the
                # difference so a single-source machine entry is never mistaken
                # for a two-source verified one, and so the evidence test can
                # hold each kind to the bar that actually applies to it.
                "auto": True,
                "api_fixture_id": c["fixture"]["id"],
                "api_status": status,
                "sources": [f"API-Football fixture {c['fixture']['id']}: "
                            f"{home} {g['home']}-{g['away']} {away}, status {status}"],
            }
            added += 1
            print(f"  {home} {g['home']}-{g['away']} {away}  ({status})")
            break

    if added:
        data["_verified_on"] = now.date().isoformat()
        tmp = OVERRIDE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(OVERRIDE)
        print(f"wrote {added} result(s) to {OVERRIDE.name}")
    else:
        print("nothing new to record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
