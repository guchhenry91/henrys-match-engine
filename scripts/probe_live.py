"""Does API-Football serve LIVE fixtures on this account? Prints, changes nothing.

WHY THIS RUNS BEFORE ANY LIVE-SCORE CODE. The NFL odds integration was designed
around bet365 being listed in `odds/bookmakers` and every market existing in
`odds/bets` -- both true, and the endpoint returned zero prices for every fixture
anyway. Days went into a feed that had nothing in it. A catalogue entry is not
coverage, and the only way to tell them apart is to ask.

`fixtures?live=all` returns EVERY in-play match in one request, which is what
makes polling affordable: one call per poll regardless of how many games are on,
and with a short server-side cache, regardless of how many people are watching.

READ THE RESULT CAREFULLY.
  * An ERROR -- "not available for your plan", an empty body, a rejected
    parameter -- is decisive: no live scores on this key.
  * RECORDS returned is decisive the other way, and shows the shape.
  * ZERO records with NO error means the endpoint is permitted and nothing is
    kicking a ball right now. That proves permission and says nothing about
    coverage of our five leagues. Re-run it during a matchday window before
    concluding anything about which competitions appear.
"""
import json
import sys
from collections import Counter

from leagues import config
from leagues.api_football import Client
from scripts.sync_rosters import API_LEAGUES

# 2 = UEFA Champions League in API-Football's own numbering, alongside the five
# domestic ids the roster sync already uses.
WANTED = {**API_LEAGUES, "UCL": 2}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    client = Client(budget=4)
    print("=== fixtures?live=all ===")
    try:
        rows = client.get("fixtures", live="all")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        print("\n  DECISIVE: the endpoint is not usable on this key.")
        print(client.report())
        return 0

    print(f"  {len(rows)} live fixture(s) right now")
    if not rows:
        print("  EMPTY, but NO ERROR -- the endpoint is permitted and nothing is")
        print("  in play at the moment. This proves PERMISSION, not coverage:")
        print("  re-run during a matchday window to see which leagues appear.")
        print(client.report())
        return 0

    leagues = Counter()
    for row in rows:
        lg = row.get("league") or {}
        leagues[(lg.get("id"), lg.get("name"))] += 1
    print("\n  leagues in play:")
    for (lid, name), n in leagues.most_common(20):
        ours = next((k for k, v in WANTED.items() if v == lid), None)
        mark = f"  <-- {ours}" if ours else ""
        print(f"    {str(lid):>5}  {str(name)[:38]:<38} {n}{mark}")

    mine = [r for r in rows
            if (r.get("league") or {}).get("id") in set(WANTED.values())]
    print(f"\n  of ours: {len(mine)}")
    if mine:
        row = mine[0]
        fixture, goals = row.get("fixture") or {}, row.get("goals") or {}
        status = fixture.get("status") or {}
        teams = row.get("teams") or {}
        print("\n  SAMPLE -- the fields a live card needs:")
        print(f"    id       {fixture.get('id')}")
        print(f"    teams    {(teams.get('home') or {}).get('name')} v "
              f"{(teams.get('away') or {}).get('name')}")
        print(f"    score    {goals.get('home')}-{goals.get('away')}")
        print(f"    status   {status.get('short')!r} elapsed={status.get('elapsed')!r}")
        print("\n    raw:", json.dumps(row)[:600])

    print(f"\n{client.report()}")
    print("\nNothing was written. Live scores must never reach the picks log or "
          "the record -- sync_results gates on FT/AET/PEN for that reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
