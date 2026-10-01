"""Which club names The Odds API uses that leagues/names.py cannot map.

FREE: one /events call per league (no credits). scripts/sync_soccer_shot_odds.py
joins a fixture to its event only when BOTH clubs map, so an unmapped spelling
leaves that fixture's shots picks unpriced. Prints the names to add to ALIASES.
"""
import sys

from leagues import shot_odds
from leagues.names import UnknownTeam, canonical
from oddsapi import client as oc


def main() -> int:
    client = oc.Client()
    missing = {}
    for league, sport in shot_odds.SPORT_KEYS.items():
        events = client.get(f"sports/{sport}/events", sport=league.lower(),
                            purpose=f"{league}: event names") or []
        names = {e.get(k) for e in events for k in ("home_team", "away_team")} - {None}
        bad = []
        for n in sorted(names):
            try:
                canonical(n, league)
            except UnknownTeam:
                bad.append(n)
        missing[league] = bad
        print(f"{league}: {len(events)} events, {len(names)} clubs, unmapped: {bad or 'none'}")
    print(client.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
