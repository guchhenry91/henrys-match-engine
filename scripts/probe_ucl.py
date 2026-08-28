"""Reconnaissance on Champions League data. Prints, changes nothing.

Three things decide whether a UCL board is buildable at all, and none can be
assumed:

  1. How many past seasons does the API actually hold? A backtest needs a decade.
  2. Are finished fixtures returned with scores, or only the fixture list?
  3. How many of the 36 qualified clubs appear in enough of that history to have a
     strength worth fitting? This is the one that matters most -- Bodo/Glimt,
     Viking and Sabah do not have ten years of European football behind them, and
     a model that quietly seeds them from nothing would publish confident nonsense
     about a third of the field.

Probing before building, because twice this week I built on an assumed feed shape
and had to tear it out: the API-NFL roster endpoint looked complete and was not,
and the odds endpoint has no prices at all.

One request per season, so a fifteen-season probe costs fifteen calls.
"""
import sys
from collections import Counter

from leagues.api_football import Client

UCL_LEAGUE_ID = 2
SEASONS = list(range(2011, 2027))

# The 36 clubs drawn on 2026-08-27, by pot. Spellings are the API's problem to
# match, not ours -- this list is what UEFA published.
POT1 = ["Paris Saint Germain", "Bayern Munich", "Real Madrid", "Liverpool",
        "Inter", "Manchester City", "Arsenal", "Barcelona", "Atletico Madrid"]
POT2 = ["Borussia Dortmund", "AS Roma", "Sporting CP", "Aston Villa", "FC Porto",
        "Manchester United", "Club Brugge KV", "Real Betis", "PSV Eindhoven"]
POT3 = ["Feyenoord", "Lille", "Bodo/Glimt", "Napoli", "RB Leipzig", "Villarreal",
        "Fenerbahce", "Shakhtar Donetsk", "Galatasaray"]
POT4 = ["Slavia Praha", "Slovan Bratislava", "VfB Stuttgart", "AEK Athens", "LASK",
        "Como", "Lens", "Viking", "Sabah"]
DRAWN = POT1 + POT2 + POT3 + POT4


def main():
    try:
        client = Client(limit=40)
    except RuntimeError as exc:
        print(f"{exc}; nothing probed")
        return 0

    appearances = Counter()
    print(f"{'season':>7} {'fixtures':>9} {'finished':>9}  teams")
    total = 0
    for season in SEASONS:
        try:
            rows = client.get("fixtures", league=UCL_LEAGUE_ID, season=season)
        except Exception as exc:
            print(f"{season:>7}  FAILED {exc}")
            continue
        finished = 0
        teams = set()
        for row in rows:
            status = ((row.get("fixture") or {}).get("status") or {}).get("short")
            home = ((row.get("teams") or {}).get("home") or {}).get("name")
            away = ((row.get("teams") or {}).get("away") or {}).get("name")
            if home and away:
                teams.update({home, away})
            if status in {"FT", "AET", "PEN"}:
                finished += 1
                for name in (home, away):
                    if name:
                        appearances[name] += 1
        total += finished
        print(f"{season:>7} {len(rows):>9} {finished:>9}  {len(teams)}")

    print(f"\ntotal finished matches across probed seasons: {total}")

    print("\n=== HOW MUCH HISTORY DOES EACH DRAWN CLUB HAVE? ===")
    print("(matches found under the API's own spelling; a 0 may be a NAME "
          "mismatch rather than a club with no European history)")
    thin = []
    for pot, names in (("1", POT1), ("2", POT2), ("3", POT3), ("4", POT4)):
        print(f"\n  Pot {pot}")
        for name in names:
            exact = appearances.get(name, 0)
            near = [k for k in appearances
                    if name.split()[0].lower() in k.lower()][:2] if not exact else []
            note = f"  (near: {near})" if near else ""
            print(f"    {name:24s} {exact:4d}{note}")
            if exact < 20:
                thin.append(name)
    print(f"\nclubs with <20 matches of history: {len(thin)} of {len(DRAWN)}")
    print(f"  {thin}")
    print(f"\nAPI-Football requests used: {client.used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
