"""Does the API serve PLAYER PROP prices, or only game lines? Prints, changes nothing.

WHY THIS IS A SEPARATE QUESTION FROM "are there odds". As of 2026-09-08 the
moneyline works: sync_nfl_odds reported "priced 2 fixture(s) from ['Bet365']" and
data-raw/nfl/odds.json holds real de-vigged prices for SEA|NE and LA|SF. But
nothing in the repo has ever asked for a PROP, and the board's four markets are
props. A game line arriving says nothing about whether Player Receiving Yards
does -- that is exactly the mistake the odds work made once already, reading
"bet365 is in the bookmaker list and 185 prop bet types exist" as evidence that
prop prices would be served. They were not.

IT ASKS ABOUT A GAME KNOWN TO BE PRICED, not whichever fixture the API lists
first. An empty answer for an unpriced game proves nothing; an empty answer for
one whose moneyline is already quoted is real evidence.

Cheap: one request to map fixtures, then one per game asked about.
"""
import json
import sys
from collections import Counter

from nfl import api, config, data, odds, publish

WANTED = {bet: market
          for market, bets in odds.PLAYER_PROP_BETS.items()
          for bet in bets}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Reconnaissance must never fail the run that hosts it.
    try:
        return probe()
    except Exception as exc:
        print(f"PROBE FAILED: {type(exc).__name__}: {exc}")
        print("  If this is a client/programming error it says nothing about "
              "the API -- read the exception type before concluding.")
        return 0


def probe() -> int:
    import pandas as pd

    client = api.Client(budget=12)   # NFL client caps with `budget`; API-Football uses `limit`
    schedule = pd.concat([data.games(), data.games(seasons=(config.CURRENT_SEASON,))],
                         ignore_index=True)
    upcoming = publish.upcoming_games(schedule)
    print(f"upcoming fixtures: {len(upcoming)}")

    from scripts.sync_nfl_odds import upcoming_ids
    ids = upcoming_ids(client, upcoming)
    print(f"matched {len(ids)} of {len(upcoming)} fixtures to API game ids")

    # The two the moneyline sync already priced, soonest first, so an empty
    # answer here cannot be explained away by "the book has not posted".
    kickoff = {(r["home_team"], r["away_team"]): r["kickoff"]
               for _, r in upcoming.iterrows()}
    order = sorted(ids, key=lambda k: str(kickoff.get(k, "")))
    for pair in order[:2]:
        gid = ids[pair]
        home, away = pair
        print(f"\n=== {away} @ {home}  (game {gid}, kickoff {kickoff.get(pair)}) ===")
        rows = client.get("odds", game=gid)
        if not rows:
            print("  no odds records at all for this fixture")
            continue
        for book in rows[0].get("bookmakers") or []:
            bets = book.get("bets") or []
            print(f"  {book.get('name')}: {len(bets)} market(s)")
            names = Counter()
            for bet in bets:
                names[(bet.get("id"), str(bet.get("name")))] += 1
            for (bid, name), _ in names.most_common(60):
                mark = f"   <-- {WANTED[bid]}" if bid in WANTED else ""
                print(f"     {str(bid):>5}  {name[:46]}{mark}")

            # THE ACTUAL QUESTION: receiving and passing yards, by id.
            for market in ("receiving_yards", "passing_yards"):
                ids_for = odds.PLAYER_PROP_BETS[market]
                hit = next((b for b in bets if b.get("id") in ids_for), None)
                if not hit:
                    print(f"    {market}: NOT OFFERED by {book.get('name')}")
                    continue
                values = hit.get("values") or []
                print(f"    {market}: {len(values)} value(s) under "
                      f"{hit.get('name')!r} (id {hit.get('id')})")
                print("      " + json.dumps(values[:6])[:500])

    print(f"\n{client.report()}")
    print("Nothing was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
