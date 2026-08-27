"""One-off reconnaissance of API-NFL's odds coverage. Prints, changes nothing.

Written before the odds integration rather than during it, because the two things
that decide whether "beat the books" is even possible here cannot be assumed:

  1. Is bet365 among the bookmakers this account can see?
  2. Are PLAYER PROP markets offered, or only game lines?

If it is only game lines, the team-winner board can be priced against a real book
and the four player markets cannot -- and it would be dishonest to ship an "edge"
column on props that is secretly computed against something else. Better to find
that out from the API than to discover it after building on the assumption.

Deliberately cheap: a handful of requests, and it reports what it spent.
"""
import json
import sys

from nfl import api, data, publish


def show(title, rows, limit=25):
    print(f"\n=== {title} ({len(rows)} rows) ===")
    for row in rows[:limit]:
        print("   ", json.dumps(row)[:220])


def main():
    if not api.available():
        print("API_NFL_KEY not set or allowance spent; nothing probed")
        return 0
    client = api.Client(budget=12)

    try:
        books = client.get("odds/bookmakers")
        show("bookmakers", books, limit=60)
        bet365 = [b for b in books
                  if "365" in str(b.get("name", "")).lower()]
        print(f"\n  bet365 present: {bool(bet365)} -> {bet365}")
    except Exception as exc:
        print(f"bookmakers lookup failed: {exc}")
        bet365 = []

    try:
        bets = client.get("odds/bets")
        show("bet types offered", bets, limit=80)
        prop_like = [b for b in bets if any(
            k in str(b.get("name", "")).lower()
            for k in ("player", "passing", "rushing", "receiving", "touchdown",
                      "anytime"))]
        print(f"\n  PLAYER-PROP bet types: {len(prop_like)}")
        for b in prop_like[:30]:
            print("     ", b)
    except Exception as exc:
        print(f"bet types lookup failed: {exc}")

    # One real upcoming game, to see the actual shape of a priced market.
    try:
        schedule = data.games(seasons=(2026,))
        upcoming = publish.upcoming_games(schedule)
        if not upcoming.empty:
            row = upcoming.iloc[0]
            print(f"\n  probing odds for {row['away_team']} @ {row['home_team']} "
                  f"week {row['week']}")
            fixtures = client.get("games", league=1, season=2026,
                                  date=str(row["gameday"].date()))
            print(f"  games that day: {len(fixtures)}")
            if fixtures:
                gid = ((fixtures[0].get("game") or {}).get("id")
                       or fixtures[0].get("id"))
                print(f"  game id: {gid}")
                odds = client.get("odds", game=gid)
                show("odds payload", odds, limit=3)
                if odds:
                    book_names = [b.get("name") for o in odds
                                  for b in (o.get("bookmakers") or [])]
                    print(f"\n  bookmakers quoting this game: {sorted(set(book_names))}")
                    for o in odds:
                        for b in (o.get("bookmakers") or []):
                            if "365" in str(b.get("name", "")).lower():
                                print(f"\n  BET365 markets: "
                                      f"{[x.get('name') for x in (b.get('bets') or [])]}")
    except Exception as exc:
        print(f"game odds probe failed: {exc}")

    print(f"\n{client.report()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
