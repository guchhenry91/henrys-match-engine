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

    # WHICH GAMES, IF ANY, ARE PRICED? Week 1 is 14 days out and returned nothing,
    # which could mean "this API has no NFL odds" or "books have not posted yet".
    # Those need completely different responses from me, so ask the odds endpoint
    # itself what it holds rather than guessing from one empty answer.
    try:
        priced = client.get("odds", league=1, season=2026)
        print(f"  odds rows for the whole season: {len(priced)}")
        if priced:
            seen = []
            for row in priced[:40]:
                game = row.get("game") or {}
                books = [b.get("name") for b in (row.get("bookmakers") or [])]
                seen.append((game.get("id"), game.get("date"), books[:6]))
            for s in seen[:12]:
                print("   ", s)
            sample = priced[0]
            for b in (sample.get("bookmakers") or []):
                names = [x.get("name") for x in (b.get("bets") or [])]
                print(f"   {b.get('name')}: {len(names)} markets -> {names[:12]}")
                for bet in (b.get("bets") or [])[:4]:
                    vals = [(v.get("value"), v.get("odd"))
                            for v in (bet.get("values") or [])[:4]]
                    print(f"      {bet.get('name')}: {vals}")
    except Exception as exc:
        print(f"  season-wide odds lookup FAILED: {exc}")

    print(f"\n{client.report()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
