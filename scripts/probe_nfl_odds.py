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

    # Does a real upcoming game actually have PRICES yet? Several lookups,
    # because the games endpoint's parameters differ by sport and a wrong one
    # returns an empty list rather than an error -- which reads exactly like "no
    # games" and is how the first probe learned nothing while spending three calls.
    gid = None
    for label, params in (("by week", {"league": 1, "season": 2026, "week": 1}),
                          ("by season", {"league": 1, "season": 2026})):
        try:
            fixtures = client.get("games", **params)
        except Exception as exc:
            print(f"  games {label}: FAILED {exc}")
            continue
        print(f"  games {label}: {len(fixtures)}")
        if fixtures:
            first = fixtures[0]
            print("   sample:", json.dumps(first)[:280])
            inner = first.get("game") if isinstance(first.get("game"), dict) else None
            gid = (inner or {}).get("id") or first.get("id")
            print(f"   game id: {gid}")
            break

    if not gid:
        print("  no game id resolved; cannot probe prices")
    else:
        for label, params in (("bet365", {"game": gid, "bookmaker": 4}),
                              ("all books", {"game": gid})):
            try:
                odds = client.get("odds", **params)
            except Exception as exc:
                print(f"  odds {label}: FAILED {exc}")
                continue
            print(f"  odds {label}: {len(odds)} row(s)")
            if not odds:
                continue
            print("   raw:", json.dumps(odds[0])[:700])
            for o in odds:
                for b in (o.get("bookmakers") or []):
                    names = [x.get("name") for x in (b.get("bets") or [])]
                    print(f"   {b.get('name')}: {len(names)} markets -> {names[:12]}")
            break

    print(f"\n{client.report()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
