"""One-off reconnaissance of API-NFL's odds coverage. Prints, changes nothing.

Written before the odds integration rather than during it, because the two things
that decide whether "beat the books" is even possible here cannot be assumed:

  1. Is bet365 among the bookmakers this account can see?
  2. Are PLAYER PROP markets offered, or only game lines?

If it is only game lines, the team-winner board can be priced against a real book
and the four player markets cannot -- and it would be dishonest to ship an "edge"
column on props that is secretly computed against something else. Better to find
that out from the API than to discover it after building on the assumption.

WHAT THE 2026-08-30 RUN ESTABLISHED. Bet365 IS visible (bookmaker id 4), and 185
bet types mention a player. But the `date` parameter this used to query by DOES
NOT EXIST -- five dates all returned "The Date field do not exist." So the odds
endpoint is now asked the way it actually accepts: by GAME, which needs the API's
own fixture id rather than nflverse's.

Deliberately cheap: a handful of requests, and it reports what it spent.
"""
import json
import sys

from nfl import api


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
        bet365 = [b for b in books if "365" in str(b.get("name", "")).lower()]
        print(f"\n  bet365 present: {bool(bet365)} -> {bet365}")
    except Exception as exc:
        print(f"bookmakers lookup failed: {exc}")

    bets = []
    try:
        bets = client.get("odds/bets")
        print(f"\n=== bet types offered: {len(bets)} ===")
        prop_like = [b for b in bets if any(
            k in str(b.get("name", "")).lower()
            for k in ("player", "passing", "rushing", "receiving", "touchdown",
                      "anytime"))]
        print(f"  player-ish bet types: {len(prop_like)}")
        # The three yardage markets the board publishes are the ones that decide
        # whether an edge column can cover props at all, so they get named.
        yard = [b for b in bets if "yard" in str(b.get("name", "")).lower()]
        print(f"\n  bet types mentioning 'yard': {len(yard)}")
        for b in yard[:40]:
            print("     ", b)
        td = [b for b in bets if "touchdown scorer" in str(b.get("name", "")).lower()
              or "anytime" in str(b.get("name", "")).lower()]
        print(f"\n  anytime/touchdown-scorer bet types: {len(td)}")
        for b in td[:20]:
            print("     ", b)
    except Exception as exc:
        print(f"bet types lookup failed: {exc}")

    print("\n=== locating an upcoming fixture in the API's own schedule ===")
    gid = None
    try:
        games = client.get("games", league=1, season=2026)
        print(f"  API returned {len(games)} games")
        upcoming = []
        for g in games:
            fixture = g.get("game") or g
            status = ((fixture.get("status") or {}).get("short")
                      or ((g.get("status") or {}).get("short")))
            teams = g.get("teams") or {}
            if status in (None, "NS"):
                upcoming.append((fixture.get("id") or g.get("id"),
                                 str(fixture.get("date"))[:40],
                                 (teams.get("away") or {}).get("name"),
                                 (teams.get("home") or {}).get("name")))
        print(f"  {len(upcoming)} not-started; first few:")
        for row in upcoming[:5]:
            print("     ", row)
        if upcoming:
            gid = upcoming[0][0]
    except Exception as exc:
        print(f"  games lookup failed: {exc}")

    if gid:
        for label, params in (("bet365 only", {"game": gid, "bookmaker": 4}),
                              ("all books", {"game": gid})):
            print(f"\n=== odds for game {gid} ({label}) ===")
            try:
                rows = client.get("odds", **params)
                print(f"  {len(rows)} record(s)")
                if not rows:
                    print("  EMPTY -- no prices posted for this fixture yet")
                    continue
                for book in (rows[0].get("bookmakers") or [])[:2]:
                    allbets = book.get("bets") or []
                    print(f"    {book.get('name')}: {len(allbets)} markets")
                    for x in allbets[:60]:
                        print(f"       - {x.get('id')} {x.get('name')}")
                    for bet in allbets:
                        nm = str(bet.get("name", "")).lower()
                        if any(k in nm for k in ("yard", "touchdown scorer",
                                                 "anytime")):
                            print(f"    SAMPLE {bet.get('name')} "
                                  f"(id {bet.get('id')}):")
                            print("      " + json.dumps(
                                (bet.get("values") or [])[:6])[:900])
                            break
            except Exception as exc:
                print(f"  FAILED: {exc}")

    print(f"\n{client.report()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
